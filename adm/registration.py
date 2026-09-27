from .io import write_array, write_json, write_npz
from pathlib import Path
import json
import time
from functools import lru_cache
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .geometry import project
import cv2
from .vocabulary import match_corners
from .io import write_image

def collect_matches(source,names,metadata,pairs,cfg,out,geo=None,tile_index=None,gpu=None,save_cache=None):
    started=time.perf_counter();descriptor_seconds=0.;enhancement_seconds=0.
    sift=cv2.SIFT_create(nfeatures=cfg.feature_count,contrastThreshold=.025)
    keys=[]; desc=[]; scales=[]
    diagnostics=cfg.export_profile=='diagnostic'
    if save_cache is None:save_cache=diagnostics
    vocab_dir=out/'vocabulary'
    if diagnostics:vocab_dir.mkdir(exist_ok=True)
    from .vocabulary import corner_vocab
    for name,m in zip(names,metadata):
        im=cv2.imread(str(source/name)); h=round(m.height*cfg.registration_width/m.width)
        if cfg.input_upscale_passes:
            from .enhancement import upscale
            tick=time.perf_counter()
            im,_=upscale(im,cfg.input_upscale_passes,cfg.fsr_strength,gpu)
            enhancement_seconds+=time.perf_counter()-tick
        small=cv2.resize(im,(cfg.registration_width,h),interpolation=cv2.INTER_AREA)
        key,d=sift.detectAndCompute(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),None)
        keys.append(np.array([k.pt for k in key])); desc.append(d); scales.append(np.diag([cfg.registration_width/m.width,h/m.height,1.]))
        if diagnostics:
            origins,raw,delta=corner_vocab(im[:,:,::-1],cfg.vocabulary_size)
            write_npz(vocab_dir/f'{Path(name).stem}.npz',origins=origins,rgb=raw,delta=delta)
            preview=np.concatenate([np.clip((p+.5)*255,0,255).astype(np.uint8) for p in delta],axis=1)
            write_image(vocab_dir/f'{Path(name).stem}_delta.png',preview[:,:,::-1])
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=64)) if gpu is None else None
    device_desc=None if gpu is None else [None if d is None else gpu.descriptors(d) for d in desc]
    if gpu is not None:gpu.sync()
    prepared=time.perf_counter()
    @lru_cache(maxsize=3)
    def image(i):
        # Decode only pairs that need full-resolution corner verification.
        return cv2.imread(str(source/names[i]))
    edges=[]; corner_report=[]; rejected=[];evaluated=accepted_corners=0
    from .match_graph import valid_transform
    for pair_index,(i,j) in enumerate(pairs):
        if pair_index%250==0:print(f'Matching {pair_index}/{len(pairs)} pairs; {len(edges)} verified',flush=True)
        candidate=None; records=[]; method='SIFT fallback'
        if geo is not None:
            predicted=np.linalg.inv(geo[j])@geo[i]
            records=match_corners(image(i),image(j),predicted,cfg,geo[i],geo[j],tile_index,j,gpu=gpu)
            accepted=[r for r in records if r['accepted']]
            if len(accepted)>=4:
                a=np.array([r['source'] for r in accepted]); b=np.array([r['target'] for r in accepted])
                candidate,mask=cv2.findHomography(a,b,cv2.RANSAC,3)
                if candidate is not None:
                    a=project(scales[i],a); b=project(scales[j],b); method='four corner delta vocabulary'
        if candidate is None and cfg.visual_fallback:
            if desc[i] is None or desc[j] is None: continue
            matching_started=time.perf_counter()
            if gpu is None:
                matches=matcher.knnMatch(desc[i],desc[j],k=2)
                good=[group[0] for group in matches if len(group)==2 and group[0].distance<.70*group[1].distance]
                query=np.array([m.queryIdx for m in good],dtype=np.intp)
                target=np.array([m.trainIdx for m in good],dtype=np.intp)
            else:
                nearest=gpu.match_descriptors(device_desc[i],device_desc[j])
                query=np.flatnonzero(nearest[:,2]<.70**2*nearest[:,3])
                target=nearest[query,0].astype(np.intp)
            descriptor_seconds+=time.perf_counter()-matching_started
            if len(query)<35: continue
            a=keys[i][query]; b=keys[j][target]
            candidate,mask=cv2.findHomography(a,b,cv2.RANSAC,2.5)
            if candidate is None: continue
            ok=mask.ravel().astype(bool); a,b=a[ok],b[ok]
            if len(a)<30 or len(a)/len(query)<.25: continue
            hull=cv2.contourArea(cv2.convexHull(a.astype(np.float32)))
            if hull<cfg.registration_width*(metadata[i].height*cfg.registration_width/metadata[i].width)*.035: continue
            if geo is not None:
                native_a=project(np.linalg.inv(scales[i]),a); native_b=project(np.linalg.inv(scales[j]),b)
                discrepancies=np.linalg.norm(project(geo[i],native_a)-project(geo[j],native_b),axis=1)
                if np.median(discrepancies)>cfg.gps_tolerance_m:
                    rejected.append({'i':i,'j':j,'reason':'visual matches contradict geographic prior','median_m':float(np.median(discrepancies))}); continue
            elif diagnostics:
                predicted=np.linalg.inv(scales[j])@candidate@scales[i]
                records=match_corners(image(i),image(j),predicted,cfg,gpu=gpu)
        if candidate is None: continue
        H,mask=cv2.findHomography(a,b,0)
        if not valid_transform(H,cfg.registration_width,
            round(metadata[i].height*cfg.registration_width/metadata[i].width),
            round(metadata[j].height*cfg.registration_width/metadata[j].width)):continue
        err=np.linalg.norm(project(H,a)-b,axis=1)
        ix=np.linspace(0,len(a)-1,min(160,len(a))).astype(int)
        edges.append(dict(i=i,j=j,H=H,a=a[ix],b=b[ix],n=len(a),median=float(np.median(err)),method=method))
        attempted=[r for r in records if 'delta_rmse' in r]
        evaluated+=len(attempted);accepted_corners+=sum(r['accepted'] for r in attempted)
        if diagnostics:corner_report.append({'source':names[i],'target':names[j],'method':method,'independent_geography':geo is not None,'corners':records})
        if diagnostics or pair_index%100==0 or pair_index+1==len(pairs):
            print('match',pair_index+1,'/',len(pairs),names[i],names[j],len(a),method,flush=True)
    from .match_graph import recover
    edges,recovery=recover(source,names,metadata,pairs,edges,cfg,out,gpu=gpu,geo=geo)
    matched=time.perf_counter()
    if diagnostics:
        write_json(out/'corner_matches.json',corner_report)
        write_json(out/'rejected_pairs.json',rejected)
    if save_cache:write_npz(out/'registration.npz',names=np.array(names),edges=np.array(edges,dtype=object))
    write_json(out/'matching_report.json',dict(candidate_pairs=len(pairs),registered_pairs=len(edges),
        corner_patches_evaluated=evaluated,corner_patches_accepted=accepted_corners,cache_written=bool(save_cache),
        input_enhancement_seconds=enhancement_seconds,
        graph_recovery=recovery,
        timings_seconds=dict(features_and_upload=prepared-started,descriptor_matching=descriptor_seconds,
            verification_and_bookkeeping=matched-prepared-descriptor_seconds,cache_and_diagnostic_export=time.perf_counter()-matched)))
    return edges

def solve(names,edges,out,width=1400,gpu=None,workspace_budget_mb=512,iterations=60,symmetric=False):
    from .match_graph import require_connected
    require_connected(names,edges,out)
    if gpu is not None:
        from .gpu_alignment import solve as solve_gpu
        return solve_gpu(names,edges,out,width,gpu,iterations=iterations,workspace_budget_mb=workspace_budget_mb,
                         **(dict(symmetric=True) if symmetric else {}))
    if symmetric:raise ValueError('Symmetric transfer alignment requires the Metal solver')
    n=len(names); anchor=n//2; transforms={anchor:np.eye(3)}
    while len(transforms)<n:
        candidates=[e for e in edges if (e['i'] in transforms) != (e['j'] in transforms)]
        if not candidates: raise RuntimeError('Disconnected match graph; refusing to omit frames silently')
        e=max(candidates,key=lambda e:e['n'])
        i,j=e['i'],e['j']
        if i in transforms: transforms[j]=transforms[i]@np.linalg.inv(e['H'])
        else: transforms[i]=transforms[j]@e['H']
    ids=[i for i in range(n) if i!=anchor]; offsets={i:k*8 for k,i in enumerate(ids)}
    # Normalize coordinates for well-conditioned projective parameters.
    S=np.diag([1/width,1/width,1.]); Si=np.linalg.inv(S)
    normalized={i:S@H@Si for i,H in transforms.items()}
    x=np.concatenate([(normalized[i]/normalized[i][2,2]).ravel()[:8] for i in ids])
    def unpack(x):
        return {**{anchor:np.eye(3)},**{i:np.r_[x[offsets[i]:offsets[i]+8],1.].reshape(3,3) for i in ids}}
    def residual(x):
        hs=unpack(x)
        return np.concatenate([(project(hs[e['i']],e['a']/width)-project(hs[e['j']],e['b']/width)).ravel() for e in edges])
    rows=sum(len(e['a'])*2 for e in edges); sparsity=lil_matrix((rows,len(x)),dtype=int); row=0
    for e in edges:
        count=len(e['a'])*2
        for i in [e['i'],e['j']]:
            if i!=anchor: sparsity[row:row+count,offsets[i]:offsets[i]+8]=1
        row+=count
    result=least_squares(residual,x,jac_sparsity=sparsity.tocsr(),loss='soft_l1',f_scale=1/width,max_nfev=600,ftol=1e-7)
    hs={i:Si@H@S for i,H in unpack(result.x).items()}
    errors=[]
    for e in edges:
        err=np.linalg.norm(project(hs[e['i']],e['a'])-project(hs[e['j']],e['b']),axis=1)
        errors.extend(err.tolist())
    report=dict(anchor=names[anchor],images=n,edges=len(edges),median_registration_error_px=float(np.median(errors)),p95_registration_error_px=float(np.percentile(errors,95)),solver_success=bool(result.success),solver_message=result.message)
    write_json(out/'registration_report.json',report); print(report,flush=True)
    write_array(out/'transforms.npy',np.array([hs[i] for i in range(n)]))
    return hs
