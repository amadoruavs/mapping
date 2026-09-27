from pathlib import Path
import json
from functools import lru_cache
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .geometry import project
import cv2
from .vocabulary import match_corners
from .io import write_image

def collect_matches(source,names,metadata,pairs,cfg,out,geo=None,tile_index=None,gpu=None):
    sift=cv2.SIFT_create(nfeatures=cfg.feature_count,contrastThreshold=.025)
    keys=[]; desc=[]; scales=[]
    vocab_dir=out/'vocabulary'; vocab_dir.mkdir(exist_ok=True)
    from .vocabulary import corner_vocab
    for name,m in zip(names,metadata):
        im=cv2.imread(str(source/name)); h=round(m.height*cfg.registration_width/m.width)
        small=cv2.resize(im,(cfg.registration_width,h),interpolation=cv2.INTER_AREA)
        key,d=sift.detectAndCompute(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),None)
        keys.append(np.array([k.pt for k in key])); desc.append(d); scales.append(np.diag([cfg.registration_width/m.width,h/m.height,1.]))
        origins,raw,delta=corner_vocab(im[:,:,::-1],cfg.vocabulary_size)
        np.savez_compressed(vocab_dir/f'{Path(name).stem}.npz',origins=origins,rgb=raw,delta=delta)
        preview=np.concatenate([np.clip((p+.5)*255,0,255).astype(np.uint8) for p in delta],axis=1)
        write_image(vocab_dir/f'{Path(name).stem}_delta.png',preview[:,:,::-1])
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=64))
    device_desc=None if gpu is None else [None if d is None else gpu.descriptors(d) for d in desc]
    @lru_cache(maxsize=3)
    def image(i):
        # Decode only pairs that need full-resolution corner verification.
        return cv2.imread(str(source/names[i]))
    edges=[]; corner_report=[]; rejected=[]
    for pair_index,(i,j) in enumerate(pairs):
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
            if gpu is None:
                matches=matcher.knnMatch(desc[i],desc[j],k=2)
                good=[group[0] for group in matches if len(group)==2 and group[0].distance<.70*group[1].distance]
            else:
                nearest=gpu.match_descriptors(device_desc[i],device_desc[j])
                good=[cv2.DMatch(k,int(row[0]),float(np.sqrt(row[2]))) for k,row in enumerate(nearest) if row[2]<.70**2*row[3]]
            if len(good)<35: continue
            a=keys[i][[m.queryIdx for m in good]]; b=keys[j][[m.trainIdx for m in good]]
            candidate,mask=cv2.findHomography(a,b,cv2.RANSAC,2.5)
            if candidate is None: continue
            ok=mask.ravel().astype(bool); a,b=a[ok],b[ok]
            if len(a)<30 or len(a)/len(good)<.25: continue
            hull=cv2.contourArea(cv2.convexHull(a.astype(np.float32)))
            if hull<cfg.registration_width*(metadata[i].height*cfg.registration_width/metadata[i].width)*.035: continue
            native_a=project(np.linalg.inv(scales[i]),a); native_b=project(np.linalg.inv(scales[j]),b)
            if geo is not None:
                discrepancies=np.linalg.norm(project(geo[i],native_a)-project(geo[j],native_b),axis=1)
                if np.median(discrepancies)>cfg.gps_tolerance_m:
                    rejected.append({'i':i,'j':j,'reason':'visual matches contradict geographic prior','median_m':float(np.median(discrepancies))}); continue
            else:
                predicted=np.linalg.inv(scales[j])@candidate@scales[i]
                records=match_corners(image(i),image(j),predicted,cfg,gpu=gpu)
        if candidate is None: continue
        H,mask=cv2.findHomography(a,b,0)
        if H is None: continue
        err=np.linalg.norm(project(H,a)-b,axis=1)
        ix=np.linspace(0,len(a)-1,min(160,len(a))).astype(int)
        edges.append(dict(i=i,j=j,H=H,a=a[ix],b=b[ix],n=len(a),median=float(np.median(err)),method=method))
        corner_report.append({'source':names[i],'target':names[j],'method':method,'independent_geography':geo is not None,'corners':records})
        print('match',pair_index+1,'/',len(pairs),names[i],names[j],len(a),method,flush=True)
    (out/'corner_matches.json').write_text(json.dumps(corner_report,indent=2))
    (out/'rejected_pairs.json').write_text(json.dumps(rejected,indent=2))
    np.savez_compressed(out/'registration.npz',names=np.array(names),edges=np.array(edges,dtype=object))
    return edges

def solve(names,edges,out,width=1400):
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
    (out/'registration_report.json').write_text(json.dumps(report,indent=2)); print(report,flush=True)
    np.save(out/'transforms.npy',np.array([hs[i] for i in range(n)]))
    return hs
