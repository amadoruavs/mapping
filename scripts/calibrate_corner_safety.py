"""Measure relaxed corner rules against geometry, context, and texture decoys.

This is an experimental SIFT-assisted validation harness, not a measured real-world
false-positive rate. Real-image candidates are unlabelled. Decoys deliberately
sample a different image region while retaining nominal geometry, so a geometry
gate alone cannot trivially reject all of them. Source-image parity holds out half
the observations from selecting a rule.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
from functools import lru_cache
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from adm.geometry import project
from adm.vocabulary import delta_map


def photometric(a,b):
    aa=a.astype(np.float32)-a.mean(axis=(0,1),keepdims=True)
    bb=b.astype(np.float32)-b.mean(axis=(0,1),keepdims=True)
    ncc=float(np.sum(aa*bb)/max(np.sqrt(np.sum(aa*aa)*np.sum(bb*bb)),1e-8))
    return ncc,float(np.sqrt(np.mean((delta_map(a)-delta_map(b))**2)))


def measure(image_a,image_b,H,center,context_sign,decoy_shift=None):
    size=50; radius=24; cx,cy=center; x,y=cx-24.5,cy-24.5
    shift=np.zeros(2) if decoy_shift is None else np.array(decoy_shift)
    yy,xx=np.mgrid[-radius:size+radius,-radius:size+radius]
    points=np.c_[xx.ravel()+x+shift[0],yy.ravel()+y+shift[1]]
    coords=project(H,points).reshape(*xx.shape,2).astype(np.float32)
    h,w=image_b.shape[:2]
    valid=((coords[:,:,0]>=0)&(coords[:,:,0]<w-1)&(coords[:,:,1]>=0)&(coords[:,:,1]<h-1)).astype(np.uint8)
    warped=cv2.remap(image_b,coords[:,:,0],coords[:,:,1],cv2.INTER_LINEAR)
    patch=cv2.getRectSubPix(image_a,(size,size),(float(cx),float(cy)))
    complete=cv2.matchTemplate(valid,np.ones((size,size),np.uint8),cv2.TM_CCORR)>size*size-.5
    if not complete.any(): return None
    scores=cv2.matchTemplate(warped,patch,cv2.TM_CCOEFF_NORMED); scores[~complete]=-1
    _,ncc,_,(dx,dy)=cv2.minMaxLoc(scores)
    other=scores.copy(); other[max(0,dy-4):dy+5,max(0,dx-4):dx+5]=-1
    margin=float(ncc-other.max()); destination=project(H,[np.array(center)+[dx-radius,dy-radius]])[0]
    rmse=photometric(patch,warped[dy:dy+size,dx:dx+size])[1]
    context=[]
    sx,sy=context_sign
    for offset in [[sx*64,0],[0,sy*64],[sx*64,sy*64]]:
        cc=np.array(center)+offset
        yy,xx=np.mgrid[:size,:size]
        src=np.c_[xx.ravel()+cc[0]-24.5,yy.ravel()+cc[1]-24.5]
        target=project(H,src+[dx-radius,dy-radius]+shift).reshape(size,size,2).astype(np.float32)
        if np.any(target[:,:,0]<0) or np.any(target[:,:,0]>=w-1) or np.any(target[:,:,1]<0) or np.any(target[:,:,1]>=h-1): continue
        dest=cv2.remap(image_b,target[:,:,0],target[:,:,1],cv2.INTER_LINEAR)
        orig=cv2.getRectSubPix(image_a,(size,size),tuple(map(float,cc)))
        score,error=photometric(orig,dest); context.append({'ncc':score,'delta_rmse':error})
    return {'ncc':float(ncc),'delta_rmse':rmse,'peak_margin':margin,'target':destination.tolist(),'context':context,'decoy_shift_source_pixels':None if decoy_shift is None else list(decoy_shift)}


def passes(row,ncc,guarded):
    basic=row['delta_rmse']<=.10 and row['ncc']>=ncc and row['peak_margin']>.02
    if not basic or not guarded: return basic
    # 20 native pixels accommodates corner extrapolation; additional evidence
    # must come from two other image paths and two neighboring patches.
    return row['supporting_paths_within_20px']>=2 and row['triangle_median_error_px']<=20 and sum(c['ncc']>=.4 and c['delta_rmse']<=.12 for c in row['context'])>=2


def main():
    p=argparse.ArgumentParser(); p.add_argument('source',type=Path); p.add_argument('--baseline',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    started=time.perf_counter(); out=args.output; out.mkdir(parents=True,exist_ok=True); cv2.setNumThreads(6)
    data=np.load(args.baseline/'registration.npz',allow_pickle=True); names=data['names'].tolist(); edges=data['edges'].tolist(); ids={n:i for i,n in enumerate(names)}
    inventory={r['name']:r for r in json.loads((args.baseline/'selection.json').read_text())}
    for name in names:
        if hashlib.sha256((args.source/name).read_bytes()).hexdigest()!=inventory[name]['sha256']: raise ValueError(f'Input changed: {name}')
    S=np.diag([1400/3840,788/2160,1.]); Si=np.linalg.inv(S); links={}
    for edge in edges:
        H=Si@edge['H']@S; links[edge['i'],edge['j']]=H; links[edge['j'],edge['i']]=np.linalg.inv(H)
    @lru_cache(maxsize=5)
    def read(i): return cv2.imread(str(args.source/names[i]))
    observations=[]; signs=[(1,1),(-1,1),(-1,-1),(1,-1)]
    baseline=json.loads((args.baseline/'corner_matches.json').read_text())
    for pair_index,pair in enumerate(baseline):
        i,j=ids[pair['source']],ids[pair['target']]; H=links[i,j]
        for patch in pair['corners']:
            if 'delta_rmse' not in patch: continue
            center=patch['source']; corner=patch['corner']; sx,sy=signs[corner]
            triangle_predictions=[project(links[k,j]@links[i,k],[center])[0] for k in range(len(names)) if k not in [i,j] and (i,k) in links and (k,j) in links]
            # Three real-texture distractors, all at least 160 pixels away from
            # the true source-plane location, preserve repetitive-field texture.
            for kind,shift in [('real',None),('decoy',[sx*160,0]),('decoy',[0,sy*160]),('decoy',[sx*192,sy*192])]:
                metrics=measure(read(i),read(j),H,center,signs[corner],shift)
                if metrics is None: continue
                errors=np.linalg.norm(np.array(triangle_predictions)-metrics['target'],axis=1) if triangle_predictions else np.array([])
                metrics.update(source=names[i],target_image=names[j],corner=corner,source_center=center,kind=kind,split='calibration' if i%2==0 else 'held_out',triangle_paths=len(errors),supporting_paths_within_20px=int(np.sum(errors<=20)),triangle_median_error_px=float(np.median(errors)) if len(errors) else None)
                observations.append(metrics)
        if pair_index%20==0: print('measured pairs',pair_index+1,'/',len(baseline),'observations',len(observations),flush=True)
    (out/'observations.json').write_text(json.dumps(observations,indent=2))
    sweep=[]
    for ncc in [.8,.7,.6,.5,.4,.3]:
        for guarded in [False,True]:
            row={'minimum_ncc':ncc,'geometry_and_context':guarded}
            for split in ['calibration','held_out']:
                for kind in ['real','decoy']:
                    subset=[r for r in observations if r['split']==split and r['kind']==kind]
                    row[f'{split}_{kind}_total']=len(subset)
                    row[f'{split}_{kind}_accepted']=sum(passes(r,ncc,guarded) for r in subset)
            sweep.append(row)
    # Select using calibration only. Held-out results never affect selection.
    viable=[r for r in sweep if r['geometry_and_context'] and r['calibration_decoy_accepted']==0]
    chosen=max(viable,key=lambda r:(r['calibration_real_accepted'],r['minimum_ncc'])) if viable else None
    if chosen:
        accepted=[r for r in observations if r['kind']=='real' and passes(r,chosen['minimum_ncc'],True)]
        (out/'accepted_real_candidates.json').write_text(json.dumps(accepted,indent=2))
    summary={'observations':len(observations),'rules':sweep,'selected_on_calibration_only':chosen,'source_image_parity_split':True,'false_positive_rate_on_real_images_known':False,'independent_geographic_ground_truth':False,'production_defaults_changed':False,'negative_controls':'Wrong nearby texture sampled 160–272 source pixels away, nominal geometry held fixed; these stress image checks rather than trivially failing the position gate.','limits':'SIFT guides candidate search and alternate image paths. Real candidates are unlabelled. Source split still shares target images and terrain. Zero decoy errors cannot prove zero false positives in deployment.','seconds':time.perf_counter()-started}
    (out/'report.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
