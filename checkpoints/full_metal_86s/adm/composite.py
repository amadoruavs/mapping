from pathlib import Path
import json
import cv2
import numpy as np
from scipy.special import ndtr
from .geometry import project
from .io import write_image

def smooth_uniform_field(rng,height,width):
    """Smooth Gaussian field mapped to uniform marginals for categorical sampling."""
    gh,gw=max(2,(height+95)//96),max(2,(width+95)//96)
    coarse=rng.standard_normal((gh,gw)).astype(np.float32)
    field=cv2.resize(coarse,(width,height),interpolation=cv2.INTER_LINEAR)
    x=np.clip((np.arange(width)+.5)*gw/width-.5,0,gw-1)
    y=np.clip((np.arange(height)+.5)*gh/height-.5,0,gh-1)
    fx=x-np.floor(x); fy=y-np.floor(y)
    variance=((1-fy)**2+fy**2)[:,None]*((1-fx)**2+fx**2)[None,:]
    return np.clip(ndtr(field/np.sqrt(variance)).astype(np.float32),.00001,.99999)

def compose(source,names,metadata,transforms,cfg,out,geographic=False,gpu=None):
    """Two streaming passes; retain every observation on disk, never allocate n×H×W."""
    shapes=[(m.width,m.height) for m in metadata]
    for H,(w,h) in zip(transforms,shapes):
        denominator=np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]])@H[2]
        if denominator.min()*denominator.max()<=0: raise ValueError('Registration folds an image across the projective horizon')
    footprints=[project(H,[[0,0],[w-1,0],[w-1,h-1],[0,h-1]]) for H,(w,h) in zip(transforms,shapes)]
    points=np.concatenate(footprints); lo=points.min(0); hi=points.max(0)
    scale=(cfg.max_output_dimension-2)/max(hi-lo)
    if not np.isfinite(scale) or scale<=0: raise ValueError('Invalid map bounds')
    if geographic: canvas=np.array([[scale,0,-lo[0]*scale+1],[0,-scale,hi[1]*scale+1],[0,0,1.]])
    else: canvas=np.array([[scale,0,-lo[0]*scale+1],[0,scale,-lo[1]*scale+1],[0,0,1.]])
    width,height=np.minimum(cfg.max_output_dimension,np.ceil((hi-lo)*scale+2-1e-9)).astype(int)
    hs=[canvas@H for H in transforms]
    if gpu is None:
        rgba,coverage,source_ids,valid=_compose_cpu(source,names,hs,width,height,cfg,out)
    else:
        from .metal_composite import compose_pixels
        rgba,coverage,source_ids,valid=compose_pixels(source,names,hs,width,height,cfg,out,gpu)
    write_image(out/'map_neutral.png',rgba)
    write_image(out/'coverage.png',coverage); write_image(out/'source_ids.png',source_ids)
    np.save(out/'image_to_map.npy',np.array(hs)); np.save(out/'map_to_world.npy',np.linalg.inv(canvas))
    geometry={'width':int(width),'height':int(height),'coordinate_system':'local WGS84 east/north tangent approximation, metres' if geographic else 'local pixels','georeferenced':geographic,'north_up':geographic,'map_to_world':np.linalg.inv(canvas).tolist(),'input_names':names,'metres_per_pixel':float(1/scale) if geographic else None,'sampling':'seeded categorical, per-channel squared distance from mean, spatially smooth correlated uniform random fields','seed':cfg.seed,'coverage_max':int(coverage.max()),'covered_pixels':int(valid.sum())}
    (out/'map_geometry.json').write_text(json.dumps(geometry,indent=2))
    return rgba,hs,geometry


def _compose_cpu(source,names,hs,width,height,cfg,out):
    total=np.zeros((height,width,3),np.float32); weights=np.zeros((height,width),np.float32)
    coverage=np.zeros((height,width),np.uint16); cache=out/'warps'; cache.mkdir(exist_ok=True)
    records=[]
    for i,(name,H) in enumerate(zip(names,hs)):
        image=cv2.imread(str(source/name)); h,w=image.shape[:2]
        p=project(H,[[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        x0,y0=np.maximum(0,np.floor(p.min(0)).astype(int)); x1,y1=np.minimum([width,height],np.ceil(p.max(0)).astype(int)+1)
        transform=np.array([[1,0,-x0],[0,1,-y0],[0,0,1.]])@H
        warped=cv2.warpPerspective(image,transform,(int(x1-x0),int(y1-y0)))
        yy,xx=np.mgrid[:h,:w]; distance=np.minimum.reduce([xx/(w/2),(w-1-xx)/(w/2),yy/(h/2),(h-1-yy)/(h/2)]).astype(np.float32)
        edge=np.clip(distance*8,0,1)
        # Trim 1% border to remove codec edges and occasional aircraft fragments.
        edge[distance<.02]=0
        confidence=cv2.warpPerspective(edge,transform,(int(x1-x0),int(y1-y0)))
        roi=np.s_[y0:y1,x0:x1]; total[roi]+=warped.astype(np.float32)*confidence[:,:,None]; weights[roi]+=confidence
        coverage[roi]+=(confidence>.001).astype(np.uint16)
        np.save(cache/f'{i:04}_rgb.npy',warped); np.save(cache/f'{i:04}_weight.npy',confidence.astype(np.float16))
        records.append([int(x0),int(y0),int(x1),int(y1)])
        print('composite warp',name,flush=True)
    mean=total/np.maximum(weights[:,:,None],1e-6); del total
    best=np.full((height,width),-np.inf,np.float32); selected=np.zeros((height,width,3),np.uint8); source_ids=np.full((height,width),65535,np.uint16)
    rng=np.random.default_rng(cfg.seed)
    for i,(x0,y0,x1,y1) in enumerate(records):
        rgb=np.load(cache/f'{i:04}_rgb.npy',mmap_mode='r'); confidence=np.load(cache/f'{i:04}_weight.npy').astype(np.float32)
        roi=np.s_[y0:y1,x0:x1]; distance=np.mean((rgb.astype(np.float32)-mean[roi])**2,axis=2)
        uniform=smooth_uniform_field(rng,height,width)[y0:y1,x0:x1]
        score=-distance/(2*cfg.selection_sigma_rgb**2)+np.log(np.maximum(confidence,1e-8))-np.log(-np.log(uniform))
        take=(score>best[roi])&(confidence>.001)
        selected[roi][take]=rgb[take]; best[roi][take]=score[take]; source_ids[roi][take]=i
    valid=weights>.001
    # Low-frequency seam correction; high-frequency detail stays source-sampled.
    numerator=cv2.GaussianBlur((mean-selected.astype(np.float32))*valid[:,:,None],(0,0),7)
    denominator=cv2.GaussianBlur(valid.astype(np.float32),(0,0),7)
    correction=numerator/np.maximum(denominator[:,:,None],1e-6)
    result=np.clip(selected.astype(np.float32)+correction,0,255).astype(np.uint8); result[~valid]=0
    rgba=np.dstack([result,valid.astype(np.uint8)*255])
    return rgba,coverage,source_ids,valid
