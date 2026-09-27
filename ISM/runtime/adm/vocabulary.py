import cv2
import numpy as np
from .geometry import project

def delta_map(rgb):
    values=rgb.astype(np.float32)
    return (values-values.mean(axis=(0,1),keepdims=True))/255.0

def corner_vocab(image,size):
    h,w=image.shape[:2]
    if min(h,w)<size: raise ValueError('Image smaller than vocabulary patch')
    origins=np.array([[0,0],[w-size,0],[w-size,h-size],[0,h-size]])
    raw=np.stack([image[y:y+size,x:x+size] for x,y in origins])
    return origins,raw,np.stack([delta_map(p) for p in raw])

def match_corners(image_i,image_j,H,cfg,geo_i=None,geo_j=None,tile_index=None,target_id=None,gpu=None):
    """Rectify candidate into source coordinates before matching the delta kernels.

    Geographic mode predicts H from metadata. Local mode predicts H from SIFT and
    reports this as a consistency check, never as independent ground truth.
    """
    size=cfg.vocabulary_size; radius=cfg.template_search_radius
    origins,patches,_=corner_vocab(image_i,size); matches=[]
    h,w=image_j.shape[:2]; resident=None
    for corner,((x,y),patch) in enumerate(zip(origins,patches)):
        record={'corner':corner,'accepted':False,'geographic_check':None}
        source_center=np.array([[x+(size-1)/2,y+(size-1)/2]])
        expected=project(H,source_center)[0]
        if not (0<=expected[0]<w and 0<=expected[1]<h):
            record['reason']='corner outside adjacent footprint'; matches.append(record); continue
        if np.mean(patch.std(axis=(0,1)))<cfg.corner_min_std:
            record['reason']='uninformative flat patch'; matches.append(record); continue
        if gpu is not None:
            if resident is None: resident=gpu.upload(image_j)
            warped,scores=gpu.corner_search(resident,patch,H,(x,y),radius)
        else:
            yy,xx=np.mgrid[y-radius:y+size+radius,x-radius:x+size+radius]
            coords=project(H,np.c_[xx.ravel(),yy.ravel()]).reshape(*xx.shape,2).astype(np.float32)
            warped=cv2.remap(image_j,coords[:,:,0],coords[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            valid=((coords[:,:,0]>=0)&(coords[:,:,0]<w-1)&(coords[:,:,1]>=0)&(coords[:,:,1]<h-1)).astype(np.uint8)
            window_valid=cv2.matchTemplate(valid,np.ones((size,size),np.uint8),cv2.TM_CCORR)>size*size-.5
            scores=cv2.matchTemplate(warped,patch,cv2.TM_CCOEFF_NORMED); scores[~window_valid]=-1
        _,peak,_,loc=cv2.minMaxLoc(scores); dx,dy=loc
        found=warped[dy:dy+size,dx:dx+size]
        rmse=float(np.sqrt(np.mean((delta_map(found)-delta_map(patch))**2)))
        second=scores.copy(); second[max(0,dy-4):dy+5,max(0,dx-4):dx+5]=-1
        margin=peak-float(second.max()); destination=project(H,source_center+np.array([[dx-radius,dy-radius]]))[0]
        record.update(ncc=float(peak),delta_rmse=rmse,peak_margin=margin,source=source_center[0].tolist(),target=destination.tolist())
        geo_ok=True
        if geo_i is not None and geo_j is not None:
            a=project(geo_i,source_center)[0]; b=project(geo_j,destination[None])[0]; distance=float(np.linalg.norm(a-b))
            hits=[] if tile_index is None else tile_index.lookup(*a,radius=cfg.gps_tolerance_m)
            support=any(row[0]==target_id for row in hits)
            geo_ok=distance<=cfg.gps_tolerance_m and support
            record['geographic_check']={'distance_m':distance,'hash_support':support,'passed':geo_ok}
        record['accepted']=bool(peak>=cfg.corner_min_ncc and rmse<=cfg.delta_rmse_threshold and margin>.02 and geo_ok)
        record['reason']='accepted' if record['accepted'] else 'delta / uniqueness / geometry threshold failed'
        matches.append(record)
    return matches
