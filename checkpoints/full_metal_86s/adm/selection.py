import hashlib
from pathlib import Path
import cv2
import numpy as np

def annotation_regions(im):
    """Conservative heuristic for solid blue detector labels; manifest overrides it."""
    b,g,r=cv2.split(im)
    mask=((b>190)&(g<100)&(r<100)).astype(np.uint8)
    count,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
    return [dict(x=int(x),y=int(y),width=int(w),height=int(h)) for x,y,w,h,area in stats[1:] if w>=60 and 8<=h<=100 and w/h>2.5 and area/(w*h)>.35]

def select_frames(source,cfg,exclusions=None):
    files=sorted(p for p in Path(source).iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png','.tif','.tiff'])
    if not files: raise ValueError('No supported images')
    chosen=[]; rows=[]; hashes={}; thumbnails=[]
    for i,p in enumerate(files):
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        reason='first frames / takeoff' if i<cfg.skip_first else (exclusions or {}).get(p.name)
        im=cv2.imread(str(p))
        if im is None: reason='unreadable image'
        if reason is None and sha in hashes: reason=f'byte duplicate of {hashes[sha]}'
        boxes=[] if im is None else annotation_regions(im)
        if reason is None and boxes: reason='detector label heuristic'
        if reason is None:
            small=cv2.resize(im,(480,270),interpolation=cv2.INTER_AREA)
            # Tiny color signature shortlist, then full thumbnail difference.
            for old,name in thumbnails:
                if abs(float(small.mean())-float(old.mean()))>cfg.near_duplicate_mae: continue
                if np.abs(small.astype(np.float32)-old.astype(np.float32)).mean()<cfg.near_duplicate_mae:
                    reason=f'near duplicate of {name}'; break
            if reason is None: chosen.append(p.name); thumbnails.append((small,p.name))
        hashes[sha]=p.name
        rows.append({'name':p.name,'sha256':sha,'excluded':reason,'annotation_regions':boxes})
    if not chosen: raise ValueError('All input images excluded')
    return chosen,rows
