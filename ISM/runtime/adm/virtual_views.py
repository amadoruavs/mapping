"""Experimental planar virtual cameras for rendering, never matching evidence."""
from functools import lru_cache
import time
import cv2
import numpy as np
from .geometry import project
from .enhancement import interpolate


class VirtualViews:
    def __init__(self,source,names,hs,gpu,levels=1,gated=True):
        self.source=source;self.names=names;self.gpu=gpu;self.gated=gated
        self.records=[(i,None,0,H) for i,H in enumerate(hs)]
        self.decode=lru_cache(maxsize=2)(self._decode)
        self.generated_seconds=0.;self.generated_count=0
        if levels not in (0,1,2):raise ValueError('Virtual-view levels must be 0, 1 or 2')
        for i in range(len(names)-1) if levels else []:
            a,b=self.decode(i),self.decode(i+1)
            if a.shape!=b.shape:continue
            h,w=a.shape[:2];corners=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
            pa,pb=project(hs[i],corners),project(hs[i+1],corners)
            if np.linalg.norm(pa.mean(0)-pb.mean(0))>np.sqrt(abs(cv2.contourArea(pa.astype(np.float32))))*.8:continue
            def add(left,right,depth):
                lp=project(self.records[left][3],corners);rp=project(self.records[right][3],corners)
                H=cv2.getPerspectiveTransform(corners,((lp+rp)*.5).astype(np.float32))
                middle=len(self.records);self.records.append((left,right,.5,H))
                if depth>1:add(left,middle,depth-1);add(middle,right,depth-1)
            add(i,i+1,levels)

    def _decode(self,i):
        image=cv2.imread(str(self.source/self.names[i]))
        if image is None:raise ValueError(f'Cannot decode {self.names[i]}')
        return image

    def __call__(self,k):
        i,j,t,H=self.records[k]
        if j is None:return self.decode(i),H
        ai,ah=self(i);bi,bh=self(j)
        started=time.perf_counter()
        def alpha(im):
            if im.shape[2]==4:return im
            a=cv2.cvtColor(im,cv2.COLOR_BGR2BGRA)
            a[:,:,3]=cv2.erode((im.max(2)>8).astype(np.uint8),np.ones((3,3),np.uint8))*255
            return a
        out=interpolate(alpha(ai),alpha(bi),np.linalg.inv(H)@ah,np.linalg.inv(H)@bh,t,self.gpu,self.gated)
        self.generated_seconds+=time.perf_counter()-started;self.generated_count+=1
        return out,H

    def report(self):
        return dict(algorithm='ADM Metal planar reprojection and color-gated interpolation; not AMD FSR3 frame generation',
            real_photos=len(self.names),virtual_views=len(self.records)-len(self.names),
            generation_seconds=self.generated_seconds,generation_calls=self.generated_count,
            interpolation_after_real_photo_alignment=True,synthetic_registration_observations=0,
            synthetic_files_saved=0,limitation='Planar motion only; no depth or occlusion recovery. Synthetic views contain correlated samples of existing photos.',
            views=[dict(id=k,parents=[i,j],fraction=t,image_to_map=H.tolist()) for k,(i,j,t,H) in enumerate(self.records) if j is not None])
