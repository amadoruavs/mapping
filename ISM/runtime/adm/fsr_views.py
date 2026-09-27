"""Recursive AMD FSR frame generation after real-photo registration."""
from functools import lru_cache
import cv2
import numpy as np
from .geometry import project
from .fsr_framegen import FrameGenerator,midpoint_transform
from .fsr_geometry import check_pair


class FSRViews:
    def __init__(self,source,names,hs,levels=2,depth_m=60.,hfov_deg=81.,resource_budget_mb=512):
        if type(levels) is not int or levels not in (1,2):raise ValueError('FSR levels must be 1 or 2')
        if len(names)!=len(hs):raise ValueError('FSR names and cameras must agree')
        self.source=source;self.names=names;self.levels=levels
        self.records=[(i,None,0,H) for i,H in enumerate(hs)]
        self.decode=lru_cache(maxsize=2)(self._decode)
        self.pairs=0;self.skipped_pairs=[]
        for i in range(len(names)-1):
            a,b=self.decode(i),self.decode(i+1)
            if a.shape!=b.shape:
                self.skipped_pairs.append(dict(pair=[i,i+1],reason='different dimensions'));continue
            h,w=a.shape[:2];corners=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
            pa,pb=project(hs[i],corners),project(hs[i+1],corners)
            if np.linalg.norm(pa.mean(0)-pb.mean(0))>np.sqrt(abs(cv2.contourArea(pa.astype(np.float32))))*.8:
                self.skipped_pairs.append(dict(pair=[i,i+1],reason='insufficient adjacent overlap'));continue
            def add(left,right,depth):
                check,H=check_pair(self.records[left][3],self.records[right][3],a.shape)
                if not check['accepted']:raise ValueError(check['reason'])
                middle=len(self.records);self.records.append((left,right,.5,H))
                if depth>1:add(left,middle,depth-1);add(middle,right,depth-1)
            start=len(self.records)
            try:add(i,i+1,levels)
            except ValueError as error:
                del self.records[start:]
                self.skipped_pairs.append(dict(pair=[i,i+1],reason=str(error)));continue
            self.pairs+=1
        print(f'FSR pairs: {self.pairs} accepted, {len(self.skipped_pairs)} skipped; {len(self.records)-len(names)} generated views planned',flush=True)
        self.engine=FrameGenerator(depth_m,hfov_deg,resource_budget_mb)
        # Midpoint and two quarter frames for one pair. Recompute on the second
        # compositor traversal rather than retaining 915 full-size images.
        self.generated=lru_cache(maxsize=3)(self._generate)

    def _decode(self,i):
        image=cv2.imread(str(self.source/self.names[i]))
        if image is None:raise ValueError(f'Cannot decode {self.names[i]}')
        out=cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
        out[:,:,3]=cv2.erode((image.max(2)>8).astype(np.uint8),np.ones((3,3),np.uint8))*255
        return out

    def _generate(self,k):
        i,j,_,H=self.records[k];a,ah=self(i);b,bh=self(j)
        out,actual=self.engine.interpolate(a,b,ah,bh)
        if not np.allclose(actual,H,rtol=1e-6,atol=1e-6):raise RuntimeError('FSR image and camera disagree')
        return out,H

    def __call__(self,k):
        i,j,_,H=self.records[k]
        return (self.decode(i),H) if j is None else self.generated(k)

    def report(self):
        return dict(**self.engine.report(),levels=self.levels,real_photos=len(self.names),
            virtual_views=len(self.records)-len(self.names),total_views=len(self.records),adjacent_pairs=self.pairs,
            skipped_pairs=self.skipped_pairs,working_cache=dict(real_images=2,generated_images=3),
            generation_calls=self.engine.frames,generation_seconds=self.engine.frame_seconds,
            interpolation_after_real_photo_alignment=True,synthetic_registration_observations=0,synthetic_files_saved=0,
            views=[dict(id=k,parents=[i,j],fraction=t,image_to_map=H.tolist()) for k,(i,j,t,H) in enumerate(self.records) if j is not None])
