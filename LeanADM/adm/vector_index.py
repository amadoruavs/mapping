"""Compact camera-footprint vectors; source pixels remain in the input images."""
from .io import write_json
import json
import cv2
import numpy as np
from .geometry import project


class FootprintIndex:
    def __init__(self,source,names,metadata,transforms,earth=None):
        self.source=source;self.names=names;self.metadata=metadata;self.transforms=transforms;self.earth=earth
        self.footprints=[project(H,[[0,0],[m.width-1,0],[m.width-1,m.height-1],[0,m.height-1]]) for H,m in zip(transforms,metadata)]
    def lookup(self,x,y,radius=0):
        # Return image support, explicitly using -1 instead of a fabricated tile ID.
        ll=self.earth.latlon([x,y]) if self.earth else [None,None]
        return [(i,-1,float(x),float(y),*ll,m.width,m.height) for i,(f,m) in enumerate(zip(self.footprints,self.metadata))
                if cv2.pointPolygonTest(f.astype(np.float32),(float(x),float(y)),True)>=-radius]
    def close(self):pass
    def save(self,out):
        value=dict(kind='camera footprint vector index',source=str(self.source),names=self.names,
                   origin_latlon=[self.earth.lat,self.earth.lon] if self.earth else None,
                   image_to_world=[h.tolist() for h in self.transforms],footprints=[f.tolist() for f in self.footprints],
                   image_dimensions=[[m.width,m.height] for m in self.metadata],raw_pixel_copies=False,
                   note='Pixels are sampled from the original images on demand; this is image-footprint support, not a materialized raw tile database.')
        write_json(out/'footprint_index.json',value)


def build_index(source,names,metadata,transforms,out,earth=None):
    index=FootprintIndex(source,names,metadata,transforms,earth);index.save(out);return index
