"""Replace only OpenSfM's reader/consumer scheduling, retaining its detector."""
import logging
import math
import os
import cv2
from .scheduler import bounded_map, worker_count, MIB

def extract(data, images, force):
    from opensfm import features_processing as fp
    if not images:return
    cameras=list(data.load_camera_models().values())
    # Existing scale-space estimate is intentionally retained conservatively.
    decoded=max((c.width*c.height*4 for c in cameras),default=64*MIB)
    per_image=int(fp.average_processing_size(data)*MIB)+4*decoded+256*MIB
    workers=worker_count(data.config['processes']);stats={}
    def detect_one(image):
        pixels=data.load_image(image)
        if data.config['features_bake_segmentation']:
            segments=data.load_segmentation(image);instances=data.load_instances(image)
        else:segments=instances=None
        fp.detect(image,pixels,segments,instances,data,force)
        return image
    previous=cv2.getNumThreads();cv2.setNumThreads(0)
    try:
        # Each detector writes only its own photograph's feature/report files.
        # A complex tree image must not leave workers idle behind completed sky
        # or runway images just to preserve an unused result ordering.
        ordered = os.environ.get('ODM_TERRAIN_RUNTIME', '1') == '0'
        for _ in bounded_map(detect_one,images,workers=workers,bytes_per_item=per_image,stats=stats,ordered=ordered):pass
    finally:
        cv2.setNumThreads(previous)
        logging.getLogger(__name__).info('Bounded feature extraction: %s',stats)
