"""The final crop must preserve finished detail and geographic sampling."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from adm.config import Config
from adm.presentation import write_presentation


@unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
class FinalLassoTests(unittest.TestCase):
    def test_finished_pixels_and_coordinates_preserved(self):
        from adm.metal import get_backend
        gpu=get_backend();rng=np.random.default_rng(24)
        image=rng.integers(20,170,(240,320,4),np.uint8);image[:,:,3]=0
        image[20:220,25:295,3]=255;image[8:35,90:150,3]=255
        image[image[:,:,3]==0]=0
        geometry=dict(width=320,height=240,georeferenced=True,metres_per_pixel=.2,
                      map_to_world=[[.2,0,100],[0,-.2,200],[0,0,1]])
        for preset in ('pro_camera','none'):
            with self.subTest(preset=preset),tempfile.TemporaryDirectory() as temp:
                plain=Path(temp)/'plain';cropped=Path(temp)/'cropped';plain.mkdir();cropped.mkdir()
                cfg=Config(presentation=True,presentation_scale=1.5,presentation_profile='siyi_low_light',color_grade=preset)
                write_presentation(image,geometry,[np.eye(3)],cfg,plain,gpu)
                cfg.lasso_crop=True
                write_presentation(image,geometry,[np.eye(3)],cfg,cropped,gpu)
                filename='map_graded.png' if preset!='none' else 'map_enhanced.png'
                a=cv2.imread(str(plain/filename),cv2.IMREAD_UNCHANGED)
                b=cv2.imread(str(cropped/filename),cv2.IMREAD_UNCHANGED);keep=b[:,:,3]>0
                self.assertEqual(a.shape,b.shape)
                self.assertLess(np.count_nonzero(keep),np.count_nonzero(a[:,:,3]))
                np.testing.assert_array_equal(b[keep],a[keep])
                before=json.loads((plain/'presentation_geometry.json').read_text())
                after=json.loads((cropped/'presentation_geometry.json').read_text())
                self.assertEqual(before['map_to_world'],after['map_to_world'])
                self.assertEqual(after['covered_pixels'],np.count_nonzero(keep))
                np.testing.assert_array_equal(np.load(plain/'image_to_presentation.npy'),np.load(cropped/'image_to_presentation.npy'))
                report=json.loads((cropped/'lasso_crop.json').read_text())
                self.assertEqual(report['applies_to'],[filename])
                self.assertEqual(report['after_color_grade'],preset!='none')
