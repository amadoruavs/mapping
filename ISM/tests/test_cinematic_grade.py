"""Presentation changes must preserve geometry and behave identically across strips."""
import unittest
import numpy as np
from PIL import Image,ImageFilter
from adm.cinematic_grade import grade_bgra,PRESETS,grade_array,classify_regions


class GradeTests(unittest.TestCase):
    def test_supplied_pro_camera_preset(self):
        p=PRESETS['pro_camera']
        self.assertEqual(len(PRESETS),7)
        self.assertEqual((p['contrast'],p['saturation'],p['vignette_strength']),(.35,.95,.08))
        self.assertEqual((p['blacks'],p['whites'],p['shadows'],p['highlights']),(.05,-.06,.02,.06))
        self.assertEqual((p['sharpness'],p['clarity']),(90,35))
        self.assertNotIn('tone_model',p)
        self.assertNotIn('white_reduction',p)

    def test_alpha_shape_and_seeded_grain_strip_invariance(self):
        rng=np.random.default_rng(99);im=rng.integers(0,256,(61,83,4),dtype=np.uint8);im[:,:,3]=255;im[:,:8]=0
        original=im.copy()
        for name in ('pro_camera','aerial_documentary'):
            a=grade_bgra(im,name,strip_rows=13);b=grade_bgra(im,name,strip_rows=99)
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[:,:,3],im[:,:,3]);self.assertEqual(a.shape,im.shape)
            self.assertTrue(np.all(a[:,:8]==0))
        np.testing.assert_array_equal(im,original)

    def test_regional_weights_and_black_pixels_are_finite(self):
        arr=np.zeros((41,53,3),np.float32);arr[10:30,10:30]=(.1,.4,.15)
        weights=classify_regions(arr)
        np.testing.assert_allclose(sum(weights.values()),1,atol=2e-7)
        for params in PRESETS.values():self.assertTrue(np.isfinite(grade_array(arr,params)).all())
        im=np.full((41,53,4),0,np.uint8);im[:,:,3]=255
        a=grade_bgra(im,'aerial_documentary','aerial_documentary_regional',strip_rows=15)
        b=grade_bgra(im,'aerial_documentary','aerial_documentary_regional',strip_rows=100)
        np.testing.assert_array_equal(a,b)
