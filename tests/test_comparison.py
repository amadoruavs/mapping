"""Basic invariants for the evaluation metrics, independent of either mapper."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_odm import distribution, pixel_agreement, alignment_support
from compare_lean import resample, sharpness, image_parity


class ComparisonTests(unittest.TestCase):
    def test_sharpness_minification_filters_aliasing(self):
        y,x=np.indices((64,64));gray=((x+y)%2*255).astype(np.uint8)
        image=np.dstack([gray,gray,gray,np.full_like(gray,255)])
        scale=.25
        transform=np.array([[scale,0,(scale-1)/2],[0,scale,(scale-1)/2],[0,0,1.]])
        small=resample(image,transform,(16,16),scale)
        np.testing.assert_allclose(small[:,:,:3],128,atol=1)
        self.assertEqual(sharpness(small,np.ones((16,16),bool))['mean_gradient'],0)

    def test_sharpness_uses_only_shared_interior_and_parity_checks_alpha(self):
        a=np.random.default_rng(2).integers(10,200,(40,40,4),dtype=np.uint8);a[:,:,3]=255
        b=a.copy();b[:5,:,:3]=255
        mask=np.zeros((40,40),bool);mask[10:30,10:30]=True
        self.assertEqual(sharpness(a,mask),sharpness(b,mask))
        b=a.copy();b[20,20,3]=0
        self.assertEqual(image_parity(a,b)['changed_pixels'],1)

    def test_identical_maps_and_mask_exclusion(self):
        a=np.random.default_rng(0).integers(10,200,(51,61,4),dtype=np.uint8);a[:,:,3]=255
        a[:5,:,3]=0;b=a.copy();b[:5,:,:3]=255
        report=pixel_agreement(a,b)
        self.assertEqual(report['mae_8bit'],0)
        self.assertEqual(report['footprint_iou'],1)
        self.assertAlmostEqual(report['ssim_luminance'],1,places=6)

    def test_constant_offset_and_disjoint_coverage(self):
        a=np.full((41,43,4),100,np.uint8);a[:,:,3]=255;b=a.copy();b[:,:,:3]+=10
        report=pixel_agreement(a,b)
        self.assertAlmostEqual(report['mae_8bit'],10)
        self.assertAlmostEqual(report['rmse_8bit'],10)
        self.assertIsNone(report['luminance_correlation'])
        b[:,:,3]=0;report=pixel_agreement(a,b)
        self.assertEqual(report['overlap_pixels'],0)
        self.assertNotIn('mae_8bit',report)

    def test_distance_statistics_known_values(self):
        report=distribution([3,4])
        self.assertEqual(report['count'],2)
        self.assertEqual(report['mean'],3.5)
        self.assertAlmostEqual(report['rmse'],np.sqrt(12.5))

    def test_minimal_and_local_homography_support_is_rejected(self):
        y,x=np.mgrid[10:100:20,10:100:20]
        points=np.c_[x.ravel(),y.ravel()].astype(np.float32)
        mask=np.zeros(len(points),bool);mask[:4]=True
        self.assertFalse(alignment_support(points,points,mask,(100,100),(100,100))['success'])
        mask[:]=True
        self.assertTrue(alignment_support(points,points,mask,(100,100),(100,100))['success'])
        self.assertFalse(alignment_support(points*.1,points*.1,mask,(100,100),(100,100))['success'])
