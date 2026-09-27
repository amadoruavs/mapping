"""GPU crop checks against known geometry, independent of the polygon algorithm."""
import os
import unittest

import cv2
import numpy as np

from adm.lasso import crop_in_place


@unittest.skipUnless(os.environ.get('ADM_TEST_METAL') == '1', 'requires Metal')
class LassoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adm.metal import get_backend
        cls.gpu = get_backend()

    def test_jagged_border_removed_without_changing_retained_pixels(self):
        image = np.zeros((400,480,4), np.uint8)
        image[30:370,35:445] = [75,100,130,255]
        image[15:55,110:190] = [75,100,130,255]
        image[350:390,280:350] = [0,0,0,255]
        # Interior shadow must be preserved, not mistaken for the border.
        image[175:210,220:255] = [0,0,0,255]
        original = image.copy()
        coverage = self.gpu.upload((image[:,:,3]>0).astype(np.uint16))
        ids = self.gpu.upload(np.where(image[:,:,3]>0,3,65535).astype(np.uint16))
        result = crop_in_place(image,self.gpu,coverage=coverage.view(),source_ids=ids.view())
        valid = result.image[:,:,3]>0
        np.testing.assert_array_equal(image,original)
        np.testing.assert_array_equal(result.image[valid],original[valid])
        self.assertFalse(valid[:30].any())
        self.assertFalse(valid[370:].any())
        self.assertTrue(valid[175:210,220:255].all())
        self.assertTrue(np.all(coverage.view()[~valid]==0))
        self.assertTrue(np.all(ids.view()[~valid]==65535))
        self.assertGreater(result.report['retained_fraction'],.8)
        self.assertLess(result.report['retained_fraction'],1)
        mask=valid.astype(np.uint8)
        count,_=cv2.connectedComponents(mask)
        self.assertEqual(count,2)

    def test_blank_and_disconnected_center_rejected(self):
        image=np.zeros((200,300,4),np.uint8)
        with self.assertRaisesRegex(ValueError,'central support'):
            crop_in_place(image,self.gpu)
        image[10:190,10:80]=[80,90,100,255]
        image[10:190,220:290]=[80,90,100,255]
        with self.assertRaisesRegex(ValueError,'central support'):
            crop_in_place(image,self.gpu)

    def test_repeatability_and_grid_unchanged(self):
        image=np.full((283,391,4),150,np.uint8);image[:,:,3]=255
        a=crop_in_place(image,self.gpu)
        b=crop_in_place(image,self.gpu)
        np.testing.assert_array_equal(a.image,b.image)
        self.assertEqual(a.image.shape,image.shape)
        self.assertFalse(a.report['canvas_transform_changed'])
