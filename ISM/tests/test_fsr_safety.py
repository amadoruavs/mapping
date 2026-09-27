"""Generated frames must have consistent geometry and real source support."""
import os
import unittest
import numpy as np
from adm.fsr_geometry import check_pair


class FSRGeometryTests(unittest.TestCase):
    def test_translation_is_accepted(self):
        check,H=check_pair(np.eye(3),np.array([[1,0,16],[0,1,0],[0,0,1.]]),(96,96))
        self.assertTrue(check['accepted']);self.assertLess(check['max_camera_error_px'],1e-5)
        np.testing.assert_allclose(H,[[1,0,8],[0,1,0],[0,0,1.]],atol=1e-5)

    def test_large_turn_and_motion_are_rejected(self):
        for H in (np.array([[0,-1,95],[1,0,0],[0,0,1.]]),np.array([[1,0,70],[0,1,0],[0,0,1.]])):
            check,_=check_pair(np.eye(3),H,(96,96));self.assertFalse(check['accepted'])

    def test_nonfinite_camera_is_rejected(self):
        H=np.eye(3);H[0,2]=np.nan
        check,_=check_pair(np.eye(3),H,(96,96));self.assertFalse(check['accepted'])

    @unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
    def test_generated_pixels_require_both_sources_and_matching_color(self):
        from adm.metal import get_backend
        gpu=get_backend();a=np.full((64,64,4),100,np.uint8);a[:,:,3]=255;b=a.copy()
        b[:,:16,3]=0;b[:,32:48,:3]=180
        output=a.copy();output[:,16:32,:3]=225
        buffer=gpu.upload(output)
        gpu.dispatch('fsr_validate_generated',[gpu.upload(a),gpu.upload(b),buffer],64*64,
                     [64,64,4,4],np.r_[np.eye(3).ravel(),np.eye(3).ravel()])
        gpu.sync();result=buffer.numpy()
        self.assertTrue(np.all(result[:,:48]==0))
        np.testing.assert_array_equal(result[:,48:],a[:,48:])
