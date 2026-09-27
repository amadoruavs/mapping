"""Check actual translated AMD frame-generation shaders on Metal hardware."""
import os
import unittest
import cv2
import numpy as np


@unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
class FrameGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adm.fsr_framegen import FrameGenerator
        cls.engine=FrameGenerator()

    def test_identical_frames_and_alpha(self):
        image=np.full((64,64,4),120,np.uint8);image[:,:,3]=255
        output,H=self.engine.interpolate(image,image,np.eye(3),np.eye(3))
        np.testing.assert_allclose(output,image,atol=1)
        np.testing.assert_allclose(H,np.eye(3),atol=1e-6)

    def test_translated_frames_generate_a_midpoint(self):
        rng=np.random.default_rng(61)
        world=cv2.GaussianBlur(rng.integers(30,210,(96,120,3),dtype=np.uint8),(3,3),.6)
        a=world[:,:96];b=world[:,16:112]
        hb=np.array([[1,0,16],[0,1,0],[0,0,1.]])
        result,H=self.engine.interpolate(a,b,np.eye(3),hb)
        np.testing.assert_allclose(H,[[1,0,8],[0,1,0],[0,0,1]],atol=1e-5)
        error=np.abs(result[12:-12,12:-12,:3].astype(float)-world[12:-12,20:92].astype(float))
        self.assertLess(error.mean(),2)
