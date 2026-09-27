"""FSR numerical and boundary checks on actual Metal, plus config validation."""
import os
import unittest
import cv2
import numpy as np
from adm.config import Config


class ConfigTests(unittest.TestCase):
    def test_enhancement_requires_supported_path_and_bounded_passes(self):
        for cfg in (Config(fsr=True),Config(input_upscale_passes=1),
                    Config(input_upscale_passes=4),Config(fsr_strength=float('nan'))):
            with self.assertRaises(ValueError):cfg.validate()


@unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
class FSRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adm.metal import get_backend
        cls.gpu=get_backend()

    def test_rcas_matches_vector_reference_and_preserves_alpha(self):
        from adm.enhancement import rcas
        rng=np.random.default_rng(8)
        im=rng.integers(0,256,(41,53,4),dtype=np.uint8);im[:,:,3]=255
        im[:3,:,3]=0;original=im.copy()
        z=np.pad(im[:,:,:3].astype(np.float64)/255,((1,1),(1,1),(0,0)),mode='edge')
        e=z[1:-1,1:-1];ring=np.stack((z[:-2,1:-1],z[1:-1,:-2],z[1:-1,2:],z[2:,1:-1]))
        mn=ring.min(0);mx=ring.max(0)
        lo=-np.minimum(mn,e)/np.maximum(4*mx,1e-8)
        hi=(1-np.maximum(mx,e))/np.minimum(4*mn-4,-1e-8)
        lobe=np.clip(np.maximum(lo,hi).max(2),-.1875,0)*.5
        lum=lambda x:x@np.array([.5,1,.5])
        lights=np.concatenate((lum(ring),lum(e)[None]))
        noise=1-.5*np.clip(abs(lum(ring).mean(0)-lum(e))/np.maximum(np.ptp(lights,axis=0),1e-8),0,1)
        lobe=(lobe*noise)[:,:,None]
        expected=np.rint(np.clip((lobe*ring.sum(0)+e)/(1+4*lobe),0,1)*255).astype(np.uint8)
        expected[:4]=im[:4,:,:3]
        result=rcas(im,.5,self.gpu)
        self.assertLessEqual(int(np.abs(result[:,:,:3].astype(int)-expected).max()),1)
        np.testing.assert_array_equal(result[:,:,3],im[:,:,3]);np.testing.assert_array_equal(im,original)
        np.testing.assert_array_equal(rcas(im,0,self.gpu),im)

    def test_easu_constants_edges_and_identity(self):
        from adm.enhancement import easu,upscale
        for c in (0,128,255):
            im=np.full((19,31,3),c,np.uint8)
            np.testing.assert_array_equal(easu(im,(38,62),self.gpu),np.full((38,62,3),c,np.uint8))
        im=np.full((32,40,3),40,np.uint8);im[:,20:]=210
        out=easu(im,(64,80),self.gpu)
        self.assertGreaterEqual(int(out.min()),40);self.assertLessEqual(int(out.max()),210)
        self.assertTrue(np.all(np.diff(out[32,:,0].astype(int))>=0))
        np.testing.assert_array_equal(easu(im,im.shape[:2],self.gpu),im)
        _,S=upscale(im,1,.5,self.gpu)
        np.testing.assert_array_equal(S,[[2,0,.5],[0,2,.5],[0,0,1]])

    def test_no_dark_halo_from_transparent_boundary(self):
        from adm.enhancement import easu
        im=np.full((25,31,4),180,np.uint8);im[:,:,3]=255;im[:,:7]=0
        out=easu(im,(50,62),self.gpu)
        np.testing.assert_array_equal(out[:,:,:3][out[:,:,3]==255],180)
        self.assertEqual(int(np.count_nonzero(out[:,:,3])),25*24*4)

    def test_interpolation_translated_views_and_disagreement_gate(self):
        from adm.enhancement import interpolate
        rng=np.random.default_rng(17);world=rng.integers(30,220,(41,75,4),dtype=np.uint8);world[:,:,3]=255
        a=world[:,:61].copy();b=world[:,14:].copy()
        ha=np.array([[1,0,-7],[0,1,0],[0,0,1.]])
        hb=np.array([[1,0,7],[0,1,0],[0,0,1.]])
        result=interpolate(a,b,ha,hb,.5,self.gpu)
        np.testing.assert_array_equal(result,world[:,7:68])
        a=np.full((41,61,4),60,np.uint8);b=np.full_like(a,210);a[:,:,3]=255;b[:,:,3]=255
        result=interpolate(a,b,np.eye(3),np.eye(3),.5,self.gpu,gated=True)
        np.testing.assert_array_equal(result,a)
        result=interpolate(a,b,np.eye(3),np.eye(3),.5,self.gpu,gated=False)
        self.assertTrue(np.all(result[:,:,:3]==135))
