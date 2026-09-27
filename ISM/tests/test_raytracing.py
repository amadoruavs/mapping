"""Ray intersections must respond to geometry, not map intensity."""
import os
import unittest
import numpy as np
from adm.raytracing import flat_plane,trace,relight_map


@unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal ray tracing')
class RayTracingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adm.metal import get_backend
        cls.gpu=get_backend()

    def test_plane_is_uniform_and_image_detail_is_unchanged(self):
        field=trace(self.gpu,flat_plane(),shape=(17,31)).numpy()
        np.testing.assert_allclose(field[:,:,:3],1,atol=.002)
        np.testing.assert_array_equal(field[:,:,3],1)
        self.assertLess(np.ptp(field[:,:,0]),1e-6)
        rng=np.random.default_rng(122);im=rng.integers(0,256,(41,63,4),np.uint8);im[:,:,3]=255;im[:,:3]=0
        actual,report=relight_map(im,self.gpu)
        np.testing.assert_array_equal(actual,im)
        self.assertFalse(report['measured_elevation_available'])

    def test_real_occluder_casts_directional_shadow_and_repeat_is_exact(self):
        # A roof at z=.2 occupies only the right side of the orthographic view.
        roof=np.array([[[.05,-.4,.2],[.45,-.4,.2],[.45,.4,.2]],
                       [[.05,-.4,.2],[.45,.4,.2],[.05,.4,.2]]],np.float32)
        scene=np.concatenate([flat_plane(),roof]);kwargs=dict(shape=(49,65),elevation=30,samples=32)
        towards=trace(self.gpu,scene,azimuth=90,**kwargs).numpy()
        repeat=trace(self.gpu,scene,azimuth=90,**kwargs).numpy()
        away=trace(self.gpu,scene,azimuth=270,**kwargs).numpy()
        np.testing.assert_array_equal(towards,repeat)
        # Ground x in [-.27,-.13] is shaded only by the eastward sun ray.
        self.assertLess(towards[22:27,15:24,:3].mean(),away[22:27,15:24,:3].mean()-.12)
        self.assertTrue(np.isfinite(towards).all())

    def test_invalid_geometry_is_rejected_before_gpu_allocation(self):
        with self.assertRaises(ValueError):trace(self.gpu,np.zeros((2,3,3)))
        with self.assertRaises(ValueError):trace(self.gpu,flat_plane(),samples=1000)
        with self.assertRaises(ValueError):trace(self.gpu,flat_plane(),materials=np.full((2,3),2.))
