from pathlib import Path
import sys
import unittest
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cinematic_finish import finish_cinematic
from openrt_backend import OpenRT
from native import plane
from run import relight


class CinematicTests(unittest.TestCase):
    def test_alpha_hidden_colors_and_strip_parity(self):
        rng=np.random.default_rng(832);a=rng.integers(20,235,(163,211,4),np.uint8);a[:,:,3]=255
        a[:23,:,3]=0;a[70:92,75:102,3]=128;a[120:140,150:170,3]=0
        b=a.copy();b[a[:,:,3]==0,:3]=rng.integers(0,256,(np.count_nonzero(a[:,:,3]==0),3),np.uint8)
        for look in ('natural_cinema','warm_documentary'):
            out,report=finish_cinematic(a,look,strip_rows=17)
            other,_=finish_cinematic(b,look,strip_rows=64)
            np.testing.assert_array_equal(out,other);np.testing.assert_array_equal(out[:,:,3],a[:,:,3])
            self.assertTrue(report['alpha_preserved']);self.assertLessEqual(report['max_virtual_exposure_adjustment_codes'],14.026)

    def test_gray_ramp_has_no_tonal_reversals_or_excess_clipping(self):
        x=np.tile(np.arange(256,dtype=np.uint8),(64,1));image=cv2.cvtColor(x,cv2.COLOR_GRAY2BGRA)
        for look in ('natural_cinema','warm_documentary'):
            output,_=finish_cinematic(image,look)
            gray=cv2.cvtColor(output,cv2.COLOR_BGRA2GRAY).astype(int)
            self.assertGreaterEqual(int(np.diff(gray,axis=1).min()),-1)
            self.assertLess(np.mean(np.any(output[:,:,:3]==255,axis=2)),.02)


class OpenRTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.backend=OpenRT(4)

    def test_actual_openrt_plane_is_neutral_and_keeps_pixels(self):
        field=self.backend.trace(plane((120,160)),(120,160),grid=48)
        np.testing.assert_array_equal(field,np.ones_like(field))
        image=np.full((120,160,4),100,np.uint8);image[:,:,3]=255
        result,report=relight(image,self.backend)
        np.testing.assert_array_equal(result,image)
        self.assertIn('Project-10/OpenRT',report['backend']);self.assertEqual(report['bounces'],0)

    def test_openrt_occluder_and_thread_repeatability(self):
        shape=(160,200)
        wall=np.array([[[90,40,0],[90,115,0],[90,115,30]],[[90,40,0],[90,115,30],[90,40,30]]],np.float32)
        scene=np.concatenate([plane(shape),wall])
        a=self.backend.trace(scene,shape,grid=64,elevation=30,azimuth=270)
        b=OpenRT(1).trace(scene,shape,grid=64,elevation=30,azimuth=270)
        np.testing.assert_array_equal(a,b);self.assertLess(float(a.min()),.7)
        self.assertTrue(np.isfinite(a).all());self.assertLessEqual(float(a.max()),1.00001)

    def test_golden_hour_is_warm_rgb_luminance_balanced_and_repeatable(self):
        backend=OpenRT(4,lighting='golden_hour_5pm');shape=(80,120)
        field=backend.trace(plane(shape),shape,grid=48,elevation=15)
        repeat=OpenRT(1,lighting='golden_hour_5pm').trace(plane(shape),shape,grid=48,elevation=15)
        np.testing.assert_array_equal(field,repeat)
        self.assertTrue(np.all(field[:,:,0]>1.1));self.assertTrue(np.all(field[:,:,2]<.9))
        np.testing.assert_allclose(field@np.array([.2126,.7152,.0722]),1,atol=.002)
        image=np.full((*shape,4),128,np.uint8);image[:,:,3]=255
        image[:10,:,3]=0;image[30:40,20:30,3]=96
        out,report=relight(image,backend,strength=.65,elevation=15)
        self.assertGreater(int(out[20,20,2]),int(out[20,20,0])) # Output BGRA.
        np.testing.assert_array_equal(out[:,:,3],image[:,:,3])
        self.assertFalse(report['flat_plane_preserves_pixels'])
        self.assertEqual(report['field_channels'],'linear RGB')
        zero,_=relight(image,backend,strength=0)
        np.testing.assert_array_equal(zero,image)

    def test_golden_highlight_protection_preserves_white_and_ramp(self):
        ramp=np.tile(np.arange(256,dtype=np.uint8),(32,1))
        image=cv2.cvtColor(ramp,cv2.COLOR_GRAY2BGRA)
        out,_=relight(image,OpenRT(2,lighting='golden_hour_5pm'),strength=.65,elevation=15)
        self.assertGreaterEqual(int(np.diff(out[:,:,:3].astype(int),axis=1).min()),0)
        np.testing.assert_array_equal(out[:,-1],image[:,-1])
        self.assertLess(np.mean(np.any(out[:,:,:3]==255,axis=2)),.01)


if __name__=='__main__':unittest.main(verbosity=2)
