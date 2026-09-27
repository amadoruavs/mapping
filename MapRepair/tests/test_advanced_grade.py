import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from advanced_grade import grade, render, settings, curve_lut, halo, preview_source, PREVIEW_CACHE

class AdvancedGradeTests(unittest.TestCase):
    def test_curves_shape_and_validation(self):
        points=[[0,0],[.25,.15],[.75,.85],[1,1]]
        lut=curve_lut(points)
        self.assertTrue(np.all(np.diff(lut)>=0))
        self.assertEqual(lut[0],0);self.assertEqual(lut[-1],1)
        self.assertLess(lut[1024],.25);self.assertGreater(lut[3072],.75)
        for value in [None,{'red':[[0,0],[0,.4],[1,1]]},{'master':[[.1,0],[1,1]]},
                      {'blue':[[0,0],[1,float('nan')]]},{'typo':[[0,0],[1,1]]}]:
            with self.assertRaises(ValueError):settings({'curves':value})
        image=np.full((12,12,4),100,np.uint8);image[:,:,3]=180
        out=grade(image,{'curves':{'red':[[0,.1],[1,1]]}})
        self.assertGreater(int(out[5,5,2]),100)
        np.testing.assert_array_equal(out[:,:,:2],image[:,:,:2])
        np.testing.assert_array_equal(out[:,:,3],image[:,:,3])

    def test_mixer_is_selective(self):
        image=np.zeros((12,12,4),np.uint8);image[:,:,3]=255
        image[:,:6,1]=200;image[:,6:,2]=200
        out=grade(image,{'green_saturation':-100})
        self.assertLess(np.ptp(out[5,2,:3]),2)
        np.testing.assert_array_equal(out[:,6:],image[:,6:])

    def test_full_frame_proxy_cache_and_alpha(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);image=np.full((1200,2400,4),[30,80,140,255],np.uint8)
            image[:200,:200]=[250,0,0,255];image[-200:,-200:]=[0,0,250,255]
            image.tofile(path/'grade_master.bgra')
            (path/'final_geometry.json').write_text(json.dumps({'width':2400,'height':1200}))
            small,scale=preview_source(path,image,2400,1200)
            self.assertEqual(small.shape,(960,1920,4));self.assertEqual(scale,.8)
            np.testing.assert_array_equal(small[10,10],[250,0,0,255])
            np.testing.assert_array_equal(small[-10,-10],[0,0,250,255])
            self.assertIs(preview_source(path,image,2400,1200)[0],small)
            self.assertEqual(Image.open(io.BytesIO(render(path,{'exposure':.1},True))).size,(1920,960))
            self.assertEqual(Image.open(io.BytesIO(render(path,{'exposure':.1},False))).size,(2400,1200))
            image[:]=[0,250,0,0];image[:,:1200]=[0,0,250,255]
            image.tofile(path/'grade_master.bgra')
            replacement,_=preview_source(path,image,2400,1200)
            self.assertIsNot(replacement,small)
            self.assertTrue(np.all(replacement[:,:959,1]==0))

    def test_complex_grade_strip_parity(self):
        image=np.random.default_rng(24).integers(0,256,(391,77,4),dtype=np.uint8)
        s=settings({'clarity':35,'texture':-20,'sharpen':80,'radius':3,'green_hue':20,
                    'shadow_blue':10,'curves':{'master':[[0,0],[.3,.22],[.7,.8],[1,1]]}})
        full=grade(image,s);out=np.empty_like(image);pad=halo(s)
        for y in range(0,len(image),128):
            end=min(len(image),y+128);lo=max(0,y-pad);hi=min(len(image),end+pad)
            out[y:end]=grade(image[lo:hi],s)[y-lo:end-lo]
        np.testing.assert_array_equal(full,out)
        np.testing.assert_array_equal(full[:,:,3],image[:,:,3])
    def test_neutral_alpha_and_strips(self):
        rng=np.random.default_rng(7)
        image=rng.integers(0,256,(300,64,4),dtype=np.uint8)
        np.testing.assert_array_equal(grade(image,{}),image)
        s={'exposure':.2,'temperature':15,'sharpen':65,'radius':3,'shadows':10}
        full=grade(image,s)
        np.testing.assert_array_equal(full[:,:,3],image[:,:,3])
        strips=np.empty_like(image)
        for y in range(0,300,128):
            end=min(300,y+128);lo=max(0,y-16);hi=min(300,end+16)
            strips[y:end]=grade(image[lo:hi],s)[y-lo:end-lo]
        np.testing.assert_array_equal(strips,full)
    def test_validation_and_exposure(self):
        for bad in ({'sharpen':151},{'exposure':float('nan')},{'typo':2}):
            with self.assertRaises(ValueError):settings(bad)
        image=np.full((20,20,4),80,np.uint8);image[:,:,3]=255
        self.assertGreater(grade(image,{'exposure':1})[10,10,0],80)
    def test_render_is_nondestructive(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);image=np.full((32,40,4),80,np.uint8);image[:,:,3]=255
            image.tofile(path/'grade_master.bgra')
            (path/'final_geometry.json').write_text(json.dumps({'width':40,'height':32}))
            before=(path/'grade_master.bgra').read_bytes()
            result=render(path,{'exposure':.5})
            self.assertEqual(Image.open(io.BytesIO(result)).size,(40,32))
            self.assertEqual(before,(path/'grade_master.bgra').read_bytes())
