import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_finish import analyze, detail_luma, finish_adaptive, suggest_grade
from advanced_grade import grade, render, settings, auto_grade


class AdaptiveFinishTests(unittest.TestCase):
    def test_parallel_strips_alpha_and_hidden_pixel_invariance(self):
        rng=np.random.default_rng(44)
        source=rng.integers(20,235,(173,211,4),np.uint8);source[:,:,3]=255
        source[:15,:,3]=0;source[40:80,80:90,3]=127
        changed=source.copy();changed[source[:,:,3]==0,:3]=0
        a,report=finish_adaptive(source,strip_rows=17,workers=1)
        b,_=finish_adaptive(changed,strip_rows=64,workers=4)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[:,:,3],source[:,:,3])
        self.assertTrue(report['offline']);self.assertFalse(report['synthetic_detail'])

    def test_sharpen_resolves_existing_texture_without_edge_overshoot(self):
        x=np.arange(192,dtype=np.float32)
        y=np.tile(.45+.035*np.sin(x*.9),(80,1))
        output=detail_luma(y,np.ones_like(y),.5/255)
        self.assertGreater(float(output[:,10:-10].std()),float(y[:,10:-10].std())*1.06)
        step=np.full((64,128),.2,np.float32);step[:,64:]=.8
        sharp=detail_luma(step,np.ones_like(step),.5/255)
        np.testing.assert_allclose(sharp,step,atol=1e-6)

    def test_noise_is_not_mistaken_for_detail(self):
        rng=np.random.default_rng(80)
        gray=np.clip(120+rng.normal(0,3,(256,256)),0,255).astype(np.uint8)
        image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGRA)
        noise=analyze(image)['noise_sigma_codes']
        self.assertGreater(noise,2.3);self.assertLess(noise,3.7)
        out=detail_luma(gray.astype(np.float32)/255,np.ones_like(gray,dtype=np.float32),noise/255)
        self.assertLess(float(out.std()),float(gray.std()/255)*1.03)

    def test_ramp_highlights_black_and_no_tonal_reversals(self):
        ramp=cv2.cvtColor(np.tile(np.arange(256,dtype=np.uint8),(72,1)),cv2.COLOR_GRAY2BGRA)
        out,_=finish_adaptive(ramp,workers=1)
        self.assertGreaterEqual(int(np.diff(out[:,:,:3].astype(int),axis=1).min()),0)
        np.testing.assert_array_equal(out[:,0],ramp[:,0])
        np.testing.assert_array_equal(out[:,-1],ramp[:,-1])
        self.assertLessEqual(np.mean(np.any(out[:,:,:3]==255,axis=2)),1/256)

    def test_auto_grade_varies_with_scene_and_is_valid(self):
        base=np.tile(np.arange(30,220,dtype=np.uint8),(64,1))
        dark=cv2.cvtColor((base*.6).astype(np.uint8),cv2.COLOR_GRAY2BGRA)
        bright=cv2.cvtColor(np.minimum(255,base.astype(int)+40).astype(np.uint8),cv2.COLOR_GRAY2BGRA)
        a=suggest_grade(dark);b=suggest_grade(bright)
        self.assertGreater(a['settings']['exposure'],b['settings']['exposure'])
        settings(a['settings']);settings(b['settings'])

    def test_lossless_export_exact_grade_alpha_and_master_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=np.random.default_rng(35).integers(0,256,(243,151,4),np.uint8)
            source.tofile(root/'grade_master.bgra')
            (root/'final_geometry.json').write_text(json.dumps(dict(width=151,height=243)))
            before=(root/'grade_master.bgra').read_bytes()
            values={'exposure':.1,'texture':30,'sharpen':40,'detail_protection':100}
            png=render(root,values,output_format='png')
            decoded=cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(decoded,grade(source,values))
            crop=cv2.imdecode(np.frombuffer(render(root,values,True,crop=[.5,.5]),np.uint8),cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(crop,decoded)
            self.assertEqual(before,(root/'grade_master.bgra').read_bytes())
            self.assertTrue(auto_grade(root)['lossless_source'])
            with self.assertRaises(ValueError):render(root,{},output_format='unknown')

    def test_detail_crop_matches_same_pixels_in_full_grade(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=np.random.default_rng(76).integers(0,256,(710,910,4),np.uint8)
            source.tofile(root/'grade_master.bgra')
            (root/'final_geometry.json').write_text(json.dumps(dict(width=910,height=710)))
            values={'clarity':30,'texture':20,'exposure':.25}
            full=grade(source,values)
            for position in [[.5,.5],[0,0],[1,1]]:
                crop=cv2.imdecode(np.frombuffer(render(root,values,True,crop=position),np.uint8),cv2.IMREAD_UNCHANGED)
                x=max(0,min(270,round(position[0]*910-320)));y=max(0,min(230,round(position[1]*710-240)))
                np.testing.assert_array_equal(crop,full[y:y+480,x:x+640])
            for invalid in [[-.1,0],[0],[float('nan'),0],{},'bad']:
                with self.assertRaises(ValueError):render(root,{},True,crop=invalid)


if __name__=='__main__':unittest.main()
