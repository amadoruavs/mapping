"""Numerical parity against the supplied OpenCV/Pillow presentation recipes."""
import os
import unittest
import cv2
import numpy as np
from PIL import Image,ImageFilter
from adm.presentation import enhance_reference
from adm.cinematic_grade import grade_bgra


@unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adm.metal import get_backend
        from adm.presentation_metal import PresentationMetal
        cls.gpu=get_backend();cls.engine=PresentationMetal(cls.gpu)

    def error(self,a,b,name,mean=.15,maximum=2):
        d=np.abs(a.astype(float)-b.astype(float))
        print(name,dict(mean=float(d.mean()),maximum=float(d.max())),flush=True)
        self.assertLessEqual(d.mean(),mean,name);self.assertLessEqual(d.max(),maximum,name)

    def test_color_conversions(self):
        rng=np.random.default_rng(312);a=rng.integers(0,256,(83,97,4),np.uint8);a[:,:,3]=255
        for srgb,codes in ((False,(cv2.COLOR_LBGR2Lab,cv2.COLOR_Lab2LBGR)),(True,(cv2.COLOR_BGR2LAB,cv2.COLOR_LAB2BGR))):
            for inverse in (False,True):
                expected=cv2.cvtColor(a[:,:,:3],codes[inverse]);actual=self.engine.color(self.gpu.upload(a),srgb,inverse).numpy()[:,:,:3]
                self.error(expected,actual,f'color {srgb=} {inverse=}',.1,1)

    def test_white_and_vibrance_adjustment(self):
        from adm.cinematic_grade import finish_tone
        a=np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(3,1,4));a[:,:,3]=255
        a[1,:,0]=180;a[1,:,2]=80;a[2,:,3]=0
        expected=finish_tone(a,.3,.1);actual=self.engine.finish_tone(a,.3,.1)
        self.error(expected,actual,'white/vibrance CPU parity',.01,1)
        np.testing.assert_array_equal(actual[0,:128],a[0,:128])
        self.assertGreaterEqual(np.diff(actual[0,:,0].astype(int)).min(),0)
        self.assertLessEqual(abs(int(actual[0,-1,0])-round(255*.7)),1)
        np.testing.assert_array_equal(actual[:,:,3],a[:,:,3])
        self.assertFalse(actual[2,:,:3].any())

    def test_pillow_blur_and_grade(self):
        rng=np.random.default_rng(313);a=rng.integers(0,256,(81,123,4),np.uint8);a[:,:,3]=255;a[:5,:,3]=0
        for radius in (1.5,30):
            expected=np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius)))
            self.error(expected,self.engine.pillow_blur(self.gpu.upload(a),radius).numpy(),f'Pillow blur {radius}',0,0)
        self.error(grade_bgra(a),self.engine.grade(a),'full pro_camera',.05,2)

    def test_all_supplied_presets_regional_hue_and_seed_parity(self):
        from adm.cinematic_grade import PRESETS,REGION_PRESETS
        rng=np.random.default_rng(918)
        image=rng.integers(0,256,(257,319,4),np.uint8);image[:,:,3]=255;image[:5]=0
        # Include smooth saturated/neutral zones as well as high-frequency noise.
        for i,color in enumerate([(0,0,0),(255,255,255),(255,0,0),(0,255,0),
                                  (0,0,255),(80,130,180),(145,145,145)]):
            image[180:230,i*40:(i+1)*40,:3]=color
        for preset in PRESETS:
            with self.subTest(preset=preset):
                expected=grade_bgra(image,preset,seed=104)
                actual=self.engine.grade(image,preset,seed=104).copy()
                self.error(expected,actual,preset,.002,3)
                np.testing.assert_array_equal(actual[:,:,3],image[:,:,3])
                np.testing.assert_array_equal(actual,self.engine.grade(image,preset,seed=104))
                self.assertFalse(actual[:5].any())
        for region in REGION_PRESETS:
            with self.subTest(region=region):
                expected=grade_bgra(image,'aerial_documentary',region,seed=104)
                actual=self.engine.grade(image,'aerial_documentary',region_preset=region,seed=104)
                self.error(expected,actual,region,.02,5)
                d=np.abs(expected.astype(float)-actual.astype(float))
                self.assertLessEqual(np.quantile(d,.999),1)
                np.testing.assert_array_equal(actual[:,:,3],image[:,:,3])

    def test_siyi_chroma_cleanup_preserves_luma_and_matches_cpu(self):
        from adm.presentation import chroma_clean_reference
        from adm.presentation_metal import PresentationMetal
        rng=np.random.default_rng(807);a=rng.integers(40,190,(71,95,4),np.uint8);a[:,:,3]=255;a[:3]=0
        engine=PresentationMetal(self.gpu,'siyi_low_light')
        actual=engine.chroma_clean(self.gpu.upload(a)).numpy();expected=chroma_clean_reference(a)
        self.error(expected,actual,'SIYI chroma parity',.02,1)
        weights=np.array([.114,.587,.299]);before=a[:,:,:3]@weights;after=actual[:,:,:3]@weights
        self.assertLess(np.max(np.abs(before[3:]-after[3:])),.7)
        self.assertFalse(actual[:3].any())
        self.error(grade_bgra(a,'pro_camera'),engine.grade(a,'pro_camera'),'Supplied pro_camera grade parity',.05,2)
        cpu={};metal={};expected=enhance_reference(a,1.5,profile='siyi_low_light',stages=cpu);actual=engine.enhance(a,1.5,stages=metal)
        for name in ('chroma','lanczos','detail','clahe','fine'):
            self.error(cpu[name],metal[name],'SIYI '+name,.1,3)
        same=metal['detail'];lab=cv2.cvtColor(same,cv2.COLOR_BGR2LAB)
        lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0])
        same=cv2.addWeighted(same,.65,cv2.cvtColor(lab,cv2.COLOR_LAB2BGR),.35,0)
        self.error(same,metal['clahe'],'SIYI CLAHE same input',.1,2)
        self.error(expected,actual,'SIYI full presentation parity',.1,3)

    def test_optimized_nlm_matches_original_kernel(self):
        rng=np.random.default_rng(918)
        for kind in ('textured','smooth','flat'):
            a=rng.integers(0,256,(71,97,3),np.uint8)
            if kind=='smooth':a=cv2.GaussianBlur(a,(9,9),2)
            if kind=='flat':a[:]=120
            source=self.gpu.upload(cv2.cvtColor(a,cv2.COLOR_BGR2BGRA))
            original=self.engine.denoise(source,optimized=False).numpy()
            fast=self.engine.denoise(source,optimized=True).numpy()
            self.error(original,fast,'NLM optimization '+kind,0,0)

    def test_full_recipe_and_alpha(self):
        rng=np.random.default_rng(314)
        a=cv2.GaussianBlur(rng.integers(0,256,(79,99,3),np.uint8),(7,7),1.5)
        a=cv2.cvtColor(a,cv2.COLOR_BGR2BGRA);a[:,:,3]=255
        cpu={};metal={};expected=enhance_reference(a,2,stages=cpu);actual=self.engine.enhance(a,2,stages=metal)
        for name in cpu:self.error(cpu[name],metal[name],name,1.,8)
        self.error(expected,actual,'full recipe',1.,8)
        a[:10,:,3]=0
        actual=self.engine.enhance(a,1.5)
        np.testing.assert_array_equal(actual[:,:,3],cv2.resize(a[:,:,3],(148,118),interpolation=cv2.INTER_NEAREST))
        self.assertFalse(actual[actual[:,:,3]==0,:3].any())
