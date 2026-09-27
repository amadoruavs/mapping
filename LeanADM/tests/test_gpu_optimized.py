"""Accuracy gates for GPU alignment, coherent source selection and optional shade."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from adm.config import Config
from adm.geometry import project
from adm.metadata import FrameMetadata


@unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
class OptimizedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from adm.metal import get_backend
        cls.gpu=get_backend()

    def test_analytic_jacobian_against_finite_difference(self):
        params=np.array([[1.02,.03,.2,-.02,.98,.1,.04,-.03],[.98,-.01,-.1,.02,1.03,.2,-.01,.02]],np.float32)
        points=np.array([[41,32,22,18],[79,55,47,39]],np.float32);owners=np.array([[0,1],[0,1]],np.uint32)
        jac=self.gpu.buffer((2,2,16));rw=self.gpu.buffer((2,4));cost=self.gpu.buffer(2)
        self.gpu.dispatch('align_linearize',[self.gpu.upload(params),self.gpu.upload(points),self.gpu.upload(owners),jac,rw,cost],2,[2],[100.])
        actual=jac.numpy()
        def residual(x):
            h=[np.r_[r,1.].reshape(3,3) for r in x]
            return project(h[0],points[:,:2]/100.)-project(h[1],points[:,2:]/100.)
        expected=np.empty_like(actual);eps=1e-6;x=params.astype(float)
        for k in range(16):
            a=x.copy();b=x.copy();a.flat[k]+=eps;b.flat[k]-=eps
            expected[:,:,k]=(residual(a)-residual(b))/(2*eps)
        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-5)

    def test_gpu_projective_solution_and_disconnected_rejection(self):
        from adm.gpu_alignment import solve
        rng=np.random.default_rng(9);width=600
        transforms=[np.array([[1,.02,70],[-.01,.98,20],[.00003,-.00002,1.]]),np.eye(3),np.array([[.99,-.01,-85],[.03,1.02,-15],[-.00002,.00001,1.]])]
        edges=[]
        for i,j in [(0,1),(1,2),(0,2)]:
            a=rng.uniform([60,60],[420,330],(90,2));H=np.linalg.inv(transforms[j])@transforms[i];H/=H[2,2]
            b=project(H,a)+rng.normal(0,.03,a.shape);initial=H.copy();initial[0,2]+=1.3
            edges.append(dict(i=i,j=j,H=initial,a=a,b=b,n=90))
        with tempfile.TemporaryDirectory() as d:
            with patch('scipy.optimize.least_squares',side_effect=AssertionError('CPU iterative solver forbidden')):
                fitted=solve(['a','b','c'],edges,Path(d),width,self.gpu,40)
            errors=np.concatenate([np.linalg.norm(project(fitted[e['i']],e['a'])-project(fitted[e['j']],e['b']),axis=1) for e in edges])
            self.assertLess(np.median(errors),.06);self.assertLess(np.percentile(errors,95),.12)
            np.testing.assert_allclose(fitted[1],np.eye(3),atol=1e-7)
            report=json.loads((Path(d)/'registration_report.json').read_text())
            self.assertTrue(report['solver_success'])
            self.assertLess(report['iterations_encoded'],40)
            self.assertLess(report['mps_operations'],40*4)
            with self.assertRaisesRegex(RuntimeError,'did not converge'):
                solve(['a','b','c'],edges,Path(d),width,self.gpu,1)
            with self.assertRaisesRegex(RuntimeError,'Disconnected'):solve(['a','b','c'],edges[:1],Path(d),width,self.gpu)

    def test_displaced_structure_keeps_one_source_without_ghost_or_erasure(self):
        from adm.coherent_composite import compose_pixels
        h,w=151,231;names=['a.png','b.png','c.png'];images=[]
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            for i,x in enumerate([65,100,135]):
                im=np.full((h,w,3),70,np.uint8);im[35:115,x:x+9]=235
                images.append(im);cv2.imwrite(str(out/names[i]),im)
            result,coverage,ids,valid=compose_pixels(out,names,[np.eye(3)]*3,np.int64(w),np.int64(h),Config(),out,self.gpu)
            report=json.loads((out/'composite_report.json').read_text())
            self.assertEqual(report['full_resolution_bytes'],w*h*12)
            interior=np.s_[15:-15,15:-15]
            np.testing.assert_array_equal(result[interior][:,:,:3],images[0][interior])
            self.assertTrue(np.all(ids[interior]==0));self.assertTrue(np.all(coverage[interior]==3))
            self.assertFalse((out/'warps').exists())

    def test_exact_initial_alignment_is_already_converged(self):
        from adm.gpu_alignment import solve
        points=np.array([[10,10],[10,90],[90,90],[90,10],[50,50]],float)
        edge=dict(i=0,j=1,H=np.eye(3),a=points,b=points.copy(),n=len(points))
        with tempfile.TemporaryDirectory() as d:
            result=solve(['a','b'],[edge],Path(d),100,self.gpu)
            np.testing.assert_array_equal(result[0],np.eye(3))
            report=json.loads((Path(d)/'registration_report.json').read_text())
            self.assertTrue(report['solver_success'])
            self.assertEqual(report['iterations_executed'],1)
            self.assertLessEqual(report['iterations_encoded'],4)

    def test_stylized_shadow_preserves_neutral_alpha_and_constant_image(self):
        from adm.coherent_composite import stylized_shadows
        cfg=Config();image=np.full((101,131,4),128,np.uint8);image[:,:,3]=255;original=image.copy()
        shaded=stylized_shadows(image,cfg,self.gpu)
        np.testing.assert_array_equal(image,original);np.testing.assert_array_equal(shaded,image)
        image[30:70,55:65,:3]=230;original=image.copy()
        shaded=stylized_shadows(image,cfg,self.gpu)
        np.testing.assert_array_equal(image,original);np.testing.assert_array_equal(shaded[:,:,3],image[:,:,3])
        self.assertTrue(np.all(shaded[:,:,:3]<=image[:,:,:3]));self.assertTrue(np.any(shaded[:,:,:3]<image[:,:,:3]))

    def test_coherent_exposure_step_is_reduced(self):
        from adm.coherent_composite import compose_pixels
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);names=['dark.png','light.png']
            for name,value in zip(names,[80,120]):cv2.imwrite(str(out/name),np.full((151,231,3),value,np.uint8))
            transforms=[np.eye(3),np.array([[1,0,100],[0,1,0],[0,0,1.]])]
            result,coverage,ids,valid=compose_pixels(out,names,transforms,331,151,Config(),out,self.gpu)
            interior=result[30:-30,120:210,0].astype(float)
            self.assertLess(np.max(np.abs(np.diff(interior,axis=1))),8)

    def test_compact_encoded_lighting_matches_previous_kernel(self):
        from adm.lighting import sun_vector
        rng=np.random.default_rng(5);im=rng.integers(20,220,(83,97,4),np.uint8);im[:,:,3]=255;cfg=Config()
        _,reference=self.gpu.lighting(im,cfg);source=self.gpu.upload(im);out=self.gpu.buffer(im.shape,np.uint8)
        self.gpu.dispatch('flat_light_encoded',[source,out],83*97,[83*97],[*sun_vector(cfg.sun_elevation_deg,cfg.sun_azimuth_deg),cfg.light_strength])
        np.testing.assert_array_equal(out.numpy(),reference)
        self.assertIs(self.gpu.shared_input(source.view()),source)

    def test_compact_pipeline_exports_and_resume(self):
        from adm.pipeline import run
        from adm.geometry import LocalEarth, intrinsics
        from test_pipeline import write_exif
        rng=np.random.default_rng(45)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();out=root/'result'
            base=cv2.GaussianBlur(rng.integers(30,210,(500,1100,3),dtype=np.uint8),(3,3),.6)
            cfg=Config(skip_first=0,registration_width=800,max_output_dimension=700,feature_count=3000)
            earth=LocalEarth(35,-97);dummy=FrameMetadata('a',800,500,35,-97,0)
            gsd=cfg.altitude_agl_m/intrinsics(dummy,cfg)[0,0]
            for i in range(3):
                lat,lon=earth.latlon(np.array([i*100*gsd,0.]))
                write_exif(source/f'{i}.jpg',base[:,i*100:i*100+800],lat,lon,0)
            config=root/'config.json';config.write_text(json.dumps(cfg.to_dict()))
            args=SimpleNamespace(source=source,output=out,config=config,altitude=None,max_dim=None,
                strict_corners=False,exclusions=None,metadata=None,allow_local=False,resume=False,
                heightmap=None,heightmap_spacing=None,backend='metal',gpu_optimized=True,stylized_shadows=True)
            run(args)
            report=json.loads((out/'report.json').read_text())
            self.assertTrue(report['georeferenced']);self.assertEqual(report['registered_pairs'],3)
            solver=json.loads((out/'registration_report.json').read_text())
            self.assertTrue(solver['solver_success']);self.assertLess(solver['iterations_encoded'],solver['iterations_budget'])
            self.assertLess(solver['p95_registration_error_px'],.25)
            self.assertFalse(json.loads((out/'footprint_index.json').read_text())['raw_pixel_copies'])
            for name in ('warps','tiles','tiles.sqlite','map_golden_linear.npy','map_golden.hdr'):
                self.assertFalse((out/name).exists())
            neutral=cv2.imread(str(out/'map_neutral.png'),-1);shadow=cv2.imread(str(out/'map_shadows.png'),-1)
            np.testing.assert_array_equal(neutral[:,:,3],shadow[:,:,3])
            cache_time=(out/'registration.npz').stat().st_mtime_ns
            args.resume=True;run(args)
            self.assertEqual(cache_time,(out/'registration.npz').stat().st_mtime_ns)
            np.testing.assert_array_equal(neutral,cv2.imread(str(out/'map_neutral.png'),-1))

            # Removing diagnostics must not change a single neutral-map pixel.
            args.resume=False;args.export_profile='png';args.max_output_mb=16
            args.output=root/'png'
            run(args)
            np.testing.assert_array_equal(neutral,cv2.imread(str(args.output/'map_neutral.png'),-1))
            for name in ('registration.npz','cache_fingerprint.txt','coverage.png','source_ids.png',
                         'vocabulary','corner_matches.json','footprint_index.json','map_golden.png','map_shadows.png'):
                self.assertFalse((args.output/name).exists(),name)
            composite=json.loads((args.output/'composite_report.json').read_text())
            self.assertEqual(composite['full_resolution_bytes'],neutral.shape[0]*neutral.shape[1]*8)

            args.cache_matches=True
            run(args)
            args.resume=True
            with patch('adm.pipeline.collect_matches',side_effect=AssertionError('Resume must reuse matches')):
                run(args)
            np.testing.assert_array_equal(neutral,cv2.imread(str(args.output/'map_neutral.png'),-1))

            # A vector export must never enter the pixel renderer or image exporter.
            args.output=root/'vectors';args.resume=False;args.cache_matches=False;args.export_profile='vectors'
            with patch('adm.coherent_composite.compose_pixels',side_effect=AssertionError('Unexpected rendering')), \
                 patch('adm.composite.write_image',side_effect=AssertionError('Unexpected image export')):
                run(args)
            recipe=json.loads((args.output/'project.json').read_text())
            self.assertEqual(len(recipe['image_to_map']),3)
            self.assertFalse(recipe['geometry']['rendered'])
            self.assertFalse(list(args.output.glob('*.png')))
            self.assertFalse((args.output/'registration.npz').exists())
