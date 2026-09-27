"""Metric pose constraints, analytic derivatives, and prevention of collapsed maps."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from adm.config import Config
from adm.geometry import LocalEarth,project
from adm.metadata import FrameMetadata
from adm.geo_priors import build_priors,weights,diagnostics
from adm.registration import solve


def fixture(strength=4.):
    earth=LocalEarth(36.,-95.)
    cfg=Config(gpu_alignment=True,gps_yaw_alignment=True,registration_width=600,
        altitude_agl_m=50.,gps_prior_strength=strength)
    metadata=[]
    for i,(x,y,yaw) in enumerate(((-12.,0.,359.),(0.,0.,0.),(12.,1.,1.))):
        lat,lon=earth.latlon([x,y]);metadata.append(FrameMetadata(str(i),1200,900,
            latitude=float(lat),longitude=float(lon),yaw_deg=yaw,altitude_agl_m=50.))
    priors=build_priors(metadata,cfg,earth);truth=priors['initial'].copy()
    # Deliberate visual/GPS disagreement: observations place one camera 2m east.
    truth[2]=np.array([[1,0,2/priors['metres_per_pixel']],[0,1,0],[0,0,1.]])@truth[2]
    rng=np.random.default_rng(73);edges=[]
    for i,j in ((0,1),(1,2),(0,2)):
        a=rng.uniform([210,100],[380,330],(90,2));H=np.linalg.inv(truth[j])@truth[i];H/=H[2,2]
        b=project(H,a)+rng.normal(0,.1,a.shape)
        edges.append(dict(i=i,j=j,H=H,a=a,b=b,n=len(a)))
    return metadata,cfg,earth,priors,edges


class GeoPriorTests(unittest.TestCase):
    def test_absolute_pose_roundtrip_and_yaw_wrap(self):
        metadata,cfg,earth,priors,_=fixture()
        for m,H in zip(metadata,priors['initial']):
            center=np.array([[(m.width-1)/2*600/m.width,(m.height-1)/2*round(m.height*600/m.width)/m.height]])
            world=project(priors['alignment_to_world']@H,center)[0]
            np.testing.assert_allclose(world,earth.xy(m.latitude,m.longitude),atol=1e-8)
            up=project(priors['alignment_to_world']@H,center+[0,-20])[0]-world
            heading=np.rad2deg(np.arctan2(up[0],up[1]))%360
            self.assertAlmostEqual((heading-m.yaw_deg+180)%360-180,0.,places=7)
        metadata[0].yaw_deg=None
        with self.assertRaisesRegex(ValueError,'absolute camera yaw'):build_priors(metadata,cfg,earth)

    def test_config_rejects_invalid_prior_settings(self):
        for key,value in (('gps_prior_sigma_m',0),('gps_prior_strength',-1),('yaw_prior_sigma_deg',float('nan')),('scale_prior_sigma',2)):
            with self.assertRaises(ValueError):Config(**{key:value}).validate()

    @unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
    def test_prior_derivatives_and_cost_against_cpu(self):
        from adm.metal import get_backend
        _,_,_,priors,edges=fixture();gpu=get_backend();w=600
        S=np.diag([1/w,1/w,1.]);Si=np.linalg.inv(S)
        x=np.array([S@H@Si for H in priors['initial']]);x=x.reshape(3,9)[:,:8].copy()
        x[0,0]*=1.015;x[0,2]+=.002;x[0,6]+=.003
        samples=priors['samples'].astype(float);ww=weights(priors,edges)
        def residual(parameters,i):
            H=np.r_[parameters,1.].reshape(3,3);points=project(H,samples[i,:,:2]);target=samples[i,:,2:]
            values=list((points[0]-target[0])*np.sqrt(ww[i,0]))
            for k in range(1,5):
                v=target[k]-target[0];length=np.linalg.norm(v);axis=v/length
                d=points[k]-points[0]-v
                values.extend([np.dot(d,axis)/length*np.sqrt(ww[i,1]),np.dot(d,[-axis[1],axis[0]])/length*np.sqrt(ww[i,2])])
            return np.array(values)
        source=gpu.upload(x.astype(np.float32));sp=gpu.upload(priors['samples']);wb=gpu.upload(ww)
        blocks=gpu.buffer((3,72));cost=gpu.buffer(3)
        gpu.dispatch('align_prior_linearize',[source,sp,wb,blocks],216,[3],[w])
        gpu.dispatch('align_prior_cost',[source,sp,wb,cost],3,[3],[w])
        actual=blocks.numpy();cost=cost.numpy()
        for i in range(3):
            parameters=x[i].astype(np.float32).astype(float);r=residual(parameters,i);J=[]
            for k in range(8):
                a=parameters.copy();b=parameters.copy();a[k]+=1e-5;b[k]-=1e-5
                J.append((residual(a,i)-residual(b,i))/2e-5)
            J=np.array(J).T
            np.testing.assert_allclose(actual[i,:64].reshape(8,8),J.T@J,rtol=2e-4,atol=2e-5)
            np.testing.assert_allclose(actual[i,64:],J.T@r,rtol=2e-3,atol=2e-5)
            self.assertAlmostEqual(float(cost[i]),.5*np.dot(r,r)*w*w,delta=.03)

    @unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
    def test_collapsed_trial_is_rejected(self):
        from adm.metal import get_backend
        _,_,_,priors,edges=fixture();gpu=get_backend();w=600
        hs=priors['initial'].copy();center=project(hs[0],priors['samples'][0,:1,:2]*w)[0]
        hs[0]=np.array([[.01,0,.99*center[0]],[0,.01,.99*center[1]],[0,0,1.]])@hs[0]
        S=np.diag([1/w,1/w,1.]);x=np.array([S@H@np.linalg.inv(S) for H in hs]).reshape(3,9)[:,:8]
        cost=gpu.buffer(3);gpu.dispatch('align_prior_cost',[gpu.upload(x.astype(np.float32)),gpu.upload(priors['samples']),gpu.upload(weights(priors,edges)),cost],3,[3],[w])
        result=cost.numpy();self.assertTrue(np.isinf(result[0]));self.assertTrue(np.isfinite(result[1:]).all())
        self.assertFalse(diagnostics(hs,priors)['valid_footprints'])

    @unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
    def test_stronger_gps_pulls_solution_toward_positions(self):
        from adm.metal import get_backend
        reports=[]
        with tempfile.TemporaryDirectory() as folder:
            for strength in (.1,4.,64.):
                _,cfg,_,priors,edges=fixture(strength)
                with contextlib.redirect_stdout(io.StringIO()):
                    hs=solve(['0','1','2'],edges,Path(folder),600,get_backend(),iterations=100,geo_priors=priors)
                report=diagnostics([hs[i] for i in range(3)],priors);reports.append(report)
                self.assertTrue(report['valid_footprints']);self.assertGreater(report['footprint_area_ratio_min'],.5)
        print('GPS median errors, weak/default/strong:',[r['gps_center_median_error_m'] for r in reports],flush=True)
        self.assertLess(reports[1]['gps_center_median_error_m'],reports[0]['gps_center_median_error_m'])
        self.assertLess(reports[2]['gps_center_median_error_m'],reports[0]['gps_center_median_error_m']*.5)
