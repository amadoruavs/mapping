"""Physical geometry, CPU/Metal residual/Jacobian parity and convergence."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from adm.config import Config
from adm.metadata import FrameMetadata
from adm.geometry import ground_homography, project
from adm.geographic_alignment import TopocentricEarth, pack_cameras, camera_model


def fixture():
    cfg=Config(altitude_agl_m=50,registration_width=640,gpu_alignment=True,
               geographic_solver='opensfm_planar',alignment_max_iterations=120)
    earth=TopocentricEarth(41.,-87.)
    ll=earth.latlon(np.array([[0,0],[15,5],[30,0]]))
    metadata=[FrameMetadata(str(i),640,480,latitude=lat,longitude=lon,yaw_deg=12+i*2,
        pitch_deg=-86+i,roll_deg=.5,altitude_agl_m=50,
        provenance={k:'XMP' for k in ('yaw_deg','pitch_deg','roll_deg')}) for i,(lat,lon) in enumerate(ll)]
    x,data,scale=pack_cameras(metadata,cfg,earth)
    return cfg,earth,metadata,x,data,scale


class GeographicCPU(unittest.TestCase):
    def test_topocentric_roundtrip_and_physical_camera(self):
        cfg,earth,metadata,x,data,scale=fixture()
        xy=np.array([[0,0],[100,250],[-350,240]])
        np.testing.assert_allclose(earth.xy(*earth.latlon(xy).T),xy,atol=1e-5)
        H,_,_,_=camera_model(x,data)
        for h,m in zip(H,metadata):
            pixels=[[0,0],[320,240],[639,479]]
            np.testing.assert_allclose(project(h,np.array(pixels)/640)*scale,
                                       project(ground_homography(m,cfg,earth),pixels),atol=1e-5)

    def test_config_rejects_invalid_uncertainties(self):
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):Config(geographic_gps_sigma_m=value).validate()


@unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
class GeographicMetal(unittest.TestCase):
    def setUp(self):
        from adm.metal import get_backend
        self.gpu=get_backend()

    def model(self,x,data):
        n=len(x);g=self.gpu
        buffers=[g.upload(np.asarray(x,np.float32)),g.upload(data),g.buffer((n,9)),g.buffer((n,8,9)),g.buffer((n,72)),g.buffer((n,3))]
        g.dispatch('geo_camera_model',buffers,n,[n,0],[640]);g.sync()
        return buffers

    def test_camera_and_prior_derivatives(self):
        cfg,earth,metadata,x,data,scale=fixture();x=x.astype(np.float64)
        x[:,3:6]=[[1e-7,-2e-7,3e-7],[.04,-.06,.08],[-.2,.3,-.15]]
        x[:,:3]+=[.05,-.02,.01]
        result=self.model(x,data);H,R,res,cost=camera_model(x,data)
        actual=result[2].numpy().reshape(-1,3,3).transpose(0,2,1)
        np.testing.assert_allclose(actual,H,rtol=2e-6,atol=3e-7)
        np.testing.assert_allclose(result[5].numpy(),cost,rtol=2e-5,atol=3e-7)
        eps=1e-5;D=np.zeros((3,8,9));J=np.zeros((3,9,8))
        for k in range(6):
            plus=x.copy();minus=x.copy();plus[:,k]+=eps;minus[:,k]-=eps
            hp,_,rp,_=camera_model(plus,data);hm,_,rm,_=camera_model(minus,data)
            D[:,k]=((hp-hm)/(2*eps)).transpose(0,2,1).reshape(-1,9)
            J[:,:,k]=(rp-rm)/(2*eps)/640
        np.testing.assert_allclose(result[3].numpy(),D,rtol=2e-5,atol=2e-6)
        weights=np.ones((3,9))
        for part in (slice(3,6),slice(6,9)):weights[:,part]=1/(1+np.sum(res[:,part]**2,axis=1,keepdims=True))
        normal=np.einsum('nrk,nr,nrl->nkl',J,weights,J)
        grad=np.einsum('nrk,nr,nr->nk',J,weights,res/640)
        expected=np.c_[normal.reshape(3,64),grad]
        np.testing.assert_allclose(result[4].numpy(),expected,rtol=3e-4,atol=2e-10)

    def test_bidirectional_visual_derivatives(self):
        cfg,earth,metadata,x,data,scale=fixture();x=x.astype(np.float64);x[:,3:6]=[.02,-.01,.03]
        H=camera_model(x,data)[0];a=np.array([[.45,.3],[.55,.4]])
        b=project(np.linalg.inv(H[1])@H[0],a)+[.003,-.002]
        points=np.repeat(np.c_[a,b]*640,2,axis=0).astype(np.float32)
        owners=np.tile([0,1],(4,1)).astype(np.uint32);model=self.model(x,data);g=self.gpu
        out=[g.buffer((4,2,16)),g.buffer((4,4)),g.buffer(4)]
        buffers=[model[2],model[3],g.upload(points),g.upload(owners),*out]
        g.dispatch('geo_visual_linearize',buffers,4,[4],[640]);g.sync()
        def residual(v):
            hs=camera_model(v,data)[0];rows=[]
            for k in range(4):
                i,j=(1,0) if k%2 else (0,1);aa,bb=(points[k,2:],points[k,:2]) if k%2 else (points[k,:2],points[k,2:])
                rows.append(project(np.linalg.inv(hs[j])@hs[i],[aa.astype(float)/640])[0]-bb.astype(float)/640)
            return np.array(rows)
        eps=1e-5;J=np.zeros((4,2,16))
        for k in range(16):
            plus=x.copy();minus=x.copy();plus.flat[k]+=eps;minus.flat[k]-=eps
            J[:,:,k]=(residual(plus)-residual(minus))/(2*eps)
        np.testing.assert_allclose(out[0].numpy(),J,rtol=2e-4,atol=1e-6)
        r=residual(x);expected=.5*np.sum(np.sqrt(1+(r*640)**2)-1,axis=1)
        np.testing.assert_allclose(out[2].numpy(),expected,rtol=1e-4,atol=3e-4)
        g.dispatch('geo_visual_cost',[model[2],buffers[2],buffers[3],out[2]],4,[4],[640]);g.sync()
        np.testing.assert_allclose(out[2].numpy(),expected,rtol=1e-4,atol=3e-4)

    def test_above_horizon_and_collapsed_height_rejected(self):
        _,_,_,x,data,_=fixture();x[0,2]=.001;x[1,3]=3.0
        cost=self.model(x,data)[5].numpy()
        self.assertTrue(np.isinf(cost[0,0]));self.assertTrue(np.isinf(cost[1,0]))

    def test_joint_solver_recovers_noisy_camera_positions(self):
        from adm.geographic_alignment import solve
        cfg,earth,metadata,x,data,scale=fixture();H=camera_model(x,data)[0]
        rng=np.random.default_rng(64);edges=[]
        for i,j in ((0,1),(1,2),(0,2)):
            a=rng.uniform([.4,.1],[.9,.6],(100,2))*640
            transform=np.diag([640,640,1.])@np.linalg.inv(H[j])@H[i]@np.diag([1/640,1/640,1.])
            b=project(transform,a)+rng.normal(0,.05,a.shape)
            edges.append(dict(i=i,j=j,n=100,a=a,b=b,H=transform))
        ll=earth.latlon(np.array([[.3,-.2],[14.6,5.4],[30.1,-.2]]))
        for m,(lat,lon) in zip(metadata,ll):m.latitude=lat;m.longitude=lon
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            fitted=solve([m.name for m in metadata],edges,metadata,earth,cfg,Path(temp),self.gpu)
            import json
            report=json.loads((Path(temp)/'registration_report.json').read_text())
        self.assertTrue(report['solver_success']);self.assertLess(report['median_registration_error_px'],.1)
        for e in edges:
            np.testing.assert_allclose(project(np.linalg.inv(fitted[e['j']])@fitted[e['i']],e['a']),e['b'],atol=.25)
