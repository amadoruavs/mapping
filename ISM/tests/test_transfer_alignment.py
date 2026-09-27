"""Image-space transfer residuals must retain scale and analytical derivatives."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from adm.geometry import project


@unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
class TransferTests(unittest.TestCase):
    def setUp(self):
        from adm.metal import get_backend
        self.gpu=get_backend()

    def test_bidirectional_derivatives_and_cost(self):
        H=np.array([[[1.1,.02,.25],[-.02,.95,.14],[.01,-.005,1]],
                    [[.94,-.03,-.18],[.03,1.05,.02],[-.006,.012,1]]])
        a=np.array([[.2,.3],[.45,.22]])
        b=project(np.linalg.inv(H[1])@H[0],a)+[[.002,-.001],[-.003,.0015]]
        pts=np.repeat(np.c_[a,b],2,axis=0);owners=np.tile([0,1],(4,1)).astype(np.uint32)
        x=H.reshape(2,9)[:,:8].copy();width=600
        def residual(v,k):
            hs=np.c_[v,np.ones(2)].reshape(2,3,3)
            i,j=(0,1) if k%2==0 else (1,0)
            aa,bb=(pts[k,:2],pts[k,2:]) if k%2==0 else (pts[k,2:],pts[k,:2])
            return project(np.linalg.inv(hs[j])@hs[i],[aa])[0]-bb
        expected=np.array([residual(x,k) for k in range(4)])
        finite=np.empty((4,2,16));eps=1e-5
        for q in range(16):
            plus=x.copy();minus=x.copy();plus.flat[q]+=eps;minus.flat[q]-=eps
            finite[:,:,q]=np.array([(residual(plus,k)-residual(minus,k))/(2*eps) for k in range(4)])
        buffers=[self.gpu.upload(x.astype(np.float32)),self.gpu.upload((pts*width).astype(np.float32)),self.gpu.upload(owners),self.gpu.buffer((4,2,16)),self.gpu.buffer((4,4)),self.gpu.buffer(4)]
        self.gpu.dispatch('align_transfer_linearize',buffers,4,[4],[width]);self.gpu.sync()
        np.testing.assert_allclose(buffers[3].numpy(),finite,rtol=5e-4,atol=2e-6)
        np.testing.assert_allclose(buffers[4].numpy()[:,:2],expected,atol=3e-7)
        cost=.5*np.sum(np.sqrt(1+(expected*width)**2)-1,axis=1)
        np.testing.assert_allclose(buffers[5].numpy(),cost,rtol=3e-4,atol=2e-4)
        self.gpu.dispatch('align_transfer_cost',buffers[:3]+[buffers[5]],4,[4],[width]);self.gpu.sync()
        np.testing.assert_allclose(buffers[5].numpy(),cost,rtol=3e-4,atol=2e-4)
        # A common 100x shrink changes mosaic units, not source-pixel error.
        shrunk=np.array([np.diag([.01,.01,1.])@h for h in H]).reshape(2,9)[:,:8]
        self.gpu.dispatch('align_transfer_cost',[self.gpu.upload(shrunk.astype(np.float32)),*buffers[1:3],buffers[5]],4,[4],[width]);self.gpu.sync()
        np.testing.assert_allclose(buffers[5].numpy(),cost,rtol=3e-4,atol=2e-4)

    def test_solver_preserves_scale_on_noisy_visual_matches(self):
        from adm.gpu_alignment import solve
        rng=np.random.default_rng(80);width=600
        hs=[np.array([[1,.02,70],[-.01,.98,20],[.00003,-.00002,1.]]),np.eye(3),
            np.array([[.99,-.01,-85],[.03,1.02,-15],[-.00002,.00001,1.]])]
        edges=[]
        for i,j in ((0,1),(1,2),(0,2)):
            a=rng.uniform([40,40],[560,420],(160,2));H=np.linalg.inv(hs[j])@hs[i];H/=H[2,2]
            b=project(H,a)+rng.normal(0,.15,a.shape)
            edges.append(dict(i=i,j=j,H=H,a=a,b=b,n=len(a)))
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            actual=solve(['a','b','c'],edges,Path(folder),width,self.gpu,symmetric=True)
        corners=np.array([[0,0],[599,0],[599,449],[0,449]])
        for i in range(3):np.testing.assert_allclose(project(actual[i],corners),project(hs[i],corners),atol=.5)

    def test_heavy_damping_is_not_convergence(self):
        state=np.zeros(20,np.float32);state[0]=15000;state[2]=8
        buffers=[self.gpu.upload(np.array([37095272],np.float32)),
                 self.gpu.upload(np.array([37095248],np.float32)),
                 self.gpu.upload(np.array([0],np.int32)),self.gpu.upload(state),
                 self.gpu.upload(np.full(8,1e-8,np.float32)),self.gpu.upload(np.ones(8,np.float32)),
                 self.gpu.upload(np.zeros(8,np.float32))]
        self.gpu.dispatch('align_accept',buffers,1,[0,8,1],[600]);self.gpu.sync()
        self.assertEqual(buffers[3].numpy()[6],0)
