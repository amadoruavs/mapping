"""Compare Metal geographic priors with actual pinned OpenSfM C++ functors.

Optional --diagnose reuses saved correspondences, never a full-process timing.
"""
import argparse
import ctypes as C
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ISM/runtime'))
from adm.geographic_alignment import TopocentricEarth,pack_cameras,camera_model,solve
from adm.metadata import FrameMetadata
from adm.config import Config


def validate():
    folder=ROOT/'ISM/benchmarks/odm_geo94'
    lib=C.CDLL(str(folder/'opensfm_prior_reference.dylib'))
    ptr=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
    lib.opensfm_position.argtypes=[ptr]*4;lib.opensfm_position.restype=None
    lib.opensfm_up.argtypes=[ptr,ptr,C.c_double,ptr,ptr];lib.opensfm_up.restype=None
    rng=np.random.default_rng(327);n=128;data=np.zeros((n,32),np.float32);x=np.zeros((n,8),np.float32)
    bases=Rotation.random(n,random_state=rng).as_matrix()
    data[:,:9]=bases.transpose(0,2,1).reshape(n,9);data[:,9:18]=np.eye(3).ravel()
    data[:,18:21]=rng.normal(0,2,(n,3));data[:,20]=2
    data[:,21:24]=rng.uniform(1,10,(n,3));data[:,24]=rng.uniform(1,20,n);data[:,25]=.75
    # Upstream prior tests are independent of the horizon constraint; compare
    # orientation costs directly, and GPS separately on downward-facing cameras.
    x[:,:3]=data[:,18:21]+rng.normal(0,.03,(n,3));x[:,3:6]=rng.normal(0,.25,(n,3))
    H,R,res,cost=camera_model(x,data)
    expected=np.empty_like(cost);max_position=0.;axis_error=0.
    for i in range(n):
        center=np.ascontiguousarray(x[i,:3],float);prior=np.ascontiguousarray(data[i,18:21],float)
        sigma=np.ascontiguousarray(1/data[i,21:24].astype(float));r=np.zeros(3)
        lib.opensfm_position(center,prior,sigma,r)
        max_position=max(max_position,float(np.max(np.abs(np.sort(r)-np.sort(res[i,:3])))))
        expected[i,0]=.5*r@r
        base=data[i,:9].reshape(3,3).T.astype(float)
        for axis in (0,1):
            Q=np.eye(3) if axis==0 else Rotation.from_euler('x',90,degrees=True).as_matrix()
            target=np.array([0,0,1]) if axis==0 else np.array([0,1,0])
            angle=np.ascontiguousarray(Rotation.from_matrix(Q@R[i]).as_rotvec())
            acceleration=np.ascontiguousarray(base.T@target);out=np.zeros(3);rho=np.zeros(2)
            lib.opensfm_up(angle,acceleration,1/float(data[i,24]),out,rho)
            axis_error=max(axis_error,float(np.max(np.abs(out-Q@res[i,3+3*axis:6+3*axis]))))
            expected[i,axis+1]=rho[0]
    np.testing.assert_allclose(cost,expected,rtol=2e-6,atol=2e-6)
    from adm.metal import get_backend
    gpu=get_backend();buffers=[gpu.upload(x),gpu.upload(data),gpu.buffer((n,9)),gpu.buffer((n,8,9)),gpu.buffer((n,72)),gpu.buffer((n,3))]
    gpu.dispatch('geo_camera_model',buffers,n,[n,0],[1280]);gpu.sync()
    actual=buffers[-1].numpy()
    np.testing.assert_allclose(actual[:,1:],expected[:,1:],rtol=3e-6,atol=3e-6)
    finite=np.isfinite(actual[:,0]);assert finite.sum()>0
    np.testing.assert_allclose(actual[finite,0],expected[finite,0],atol=2e-6)
    spec=importlib.util.spec_from_file_location('actual_opensfm_geo',ROOT/'third_party/opensfm_reference/opensfm/geo.py')
    geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
    earth=TopocentricEarth(41.,-87.);ll=np.array([[41,-87],[41.002,-87.004],[40.999,-86.999]])
    ref=np.array(geo.topocentric_from_lla(ll[:,0],ll[:,1],0,41,-87,0)).T[:,:2]
    diff=float(np.max(np.abs(earth.xy(ll[:,0],ll[:,1])-ref)));assert diff<2e-8
    report=dict(samples=n,passed=True,actual_unmodified_cpp_functors=['DataPriorError','UpVectorError','ceres::CauchyLoss'],
        valid_ground_cameras_checked=int(finite.sum()),
        position_cpu_max_error=max_position,axis_cpu_max_error=axis_error,
        metal_orientation_cost_max_error=float(np.max(np.abs(actual[:,1:]-expected[:,1:]))),
        wgs84_against_unmodified_opensfm_max_error_m=diff,
        note='Residual/cost parity, not a claim of identical full Ceres/OpenSfM bundle solutions')
    (folder/'upstream_parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


def diagnose(reuse,output):
    if output.exists():raise ValueError('Use a new diagnostic output directory')
    output.mkdir(parents=True)
    md=[FrameMetadata(**{k:v for k,v in m.items() if k in FrameMetadata.__dataclass_fields__}) for m in json.loads((reuse/'metadata.json').read_text())]
    spec=importlib.util.spec_from_file_location('runner',ROOT/'ISM/run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    cfg=Config(**runner.SETTINGS);cfg.gpu_alignment=True;cfg.coherent_composite=True;cfg.export_profile='png';cfg.validate()
    with np.load(reuse/'registration.npz',allow_pickle=True) as cache:edges=cache['edges'].tolist()
    from adm.metal import get_backend
    earth=TopocentricEarth(md[0].latitude,md[0].longitude)
    solve([m.name for m in md],edges,md,earth,cfg,output,get_backend())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reuse',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.reuse:
        if not args.output:parser.error('--output required')
        diagnose(args.reuse,args.output)
    else:validate()
