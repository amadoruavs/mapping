"""Numerically compare the actual Metal kernel to the NumPy reference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, json, os, unittest
import numpy as np
from adm.metal import shade_metal
from adm.lighting import shade_linear

rng=np.random.default_rng(77)
rgb=rng.random((31,47,3),dtype=np.float32)
normals=rng.normal(size=rgb.shape).astype(np.float32)
normals[:,:,2]=np.abs(normals[:,:,2]); normals/=np.linalg.norm(normals,axis=2,keepdims=True)
visibility=rng.random((31,47),dtype=np.float32)
a=shade_linear(rgb,normals,visibility,30,255,.22)
b=shade_metal(rgb,normals,visibility,30,255,.22)
error=float(np.abs(a-b).max())
report={'max_abs_error':error,'pixels':1457,'tolerance':2e-6,'passed':error<2e-6}
Path('output').mkdir(exist_ok=True)
Path('output/metal_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2)); assert report['passed']

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--quick',action='store_true',help='Only run the original lighting equation check')
parser.add_argument('--pattern',default='test*.py',help='Unittest file pattern for targeted hardware validation')
args=parser.parse_args()
if not args.quick:
    os.environ['ADM_TEST_METAL']='1'
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),pattern=args.pattern)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    from adm.metal import get_backend
    report['suite']={'pattern':args.pattern,'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()}
    report['acceleration']=get_backend().stats()
    Path('output/metal_validation.json').write_text(json.dumps(report,indent=2))
    if not result.wasSuccessful(): raise SystemExit(1)
