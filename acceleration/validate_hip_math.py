"""Compile HIP device equations as scalar host code to check the translation.

This checks shared source math, NOT HIP code generation, drivers or GPU execution.
"""
from pathlib import Path
import ctypes as C, json, subprocess, sys, tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];FSR=ROOT/'ODM-Fedora/fedora/finish';sys.path.insert(0,str(FSR))
from fsr import Kernel
code='''#include <algorithm>
#include <cmath>
#include <cstdint>
#define __device__
#include "fsr_device.h"
extern "C" void evaluate(const uint8_t*s,int w,int h,uint8_t*t,uint8_t*d,float sharp){
 for(int y=0;y<2*h;++y)for(int x=0;x<2*w;++x)store(t+(size_t(y)*2*w+x)*4,easu(s,w,h,x,y));
 for(int y=0;y<2*h;++y)for(int x=0;x<2*w;++x)store(d+(size_t(y)*2*w+x)*4,rcas(t,2*w,2*h,x,y,sharp));
}
'''
report={'hip_hardware_tested':False,'test':'HIP device equations compiled as host C++','passed':False}
with tempfile.TemporaryDirectory() as name:
 root=Path(name);(root/'math.cpp').write_text(code);lib=root/'math.dylib'
 subprocess.run(['c++','-std=c++17','-O3','-ffp-contract=off','-fno-fast-math','-shared','-fPIC','-I',str(FSR),str(root/'math.cpp'),'-o',str(lib)],check=True)
 f=C.CDLL(str(lib)).evaluate;f.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_float]
 cpu=Kernel('cpu',2);rng=np.random.default_rng(365);cases=0;max_error=0
 for h,w in [(1,1),(2,3),(17,31),(129,137)]:
  for alpha in ['opaque','transparent','random']:
   image=rng.integers(0,256,(h,w,4),dtype=np.uint8)
   if alpha!='random':image[:,:,3]=255 if alpha=='opaque' else 0
   for sharp in [0.,.25,1.]:
    tmp=np.empty((2*h,2*w,4),np.uint8);out=np.empty_like(tmp)
    f(image.ctypes.data,w,h,tmp.ctypes.data,out.ctypes.data,sharp);reference=cpu.run(image,sharp)
    max_error=max(max_error,int(np.abs(out.astype(np.int16)-reference.astype(np.int16)).max()));cases+=1
 report.update(passed=max_error==0,cases=cases,max_pixel_error=max_error)
OUT=ROOT/'acceleration/validation/hip_host_math.json';OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));sys.exit(0 if report['passed'] else 1)
