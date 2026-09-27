"""Compile actual portable kernel sources on the host; not a GPU/NPU test."""
import ctypes
from pathlib import Path
import os, shutil, subprocess, tempfile, unittest
import numpy as np
from .pixels import cpu_composite, composite_bgra
from unittest.mock import patch

KERNELS=Path(__file__).parent/'kernels'

def flat_cases():
    h,w=161,193;y,x=np.mgrid[:h,:w]
    plane=(1/(.01+x*1e-7+y*2e-7)).astype(np.float32)
    confidence=np.full((h,w),.01,np.float32)
    yield 'plane',plane,confidence
    for name in ('roof_edge','rough_canopy','hole','uncertain','nan','infinity'):
        d=plane.copy();c=confidence.copy()
        if name=='roof_edge':d[:,80:]-=3
        if name=='rough_canopy':d[32:128,32:128]+=np.random.default_rng(5).normal(0,.2,(96,96))
        if name=='hole':d[64,64]=0
        if name=='uncertain':c[64,64]=.031
        if name=='nan':d[64,64]=np.nan
        if name=='infinity':c[64,64]=np.inf
        yield name,d,c

class NativeMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();root=Path(cls.temp.name)
        cxx=os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
        if not cxx:raise RuntimeError('A C++ compiler is required to check the kernel sources')
        wrapper=root/'host.cpp'
        wrapper.write_text('''#include "composite.h"
#include "flat.h"
extern "C" void pixels(const uint32_t* s,uint32_t* d,int n,uint32_t bg){for(int i=0;i<n;++i)d[i]=adm_pixel(s[i],bg);}
extern "C" void flat(const float* d,const float* c,int w,int h,uint8_t* o){adm_flat_cpu(d,c,w,h,o);}
extern "C" bool interior(const uint8_t* m,int w,int h,int x,int y){return adm_flat_interior(m,w,h,x,y);}
extern "C" unsigned divide(unsigned t){return (t+1+(t>>8))>>8;}
''')
        output=root/'host.so'
        subprocess.run([cxx,'-std=c++17','-O3','-ffp-contract=off','-fno-fast-math','-shared','-fPIC','-I',str(KERNELS),str(wrapper),str(KERNELS/'xdna_composite.cpp'),'-o',str(output)],check=True)
        cls.lib=ctypes.CDLL(str(output))
        cls.lib.pixels.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_uint32]
        cls.lib.adm_xdna_composite.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int32]
        cls.lib.flat.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        cls.lib.interior.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        cls.lib.interior.restype=ctypes.c_bool
        cls.lib.divide.argtypes=[ctypes.c_uint];cls.lib.divide.restype=ctypes.c_uint
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_division_exhaustive(self):
        for t in range(65153):self.assertEqual(self.lib.divide(t),t//255)
    def test_actual_vector_and_scalar_sources(self):
        a,f=np.mgrid[:256,:256]
        src=np.stack([f,255-f,17*f%256,a],axis=-1).astype(np.uint8)
        out=np.empty_like(src)
        for bg in ([0,0,0],[255,255,255],[17,128,241]):
            b=bg[0]|(bg[1]<<8)|(bg[2]<<16)
            self.lib.pixels(src.ctypes.data,out.ctypes.data,65536,b)
            np.testing.assert_array_equal(out[:,:,:3],cpu_composite(src,bg))
            self.assertTrue(np.all(out[:,:,3]==255))
        self.lib.adm_xdna_composite(src.ctypes.data,out.ctypes.data,65536)
        np.testing.assert_array_equal(out[:,:,:3],cpu_composite(src,[255]*3))
        # Multiple vector tiles, all byte values, and a non-aligned input address.
        rng=np.random.default_rng(365)
        for n in (16,1024,65536):
            storage=rng.integers(0,256,n*4+1,dtype=np.uint8);src=storage[1:];out=np.empty_like(src)
            self.lib.adm_xdna_composite(src.ctypes.data,out.ctypes.data,n)
            np.testing.assert_array_equal(out.reshape(1,n,4)[:,:,:3],cpu_composite(src.reshape(1,n,4),[255]*3))
    def test_plane_and_hazard_rejection(self):
        for name,d,c in flat_cases():
            h,w=d.shape;mask=np.zeros(((h+31)//32,(w+31)//32),np.uint8)
            self.lib.flat(d.ctypes.data,c.ctypes.data,w,h,mask.ctypes.data)
            self.assertFalse(mask[-1,:].any());self.assertFalse(mask[:,-1].any())
            if name=='plane':self.assertTrue(mask[:-1,:-1].all())
            else:self.assertEqual(mask[2,2],0,name)
            for yy,xx in [(0,0),(0,2),(2,0),(mask.shape[0]-1,2)]:
                self.assertFalse(self.lib.interior(mask.ctypes.data,mask.shape[1],mask.shape[0],xx,yy))
            if name!='plane':self.assertFalse(self.lib.interior(mask.ctypes.data,mask.shape[1],mask.shape[0],3,2))
    def test_native_failure_cpu_fallback(self):
        src=np.full((256,256,4),80,np.uint8)
        with patch('parallel_runtime.native.automatic',return_value='native-hip'),patch('parallel_runtime.native.run',side_effect=RuntimeError('lost device')):
            np.testing.assert_array_equal(composite_bgra(src,backend='auto'),cpu_composite(src,[255]*3))

if __name__=='__main__':unittest.main()
