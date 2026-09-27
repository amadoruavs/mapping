"""Build/load the bundled C++17 CPU code. No Metal, CUDA or GPU required."""
import ctypes as C
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent


class Native:
    def __init__(self,threads=None):
        self.threads=max(1,min(32,threads or os.cpu_count() or 1))
        source=ROOT/'native.cpp'
        digest=hashlib.sha256(source.read_bytes()+str((platform.system(),platform.machine())).encode()).hexdigest()[:16]
        suffix='.dll' if sys.platform=='win32' else '.dylib' if sys.platform=='darwin' else '.so'
        binary=ROOT/'build'/('maprepair-'+digest+suffix);binary.parent.mkdir(exist_ok=True)
        if not binary.exists():
            candidates=[os.environ.get('CXX',''),'clang++','g++','c++','cl']
            compiler=next((shlex.split(c) for c in candidates if c and shutil.which(shlex.split(c)[0])),None)
            if compiler is None:raise RuntimeError('Install a C++17 compiler (Xcode tools, GCC/Clang, or Visual Studio C++), or set CXX.')
            temporary=binary.with_name(binary.stem+f'-{os.getpid()}'+suffix)
            if Path(compiler[0]).stem.lower()=='cl':
                command=compiler+['/nologo','/O2','/std:c++17','/EHsc','/LD',str(source),'/link','/OUT:'+str(temporary)]
            else:
                command=compiler+['-std=c++17','-O3','-ffp-contract=off','-pthread',
                    '-dynamiclib' if sys.platform=='darwin' else '-shared','-fPIC',str(source),'-o',str(temporary)]
            print('[native] Compiling portable C++ kernels',flush=True)
            result=subprocess.run(command,cwd=binary.parent,capture_output=True,text=True)
            if result.returncode:raise RuntimeError('C++ build failed:\n'+result.stdout+result.stderr)
            temporary.replace(binary)
        self.binary=binary;self.lib=C.CDLL(str(binary));ptr=C.c_void_p
        self.lib.mr_error.restype=C.c_char_p
        self.lib.mr_fsr_rows.argtypes=[ptr,C.c_int,C.c_int,ptr,C.c_int,C.c_int,C.c_int,C.c_float,C.c_int]
        self.lib.mr_trace.argtypes=[ptr,C.c_int,ptr,C.c_int,C.c_int,*([C.c_float]*5),C.c_int,C.c_int]

    def check(self,result):
        if result:raise RuntimeError(self.lib.mr_error().decode())

    def fsr2x(self,image,workdir,sharpness=.2,row_chunk=256,progress=print):
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=4:raise ValueError('FSR expects BGRA8')
        if not np.isfinite(sharpness) or not 0<=sharpness<=1:raise ValueError('FSR sharpness must be 0..1')
        source=np.ascontiguousarray(image);h,w=source.shape[:2];shape=(h*2,w*2,4)
        workdir=Path(workdir);workdir.mkdir(parents=True,exist_ok=True)
        # Disk-backed image storage also works on machines with less RAM.
        temp=np.memmap(workdir/'easu.bgra',mode='w+',dtype=np.uint8,shape=shape)
        output=np.memmap(workdir/'final.bgra',mode='w+',dtype=np.uint8,shape=shape)
        for step,(src,dst,pass_id) in enumerate(((source,temp,0),(temp,output,1))):
            sh,sw=src.shape[:2];last=-1
            for y in range(0,shape[0],row_chunk):
                count=min(row_chunk,shape[0]-y)
                self.check(self.lib.mr_fsr_rows(src.ctypes.data,sw,sh,dst[y:].ctypes.data,y,count,pass_id,sharpness,self.threads))
                percent=(y+count)*100//shape[0]
                if percent//20!=last:
                    progress(f'[FSR {"EASU" if step==0 else "RCAS"}] {percent}%');last=percent//20
            dst.flush()
        temp._mmap.close();(workdir/'easu.bgra').unlink()
        return output

    def trace(self,vertices,map_shape,grid=128,samples=8,elevation=8.,azimuth=255.):
        vertices=np.ascontiguousarray(vertices,np.float32)
        if vertices.ndim!=3 or vertices.shape[1:]!=(3,3) or not 0<len(vertices)<=1_000_000 or not np.isfinite(vertices).all():
            raise ValueError('Expected finite triangles, shape N x 3 x 3')
        if np.any(np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)<1e-9):
            raise ValueError('Degenerate mesh triangle')
        if not 1<=grid<=512 or not 1<=samples<=64 or not np.isfinite([elevation,azimuth]).all() or not 1<=elevation<=89:
            raise ValueError('Invalid ray grid, samples or sun')
        h,w=map_shape[:2];gw=grid;gh=max(1,round(grid*h/w))
        if gh>512:gw=max(1,round(512*w/h));gh=512
        result=np.empty((gh,gw,3),np.float32)
        camera=float(vertices[:,:,2].max()+max(w,h))
        self.check(self.lib.mr_trace(vertices.ctypes.data,len(vertices),result.ctypes.data,gw,gh,w,h,camera,elevation,azimuth,samples,self.threads))
        return result


def plane(shape):
    h,w=shape[:2];d=float(max(w,h))*4
    return np.array([[[-d,-d,0],[d,-d,0],[d,d,0]],[[-d,-d,0],[d,d,0],[-d,d,0]]],np.float32)


def load_obj(path):
    """OBJ vertices/triangles in input-map pixel XY coordinates, Z in pixel units."""
    vertices=[];triangles=[]
    for line in Path(path).read_text().splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]=='v':vertices.append([float(x) for x in fields[1:4]])
        elif fields[0]=='f':
            ids=[int(x.split('/')[0]) for x in fields[1:]]
            ids=[i-1 if i>0 else len(vertices)+i for i in ids]
            for j in range(1,len(ids)-1):triangles.append([vertices[ids[0]],vertices[ids[j]],vertices[ids[j+1]]])
    return np.asarray(triangles,np.float32)
