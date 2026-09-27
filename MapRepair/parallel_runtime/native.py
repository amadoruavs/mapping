"""Optional compiled HIP and IRON kernels; no mandatory PyTorch dependency."""
import ctypes
import functools
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent

@functools.lru_cache(maxsize=1)
def hip_library():
    if not sys.platform.startswith('linux'):raise RuntimeError('HIP requires Linux')
    lib=ctypes.CDLL(str(ROOT/'kernels/libadm_hip.so'))
    lib.adm_hip_error.restype=ctypes.c_char_p
    lib.adm_hip_composite.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint32]
    lib.adm_flat_tiles.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    return lib

def hip_composite(image,bg):
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=4:
        raise ValueError('Expected uint8 BGRA')
    if image.shape[0]*image.shape[1]>16777216:raise ValueError('HIP composite tile exceeds 16M pixels')
    src=np.ascontiguousarray(image);out=np.empty_like(src)
    lib=hip_library()
    background=int(bg[0])|(int(bg[1])<<8)|(int(bg[2])<<16)
    if lib.adm_hip_composite(src.ctypes.data,out.ctypes.data,src.size//4,background):
        raise RuntimeError(lib.adm_hip_error().decode())
    return out[:,:,:3]

def xdna_composite(image,bg):
    from .xdna import composite
    return composite(image,bg)

def run(image,bg,backend):
    return (hip_composite if backend=='native-hip' else xdna_composite)(image,bg)

@functools.lru_cache(maxsize=4)
def certify(backend,require_speed=False):
    from .pixels import cpu_composite
    fg,a=np.mgrid[:256,:256]
    probe=np.stack([fg,255-fg,(17*fg)%256,a],axis=-1).astype(np.uint8)
    for bg in ([0,0,0],[255,255,255],[17,128,241]):
        if not np.array_equal(run(probe,bg,backend),cpu_composite(probe,bg)):
            raise RuntimeError(backend+' failed exact CPU parity')
    if require_speed:
        image=np.tile(probe,(2,4,1));cpu=[];accelerator=[]
        for _ in range(5):
            start=time.perf_counter();cpu_composite(image,[255]*3);cpu.append(time.perf_counter()-start)
            start=time.perf_counter();run(image,[255]*3,backend);accelerator.append(time.perf_counter()-start)
        if np.median(accelerator)>=.9*np.median(cpu):return None
    return backend

@functools.lru_cache(maxsize=1)
def automatic():
    if not sys.platform.startswith('linux'):return None
    # NPU JIT is opt-in until its toolchain has been installed and checked locally.
    backends=['native-hip']
    if os.environ.get('ADM_ENABLE_XDNA')=='1':backends.append('xdna')
    for backend in backends:
        try:
            result=certify(backend,True)
            if result:return result
        except (ImportError,OSError,RuntimeError,ValueError):pass
    return None
