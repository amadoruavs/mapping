"""Experimental XDNA2 packed-pixel kernel. Requires matching IRON/Peano/XRT.

Uses the pinned upstream transform_parallel API to distribute 1024-pixel tiles
over compute workers and two DMA channels per column. Hardware tests are mandatory.
"""
from pathlib import Path
import threading
import os, sys
from importlib.metadata import distribution
# Match the layout documented by upstream env_setup.sh.
mlir=Path(distribution("mlir_aie").locate_file("mlir_aie"))
peano=Path(distribution("llvm-aie").locate_file("llvm-aie"))
os.environ.setdefault("MLIR_AIE_INSTALL_DIR",str(mlir))
os.environ.setdefault("PEANO_INSTALL_DIR",str(peano))
os.environ["PATH"]=str(mlir/"bin")+":/opt/xilinx/xrt/bin:"+os.environ.get("PATH","")
sys.path[:0]=[str(mlir/"python"),"/opt/xilinx/xrt/python"]
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ExternalFunction
from aie.iron.algorithms import transform_parallel

CHUNK=65536
LOCK=threading.Lock()
SOURCE=(Path(__file__).parent/'kernels/xdna_composite.cpp').read_text()

@iron.jit
def design(A:In,B:Out,*,background:CompileTime[int]):
    tile=np.ndarray[(1024,),np.dtype[np.uint32]]
    tensor=np.ndarray[(CHUNK,),np.dtype[np.uint32]]
    kernel=ExternalFunction('adm_xdna_composite',
        source_string=f'#define ADM_BACKGROUND {background}u\n'+SOURCE,
        arg_types=[tile,tile,np.int32],compile_flags=['-O3'])
    return transform_parallel(kernel,tensor,tile_size=1024,num_channels=2)

def composite(image,bg):
    packed=np.ascontiguousarray(image).view(np.uint32).reshape(-1)
    result=np.empty(packed.size,np.uint32)
    background=int(bg[0])|(int(bg[1])<<8)|(int(bg[2])<<16)
    with LOCK:
        # Fixed-size buffers and designs prevent compilation per image shape.
        for offset in range(0,len(packed),CHUNK):
            n=min(CHUNK,len(packed)-offset);host=np.zeros(CHUNK,np.uint32)
            host[:n]=packed[offset:offset+n]
            a=iron.tensor(host,dtype=np.uint32,device='npu')
            b=iron.zeros(CHUNK,dtype=np.uint32,device='npu')
            design(a,b,background=background)
            result[offset:offset+n]=b.numpy()[:n]
    return result.view(np.uint8).reshape(image.shape)[:,:,:3]
