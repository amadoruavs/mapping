"""Exact integer pixel math, expressed in Python for CPU, Metal and ROCm PyTorch.

No reduced precision, approximate reciprocal, or implicit CPU device fallback.
The optional GPU path is limited to these verified operations, not reconstruction.
"""
import functools
import os
import sys
import time
import numpy as np
from .scheduler import memory_budget

def cpu_composite(image, background):
    rgb=image[:,:,:3].astype(np.int32);alpha=image[:,:,3:4].astype(np.int32)
    return ((rgb*alpha+np.asarray(background,np.int32)*(255-alpha)+127)//255).astype(np.uint8)

def torch_composite(image,background,device):
    import torch
    with torch.inference_mode():
        src=torch.as_tensor(np.ascontiguousarray(image),device=device,dtype=torch.int32)
        bg=torch.tensor(background,device=device,dtype=torch.int32)
        a=src[:,:,3:4]
        result=torch.div(src[:,:,:3]*a+bg*(255-a)+127,255,rounding_mode='floor')
        return result.to(dtype=torch.uint8).cpu().numpy()

def torch_repeat2x(image,device):
    import torch
    with torch.inference_mode():
        src=torch.as_tensor(np.ascontiguousarray(image),device=device)
        return src.repeat_interleave(2,dim=0).repeat_interleave(2,dim=1).cpu().numpy()

def available_device(requested):
    import torch
    if requested in ('metal','auto') and sys.platform=='darwin' and torch.backends.mps.is_available():return 'mps'
    if requested in ('hip','auto') and sys.platform.startswith('linux') and torch.version.hip and torch.cuda.is_available():return 'cuda'
    raise RuntimeError(f'{requested} accelerator/runtime is unavailable')

@functools.lru_cache(maxsize=4)
def certified_device(requested):
    device=available_device(requested)
    # Every foreground/alpha combination, three contrasting backgrounds.
    alpha,fg=np.mgrid[:256,:256]
    image=np.stack((fg,255-fg,(fg*17)%256,alpha),axis=-1).astype(np.uint8)
    for bg in ([0,0,0],[255,255,255],[17,128,241]):
        if not np.array_equal(torch_composite(image,bg,device),cpu_composite(image,bg)):
            raise RuntimeError('Integer alpha compositing failed exact CPU parity')
    if requested=='auto':
        # Include transfers/synchronization; an available GPU can still be slower.
        probe=np.tile(image,(2,4,1));timings={'cpu':[],'gpu':[]}
        for _ in range(5):
            start=time.perf_counter();cpu_composite(probe,[255]*3);timings['cpu'].append(time.perf_counter()-start)
            start=time.perf_counter();torch_composite(probe,[255]*3,device);timings['gpu'].append(time.perf_counter()-start)
        if np.median(timings['gpu'])>=.9*np.median(timings['cpu']):return None
    return device

def composite_bgra(image,background=(255,255,255),backend=None):
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=4:raise ValueError('Expected uint8 BGRA')
    bg=np.asarray(background)
    if bg.shape!=(3,) or not np.isfinite(bg).all() or np.any(bg<0) or np.any(bg>255) or np.any(bg!=np.floor(bg)):
        raise ValueError('Background must contain three integer bytes')
    backend=backend or os.environ.get('ADM_PIXEL_BACKEND','auto')
    if backend not in ('auto','cpu','metal','hip','native-hip','xdna'):raise ValueError('Unknown pixel backend')
    h,w=image.shape[:2]
    # Account for result plus conservative host/device temporary reservations.
    budget=memory_budget()
    result_bytes=h*w*3
    row_bytes=max(1,w*96)
    if result_bytes+row_bytes>budget:raise MemoryError('Composite output and one work row exceed available memory budget')
    rows=min(512,max(1,(budget-result_bytes)//row_bytes))
    device=None
    native_backend=None
    accelerator_room=budget-result_bytes>=64*1024**2
    if backend not in ('cpu','auto') and not accelerator_room:
        raise MemoryError('Accelerator probe/staging requires 64 MiB of work budget')
    if backend!='cpu' and accelerator_room and (backend!='auto' or image.shape[0]*image.shape[1]>=65536):
        try:
            from . import native
            if backend in ('native-hip','xdna'):native_backend=native.certify(backend)
            elif backend=='auto':native_backend=native.automatic()
            if not native_backend and backend not in ('native-hip','xdna'):device=certified_device(backend)
        except (ImportError,OSError,RuntimeError,ValueError):
            if backend!='auto':raise
    result=np.empty((h,w,3),np.uint8)
    for y in range(0,h,rows):
        block=image[y:y+rows]
        try:
            result[y:y+rows]=native.run(block,bg.tolist(),native_backend) if native_backend else (torch_composite(block,bg.tolist(),device) if device else cpu_composite(block,bg))
        except (OSError,RuntimeError,ValueError):
            if backend!='auto':raise
            device=None
            native_backend=None
            result[y:y+rows]=cpu_composite(block,bg)
    return result
