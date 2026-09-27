"""Actual AMD FSR3 frame-generation passes translated to Metal.

The ADM adapter supplies game motion vectors from registered planar cameras and
assumed flat-ground depth. AMD shaders perform reconstruction, vector-field
priority, disocclusion, interpolation and inpainting. No swapchain is needed
for offline photos. Optical-flow assistance is disabled in this first adapter.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import weakref
import cv2
import numpy as np
from .geometry import project
from .fsr_geometry import midpoint_camera,check_pair

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'build/fsr3'


def midpoint_transform(previous_to_world,current_to_world,shape):
    """Fit a planar camera to FSR's half-motion pixel coordinates."""
    return midpoint_camera(previous_to_world,current_to_world,shape)


class Resource:
    def __init__(self,engine,handle,shape=None,dtype=None,byte_size=None):
        if not handle:engine.fail()
        self.handle=handle;self.engine=engine;self.shape=shape;self.dtype=dtype;self.byte_size=byte_size
        self._release=weakref.finalize(self,engine.lib.fsr_release,handle)

    def write(self,value):
        a=np.ascontiguousarray(value)
        if self.shape is None:
            if a.nbytes!=self.byte_size:raise ValueError('FSR buffer size mismatch')
            self.engine.lib.fsr_buffer_write(self.handle,a.ctypes.data,a.nbytes)
        else:
            if a.shape!=self.shape or a.dtype!=self.dtype:raise ValueError('FSR texture data shape/type mismatch')
            self.engine.lib.fsr_texture_write(self.handle,a.ctypes.data,a.strides[0])

    def read(self):
        self.engine.sync();a=np.empty(self.shape,self.dtype)
        self.engine.lib.fsr_texture_read(self.handle,a.ctypes.data,a.strides[0]);return a


class FrameGenerator:
    def __init__(self,depth_m=60.,hfov_deg=81.,resource_budget_mb=512):
        started=time.perf_counter();self.thread=threading.get_ident()
        if not np.isfinite(depth_m) or not .1<depth_m<10000 or not np.isfinite(hfov_deg) or not 0<hfov_deg<175:
            raise ValueError('Invalid FSR flat depth or FOV')
        self.depth_m=float(depth_m);self.hfov_deg=float(hfov_deg)
        if type(resource_budget_mb) is not int or not 128<=resource_budget_mb<=1024:raise ValueError('FSR resource budget must be 128..1024 MiB')
        self.resource_budget_mb=resource_budget_mb
        if not (BUILD/'manifest.json').exists():
            subprocess.run([__import__('sys').executable,str(ROOT/'scripts/build_fsr3.py')],check=True)
        self.manifest=json.loads((BUILD/'manifest.json').read_text())
        source=ROOT/'shaders/FSRMetal.m';digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        library=BUILD/f'libFSRMetal-{digest}.dylib'
        if not library.exists():
            result=subprocess.run(['xcrun','clang','-O3','-fobjc-arc','-fno-modules','-dynamiclib','-framework','Foundation','-framework','Metal',str(source),'-o',str(library)],capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stderr)
        self.lib=C.CDLL(str(library));ptr=C.c_void_p;u=C.c_uint32
        signatures={
            'fsr_create':(ptr,[]),'fsr_error':(C.c_char_p,[ptr]),'fsr_release':(None,[ptr]),
            'fsr_compile':(C.c_int,[ptr,C.c_char_p,C.c_char_p]),
            'fsr_texture':(ptr,[ptr,u,u,u,u]),'fsr_mip':(ptr,[ptr,u]),
            'fsr_texture_write':(None,[ptr,ptr,u]),'fsr_texture_read':(None,[ptr,ptr,u]),
            'fsr_buffer':(ptr,[ptr,ptr,u]),'fsr_buffer_write':(None,[ptr,ptr,u]),'fsr_sync':(C.c_int,[ptr]),
            'fsr_dispatch':(C.c_int,[ptr,C.c_char_p,C.POINTER(ptr),u,C.POINTER(ptr),u,C.POINTER(u),u,C.POINTER(u),C.POINTER(u)]),
        }
        for name,(restype,args) in signatures.items():
            fn=getattr(self.lib,name);fn.restype=restype;fn.argtypes=args
        self.handle=self.lib.fsr_create()
        if not self.handle:self.fail()
        self._release=weakref.finalize(self,self.lib.fsr_release,self.handle)
        self.passes=self.manifest['passes']
        for name in self.passes:
            shader=(BUILD/(name+'.metal')).read_bytes()
            if hashlib.sha256(shader).hexdigest()!=self.passes[name]['metal_sha256']:raise RuntimeError('FSR build manifest is stale; rebuild shaders')
            self.check(self.lib.fsr_compile(self.handle,name.encode(),shader))
        self.shape=None;self.resources={};self.frames=0;self.dispatches=0;self.frame_seconds=0.;self.stage_seconds=dict(upload=0.,motion=0.,dispatch_and_read=0.,alpha=0.)
        self.initialization_seconds=time.perf_counter()-started

    def fail(self):
        message=self.lib.fsr_error(getattr(self,'handle',None))
        decoded=message.decode() if message else 'FSR Metal operation failed'
        if len(decoded)>8000:
            (BUILD/'last_error.log').write_text(decoded)
            decoded=decoded[:8000]+f'\nFull error: {BUILD}/last_error.log'
        raise RuntimeError(decoded)

    def check(self,status):
        if status:self.fail()

    def sync(self):self.check(self.lib.fsr_sync(self.handle))

    def buffer(self,array):
        a=np.ascontiguousarray(array)
        return Resource(self,self.lib.fsr_buffer(self.handle,a.ctypes.data,a.nbytes),byte_size=a.nbytes)

    def texture(self,shape,dtype,format,levels=1):
        h,w=shape[:2]
        return Resource(self,self.lib.fsr_texture(self.handle,w,h,format,levels),shape,np.dtype(dtype))

    def constant_buffer(self,pass_name,buffer_name,values):
        info=json.loads((BUILD/(pass_name+'.json')).read_text())
        block=next(b for b in info['ubos'] if b['name']==buffer_name)
        unknown=set(values)-{f['name'] for f in info['types'][block['type']]['members']}
        if unknown:raise ValueError(f'Unknown FSR constants: {unknown}')
        data=np.zeros(block['block_size'],np.uint8)
        for field in info['types'][block['type']]['members']:
            if field['name'] not in values:continue
            dtype=np.float32 if field['type'].startswith(('vec','float')) else np.uint32 if field['type'].startswith(('uvec','uint')) else np.int32
            value=np.atleast_1d(np.array(values[field['name']],dtype)).view(np.uint8)
            data[field['offset']:field['offset']+value.size]=value
        return self.buffer(data)

    def allocate(self,h,w):
        if self.shape==(h,w):return
        if h<64 or w<64 or h>4096 or w>4096:raise ValueError('FSR adapter supports images from 64 to 4096 pixels per axis')
        if h*w*100>self.resource_budget_mb*1024**2:raise ValueError(f'FSR frame resources exceed {self.resource_budget_mb} MiB working budget')
        self.sync();self.resources={};r=self.resources;self.shape=(h,w)
        for name in ('previous_interpolation_source','current_interpolation_source','output'):
            r[name]=self.texture((h,w,4),np.uint8,70)
        for name in ('input_depth','dilated_depth'):r[name]=self.texture((h,w),np.float32,55)
        for name in ('input_motion_vectors','dilated_motion_vectors'):r[name]=self.texture((h,w,2),np.float32,105)
        for name in ('reconstructed_depth_previous_frame','reconstructed_depth_interpolated_frame',
                     'game_motion_vector_field_x','game_motion_vector_field_y','optical_flow_motion_vector_field_x','optical_flow_motion_vector_field_y'):
            r[name]=self.texture((h,w),np.uint32,53)
        r['disocclusion_mask']=self.texture((h,w,2),np.uint8,30)
        r['input_distortion_field']=self.texture((1,1,2),np.float32,105);r['input_distortion_field'].write(np.zeros((1,1,2),np.float32))
        r['optical_flow_scd']=self.texture((1,2),np.uint32,53);r['optical_flow_scd'].write(np.zeros((1,2),np.uint32))
        # Disabled optical-flow resources remain well-defined zero inputs.
        r['optical_flow']=self.texture((h,w,2),np.int32,104);r['optical_flow'].write(np.zeros((h,w,2),np.int32))
        r['optical_flow_confidence']=self.texture((h,w),np.uint32,53);r['optical_flow_confidence'].write(np.zeros((h,w),np.uint32))
        levels=int(np.floor(np.log2(max(w//2,h//2))))+1
        r['inpainting_pyramid']=self.texture((h//2,w//2,4),np.float16,115,levels)
        for i in range(13):r[f'inpainting_pyramid{i}']=Resource(self,self.lib.fsr_mip(r['inpainting_pyramid'].handle,min(i,levels-1)))
        r['counters']=self.buffer(np.zeros(2,np.uint32))
        r['present_backbuffer']=r['current_interpolation_source']
        near,far=.1,10000.;a=far/(far-near);b=-far*near/(far-near)
        tan_h=np.tan(np.deg2rad(self.hfov_deg/2));tan_v=tan_h*h/w
        r['cbFI']=self.constant_buffer('interpolation','cbFI_t',dict(renderSize=[w,h],displaySize=[w,h],
            displaySizeRcp=[1/w,1/h],cameraNear=near,cameraFar=far,upscalerTargetSize=[w,h],reset=0,
            fDeviceToViewDepth=[a,b,tan_h,tan_v],deltaTime=1/30,
            distortionFieldSize=[1,1],opticalFlowScale=[1/w,1/h],opticalFlowBlockSize=1,maxRenderSize=[w,h],
            NumInstances=1,interpolationRectSize=[w,h],backBufferTransferFunction=0,minMaxLuminance=[0,1],
            fTanHalfFOV=tan_h,fMotionVectorScale=[1/w,1/h]))
        r['cbInpaintingPyramid']=self.constant_buffer('compute_inpainting_pyramid','cbInpaintingPyramid_t',
            dict(mips=min(int(np.floor(np.log2(max(w,h)))),12),numWorkGroups=((w+63)//64)*((h+63)//64),workGroupOffset=[0,0]))
        yy,xx=np.mgrid[:h,:w];self.points=np.stack((xx,yy),axis=-1).astype(np.float32)
        self.depth_value=a+b/self.depth_m
        self.clear_depth=np.full((h,w),np.float32(1).view(np.uint32),np.uint32)

    def dispatch(self,name):
        h,w=self.shape;meta=self.passes[name];bindings=meta['bindings']
        tables={kind:[0]*(1+max((x['index'] for x in bindings if x['kind']==kind),default=-1)) for kind in ('buffer','texture','sampler')}
        for item in bindings:
            key=item['name'];kind=item['kind']
            if kind=='sampler':value=int('Linear' in key)
            else:
                key=key.removeprefix('rw_').removeprefix('r_')
                value=self.resources[key].handle
            tables[kind][item['index']]=value
        textures=(C.c_void_p*len(tables['texture']))(*tables['texture']);buffers=(C.c_void_p*len(tables['buffer']))(*tables['buffer'])
        samplers=(C.c_uint32*len(tables['sampler']))(*tables['sampler'])
        groups=((w+63)//64,(h+63)//64,1) if 'pyramid' in name else ((w+7)//8,(h+7)//8,1)
        self.check(self.lib.fsr_dispatch(self.handle,name.encode(),textures,len(textures),buffers,len(buffers),samplers,len(samplers),
            (C.c_uint32*3)(*groups),(C.c_uint32*3)(*meta['threads'])))
        self.dispatches+=1

    def interpolate(self,previous,current,previous_to_world,current_to_world):
        if threading.get_ident()!=self.thread:raise RuntimeError('FSR engine must stay on its creating thread')
        if previous.ndim!=3 or current.ndim!=3 or previous.shape[2] not in (3,4) or current.shape[2] not in (3,4) or previous.shape[:2]!=current.shape[:2] or previous.dtype!=np.uint8 or current.dtype!=np.uint8:
            raise ValueError('FSR frame inputs must be equal-size uint8 images')
        geometry,middle_to_world=check_pair(previous_to_world,current_to_world,previous.shape)
        if not geometry['accepted']:raise ValueError('Unsupported FSR pair: '+geometry['reason'])
        started=time.perf_counter();h,w=previous.shape[:2];self.allocate(h,w);r=self.resources
        rgba=lambda im:cv2.cvtColor(im,cv2.COLOR_BGRA2RGBA if im.shape[2]==4 else cv2.COLOR_BGR2RGBA)
        r['previous_interpolation_source'].write(rgba(previous));r['current_interpolation_source'].write(rgba(current))
        uploaded=time.perf_counter()
        relative=np.linalg.inv(previous_to_world)@current_to_world
        old=project(relative,self.points.reshape(-1,2)).reshape(h,w,2)
        motion=(old-self.points).astype(np.float32)
        if not np.isfinite(motion).all():raise ValueError('FSR motion field is nonfinite')
        r['input_motion_vectors'].write(motion);r['input_depth'].write(np.full((h,w),self.depth_value,np.float32))
        r['reconstructed_depth_previous_frame'].write(self.clear_depth)
        r['reconstructed_depth_interpolated_frame'].write(self.clear_depth)
        r['counters'].write(np.zeros(2,np.uint32))
        prepared=time.perf_counter()
        for name in ('reconstruct_and_dilate','setup','reconstruct_previous_depth','game_motion_vector_field',
                     'compute_game_vector_field_inpainting_pyramid','disocclusion_mask','interpolation',
                     'compute_inpainting_pyramid','inpainting'):self.dispatch(name)
        output=cv2.cvtColor(r['output'].read(),cv2.COLOR_RGBA2BGRA)
        rendered=time.perf_counter()
        # Only pixels supported by BOTH parent photographs may enter a map.
        # Reject disocclusion/inpainting and colors inconsistent with the sources.
        from .metal import get_backend
        gpu=get_backend();generated=gpu.shared_input(output)
        matrices=[]
        for H in (previous_to_world,current_to_world):
            transform=np.linalg.inv(H)@middle_to_world;transform/=transform[2,2];matrices.extend(transform.ravel())
        gpu.dispatch('fsr_validate_generated',[gpu.shared_input(previous),gpu.shared_input(current),generated],w*h,
                     [w,h,previous.shape[2],current.shape[2]],matrices)
        gpu.sync();output=generated.numpy()
        self.frames+=1;self.frame_seconds+=time.perf_counter()-started
        self.stage_seconds['upload']+=uploaded-started;self.stage_seconds['motion']+=prepared-uploaded
        self.stage_seconds['dispatch_and_read']+=rendered-prepared;self.stage_seconds['alpha']+=time.perf_counter()-rendered
        return output,middle_to_world

    def report(self):
        return dict(algorithm=self.manifest['algorithm'],upstream_revision=self.manifest['upstream_revision'],
            translated_upstream_shaders=True,frames_generated=self.frames,dispatches=self.dispatches,resource_budget_mb=self.resource_budget_mb,
            initialization_seconds=self.initialization_seconds,frame_seconds=self.frame_seconds,
            stage_seconds=self.stage_seconds,
            optical_flow_assistance=False,motion='Current-to-previous pixel motion from ADM homographies',
            map_validation='Geometry within 1 source pixel; rotation <=15 degrees; motion <=quarter-frame; both parents observed and color-consistent',
            depth=f'Assumed constant {self.depth_m:g} m flat ground; not reconstructed depth',
            limitation='Offline FSR3 motion-vector path, without game-engine depth truth, temporal upscaling, or swapchain pacing')
