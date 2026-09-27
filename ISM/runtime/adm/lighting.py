"""Linear-light golden-hour treatment; optional real height-field shading.

    JPEG inputs cannot recover clipped radiance. HDR here means float scene-linear
    storage, not recovered bracketed exposure information.
"""
from .io import write_array, write_json
from pathlib import Path
from dataclasses import replace
import json
import cv2
import numpy as np
from .io import write_image

def srgb_to_linear(x): return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def linear_to_srgb(x): return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)

def normals_from_height(height,metres_per_pixel):
    if metres_per_pixel<=0: raise ValueError('Positive height-map pixel spacing required')
    # Map x=east and map y=south for a north-up mosaic.
    dy,dx=np.gradient(height.astype(np.float32),metres_per_pixel)
    normals=np.stack([-dx,dy,np.ones_like(dx)],axis=2)
    return normals/np.maximum(np.linalg.norm(normals,axis=2,keepdims=True),1e-6)

def sun_vector(elevation,azimuth):
    el,az=np.deg2rad([elevation,azimuth]); return np.array([np.cos(el)*np.sin(az),np.cos(el)*np.cos(az),np.sin(el)],np.float32)

def visibility_from_height(height,valid,spacing,cfg):
    h,w=height.shape; yy,xx=np.mgrid[:h,:w].astype(np.float32); visibility=np.ones((h,w),np.float32)
    az=np.deg2rad(cfg.sun_azimuth_deg); slope=np.tan(np.deg2rad(cfg.sun_elevation_deg))
    for distance in np.linspace(spacing,cfg.shadow_max_distance_m,cfg.shadow_steps):
        mx=(xx+np.sin(az)*distance/spacing).astype(np.float32); my=(yy-np.cos(az)*distance/spacing).astype(np.float32)
        terrain=cv2.remap(height,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-1e6)
        known=cv2.remap(valid.astype(np.float32),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        blocked=(terrain>height+distance*slope+.05)&(known>.5)
        visibility[blocked]=0
    return cv2.GaussianBlur(visibility,(0,0),.8)

def shade_linear(rgb,normals,visibility,elevation,azimuth,strength):
    sun=sun_vector(elevation,azimuth); cosine=np.maximum(normals@sun,0)
    # Normalize flat ground so the relighting pass does not darken the whole map.
    direct=cosine/max(float(sun[2]),.15)*visibility
    shade=1+strength*(np.clip(.65+.35*direct,.35,1.45)-1)
    warm=np.array([1.065,1.015,.94],np.float32)
    luminance=rgb@np.array([.2126,.7152,.0722],np.float32)
    return rgb*warm*shade[:,:,None]+np.stack([.008*(1-luminance),.004*(1-luminance),np.zeros_like(luminance)],axis=2)

def render_lighting(rgba,cfg,out,height_path=None,spacing=None,backend='cpu',gpu=None):
    valid=rgba[:,:,3]>0; height=None
    if backend not in ('cpu','metal'): raise ValueError('Unknown lighting backend')
    if height_path:
        height=np.load(height_path).astype(np.float32)
        if height.shape!=valid.shape or not np.isfinite(height).all(): raise ValueError('Height map must be finite metres and match mosaic dimensions exactly')
        if spacing is None or not np.isfinite(spacing) or spacing<=0: raise ValueError('Positive height-map pixel spacing required')
        mode='height-field directional relighting with cast-shadow approximation'
    else:
        mode='flat-ground SDR color treatment; original shadows retained'
    if backend=='metal':
        from .metal import get_backend
        gpu=gpu or get_backend()
        if cfg.compact_outputs and height is None:
            source=gpu.shared_input(rgba);output=gpu.buffer(rgba.shape,np.uint8)
            gpu.dispatch('flat_light_encoded',[source,output],valid.size,[valid.size],[*sun_vector(cfg.sun_elevation_deg,cfg.sun_azimuth_deg),cfg.light_strength])
            chosen=None;encoded=output.numpy(False)
        else:chosen,encoded=gpu.lighting(rgba,cfg,height,spacing)
        bgr=encoded[:,:,:3]
    else:
        chosen=_choose_cpu(rgba,cfg,height,spacing)
        encoded=np.clip(linear_to_srgb(chosen),0,1)
        bgr=np.round(encoded[:,:,::-1]*255).astype(np.uint8)
    write_image(out/'map_golden.png',np.dstack([bgr,rgba[:,:,3]]))
    if not cfg.compact_outputs:
        write_image(out/'map_golden_linear.hdr',chosen[:,:,::-1]); write_array(out/'map_golden_linear.npy',chosen)
    preview_scale=min(1,1600/max(valid.shape)); preview=cv2.resize(bgr,(round(valid.shape[1]*preview_scale),round(valid.shape[0]*preview_scale)),interpolation=cv2.INTER_AREA)
    write_image(out/'preview.jpg',preview)
    report={'mode':mode,'backend':backend,'sun_elevation_deg':cfg.sun_elevation_deg,'sun_azimuth_deg':cfg.sun_azimuth_deg,'environments':cfg.lighting_candidates,'seed':cfg.seed+1,'linear_peak':float(chosen.max()) if chosen is not None else None,'physical_hdr_recovery':False,'new_geometry_invented':False,'note':'This is a relighting approximation applied to already-lit photographs. It cannot remove old shadows or recover clipped sensor highlights.'}
    if cfg.stylized_shadows:
        if gpu is None:raise ValueError('Stylized shadows require Metal')
        from .coherent_composite import stylized_shadows
        shadow=stylized_shadows(rgba,cfg,gpu);write_image(out/'map_shadows.png',shadow)
        report['stylized_shadow_export']={'file':'map_shadows.png','method':'three short screen-space rays over a contrast-derived relief proxy, with MPS filtering','physical_geometry':False,'neutral_map_modified':False}
    write_json(out/'lighting_report.json',report); return report


def _choose_cpu(rgba,cfg,height=None,spacing=None):
    rgb=rgba[:,:,:3][:,:,::-1].astype(np.float32)/255; valid=rgba[:,:,3]>0; linear=srgb_to_linear(rgb)
    if height is not None: normals=normals_from_height(height,spacing)
    else:
        normals=np.zeros_like(linear); normals[:,:,2]=1; visibility=np.ones(valid.shape,np.float32)
    renderer=shade_linear
    # Mean-distance categorical selection among restrained sun environments.
    offsets=np.linspace(-4,4,cfg.lighting_candidates) if cfg.lighting_candidates>1 else np.array([0.])
    mean=np.zeros_like(linear); environments=[]
    for offset in offsets:
        elevation=float(np.clip(cfg.sun_elevation_deg+offset,1,89)); azimuth=cfg.sun_azimuth_deg+offset*2
        if height is not None: visibility=visibility_from_height(height,valid,spacing,replace(cfg,sun_elevation_deg=elevation,sun_azimuth_deg=azimuth))
        environments.append((elevation,azimuth,visibility))
        mean+=renderer(linear,normals,visibility,elevation,azimuth,cfg.light_strength)/len(offsets)
    rng=np.random.default_rng(cfg.seed+1); best=np.full(valid.shape,-np.inf,np.float32); chosen=np.zeros_like(linear)
    for elevation,azimuth,visibility in environments:
        candidate=renderer(linear,normals,visibility,elevation,azimuth,cfg.light_strength)
        error=np.mean((candidate-mean)**2,axis=2); uniform=rng.uniform(.0001,.9999,valid.shape).astype(np.float32)
        score=-error/(2*.025**2)-np.log(-np.log(uniform)); take=score>best; best[take]=score[take]; chosen[take]=candidate[take]
    chosen[~valid]=0
    return chosen
