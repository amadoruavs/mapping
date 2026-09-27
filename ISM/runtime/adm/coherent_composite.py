"""GPU source-label optimization; bounded coarse vectors, one full-size output."""
from .io import write_json
import json
import cv2
import numpy as np
from .geometry import project


def compose_pixels(source,names,hs,width,height,cfg,out,gpu,frame_provider=None):
    width,height=int(width),int(height)
    n=len(names)
    if n>65535:raise ValueError('Source IDs exceed uint16 capacity')
    upscale_passes=cfg.input_upscale_passes if cfg.render_upscale_passes is None else cfg.render_upscale_passes
    def load(i):
        image,H=frame_provider(i) if frame_provider else (cv2.imread(str(source/names[i])),hs[i])
        if image is None:raise ValueError(f'Cannot decode {names[i]}')
        if upscale_passes:
            from .enhancement import upscale
            image,scale=upscale(image,upscale_passes,cfg.fsr_strength,gpu)
            H=H@np.linalg.inv(scale)
        return image,H
    # Coarse observations plus unary/confidence stay within the budget, independently of map size.
    budget=int(cfg.coherent_budget_mb)*1024**2
    coarse_scale=min(1.,384/max(width,height),np.sqrt(budget/(max(1,n)*8*width*height)))
    gw,gh=max(2,int(round(width*coarse_scale))),max(2,int(round(height*coarse_scale)))
    count=gw*gh;scale=np.array([[gw/width,0,(gw/width-1)/2],[0,gh/height,(gh/height-1)/2],[0,0,1.]])
    observations=gpu.buffer((n,gh,gw,4),np.uint8);confidence=gpu.buffer((n,gh,gw),np.float16)
    mean=gpu.buffer((gh,gw,4),zero=True)
    for i in range(n):
        if i%25==0:print(f'Composition observations: {i}/{n}',flush=True)
        image,H=load(i)
        h,w=image.shape[:2];inverse=np.linalg.inv(scale@H);inverse/=inverse[2,2]
        gpu.dispatch('coherent_observation',[gpu.shared_input(image),observations,confidence,mean],count,[w,h,gw,gh,i,image.shape[2]],inverse.ravel())
        gpu.sync()
    gpu.dispatch('coherent_normalize',[mean],count,[count])
    unary=gpu.buffer((n,gh,gw),np.float16);labels=gpu.buffer((gh,gw),np.uint16)
    gpu.dispatch('coherent_unary',[observations,confidence,mean,unary,labels],count,[count,n])
    for _ in range(32):
        for parity in (0,1):
            gpu.dispatch('coherent_relax',[unary,labels,mean],count,[gw,gh,n,parity],[2.5])
    gpu.sync();del unary
    shape=(height,width);best=gpu.buffer(shape,zero=True)
    selected=gpu.buffer((*shape,4),np.uint8,zero=True)
    support=cfg.export_profile=='diagnostic'
    coverage=gpu.buffer(shape if support else 1,np.uint16,zero=True)
    ids=gpu.upload(np.full(shape,65535,np.uint16)) if support else gpu.buffer(1,np.uint16)
    bias=gpu.buffer((gh,gw,4))
    for i in range(n):
        if i%25==0:print(f'Composition rendering: {i}/{n}',flush=True)
        image,H=load(i);h,w=image.shape[:2]
        bounds=project(H,[[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        x0,y0=np.maximum(0,np.floor(bounds.min(0)).astype(int));x1,y1=np.minimum([width,height],np.ceil(bounds.max(0)).astype(int)+1)
        rw,rh=int(x1-x0),int(y1-y0)
        if rw<=0 or rh<=0:continue
        inverse=np.linalg.inv(np.array([[1,0,-x0],[0,1,-y0],[0,0,1.]])@H);inverse/=inverse[2,2]
        gpu.dispatch('coherent_bias',[observations,confidence,mean,bias],count,[count,i])
        smooth_bias=gpu.gaussian(bias,6)
        gpu.dispatch('coherent_render',[gpu.shared_input(image),labels,best,selected,coverage,ids,smooth_bias],rw*rh,[w,h,rw,rh,x0,y0,width,height,gw,gh,i,int(support),int(cfg.fsr),image.shape[2]],inverse.ravel())
        gpu.sync()
    result=selected.view()
    if cfg.fsr:
        from .enhancement import rcas
        for _ in range(cfg.output_sharpen_passes):result=rcas(result,cfg.fsr_strength,gpu)
    valid=result[:,:,3]>0
    report=dict(method='coherent source regions, GPU red-black Potts label optimization',coarse_dimensions=[gw,gh],
                full_resolution_bytes=width*height*(12 if support else 8),support_rasters=support,coarse_observation_budget_bytes=budget,
                mixed_source_color_blending=False,random_source_switching=False,intermediate_warp_files=0,
                exposure_correction='Per-source MPS Gaussian exposure bias, sigma 6 coarse pixels, bounded to 32/255 per channel; excludes cross-view variance above 900',
                spatial_enhancement=dict(enabled=cfg.fsr,sampler='FSR1 EASU' if cfg.fsr else 'bilinear',
                    rcas_strength=cfg.fsr_strength,output_sharpen_passes=cfg.output_sharpen_passes if cfg.fsr else 0,
                    source_upscale_passes=upscale_passes,temporal_super_resolution=False),
                limitation='A planar warp cannot remove 3D parallax. Coherent source selection prevents averaging displaced structures, but source boundaries can still cross elevated objects.')
    write_json(out/'composite_report.json',report)
    return result,coverage.view() if support else None,ids.view() if support else None,valid


def stylized_shadows(rgba,cfg,gpu):
    h,w=rgba.shape[:2];scale=min(1,1600/max(w,h));sw,sh=max(1,round(w*scale)),max(1,round(h*scale))
    source=gpu.shared_input(rgba);low=gpu.buffer((sh,sw,4))
    gpu.dispatch('shadow_downsample',[source,low],sw*sh,[w,h,sw,sh])
    small=gpu.gaussian(low,1.2);large=gpu.gaussian(low,7)
    relief=gpu.buffer((sh,sw,4));shade=gpu.buffer((sh,sw,4))
    gpu.dispatch('shadow_relief',[small,large,relief],sw*sh,[sw*sh])
    gpu.dispatch('shadow_rays',[relief,shade],sw*sh,[sw,sh],[np.deg2rad(cfg.sun_azimuth_deg),np.tan(np.deg2rad(cfg.sun_elevation_deg)),.18])
    smooth=gpu.gaussian(shade,.8);output=gpu.buffer(rgba.shape,np.uint8)
    gpu.dispatch('shadow_apply',[source,smooth,output],w*h,[w,h,sw,sh])
    return output.numpy(False)
