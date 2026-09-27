#!/usr/bin/env python3
"""Edit the paths below, then run python run.py. Outputs go in a new folder."""
from pathlib import Path

# ----------------------- EDIT THESE -----------------------
PHOTOS_FOLDER=Path('/Users/avneh/Downloads/101SIYI_IMG')
MAP_IMAGE=Path('/Users/avneh/Downloads/odm.png')
OUTPUT_FOLDER=Path(__file__).resolve().parent/'output'
MAP_CRS=None  # Required for a world file without .prj, e.g. 'EPSG:32615'.
MAP_BOUNDS=None  # Optional (min_lon,min_lat,max_lon,max_lat); use real map bounds.
SHARPEN_PROFILE='siyi_low_light'  # Restored previous finish: detail .65, fine .15, CLAHE blend .35.
FINISH_MODE='adaptive'  # Offline scene analysis + protected detail; 'legacy' keeps the old recipe.
ADAPTIVE_RAY_STRENGTH=.35  # Restrained warm sun/sky; 0..1.
ADAPTIVE_FSR_SHARPNESS=.10  # Fine detail is already enhanced before upscaling.
CINEMATIC_LOOK=None  # Previous COLOR_PRESET recipe; optional experimental 'warm_documentary' / 'natural_cinema'.
COLOR_PRESET='pro_camera'
REGIONAL_PRESET=None  # e.g. 'aerial_documentary_regional'
GRADE_SHARPEN=False  # Already sharpened in the preceding pass; avoid stacked unsharp masks.
TIFF_VALUE_RANGE=None  # Optional color-sample range (e.g. (0,4095) for 12-bit data).
JPEG_QUALITY=97
JPEG_BACKGROUND=(255,255,255)  # RGB; JPEG cannot carry the input TIFF's alpha channel.
SAVE_ALPHA_MASK=True  # Lossless final_2x_alpha.tif alongside the JPEG.
RAYTRACE=True
RAYTRACE_BACKEND='openrt'  # Actual OpenRT C++ library; 'native' selects the bundled compact tracer.
OPENRT_LIGHTING='golden_hour_5pm'  # Warm sun/cool sky; 'neutral' retains the original plane-neutral lighting.
MESH_OBJ=None  # Optional registered mesh: X=map column, Y=map row, Z=height in pixel units.
RAY_STRENGTH=.65
SUN_ELEVATION_DEG=15.0  # Artistic 5 pm golden-hour look, not a computed solar position.
SUN_AZIMUTH_DEG=255.0
HIGHLIGHT_BLOOM=.02  # Previous subtle highlight glow.
FSR_SHARPNESS=.2  # Final 2x FSR1 EASU + noise-aware RCAS, range 0..1.
CPU_THREADS=8
SAVE_STAGES=True
MAX_OUTPUT_PIXELS=200_000_000  # 2x each axis means four times as many pixels.
# ---------------------------------------------------------

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
import shutil
import time
import cv2
import numpy as np
from cinematic_grade import grade_bgra
from georeference import load_georeference
from map_geo_checker import Checker,Settings,resolve_photo_paths
from native import Native,plane,load_obj
from presentation import enhance_reference
from map_io import read_map,flatten_original_extent,save_jpeg,save_alpha_tiff
from cinematic_finish import finish_cinematic
from adaptive_finish import finish_adaptive


@contextmanager
def timer(name,timings):
    start=time.perf_counter();print(f'[{name}] starting',flush=True)
    try:yield
    finally:timings[name]=time.perf_counter()-start;print(f'[{name}] {timings[name]:.3f}s',flush=True)


def save_image(path,image):
    if not cv2.imwrite(str(path),image):raise OSError(f'Could not save {path}')


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def relight(image,native,mesh=None,strength=.18,elevation=8.,azimuth=255.):
    if not np.isfinite(strength) or not 0<=strength<=1:raise ValueError('Ray strength must be 0..1')
    triangles=plane(image.shape)
    if mesh is not None:triangles=np.concatenate([triangles,load_obj(mesh)])
    field=native.trace(triangles,image.shape,elevation=elevation,azimuth=azimuth)
    h,w=image.shape[:2];result=image.copy()
    golden=getattr(native,'lighting','neutral')=='golden_hour_5pm'
    # Neutral plane lighting remains exact; a colored sun differs from the
    # neutral reference even on a plane. The field is explicitly linear RGB.
    if strength and np.max(np.abs(field-1))>1e-6:
        # Sample the small ray field per strip; no full-map float RGB allocation.
        gh,gw=field.shape[:2];xx=(np.arange(w,dtype=np.float32)+.5)*gw/w-.5
        for y in range(0,h,128):
            end=min(h,y+128);yy=(np.arange(y,end,dtype=np.float32)+.5)*gh/h-.5
            mx,my=np.meshgrid(xx,yy)
            light=cv2.remap(field,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            rgb=image[y:y+128,:,2::-1].astype(np.float32)/255
            # Ease colored illumination out near white to retain highlight
            # texture instead of clipping the warm red channel.
            headroom=np.clip((1-rgb.max(2,keepdims=True))/.15,0,1) if golden else 1
            factor=1+strength*headroom*(np.clip(light,.65,1.25)-1)
            linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)*factor
            rgb=np.where(linear<=.0031308,12.92*linear,1.055*np.maximum(linear,0)**(1/2.4)-.055)
            result[y:y+128,:,:3]=np.rint(np.clip(rgb[:,:,::-1],0,1)*255).astype(np.uint8)
        result[result[:,:,3]==0,:3]=0
    report=dict(backend='portable C++ BVH triangle ray tracing',geometry='flat map plane' if mesh is None else str(mesh),
        samples=8,bounces=2,field_shape=list(field.shape[:2]),irradiance_min=float(field.min()),irradiance_max=float(field.max()),
        strength=strength,sun_elevation_deg=elevation,sun_azimuth_deg=azimuth,
        flat_plane_preserves_pixels=mesh is None and (not golden or strength==0),provided_scene_geometry=mesh is not None,
        mean_irradiance_rgb=field.mean(axis=(0,1),dtype=np.float64).tolist(),highlight_protection=golden,
        note='Artistic warm sun/cool sky on a flat plane; existing shadows retained, no inferred tree/building shadows. The 5 pm name is a look, not a calculated sun position.' if golden else 'A flat plane adds no scene-specific shadows. Existing photographed lighting is retained.')
    if hasattr(native,'description'):
        report.update(native.description());report.update(samples=16,bounces=0)
    return result,report


def highlight_bloom(image,strength):
    if not 0<=strength<=.1:raise ValueError('Bloom strength must be 0..0.1')
    if not strength:return image
    # Quarter-size highlight field; broad glow remains separate from fine detail.
    h,w=image.shape[:2];small=cv2.resize(image,(max(1,w//4),max(1,h//4)),interpolation=cv2.INTER_AREA)
    rgb=small[:,:,:3].astype(np.float32)/255;alpha=small[:,:,3].astype(np.float32)/255
    bright=np.clip((rgb.max(2)-.85)/.15,0,1)*alpha
    glow=cv2.GaussianBlur(rgb*bright[...,None],(0,0),2.5)
    glow=cv2.resize(glow,(w,h),interpolation=cv2.INTER_LINEAR)
    result=image.copy()
    for y in range(0,h,128):
        src=image[y:y+128,:,:3].astype(np.float32)/255
        result[y:y+128,:,:3]=np.rint(np.clip(src+strength*glow[y:y+128]*(1-src),0,1)*255).astype(np.uint8)
    result[result[:,:,3]==0,:3]=0
    return result


def pipeline(map_path,photos,output,*,cfg=None,check_only=False,limit=None,map_crs=None,map_bounds=None,save_stages=None,finish_mode=None):
    finish_mode=finish_mode or FINISH_MODE
    if finish_mode not in ('adaptive','legacy'):raise ValueError('Unknown finish mode')
    ray_strength=ADAPTIVE_RAY_STRENGTH if finish_mode=='adaptive' else RAY_STRENGTH
    bloom=0. if finish_mode=='adaptive' else HIGHLIGHT_BLOOM
    fsr_sharpness=ADAPTIVE_FSR_SHARPNESS if finish_mode=='adaptive' else FSR_SHARPNESS
    cfg=cfg or Settings();map_path=Path(map_path).expanduser().resolve();output=Path(output).expanduser().resolve()
    paths=resolve_photo_paths([photos]);paths=paths[:limit] if limit else paths
    if not paths:raise ValueError('No source photos found')
    if map_path in paths:raise ValueError('The map cannot also be an input photo')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise ValueError('Choose a new or empty output folder')
    if map_path==output or output in map_path.parents:raise ValueError('Output cannot replace the input map directory')
    output.mkdir(parents=True,exist_ok=True);timings={};total=time.perf_counter();status={'status':'running'}
    def progress(text):print(text,flush=True)
    save_stages=SAVE_STAGES if save_stages is None else save_stages
    cv2.setNumThreads(CPU_THREADS);cv2.setRNGSeed(cfg.seed)
    try:
        with timer('load',timings):
            image,input_raster=read_map(map_path,TIFF_VALUE_RANGE)
            if not check_only and image.shape[0]*image.shape[1]*4>MAX_OUTPUT_PIXELS:raise ValueError('Final 2x image exceeds MAX_OUTPUT_PIXELS; use a map crop or increase the configured limit')
            geo=load_georeference(map_path,image.shape,map_bounds,map_crs)
            if geo is None:print('[GPS] No explicit map georeference; using visual checks. Camera extents are not treated as map ground truth.',flush=True)
        with timer('scan_and_source_cross_checks',timings):
            checker=Checker(image,geo,cfg,progress);checker.scan(paths)
        if check_only:
            report=dict(photos=checker.entries,pairwise=checker.pairs,mode='check',repaired_tiles=0)
            current=image
        else:
            with timer('repair',timings):current,report,owners=checker.repair()
            save_image(output/'repair_tiles.png',owners)
            if save_stages:save_image(output/'01_repaired.png',current)
        (output/'geometry_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if check_only:status.update(mode='check',final_image=None)
        else:
            with timer('native_build',timings):native=Native(CPU_THREADS)
            ray_backend=native
            if RAYTRACE and RAYTRACE_BACKEND=='openrt':
                from openrt_backend import OpenRT
                with timer('openrt_build',timings):ray_backend=OpenRT(CPU_THREADS,lighting=OPENRT_LIGHTING)
            elif RAYTRACE and RAYTRACE_BACKEND!='native':raise ValueError('RAYTRACE_BACKEND must be openrt or native')
            if finish_mode=='adaptive':
                with timer('adaptive_finish',timings):current,adaptive=finish_adaptive(current)
                (output/'adaptive_finish.json').write_text(json.dumps(adaptive,indent=2)+'\n')
            else:
                with timer('ism_sharpen',timings):current=enhance_reference(current,1.,profile=SHARPEN_PROFILE)
                if save_stages:save_image(output/'02_sharpened.png',current)
            if finish_mode=='legacy' and CINEMATIC_LOOK:
                with timer('cinematic_finish',timings):current,cinematic=finish_cinematic(current,CINEMATIC_LOOK)
                (output/'cinematic_report.json').write_text(json.dumps(cinematic,indent=2)+'\n')
            elif finish_mode=='legacy':
                with timer('ism_color_grade',timings):current=grade_bgra(current,COLOR_PRESET,REGIONAL_PRESET,seed=cfg.seed,
                    overrides=None if GRADE_SHARPEN else dict(sharpness=0.,clarity=0.))
            if save_stages:save_image(output/'03_graded.png',current)
            if RAYTRACE:
                with timer('raytrace',timings):current,ray=relight(current,ray_backend,MESH_OBJ,ray_strength,SUN_ELEVATION_DEG,SUN_AZIMUTH_DEG)
                (output/'raytrace_report.json').write_text(json.dumps(ray,indent=2)+'\n')
            if bloom:
                with timer('highlight_bloom',timings):current=highlight_bloom(current,bloom)
            if save_stages:save_image(output/'04_lit.png',current)
            with timer('fsr_2x',timings):final=native.fsr2x(current,output/'_work',fsr_sharpness,progress=progress)
            with timer('final_export',timings):
                jpeg=flatten_original_extent(final,image[:,:,3],2,JPEG_BACKGROUND)
                save_jpeg(output/'final_2x.jpg',jpeg,JPEG_QUALITY)
                pw=min(1600,jpeg.shape[1]);preview=cv2.resize(jpeg,(pw,max(1,round(pw*jpeg.shape[0]/jpeg.shape[1]))),interpolation=cv2.INTER_AREA)
                save_jpeg(output/'preview.jpg',preview,JPEG_QUALITY)
                if SAVE_ALPHA_MASK:save_alpha_tiff(output/'final_2x_alpha.tif',image[:,:,3],geo,2)
                geometry=dict(width=final.shape[1],height=final.shape[0],pixel_center_to_input=[[.5,0,-.25],[0,.5,-.25],[0,0,1]],
                    input_map=str(map_path),georeferenced=geo is not None,resolution_note='2x spatial reconstruction; no new measured ground detail')
                geometry.update(input_raster=input_raster,crop_input_bounds_xyxy=[0,0,image.shape[1],image.shape[0]],
                    alpha_policy='Exact original mask, nearest-neighbor 2x; composited onto JPEG background',
                    jpeg_background_rgb=list(JPEG_BACKGROUND),alpha_sidecar='final_2x_alpha.tif' if SAVE_ALPHA_MASK else None)
                if geo:geometry.update(geo.export(output,'final_2x',scale=2,world_suffix='.jgw'))
                (output/'final_geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
                final.flush();final._mmap.close();del final
                (output/'_work/final.bgra').replace(output/'grade_master.bgra')
                shutil.rmtree(output/'_work')
            status.update(final_image=str(output/'final_2x.jpg'),repaired_tiles=report['repaired_tiles'])
        status.update(status='complete')
    except Exception as exc:
        status.update(status='failed',error=str(exc));raise
    finally:
        status.update(total_seconds=time.perf_counter()-total,stage_seconds=timings,map=str(map_path),photos=len(paths),settings=asdict(cfg),
            appearance=dict(finish_mode=finish_mode,sharpen='noise-gated local extrema' if finish_mode=='adaptive' else SHARPEN_PROFILE,
                color_grade='offline_adaptive_cinema' if finish_mode=='adaptive' else CINEMATIC_LOOK or COLOR_PRESET,regional=None if finish_mode=='adaptive' or CINEMATIC_LOOK else REGIONAL_PRESET,
                cinematic_look=CINEMATIC_LOOK,raytrace=RAYTRACE,raytrace_backend=RAYTRACE_BACKEND,
                openrt_lighting=OPENRT_LIGHTING if RAYTRACE_BACKEND=='openrt' else None,
                ray_strength=ray_strength,sun_elevation_deg=SUN_ELEVATION_DEG,sun_azimuth_deg=SUN_AZIMUTH_DEG,mesh=str(MESH_OBJ) if MESH_OBJ else None,
                bloom=bloom,fsr='FSR1 EASU + noise-aware RCAS',scale=2,fsr_sharpness=fsr_sharpness,
                grade_sharpen=GRADE_SHARPEN,jpeg_quality=JPEG_QUALITY,jpeg_background_rgb=list(JPEG_BACKGROUND)),
            input_map_sha256=file_sha256(map_path) if map_path.is_file() else None)
        (output/'timing.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n')
    print(f'Complete: {output} ({status["total_seconds"]:.2f}s)',flush=True)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',type=Path,default=MAP_IMAGE);p.add_argument('--photos',type=Path,default=PHOTOS_FOLDER)
    p.add_argument('--output',type=Path,default=OUTPUT_FOLDER);p.add_argument('--check-only',action='store_true')
    p.add_argument('--limit',type=int,help='Small validation subset');p.add_argument('--compile-only',action='store_true')
    p.add_argument('--map-crs',default=MAP_CRS);p.add_argument('--map-bounds',nargs=4,type=float,default=MAP_BOUNDS)
    p.add_argument('--finish',choices=['adaptive','legacy'],default=FINISH_MODE)
    args=p.parse_args()
    if args.compile_only:
        print(Native(CPU_THREADS).binary)
        if RAYTRACE and RAYTRACE_BACKEND=='openrt':
            from openrt_backend import OpenRT
            print(OpenRT(CPU_THREADS,lighting=OPENRT_LIGHTING).binary)
        return
    if args.limit is not None and args.limit<1:p.error('--limit must be positive')
    pipeline(args.map,args.photos,args.output,check_only=args.check_only,limit=args.limit,map_crs=args.map_crs,map_bounds=args.map_bounds,finish_mode=args.finish)


if __name__=='__main__':main()
