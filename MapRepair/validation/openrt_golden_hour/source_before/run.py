#!/usr/bin/env python3
"""Edit the paths below, then run python run.py. Outputs go in a new folder."""
from pathlib import Path

# ----------------------- EDIT THESE -----------------------
PHOTOS_FOLDER=Path('/Users/avneh/Downloads/mapv2')
MAP_IMAGE=Path('/Users/avneh/Downloads/South-1st-Place-9-15-2026-orthophoto (1).tif')
OUTPUT_FOLDER=Path(__file__).resolve().parent/'output'
MAP_CRS=None  # Required for a world file without .prj, e.g. 'EPSG:32615'.
MAP_BOUNDS=None  # Optional (min_lon,min_lat,max_lon,max_lat); use real map bounds.
SHARPEN_PROFILE='natural_detail'  # Gentle detail; legacy 'standard' / 'siyi_low_light' also available.
CINEMATIC_LOOK='warm_documentary'  # 'natural_cinema', 'warm_documentary', or None for legacy COLOR_PRESET.
COLOR_PRESET='pro_camera'
REGIONAL_PRESET=None  # e.g. 'aerial_documentary_regional'
GRADE_SHARPEN=False  # Already sharpened in the preceding pass; avoid stacked unsharp masks.
TIFF_VALUE_RANGE=None  # Optional color-sample range (e.g. (0,4095) for 12-bit data).
JPEG_QUALITY=95
JPEG_BACKGROUND=(255,255,255)  # RGB; JPEG cannot carry the input TIFF's alpha channel.
SAVE_ALPHA_MASK=True  # Lossless final_2x_alpha.tif alongside the JPEG.
RAYTRACE=True
RAYTRACE_BACKEND='openrt'  # Actual OpenRT C++ library; 'native' selects the bundled compact tracer.
MESH_OBJ=None  # Optional registered mesh: X=map column, Y=map row, Z=height in pixel units.
RAY_STRENGTH=.18
SUN_ELEVATION_DEG=8.0  # Artistic low-sun lighting, not a computed solar position.
SUN_AZIMUTH_DEG=255.0
HIGHLIGHT_BLOOM=0.0  # New cinematic finish already includes subtle diffusion/halation.
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


@contextmanager
def timer(name,timings):
    start=time.perf_counter();print(f'[{name}] starting',flush=True)
    try:yield
    finally:timings[name]=time.perf_counter()-start;print(f'[{name}] {timings[name]:.3f}s',flush=True)


def save_image(path,image):
    if not cv2.imwrite(str(path),image):raise OSError(f'Could not save {path}')


def relight(image,native,mesh=None,strength=.18,elevation=8.,azimuth=255.):
    triangles=plane(image.shape)
    if mesh is not None:triangles=np.concatenate([triangles,load_obj(mesh)])
    field=native.trace(triangles,image.shape,elevation=elevation,azimuth=azimuth)
    h,w=image.shape[:2];result=image.copy()
    # An unoccluded plane is exactly the lighting reference. Keep this a no-op
    # instead of claiming that tracing a flat photograph creates real relief.
    if np.max(np.abs(field-1))>1e-6:
        light=cv2.resize(field,(w,h),interpolation=cv2.INTER_LINEAR)
        for y in range(0,h,128):
            rgb=image[y:y+128,:,2::-1].astype(np.float32)/255
            factor=1+strength*(np.clip(light[y:y+128],.65,1.25)-1)
            linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)*factor
            rgb=np.where(linear<=.0031308,12.92*linear,1.055*np.maximum(linear,0)**(1/2.4)-.055)
            result[y:y+128,:,:3]=np.rint(np.clip(rgb[:,:,::-1],0,1)*255).astype(np.uint8)
        result[result[:,:,3]==0,:3]=0
    report=dict(backend='portable C++ BVH triangle ray tracing',geometry='flat map plane' if mesh is None else str(mesh),
        samples=8,bounces=2,field_shape=list(field.shape[:2]),irradiance_min=float(field.min()),irradiance_max=float(field.max()),
        strength=strength,sun_elevation_deg=elevation,sun_azimuth_deg=azimuth,
        flat_plane_preserves_pixels=mesh is None,provided_scene_geometry=mesh is not None,
        note='A flat plane adds no scene-specific shadows. Existing photographed lighting is retained.')
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


def pipeline(map_path,photos,output,*,cfg=None,check_only=False,limit=None,map_crs=None,map_bounds=None,save_stages=None):
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
                with timer('openrt_build',timings):ray_backend=OpenRT(CPU_THREADS)
            elif RAYTRACE and RAYTRACE_BACKEND!='native':raise ValueError('RAYTRACE_BACKEND must be openrt or native')
            with timer('ism_sharpen',timings):current=enhance_reference(current,1.,profile=SHARPEN_PROFILE)
            if save_stages:save_image(output/'02_sharpened.png',current)
            if CINEMATIC_LOOK:
                with timer('cinematic_finish',timings):current,cinematic=finish_cinematic(current,CINEMATIC_LOOK)
                (output/'cinematic_report.json').write_text(json.dumps(cinematic,indent=2)+'\n')
            else:
                with timer('ism_color_grade',timings):current=grade_bgra(current,COLOR_PRESET,REGIONAL_PRESET,seed=cfg.seed,
                    overrides=None if GRADE_SHARPEN else dict(sharpness=0.,clarity=0.))
            if save_stages:save_image(output/'03_graded.png',current)
            if RAYTRACE:
                with timer('raytrace',timings):current,ray=relight(current,ray_backend,MESH_OBJ,RAY_STRENGTH,SUN_ELEVATION_DEG,SUN_AZIMUTH_DEG)
                (output/'raytrace_report.json').write_text(json.dumps(ray,indent=2)+'\n')
            with timer('highlight_bloom',timings):current=highlight_bloom(current,HIGHLIGHT_BLOOM)
            if save_stages:save_image(output/'04_lit.png',current)
            with timer('fsr_2x',timings):final=native.fsr2x(current,output/'_work',FSR_SHARPNESS,progress=progress)
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
                final._mmap.close();del final;shutil.rmtree(output/'_work')
            status.update(final_image=str(output/'final_2x.jpg'),repaired_tiles=report['repaired_tiles'])
        status.update(status='complete')
    except Exception as exc:
        status.update(status='failed',error=str(exc));raise
    finally:
        status.update(total_seconds=time.perf_counter()-total,stage_seconds=timings,map=str(map_path),photos=len(paths),settings=asdict(cfg),
            appearance=dict(sharpen=SHARPEN_PROFILE,color_grade=CINEMATIC_LOOK or COLOR_PRESET,regional=None if CINEMATIC_LOOK else REGIONAL_PRESET,
                cinematic_look=CINEMATIC_LOOK,raytrace=RAYTRACE,raytrace_backend=RAYTRACE_BACKEND,mesh=str(MESH_OBJ) if MESH_OBJ else None,
                bloom=HIGHLIGHT_BLOOM,fsr='FSR1 EASU + noise-aware RCAS',scale=2,fsr_sharpness=FSR_SHARPNESS,
                grade_sharpen=GRADE_SHARPEN,jpeg_quality=JPEG_QUALITY,jpeg_background_rgb=list(JPEG_BACKGROUND)),
            input_map_sha256=hashlib.sha256(map_path.read_bytes()).hexdigest() if map_path.is_file() else None)
        (output/'timing.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n')
    print(f'Complete: {output} ({status["total_seconds"]:.2f}s)',flush=True)
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',type=Path,default=MAP_IMAGE);p.add_argument('--photos',type=Path,default=PHOTOS_FOLDER)
    p.add_argument('--output',type=Path,default=OUTPUT_FOLDER);p.add_argument('--check-only',action='store_true')
    p.add_argument('--limit',type=int,help='Small validation subset');p.add_argument('--compile-only',action='store_true')
    p.add_argument('--map-crs',default=MAP_CRS);p.add_argument('--map-bounds',nargs=4,type=float,default=MAP_BOUNDS)
    args=p.parse_args()
    if args.compile_only:print(Native(CPU_THREADS).binary);return
    if args.limit is not None and args.limit<1:p.error('--limit must be positive')
    pipeline(args.map,args.photos,args.output,check_only=args.check_only,limit=args.limit,map_crs=args.map_crs,map_bounds=args.map_bounds)


if __name__=='__main__':main()
