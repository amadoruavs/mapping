"""Compare completed GPU runs with saved ODM and the preserved good Metal run.

Evaluation is outside the pipeline timer. No Docker run or image duplication.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import compare_odm as metrics
from adm.geometry import LocalEarth, project
from adm.io import write_image


def load(path):
    return json.loads(path.read_text())


def disk_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def removed_bytes(path):
    manifest=metrics.BASE/'gpu_revision_cleanup.json'
    if not manifest.exists():return 0
    return sum(row['bytes'] for row in load(manifest)['files'] if Path(row['path']).is_relative_to(path.resolve()))


def boundary_statistics(path):
    ids=cv2.imread(str(path/'source_ids.png'),-1)
    valid=ids!=65535;comparisons=transitions=0
    for a,b,mask in [(ids[1:],ids[:-1],valid[1:]&valid[:-1]),
                     (ids[:,1:],ids[:,:-1],valid[:,1:]&valid[:,:-1])]:
        comparisons+=int(mask.sum());transitions+=int(((a!=b)&mask).sum())
    return dict(valid_neighbor_pairs=comparisons,source_transitions=transitions,
                transition_fraction=transitions/max(1,comparisons),
                note='Source-boundary density is a diagnostic, not an accuracy score.')


def same_observations(a,b):
    with np.load(a/'registration.npz',allow_pickle=True) as x,np.load(b/'registration.npz',allow_pickle=True) as y:
        ea,eb=x['edges'].tolist(),y['edges'].tolist()
        same=np.array_equal(x['names'],y['names']) and len(ea)==len(eb)
        same=bool(same and all(u['i']==v['i'] and u['j']==v['j'] and all(np.array_equal(u[k],v[k]) for k in ('H','a','b')) for u,v in zip(ea,eb)))
        return dict(identical=same,previous_edges=len(ea),new_edges=len(eb),
                    previous_points=sum(len(e['a']) for e in ea),new_points=sum(len(e['a']) for e in eb))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',choices=['siyi','archive']);args=p.parse_args()
    cv2.setNumThreads(4);cv2.setRNGSeed(2026)
    folder=metrics.BASE/args.dataset;new=folder/'metal_optimized';old=folder/'metal'
    out=folder/'evaluation_optimized';out.mkdir(exist_ok=True)
    report=load(new/'report.json');geometry=load(new/'map_geometry.json')
    timing=load(folder/'metal_optimized_timing.json');odm_timing=load(folder/'odm/timing.json')
    if timing['exit_code']!=0 or odm_timing['exit_code']!=0:raise RuntimeError('Only completed runs can be compared')
    raster_path=folder/'evaluation/raster.json'
    if not raster_path.exists():raster_path=folder/'evaluation_fast/raster.json'
    raster=load(raster_path);earth=LocalEarth(*geometry['origin_latlon'])
    profiles={'optimized':new}
    if (old/'report.json').exists():profiles['previous']=old
    geometries={key:load(path/'map_geometry.json') for key,path in profiles.items()}
    matrices={key:np.asarray(g['map_to_world']) for key,g in geometries.items()}
    all_xy=[]
    for key,g in geometries.items():
        if g['origin_latlon']!=geometry['origin_latlon']:raise RuntimeError('Comparison requires a shared local origin')
        w,h=g['width'],g['height']
        all_xy.extend(project(matrices[key],[[-.5,-.5],[w-.5,-.5],[w-.5,h-.5],[-.5,h-.5]]))
    corners=np.asarray(raster['corners_lonlat']);odm_xy=earth.xy(corners[:,1],corners[:,0]);all_xy=np.r_[all_xy,odm_xy]
    lo,hi=all_xy.min(0),all_xy.max(0)
    resolution=max(geometry['metres_per_pixel'],np.sqrt(raster['pixel_area_crs_units_squared'])*(raster['linear_units_m'] or 1),max(hi-lo)/1800)
    w,h=np.ceil((hi-lo)/resolution).astype(int);hi=lo+np.array([w,h])*resolution
    to_common=np.array([[1/resolution,0,-lo[0]/resolution-.5],[0,-1/resolution,hi[1]/resolution-.5],[0,0,1.]])
    plan=dict(width=int(w),height=int(h),metres_per_pixel=float(resolution),origin_latlon=geometry['origin_latlon'],lower_xy=lo.tolist(),upper_xy=hi.tolist())
    (out/'common_grid.json').write_text(json.dumps(plan,indent=2))
    odm=cv2.imread(str(folder/'odm.png'),-1)
    if odm is None:raise RuntimeError('Saved ODM PNG missing')
    if odm.shape[2]==3:odm=np.dstack([odm,np.full(odm.shape[:2],255,np.uint8)])
    ow,oh=raster['width'],raster['height'];src=np.float32([[-.5,-.5],[ow-.5,-.5],[ow-.5,oh-.5],[-.5,oh-.5]])
    warp=cv2.getPerspectiveTransform(src,project(to_common,odm_xy).astype(np.float32))
    images={'odm':cv2.warpPerspective(odm,warp,(int(w),int(h)))}
    for key,path in profiles.items():
        image=cv2.imread(str(path/'map_neutral.png'),-1)
        images[key]=cv2.warpPerspective(image,to_common@matrices[key],(int(w),int(h)))
        del image
    for key,image in images.items():write_image(out/f'common_{key}.png',image)
    agreement={key:metrics.pixel_agreement(images['odm'],images[key]) for key in profiles}
    registration,aligned=metrics.visual_registration(images['odm'],images['optimized'])
    aligned_metrics=None
    if aligned is not None:
        aligned_metrics=metrics.pixel_agreement(images['odm'],aligned);write_image(out/'visually_aligned_optimized.png',aligned)
    metrics.PROFILE='metal_optimized';positions=metrics.position_metrics(folder)
    (out/'position_errors.json').write_text(json.dumps(positions,indent=2))
    with (out/'position_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['name','recorded_error_m','odm_error_m','metal_error_m'],extrasaction='ignore');writer.writeheader();writer.writerows(positions['rows'])
    columns=['odm']+(['previous'] if 'previous' in profiles else [])+['optimized']
    panels=[]
    for key in columns:
        rgb=images[key][:,:,:3].copy();rgb[images[key][:,:,3]<128]=35
        rgb=cv2.resize(rgb,(700,round(h*700/w)));rgb=cv2.copyMakeBorder(rgb,40,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
        label={'odm':'ODM default','previous':'Previous successful Metal','optimized':'GPU alignment + coherent regions'}[key]
        cv2.putText(rgb,label,(12,27),cv2.FONT_HERSHEY_SIMPLEX,.63,(255,255,255),1,cv2.LINE_AA);panels.append(rgb)
    write_image(folder/'side_by_side_optimized.png',np.hstack(panels))
    delta=np.mean(np.abs(images['odm'][:,:,:3].astype(float)-images['optimized'][:,:,:3]),axis=2)
    heat=cv2.applyColorMap(np.clip(delta*3,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
    heat[(images['odm'][:,:,3]<128)|(images['optimized'][:,:,3]<128)]=0;write_image(out/'difference_geographic.png',heat)
    result=dict(dataset=args.dataset,timing=dict(optimized=timing,odm=odm_timing,odm_over_optimized_wall_ratio=odm_timing['wall_seconds']/timing['wall_seconds']),
        output_bytes=dict(optimized=disk_bytes(new)),pipeline_report=report,alignment=load(new/'registration_report.json'),
        geographic_alignment=load(new/'geographic_alignment.json'),positions=positions,common_grid=plan,
        geographic_pixel_agreement=agreement,visual_registration=registration,visually_registered_pixel_agreement=aligned_metrics,
        image_statistics_at_common_resolution={key:metrics.image_statistics(image) for key,image in images.items()},
        source_boundaries={key:boundary_statistics(path) for key,path in profiles.items()},
        limitations=['One fresh run of the final revision per dataset; filesystem caches and desktop background work are uncontrolled.',
            'ODM is a full 3D reconstruction in ARM64 Docker; ADM is a native planar mosaic with different output dimensions.',
            'Common-grid pixel metrics compare neutral PNGs at equal physical resolution. Optional artistic shadows are excluded.',
            'Saved ODM corner coordinates approximate its small-site projection with a homography; all versions use the same evaluation warp.',
            'Alignment residuals are training errors. SIYI GPS is a prior; Archive supplied synthetic centers are held out from reconstruction, not surveyed checkpoints.',
            'CPU feature extraction, RANSAC, geographic conversion and codecs remain. No claim of an entirely GPU pipeline.',
            'Coherent source regions reduce parallax ghosting but cannot reconstruct missing depth or guarantee every elevated object is uncut.'])
    if 'previous' in profiles:
        previous=load(folder/'metal_timing.json');a=load(old/'registration_report.json');b=result['alignment']
        result['timing']['previous']=previous;result['timing']['previous_over_optimized_wall_ratio']=previous['wall_seconds']/timing['wall_seconds']
        result['output_bytes']['previous']=disk_bytes(old)+removed_bytes(old)
        result['output_bytes']['previous_retained_after_cleanup']=disk_bytes(old)
        observations=same_observations(old,new)
        gate=observations['identical'] and all(b[k]<=a[k]*1.01+.01 for k in ['median_registration_error_px','p95_registration_error_px'])
        result['alignment_nonregression']=dict(passed=gate,observations=observations,previous=a,
            tolerance='Each median/p95 residual <= previous * 1.01 + 0.01 pixels, with identical observations; not independent accuracy')
        fit,aligned_old=metrics.visual_registration(images['odm'],images['previous'])
        result['previous_visual_registration']=fit
        result['previous_visually_registered_pixel_agreement']=metrics.pixel_agreement(images['odm'],aligned_old) if aligned_old is not None else None
        result['quality_conclusion']=dict(alignment_nonregression=gate,independent_accuracy_verified=False,
            odm_geographic_ssim_improved=agreement['optimized']['ssim_luminance']>=agreement['previous']['ssim_luminance'],
            note='Alignment and coherent structure detail improve; this does not establish better absolute accuracy or improvement on all ODM agreement metrics.')
    (folder/'comparison_optimized.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    fields=['dataset','profile','wall_seconds','peak_child_rss_bytes','alignment_seconds','median_alignment_px','p95_alignment_px','output_bytes']
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerow(dict(dataset=args.dataset,profile='ODM default',wall_seconds=odm_timing['wall_seconds'],peak_child_rss_bytes=odm_timing['peak_child_rss_bytes']))
        for key,path in profiles.items():
            t=timing if key=='optimized' else result['timing']['previous'];r=load(path/'report.json');s=load(path/'registration_report.json')
            writer.writerow(dict(dataset=args.dataset,profile=key,wall_seconds=t['wall_seconds'],peak_child_rss_bytes=t['peak_child_rss_bytes'],alignment_seconds=r['timings_seconds']['global_alignment'],median_alignment_px=s['median_registration_error_px'],p95_alignment_px=s['p95_registration_error_px'],output_bytes=disk_bytes(path)+removed_bytes(path)))
    print(json.dumps(dict(dataset=args.dataset,wall_seconds=timing['wall_seconds'],alignment_nonregression=result.get('alignment_nonregression'),
        geographic_pixel_agreement=agreement,positions={k:v for k,v in positions.items() if k.endswith('horizontal_error_m')}),indent=2))
    if 'alignment_nonregression' in result and not result['alignment_nonregression']['passed']:raise SystemExit('Alignment accuracy gate failed')


if __name__=='__main__':main()
