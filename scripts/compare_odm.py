"""Evaluate saved default-ODM and Metal maps without feeding truth into either run."""
import argparse
import csv
import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from adm.geometry import LocalEarth, project

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output'/'comparison'
IMAGE='opendronemap/odm:3.6.2'
PROFILE='metal'
EVALUATION='evaluation'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def distribution(values):
    a=np.asarray(values,dtype=float)
    if not len(a): return {'count':0}
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),
                rmse=float(np.sqrt(np.mean(a*a))),p90=float(np.percentile(a,90)),
                p95=float(np.percentile(a,95)),max=float(a.max()),min=float(a.min()))


def run_gdal(dataset, plan=False):
    folder=BASE/dataset
    cmd=['docker','run','--rm','--entrypoint','python3',
         '--mount',f'type=bind,source={folder},target=/data',
         '--mount',f'type=bind,source={ROOT/"scripts"},target=/bench,readonly',IMAGE,
         '/bench/odm_raster_metrics.py','/data/odm/odm_orthophoto/odm_orthophoto.tif',f'/data/{EVALUATION}']
    if plan: cmd.extend(['--plan',f'/data/{EVALUATION}/common_grid.json'])
    subprocess.run(cmd,check=True)


def prepare_common(dataset):
    folder=BASE/dataset; out=folder/EVALUATION; out.mkdir(exist_ok=True)
    run_gdal(dataset)
    geometry=read(folder/PROFILE/'map_geometry.json'); raster=read(out/'raster.json')
    earth=LocalEarth(*geometry['origin_latlon']); matrix=np.asarray(geometry['map_to_world'])
    corners=project(matrix,[[-.5,-.5],[geometry['width']-.5,-.5],
                            [geometry['width']-.5,geometry['height']-.5],[-.5,geometry['height']-.5]])
    odm=np.asarray(raster['corners_lonlat']); odmxy=earth.xy(odm[:,1],odm[:,0])
    points=np.r_[corners,odmxy]; lo=points.min(0); hi=points.max(0)
    # Same geographic grid and physical resolution for all pairwise image metrics.
    resolution=max(float(geometry['metres_per_pixel']),
                   math.sqrt(raster['pixel_area_crs_units_squared'])*(raster['linear_units_m'] or 1),
                   float(max(hi-lo)/1800))
    width,height=np.ceil((hi-lo)/resolution).astype(int); hi=lo+np.array([width,height])*resolution
    ll=earth.latlon(np.array([lo,hi])); bounds=[ll[0,1],ll[0,0],ll[1,1],ll[1,0]]
    plan=dict(bounds_lonlat=list(map(float,bounds)),width=int(width),height=int(height),
              metres_per_pixel=resolution,origin_latlon=geometry['origin_latlon'],
              lower_xy=lo.tolist(),upper_xy=hi.tolist(),
              note='Union bounds, north-up WGS84 resampling; local pixel size reported in metres. Evaluation only.')
    (out/'common_grid.json').write_text(json.dumps(plan,indent=2)); run_gdal(dataset,True)
    to_common=np.array([[1/resolution,0,-lo[0]/resolution-.5],
                        [0,-1/resolution,hi[1]/resolution-.5],[0,0,1.]])
    metal=cv2.imread(str(folder/PROFILE/'map_neutral.png'),-1)
    if metal is None: raise RuntimeError('Missing neutral Metal PNG')
    warped=cv2.warpPerspective(metal,to_common@matrix,(int(width),int(height)),flags=cv2.INTER_LINEAR)
    cv2.imwrite(str(out/'common_metal.png'),warped)
    return plan


def image_statistics(image):
    valid=image[:,:,3]>127; gray=cv2.cvtColor(image[:,:,:3],cv2.COLOR_BGR2GRAY).astype(np.float32)
    inside=cv2.erode(valid.astype(np.uint8),np.ones((5,5),np.uint8))>0
    lap=cv2.Laplacian(gray,cv2.CV_32F); gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    if not inside.any(): return {'covered_pixels':int(valid.sum())}
    return dict(covered_pixels=int(valid.sum()),coverage_fraction=float(valid.mean()),
                mean_bgr=image[:,:,:3][inside].mean(0).tolist(),std_bgr=image[:,:,:3][inside].std(0).tolist(),
                laplacian_variance=float(lap[inside].var()),mean_gradient=float(gradient[inside].mean()),
                clipped_black_fraction=float(np.mean(np.all(image[:,:,:3][inside]==0,axis=1))),
                clipped_white_fraction=float(np.mean(np.all(image[:,:,:3][inside]==255,axis=1))))


def pixel_agreement(a,b):
    av=a[:,:,3]>127; bv=b[:,:,3]>127; overlap=av&bv; union=av|bv
    result=dict(overlap_pixels=int(overlap.sum()),union_pixels=int(union.sum()),
                footprint_iou=float(overlap.sum()/max(1,union.sum())),
                odm_only_pixels=int((av&~bv).sum()),metal_only_pixels=int((bv&~av).sum()))
    mask=cv2.erode(overlap.astype(np.uint8),np.ones((11,11),np.uint8))>0
    result['evaluated_interior_pixels']=int(mask.sum())
    if not mask.any(): return result
    x=a[:,:,:3].astype(np.float32); y=b[:,:,:3].astype(np.float32)
    delta=x[mask]-y[mask]; mse=float(np.mean(delta**2))
    result.update(mae_8bit=float(np.mean(np.abs(delta))),rmse_8bit=math.sqrt(mse),
                  psnr_db=None if mse==0 else 10*math.log10(255**2/mse),
                  absolute_error_p95_8bit=float(np.percentile(np.abs(delta),95)),
                  per_channel_bias_bgr=delta.mean(0).tolist())
    gx=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY); gy=cv2.cvtColor(y,cv2.COLOR_BGR2GRAY)
    mx=cv2.GaussianBlur(gx,(11,11),1.5); my=cv2.GaussianBlur(gy,(11,11),1.5)
    vx=cv2.GaussianBlur(gx*gx,(11,11),1.5)-mx*mx
    vy=cv2.GaussianBlur(gy*gy,(11,11),1.5)-my*my
    covariance=cv2.GaussianBlur(gx*gy,(11,11),1.5)-mx*my
    ssim=((2*mx*my+6.5025)*(2*covariance+58.5225))/((mx*mx+my*my+6.5025)*(vx+vy+58.5225))
    result['ssim_luminance']=float(ssim[mask].mean())
    result['luminance_correlation']=float(np.corrcoef(gx[mask],gy[mask])[0,1]) if gx[mask].std()>0 and gy[mask].std()>0 else None
    return result


def alignment_support(source, target, ok, shape_a, shape_b):
    """A minimal four-point fit or a tiny local patch cannot validate a map."""
    count = int(np.count_nonzero(ok))
    source_area = cv2.contourArea(cv2.convexHull(source[ok])) if count >= 3 else 0.
    target_area = cv2.contourArea(cv2.convexHull(target[ok])) if count >= 3 else 0.
    fractions = [source_area/(shape_b[0]*shape_b[1]), target_area/(shape_a[0]*shape_a[1])]
    accepted = count >= 12 and float(np.mean(ok)) >= .25 and min(fractions) >= .05
    return dict(success=accepted, inliers=count, inlier_fraction=float(np.mean(ok)),
                inlier_hull_fraction_metal_odm=fractions,
                acceptance='At least 12 inliers, 25% of ratio matches, and 5% canvas hull coverage in each map')


def visual_registration(a,b):
    sift=cv2.SIFT_create(nfeatures=7000)
    ka,da=sift.detectAndCompute(cv2.cvtColor(a[:,:,:3],cv2.COLOR_BGR2GRAY),a[:,:,3])
    kb,db=sift.detectAndCompute(cv2.cvtColor(b[:,:,:3],cv2.COLOR_BGR2GRAY),b[:,:,3])
    if da is None or db is None: return {'success':False},None
    groups=cv2.BFMatcher().knnMatch(db,da,k=2)
    good=[g[0] for g in groups if len(g)==2 and g[0].distance<.7*g[1].distance]
    if len(good)<12: return {'success':False,'ratio_matches':len(good)},None
    source=np.float32([kb[g.queryIdx].pt for g in good]);target=np.float32([ka[g.trainIdx].pt for g in good])
    H,mask=cv2.findHomography(source,target,cv2.RANSAC,3.,maxIters=5000)
    if H is None: return {'success':False,'ratio_matches':len(good)},None
    ok=mask.ravel()>0
    support=alignment_support(source,target,ok,a.shape,b.shape)
    if not support['success']:
        return {**support,'ratio_matches':len(good),'reason':'Insufficient global visual correspondence support'},None
    report=dict(**support,ratio_matches=len(good),
                transform_metal_to_odm=H.tolist(),inlier_residual_pixels=distribution(np.linalg.norm(project(H,source[ok])-target[ok],axis=1)),
                note='Fitted for relative visual agreement only. This alignment removes part of the geographic error; it is not ground-truth accuracy.')
    aligned=cv2.warpPerspective(b,H,(a.shape[1],a.shape[0]))
    return report,aligned


def position_metrics(folder):
    inputs=read(folder/'inputs.json'); geometry=read(folder/PROFILE/'map_geometry.json')
    earth=LocalEarth(*geometry['origin_latlon']); matrix=np.array(geometry['map_to_world'])
    hs=np.load(folder/PROFILE/'image_to_map.npy'); names=geometry['input_names']; meta={r['name']:r['metadata'] for r in inputs['images']}
    metal={}
    for name,H in zip(names,hs):
        m=meta[name];xy=project(matrix@H,[[(m['width']-1)/2,(m['height']-1)/2]])[0]
        metal[name]=earth.latlon(xy)
    shots=read(folder/'odm/odm_report/shots.geojson',{'features':[]})
    odm={f['properties']['filename']:np.array(f['geometry']['coordinates'][1::-1]) for f in shots['features']}
    truth=read(folder/'evaluation_truth.json')
    reference={r['file']:np.array([float(r['true_lat']),float(r['true_lon'])]) for r in truth} if truth else {
        n:np.array([m['latitude'],m['longitude']]) for n,m in meta.items()}
    rows=[]
    for name,ref in reference.items():
        if name not in meta: continue
        m=meta[name]; observed=np.array([m['latitude'],m['longitude']]); target=earth.xy(*ref)
        row=dict(name=name,reference_latlon=ref.tolist(),recorded_latlon=observed.tolist(),
                 recorded_error_m=float(np.linalg.norm(earth.xy(*observed)-target)))
        for label,positions in [('metal',metal),('odm',odm)]:
            if name in positions:
                estimated=positions[name];delta=earth.xy(*estimated)-target
                row[label+'_latlon']=estimated.tolist();row[label+'_error_m']=float(np.linalg.norm(delta))
                row[label+'_east_north_error_m']=delta.tolist()
        rows.append(row)
    report=dict(reference='held-out nominal true centers from supplied flight_log.csv' if truth else 'recorded input GPS, used as a reconstruction prior',
                independent_survey_ground_truth=False,
                note='Archive truth is supplied synthetic/crop metadata, not a surveyed check-point network.' if truth else 'SIYI GPS agreement is an internal prior residual, not absolute accuracy.',
                matched_image_count=len(names),odm_reconstructed_image_count=len(odm),rows=rows)
    for label in ('recorded','odm','metal'):
        report[label+'_horizontal_error_m']=distribution([r[label+'_error_m'] for r in rows if label+'_error_m' in r])
    if truth:
        expected={r['file']:float(r['gsd_m_per_px']) for r in truth}
        scales=[]
        for name,H in zip(names,hs):
            m=meta[name];c=np.array([(m['width']-1)/2,(m['height']-1)/2]);q=project(matrix@H,np.array([c,c+[1,0],c+[0,1]]))
            area=abs(np.linalg.det(np.stack([q[1]-q[0],q[2]-q[0]])))
            scales.append(dict(name=name,metal_source_gsd_m=math.sqrt(area),reference_gsd_m=expected[name],
                               relative_error=(math.sqrt(area)-expected[name])/expected[name]))
        report['metal_source_scale']=scales
        report['metal_absolute_relative_scale_error']=distribution([abs(r['relative_error']) for r in scales])
    return report


def main():
    global PROFILE, EVALUATION
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset',choices=['siyi','archive'])
    parser.add_argument('--profile',choices=['metal','metal_fast'],default='metal');args=parser.parse_args()
    PROFILE=args.profile; EVALUATION='evaluation_fast' if PROFILE=='metal_fast' else 'evaluation'
    suffix='_fast' if PROFILE=='metal_fast' else ''
    cv2.setNumThreads(6);cv2.setRNGSeed(2026)
    folder=BASE/args.dataset;out=folder/EVALUATION;plan=prepare_common(args.dataset)
    a=cv2.imread(str(out/'common_odm.png'),-1);b=cv2.imread(str(out/'common_metal.png'),-1)
    if a.shape[2]==3:a=np.dstack([a,np.full(a.shape[:2],255,np.uint8)])
    raw=pixel_agreement(a,b);registration,aligned=visual_registration(a,b)
    aligned_metrics=None
    if aligned is not None:
        aligned_metrics=pixel_agreement(a,aligned);cv2.imwrite(str(out/'visually_aligned_metal.png'),aligned)
    else:
        (out/'visually_aligned_metal.png').unlink(missing_ok=True)
    positions=position_metrics(folder);(out/'position_errors.json').write_text(json.dumps(positions,indent=2))
    fields=['name','recorded_error_m','odm_error_m','metal_error_m']
    with (out/'position_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(positions['rows'])
    difference=np.mean(np.abs(a[:,:,:3].astype(float)-b[:,:,:3]),axis=2)
    heat=cv2.applyColorMap(np.clip(difference*3,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
    heat[(a[:,:,3]<128)|(b[:,:,3]<128)]=0;cv2.imwrite(str(out/'difference_geographic.png'),heat)
    panel=np.hstack([a[:,:,:3],b[:,:,:3]])
    panel=cv2.copyMakeBorder(panel,48,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30))
    cv2.putText(panel,'ODM 3.6.2 default reconstruction',(15,31),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(panel,'ADM GPS-only Metal preview' if PROFILE=='metal_fast' else 'ADM Metal / MPS neutral map',(a.shape[1]+15,31),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
    cv2.imwrite(str(folder/f'side_by_side{suffix}.png'),panel)
    for source,target in [(out/'odm.png',folder/'odm.png'),(folder/PROFILE/'map_neutral.png',folder/f'{PROFILE}.png'),
                          (folder/PROFILE/'map_golden.png',folder/f'{PROFILE}_golden.png')]:
        if source.exists() and not target.exists():target.hardlink_to(source)
    timing_odm=read(folder/'odm/timing.json');timing_metal=read(folder/f'{PROFILE}_timing.json')
    for label,timing,png in [('odm',timing_odm,folder/'odm/odm_orthophoto/odm_orthophoto.png'),
                             ('metal',timing_metal,folder/PROFILE/'map_neutral.png')]:
        stamp=datetime.datetime.fromisoformat(timing['start_utc']).timestamp()
        ready=png.stat().st_mtime-stamp
        timing['primary_png_ready_seconds_from_mtime']=ready if 0<=ready<=timing['wall_seconds']+2 else None
        timing['logical_output_bytes_before_cleanup']=sum(p.stat().st_size for p in (folder/('odm' if label=='odm' else PROFILE)).rglob('*') if p.is_file() and 'images' not in p.relative_to(folder).parts)
        timing['input_megapixels_per_second']=read(folder/'inputs.json')['total_pixels']/1e6/timing['wall_seconds']
    report=dict(dataset=args.dataset,profile=PROFILE,inputs=read(folder/'inputs.json'),
                timing=dict(odm=timing_odm,metal=timing_metal,
                            odm_over_metal_wall_ratio=timing_odm['wall_seconds']/timing_metal['wall_seconds']),
                odm_raster=read(out/'raster.json'),metal_geometry=read(folder/PROFILE/'map_geometry.json'),
                odm_quality=read(folder/'odm/odm_report/stats.json'),metal_report=read(folder/PROFILE/'report.json'),
                metal_registration=read(folder/PROFILE/'registration_report.json'),
                metal_geographic_alignment=read(folder/PROFILE/'geographic_alignment.json'),
                positions=positions,common_grid=plan,
                image_statistics_at_common_resolution={'odm':image_statistics(a),'metal':image_statistics(b)},
                geographic_pixel_agreement=raw,visual_registration=registration,
                visually_registered_pixel_agreement=aligned_metrics,
                limitations=['One timed fresh run per pipeline and dataset; no statistical confidence interval.',
                             'ODM performs 3D reconstruction; ADM is a planar mosaic. These are different workloads.',
                             'ODM runs ARM64 in Docker with 12 visible CPUs and 7.75 GiB guest RAM; Metal runs natively with six OpenCV workers and the GPU.',
                             'Desktop apps/background services were not controlled. Docker crashed during disk exhaustion before the full SIYI Metal run and was restarted before Archive ODM and GPS-only benchmarks.',
                             'ODM quality defaults are unchanged; PNG export and intermediate disk cleanup are enabled.',
                             'Neutral maps are compared. GPS-only previews skip all matching and use recorded camera pose with nominal configured AGL/FOV; the full Metal pipeline performs visual alignment.',
                             'Image agreement, sharpness and training reprojection residuals are not independent survey accuracy.'])
    (folder/f'comparison{suffix}.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(dataset=args.dataset,timing=report['timing'],positions={k:v for k,v in positions.items() if k.endswith('_horizontal_error_m')},geographic_pixel_agreement=raw),indent=2))


if __name__=='__main__':main()
