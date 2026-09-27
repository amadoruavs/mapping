"""Compare lean runs to saved maps; measure sharpness on shared physical grids."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from adm.geometry import LocalEarth, project
import compare_odm as metrics

ROOT=Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def bytes_used(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def image_parity(a,b):
    if a.shape!=b.shape:return dict(identical=False,shapes=[list(a.shape),list(b.shape)])
    changed=error=maximum=0
    for y in range(0,len(a),128):
        d=np.abs(a[y:y+128].astype(np.int16)-b[y:y+128].astype(np.int16))
        changed+=int(np.count_nonzero(np.any(d!=0,axis=2)))
        error+=int(d.sum());maximum=max(maximum,int(d.max()))
    return dict(identical=changed==0,changed_pixels=changed,total_pixels=a.shape[0]*a.shape[1],
                mean_absolute_channel_error=error/a.size,max_channel_error=maximum)


def resample(image,transform,shape,scale):
    """Area-filter minification before the final geographic warp; pixel centers align."""
    h,w=image.shape[:2]
    if scale<1:
        nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
        image=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA)
        rx,ry=nw/w,nh/h
        transform=transform@np.array([[1/rx,0,(1/rx-1)/2],[0,1/ry,(1/ry-1)/2],[0,0,1.]])
    return cv2.warpPerspective(image,transform,shape,flags=cv2.INTER_LINEAR)


def sharpness(image,mask):
    if not mask.any():raise ValueError('No shared interior pixels for sharpness comparison')
    gray=cv2.cvtColor(image[:,:,:3],cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    laplacian=cv2.Laplacian(gray,cv2.CV_32F)
    return dict(mean_gradient=float(gradient[mask].mean()),laplacian_variance=float(laplacian[mask].var()))


def comparison_panel(a,b,mask,path,resolution):
    # Fixed spatial quantiles of shared support, not selected for favorable sharpness.
    yy,xx=np.nonzero(mask);rows=[];crop=100
    for fraction in (.25,.5,.75):
        y=int(np.quantile(yy,fraction));near=xx[abs(yy-y)<5]
        x=int(np.median(near)) if len(near) else int(np.median(xx))
        x=max(crop,min(a.shape[1]-crop,x));y=max(crop,min(a.shape[0]-crop,y))
        panels=[]
        for label,im in [('ODM',a),('ADM lean',b)]:
            patch=im[y-crop:y+crop,x-crop:x+crop,:3]
            patch=cv2.resize(patch,(400,400),interpolation=cv2.INTER_NEAREST)
            patch=cv2.copyMakeBorder(patch,34,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
            cv2.putText(patch,f'{label} | ground-grid crop {fraction:.0%}',(10,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            panels.append(patch)
        rows.append(np.hstack(panels))
    panel=np.vstack(rows)
    panel=cv2.copyMakeBorder(panel,40,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
    cv2.putText(panel,f'Same {resolution*100:.2f} cm/pixel grid; 2x pixel display; geographic alignment',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    if not cv2.imwrite(str(path),panel):raise OSError(path)


def compare(dataset):
    base=ROOT/'output'/('killerofmaps' if dataset=='killerofmaps' else 'comparison/'+dataset)
    old=base/('metal_optimized_128mb' if dataset=='killerofmaps' else 'metal_optimized')
    new=base/'metal_lean';out=base/'evaluation_lean';out.mkdir(exist_ok=True)
    timing=load(base/'metal_lean_timing.json');previous=load(old.with_name(old.name+'_timing.json'))
    if timing['exit_code'] or previous['exit_code']:raise ValueError('Cannot compare a failed run')
    a=cv2.imread(str(old/'map_neutral.png'),-1);b=cv2.imread(str(new/'map_neutral.png'),-1)
    if a is None or b is None:raise ValueError('Missing neutral map')
    result=dict(dataset=dataset,timing=dict(previous_seconds=previous['wall_seconds'],lean_seconds=timing['wall_seconds']),
                output_bytes=dict(previous=bytes_used(old),lean=bytes_used(new)),pixel_parity=image_parity(a,b),
                transforms_identical=bool(np.array_equal(np.load(old/'image_to_map.npy'),np.load(new/'image_to_map.npy'))),
                pipeline=load(new/'report.json'),registration=load(new/'registration_report.json'))
    del a
    if dataset!='killerofmaps':
        plan=load(base/'evaluation_optimized/common_grid.json');g=load(new/'map_geometry.json')
        if g['origin_latlon']!=plan['origin_latlon']:raise ValueError('Geographic origins differ')
        resolution=plan['metres_per_pixel'];w,h=plan['width'],plan['height'];lo,hi=plan['lower_xy'],plan['upper_xy']
        to_grid=np.array([[1/resolution,0,-lo[0]/resolution-.5],[0,-1/resolution,hi[1]/resolution-.5],[0,0,1.]])
        adm=resample(b,to_grid@np.asarray(g['map_to_world']),(w,h),min(1,g['metres_per_pixel']/resolution))
        raster_path=base/'evaluation/raster.json'
        if not raster_path.exists():raster_path=base/'evaluation_fast/raster.json'
        raster=load(raster_path);odm=cv2.imread(str(base/'odm.png'),-1)
        if odm is None:raise ValueError('Missing saved ODM raster')
        if odm.shape[2]==3:odm=np.dstack([odm,np.full(odm.shape[:2],255,np.uint8)])
        ow,oh=raster['width'],raster['height'];corners=np.asarray(raster['corners_lonlat'])
        earth=LocalEarth(*g['origin_latlon']);xy=earth.xy(corners[:,1],corners[:,0])
        source=np.float32([[-.5,-.5],[ow-.5,-.5],[ow-.5,oh-.5],[-.5,oh-.5]])
        warp=cv2.getPerspectiveTransform(source,project(to_grid,xy).astype(np.float32))
        odm_spacing=np.sqrt(raster['pixel_area_crs_units_squared'])*(raster['linear_units_m'] or 1)
        odm=resample(odm,warp,(w,h),min(1,odm_spacing/resolution))
        mask=cv2.erode(((adm[:,:,3]>127)&(odm[:,:,3]>127)).astype(np.uint8),np.ones((11,11),np.uint8))>0
        values={'adm':sharpness(adm,mask),'odm':sharpness(odm,mask)}
        result['sharpness']=dict(metres_per_pixel=resolution,shared_interior_pixels=int(mask.sum()),
            minification='INTER_AREA prefilter, then bilinear warp; identical eroded geographic overlap mask',
            values=values,adm_over_odm={k:values['adm'][k]/values['odm'][k] for k in values['adm']},
            interpretation='Edge/texture statistics, not resolved-detail or survey-accuracy ratios; noise, seams and geographic differences can affect them.')
        result['geographic_agreement']=metrics.pixel_agreement(odm,adm)
        result['timing']['odm_seconds']=load(base/'odm/timing.json')['wall_seconds']
        comparison_panel(odm,adm,mask,out/'crispness_crops.png',resolution)
        cv2.imwrite(str(out/'common_adm.png'),adm);cv2.imwrite(str(out/'common_odm.png'),odm)
    result['limitations']=['Single fresh run versus saved baseline; caches, machine load and thermal state uncontrolled.',
        'ADM is native Metal planar mapping; saved ODM is CPU 3D reconstruction in ARM64 Docker.',
        'Previous SIYI/Archive exports include optional lighting images, lean exports only the neutral map.']
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ['dataset','timing','output_bytes','pixel_parity','transforms_identical','sharpness'] if k in result},indent=2))
    if not result['pixel_parity']['identical']:raise SystemExit('Neutral-map parity failed; investigate before claiming unchanged quality')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',choices=['siyi','archive','killerofmaps'])
    compare(p.parse_args().dataset)
