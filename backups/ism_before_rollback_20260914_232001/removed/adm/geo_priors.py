"""GPS and camera-pose constraints in a fixed, metric alignment coordinate frame."""
import cv2
import numpy as np
from .geometry import ground_homography,project


def build_priors(metadata,cfg,earth):
    missing=[m.name for m in metadata if not m.geographic]
    if missing:raise ValueError('GPS/yaw alignment requires GPS and absolute camera yaw for every retained photo: '+', '.join(missing[:8]))
    if any(abs(m.pitch_deg+90)>5 for m in metadata):raise ValueError('GPS/yaw alignment currently requires near-nadir images (pitch within 5 degrees of -90)')
    width=cfg.registration_width
    mpp=float(np.median([2*m.altitude_agl_m*np.tan(np.deg2rad(cfg.hfov_deg)/2)/(width*cfg.digital_zoom) for m in metadata]))
    origin=earth.xy(metadata[len(metadata)//2].latitude,metadata[len(metadata)//2].longitude)
    world_to_alignment=np.array([[1/mpp,0,-origin[0]/mpp],[0,-1/mpp,origin[1]/mpp],[0,0,1.]])
    initial=[];samples=[];areas=[];shapes=[]
    for m in metadata:
        height=round(m.height*width/m.width);D=np.diag([width/m.width,height/m.height,1.])
        H=world_to_alignment@ground_homography(m,cfg,earth)@np.linalg.inv(D);H/=H[2,2]
        # Optical center plus right/up/left/down axis endpoints. Offsets constrain
        # yaw and size without incorrectly pinning every corner to noisy GPS.
        cx=(m.width-1)/2*D[0,0];cy=(m.height-1)/2*D[1,1]
        x=(m.width-1)*D[0,0];y=(m.height-1)*D[1,1]
        points=np.array([[cx,cy],[x,cy],[cx,0],[0,cy],[cx,y]])
        target=project(H,points)
        corners=project(H,[[0,0],[x,0],[x,y],[0,y]])/width
        area=float(cv2.contourArea(corners.astype(np.float32),oriented=True))
        if not np.isfinite(area) or area<=0:raise ValueError(f'{m.name}: invalid camera footprint prior')
        initial.append(H);samples.append(np.c_[points/width,target/width]);areas.append(area);shapes.append([x,y])
    return dict(initial=np.array(initial),samples=np.array(samples,np.float32),areas=np.array(areas,np.float32),
        shapes=np.array(shapes),alignment_to_world=np.linalg.inv(world_to_alignment),metres_per_pixel=mpp,
        gps_sigma_m=cfg.gps_prior_sigma_m,gps_strength=cfg.gps_prior_strength,
        yaw_sigma_deg=cfg.yaw_prior_sigma_deg,scale_sigma=cfg.scale_prior_sigma,camera_strength=cfg.camera_prior_strength,
        registration_width=width)


def weights(priors,edges):
    count=len(priors['initial']);degree=np.zeros(count)
    for edge in edges:
        degree[edge['i']]+=len(edge['a']);degree[edge['j']]+=len(edge['a'])
    degree=np.maximum(degree,1);width=priors['registration_width']
    result=np.zeros((count,4),np.float32)
    # Divide by width² because the normal equations use normalized coordinates,
    # while acceptance costs use the equivalent registration-pixel objective.
    result[:,0]=degree*priors['gps_strength']/(priors['gps_sigma_m']/priors['metres_per_pixel'])**2
    result[:,1]=degree*priors['camera_strength']/(4*priors['scale_sigma']**2*width**2)
    result[:,2]=degree*priors['camera_strength']/(4*np.deg2rad(priors['yaw_sigma_deg'])**2*width**2)
    result[:,3]=priors['areas']
    return result


def diagnostics(hs,priors):
    width=priors['registration_width'];gps=[];yaw=[];area=[]
    for i,H in enumerate(hs):
        points=priors['samples'][i,:,:2].astype(float)*width
        target=priors['samples'][i,:,2:].astype(float)*width
        actual=project(H,points);gps.append(float(np.linalg.norm(actual[0]-target[0])*priors['metres_per_pixel']))
        a=actual[2]-actual[0];b=target[2]-target[0]
        yaw.append(float(np.rad2deg(np.arctan2(a[0]*b[1]-a[1]*b[0],np.dot(a,b)))))
        x,y=priors['shapes'][i];corners=project(H,[[0,0],[x,0],[x,y],[0,y]])/width
        area.append(float(cv2.contourArea(corners.astype(np.float32),oriented=True)/priors['areas'][i]))
    return dict(enabled=True,gps_sigma_m=priors['gps_sigma_m'],gps_strength=priors['gps_strength'],
        yaw_sigma_deg=priors['yaw_sigma_deg'],scale_sigma=priors['scale_sigma'],camera_strength=priors['camera_strength'],
        gps_center_median_error_m=float(np.median(gps)),gps_center_p95_error_m=float(np.percentile(gps,95)),
        gps_center_max_error_m=float(np.max(gps)),yaw_median_abs_error_deg=float(np.median(np.abs(yaw))),
        yaw_p95_abs_error_deg=float(np.percentile(np.abs(yaw),95)),footprint_area_ratio_min=float(min(area)),
        footprint_area_ratio_median=float(np.median(area)),footprint_area_ratio_max=float(max(area)),
        valid_footprints=bool(np.isfinite(area).all() and min(area)>=.2499 and max(area)<=4.0001),
        reference='GPS camera position projected through measured yaw/pitch/roll and configured AGL/FOV onto a flat ground plane',
        scale_and_yaw_constrained_during_optimization=True,survey_accuracy_verified=False)
