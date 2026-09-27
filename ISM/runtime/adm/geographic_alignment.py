"""OpenSfM geographic residuals + ISM image matching, solved by Metal/MPS.

This is a fixed-ground-plane adaptation, not OpenSfM's full 3D bundle adjuster.
Positions are true camera centres, not projected image centres. Metadata parsing,
double precision WGS84 conversion and buffer packing remain on the CPU.
"""
import ctypes as C
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import camera_rotation, intrinsics, project, ground_homography
from .io import write_array, write_json

UPSTREAM = 'c5328439465e6ace011f39077d1077d7b1cdd65d'


class TopocentricEarth:
    """OpenSfM WGS84 ECEF/topocentric equations, evaluated in float64.

    Ellipsoid height zero supplies horizontal coordinates only. Configured AGL
    supplies camera height over the fixed plane; the two datums are never mixed.
    """
    def __init__(self, lat, lon):
        self.lat,self.lon=float(lat),float(lon)
        a,o=np.deg2rad([lat,lon]);sa,ca,so,co=np.sin(a),np.cos(a),np.sin(o),np.cos(o)
        self.axes=np.array([[-so,-sa*co,ca*co],[co,-sa*so,ca*so],[0,ca,sa]])
        self.origin=self.ecef(lat,lon)

    @staticmethod
    def ecef(lat,lon):
        a2=6378137.0**2;b2=6356752.314245**2
        lat,lon=np.deg2rad(lat),np.deg2rad(lon)
        L=1/np.sqrt(a2*np.cos(lat)**2+b2*np.sin(lat)**2)
        return np.stack([a2*L*np.cos(lat)*np.cos(lon),a2*L*np.cos(lat)*np.sin(lon),b2*L*np.sin(lat)],axis=-1)

    def xy(self,lat,lon):
        return ((self.ecef(lat,lon)-self.origin)@self.axes)[...,:2]

    def latlon(self,xy):
        p=np.asarray(xy);p=np.concatenate([p,np.zeros((*p.shape[:-1],1))],axis=-1)
        ecef=p@self.axes.T+self.origin;x,y,z=np.moveaxis(ecef,-1,0)
        a,b=6378137.0,6356752.314245;ea2=(a*a-b*b)/(a*a);eb2=(a*a-b*b)/(b*b)
        radius=np.hypot(x,y);theta=np.arctan2(z*a,radius*b)
        lat=np.arctan2(z+eb2*b*np.sin(theta)**3,radius-ea2*a*np.cos(theta)**3)
        return np.stack([np.rad2deg(lat),np.rad2deg(np.arctan2(y,x))],axis=-1)


def pack_cameras(metadata,cfg,earth):
    n=len(metadata);scale=float(np.median([m.altitude_agl_m for m in metadata]))
    data=np.zeros((n,32),np.float32);x=np.zeros((n,8),np.float32)
    for i,m in enumerate(metadata):
        if not m.geographic or m.gps_reference!='camera':
            raise ValueError(f'{m.name}: geographic solver needs GPS camera position and camera yaw')
        sh=round(m.height*cfg.registration_width/m.width)
        S=np.diag([1/m.width,sh/(m.height*cfg.registration_width),1.])
        K=S@intrinsics(m,cfg)
        R=camera_rotation(m.yaw_deg,m.pitch_deg,m.roll_deg)
        center=np.r_[earth.xy(m.latitude,m.longitude),m.altitude_agl_m]/scale
        data[i,:9]=R.ravel(order='F');data[i,9:18]=np.linalg.inv(K).ravel(order='F')
        data[i,18:21]=center;x[i,:3]=center
        sigma=m.gps_accuracy_m or cfg.geographic_gps_sigma_m
        data[i,21:24]=scale/np.array([sigma,sigma,cfg.geographic_altitude_sigma_m])
        # Assumed orientations are not recorded observations: use a weak 45deg prior.
        measured=all(not (m.provenance or {}).get(k,'configured').startswith('configured') for k in ('yaw_deg','pitch_deg','roll_deg'))
        data[i,24]=1/np.deg2rad(cfg.geographic_orientation_sigma_deg if measured else 45.)
        data[i,25]=sh/cfg.registration_width
    return x,data,scale


def camera_model(x,data):
    """Independent float64 physical projection / OpenSfM prior reference."""
    x=np.asarray(x,dtype=np.float64);data=np.asarray(data,dtype=np.float64)
    base=data[:,:9].reshape(-1,3,3).transpose(0,2,1)
    R=Rotation.from_rotvec(x[:,3:6]).as_matrix()@base
    inverse=data[:,9:18].reshape(-1,3,3).transpose(0,2,1);rays=R@inverse
    center=x[:,:3]
    H=np.stack([center[:,2,None]*rays[:,0]-center[:,0,None]*rays[:,2],
                center[:,2,None]*rays[:,1]-center[:,1,None]*rays[:,2],-rays[:,2]],axis=1)
    gps=(center-data[:,18:21])*data[:,21:24]
    up=(R@base.transpose(0,2,1))[:,:,2]-[0,0,1]
    north=(R@base.transpose(0,2,1))[:,:,1]-[0,1,0]
    residual=np.c_[gps,up*data[:,24,None],north*data[:,24,None]]
    costs=np.stack([.5*np.sum(gps**2,axis=1),.5*np.log1p(np.sum(residual[:,3:6]**2,axis=1)),
                    .5*np.log1p(np.sum(residual[:,6:9]**2,axis=1))],axis=1)
    return H,R,residual,costs


def pack_edges(names,edges,width):
    n=len(names);count=sum(2*len(e['a']) for e in edges)
    points=np.empty((count,4),np.float32);owners=np.empty((count,2),np.uint32)
    links=np.full((n,n),-1,np.int32);incidents=[[] for _ in names];rows=[];offset=0
    for k,e in enumerate(edges):
        i,j=e['i'],e['j'];m=2*len(e['a'])
        if i==j or links[i,j]>=0:raise ValueError('Duplicate/self edge')
        points[offset:offset+m]=np.repeat(np.c_[e['a'],e['b']],2,axis=0);owners[offset:offset+m]=[i,j]
        rows.append([i,j,offset,m]);offset+=m;links[i,j]=links[j,i]=k
        incidents[i].append([k,0]);incidents[j].append([k,1])
    offsets=np.r_[0,np.cumsum([len(v) for v in incidents])].astype(np.uint32)
    incidence=np.array([item for values in incidents for item in values],np.uint32)
    if not np.isfinite(points).all():raise ValueError('Nonfinite visual correspondence')
    return points,owners,np.array(rows,np.uint32),links,offsets,incidence


def initialize_from_visual(names,edges,metadata,earth,cfg,out,gpu,data,scale):
    """Decompose the verified visual mosaic into physical cameras before bundle.

    Metadata yaw may be 180 degrees away from the pixel orientation. Its robust
    prior must not select the starting basin of the visual optimization.
    """
    from .gpu_alignment import solve as visual_solve
    from scipy.optimize import least_squares
    folder=out/'visual_initialization';folder.mkdir(parents=True,exist_ok=True)
    fitted=visual_solve(names,edges,folder,cfg.registration_width,gpu,
        iterations=cfg.alignment_max_iterations,workspace_budget_mb=cfg.alignment_budget_mb,symmetric=True)
    native=[fitted[i]@np.diag([cfg.registration_width/m.width,round(m.height*cfg.registration_width/m.width)/m.height,1.]) for i,m in enumerate(metadata)]
    centers=[[(m.width-1)/2,(m.height-1)/2] for m in metadata]
    source=np.array([project(h,[c])[0] for h,c in zip(native,centers)]);source[:,1]*=-1
    target=np.array([project(ground_homography(m,cfg,earth),[c])[0] for m,c in zip(metadata,centers)])
    sx,sy=source.T;A=np.zeros((len(names)*2,4));A[0::2]=np.c_[sx,-sy,np.ones(len(names)),np.zeros(len(names))]
    A[1::2]=np.c_[sy,sx,np.zeros(len(names)),np.ones(len(names))]
    initial=np.linalg.lstsq(A,target.ravel(),rcond=None)[0]
    a,b,tx,ty=least_squares(lambda v:A@v-target.ravel(),initial,loss='soft_l1',f_scale=.75).x
    similarity=np.array([[a,b,tx],[b,-a,ty],[0,0,1.]])
    x=np.zeros((len(names),8),np.float32)
    for i,(h,m) in enumerate(zip(native,metadata)):
        B=np.linalg.inv(intrinsics(m,cfg))@np.linalg.inv(similarity@h)
        B*=2/(np.linalg.norm(B[:,0])+np.linalg.norm(B[:,1]))
        # Choose the physically visible side of the fixed plane.
        if np.linalg.det(B)>0:B=-B
        U,_,Vt=np.linalg.svd(np.c_[B[:,0],B[:,1],np.cross(B[:,0],B[:,1])])
        R=(U@Vt).T;center=-R@B[:,2]
        if center[2]<=0:raise ValueError('Visual initialization puts camera below ground')
        base=data[i,:9].reshape(3,3).T
        x[i,:3]=center/scale;x[i,3:6]=Rotation.from_matrix(R@base.T).as_rotvec()
    return x


def solve(names,edges,metadata,earth,cfg,out,gpu):
    from .match_graph import require_connected
    require_connected(names,edges,out)
    started=time.perf_counter();n=len(names);width=cfg.registration_width;iterations=cfg.alignment_max_iterations
    x,data,scale=pack_cameras(metadata,cfg,earth)
    x=initialize_from_visual(names,edges,metadata,earth,cfg,out,gpu,data,scale)
    initialized=time.perf_counter()
    points,owners,rows,links,offsets,incidence=pack_edges(names,edges,width)
    count=len(points);d=n*8
    workspace=count*156+len(edges)*1088+d*d*8+d*12+n*(9+72+72+9)*4
    if workspace>cfg.alignment_budget_mb*1024**2:raise ValueError('Geographic alignment exceeds alignment_budget_mb')
    state=np.zeros(16+iterations*4,np.float32);state[0]=1e-3
    buffers=[gpu.upload(v) for v in (x,points,owners,rows,links,offsets,incidence,state,data)]
    handles=(C.c_void_p*len(buffers))(*(b.handle for b in buffers));before=gpu.stats();prepared=time.perf_counter()
    gpu.check(gpu.lib.adm_align_geographic(gpu.handle,handles,n,len(edges),count,0,iterations,float(width)))
    finished=time.perf_counter();state=buffers[7].numpy();x=buffers[0].numpy().astype(np.float64)
    H,R,residual,costs=camera_model(x,data)
    hs=[np.diag([scale,scale,1.])@h@np.diag([1/width,1/width,1.]) for h in H]
    errors=np.concatenate([np.linalg.norm(project(np.linalg.inv(hs[e[j]])@hs[e[i]],e[a])-e[b],axis=1)
        for e in edges for i,j,a,b in [('i','j','a','b'),('j','i','b','a')]])
    gps_error=np.linalg.norm((x[:,:2]-data[:,18:20])*scale,axis=1)
    tilt=np.rad2deg(np.arccos(np.clip(-R[:,2,2],-1,1)))
    heading=np.rad2deg(np.arctan2(-R[:,0,1],-R[:,1,1]))
    yaw_error=(heading-np.array([m.yaw_deg for m in metadata])+180)%360-180
    after=gpu.stats()
    report=dict(images=n,edges=len(edges),observations=count//2,residual_observations=count,solver_success=bool(state[6]),
        solver='ISM symmetric visual transfer + OpenSfM position/axis priors, Metal analytical derivatives / MPS Cholesky',
        solver_message={1:'Objective tolerance',2:'Step tolerance',3:'Predicted reduction tolerance'}.get(int(state[6]),'Iteration limit'),
        objective='bidirectional soft_l1 image-pixel transfer + uncertainty-normalized GPS/AGL least squares + Cauchy camera-axis priors',
        median_registration_error_px=float(np.median(errors)),p95_registration_error_px=float(np.percentile(errors,95)),
        iterations_executed=int(state[5]),iterations_encoded=int(state[9]),iterations_budget=iterations,
        accepted_steps=int(state[2]),rejected_steps=int(state[3]),initial_objective=float(state[7]),final_objective=float(state[4]),
        relative_objective_tolerance=1e-5,convergence_requires_low_damping=True,convergence_check_interval=4,
        optimization_trace=state[16:16+int(state[5])*4].reshape(-1,4).tolist(),
        optimization_trace_columns=['old_objective','trial_objective','updated_damping','relative_step_norm'],
        host_setup_seconds=prepared-started,device_operation_wall_seconds=finished-prepared,
        visual_initialization_seconds=initialized-started,geographic_host_setup_seconds=prepared-initialized,
        planned_scratch_bytes=workspace,workspace_budget_mb=cfg.alignment_budget_mb,
        gpu_seconds=after['gpu_seconds']-before['gpu_seconds'],mps_operations=after['mps_operations']-before['mps_operations'],
        upstream_revision=UPSTREAM,geographic_prior_costs=dict(zip(('position','up_axis','north_axis'),costs.sum(0).tolist())),
        scene_model='fixed ground plane, 6 DOF physical camera poses, fixed intrinsics; no triangulated terrain',
        measured_gps_accuracy_images=sum(m.gps_accuracy_m is not None for m in metadata),
        gps_status_disconnected_images=sum('NO GPS' in (m.gps_status or '').upper() for m in metadata))
    write_json(out/'registration_report.json',report)
    write_array(out/'transforms.npy',np.array(hs));write_array(out/'camera_poses.npy',np.c_[x[:,:3]*scale,Rotation.from_matrix(R).as_rotvec()])
    write_json(out/'geographic_alignment.json',dict(method=report['solver'],median_center_residual_m=float(np.median(gps_error)),
        p95_center_residual_m=float(np.percentile(gps_error,95)),gps_tolerance_m=cfg.gps_tolerance_m,
        gps_fit_within_tolerance=bool(np.median(gps_error)<=cfg.gps_tolerance_m),gps_mismatch_bypassed=False,
        survey_accuracy_verified=False,median_heading_disagreement_deg=float(np.median(np.abs(yaw_error))),
        p95_heading_disagreement_deg=float(np.percentile(np.abs(yaw_error),95)),median_tilt_deg=float(np.median(tilt)),
        coordinate_system='WGS84 topocentric ENU; local flat ground z=0',altitude_source='configured AGL, not EXIF sea-level altitude',
        gps_sigma_m=cfg.geographic_gps_sigma_m,altitude_sigma_m=cfg.geographic_altitude_sigma_m,
        orientation_sigma_deg=cfg.geographic_orientation_sigma_deg,upstream_revision=UPSTREAM))
    write_json(out/'camera_geography.json',[dict(name=m.name,camera_enu_m=(x[i,:3]*scale).tolist(),
        gps_error_m=float(gps_error[i]),heading_error_deg=float(yaw_error[i]),tilt_deg=float(tilt[i]),
        gps_status=m.gps_status,gps_accuracy_m=m.gps_accuracy_m) for i,m in enumerate(metadata)])
    print(json.dumps({k:v for k,v in report.items() if k!='optimization_trace'},indent=2),flush=True)
    if not report['solver_success'] or not np.isfinite(errors).all():raise RuntimeError('Geographic solver did not converge to finite geometry')
    if np.median(gps_error)>cfg.gps_tolerance_m:raise ValueError('Geographic camera fit exceeds GPS tolerance; see geographic_alignment.json')
    # Physical model guarantees positive area only when all rays meet ground.
    for h,m in zip(hs,metadata):
        sh=round(m.height*width/m.width);corners=np.array([[0,0],[width,0],[width,sh],[0,sh]])
        z=(np.c_[corners,np.ones(4)]@h.T)[:,2]
        if np.any(z<=1e-3):raise ValueError('Physical camera footprint crosses the ground horizon')
    return {i:h for i,h in enumerate(hs)}
