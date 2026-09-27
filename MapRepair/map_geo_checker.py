"""Geometry diagnostics and local repairs, adapted from the supplied checker.

Fits are evidence of consistency, not ground truth. Weak/unsupported fits and
GPS-only disagreements are reported, never used as automatic replacement poses.
"""
from dataclasses import dataclass
from pathlib import Path
import math
import time
import cv2
import numpy as np
from georeference import read_gps
from patch_blend import blend_patch
from parallel_runtime import bounded_map, worker_count
from PIL import Image
from contextlib import closing


@dataclass
class Settings:
    feature_method: str='SIFT'
    map_width: int=2600
    photo_width: int=1400
    map_features: int=35000
    photo_features: int=5000
    ratio: float=.72
    min_inliers: int=30
    min_inlier_ratio: float=.25
    min_support: float=.06
    max_reprojection_px: float=4.
    max_local_shift_px: float=35.
    max_pairwise_px: float=4.
    gps_search_m: float=85.
    max_gps_discrepancy_m: float=15.
    tile_px: int=64  # Working-map pixels; report records the native equivalent.
    pair_neighbors: int=3
    min_confirming_neighbors: int=1
    feather_px: int=32  # Native-map pixels; smooth inward blend, twice as broad for color.
    appearance_context_px: int=64
    match_patch_detail: bool=True
    seed: int=2026

    def __post_init__(self):
        if self.feature_method not in ('SIFT','ORB'):raise ValueError('Feature method must be SIFT or ORB')
        if not 0<self.ratio<1 or not 0<self.min_inlier_ratio<=1:raise ValueError('Invalid match ratios')
        if min(self.map_width,self.photo_width,self.tile_px)<16 or self.min_inliers<8:raise ValueError('Invalid working sizes or inlier floor')
        if not 0<=self.min_confirming_neighbors<=self.pair_neighbors:raise ValueError('Invalid source-pair confirmation count')
        if self.feather_px<1 or self.appearance_context_px<16:raise ValueError('Invalid patch blending sizes')


def resolve_photo_paths(patterns):
    import glob
    paths=[]
    for pattern in patterns:
        path=Path(pattern)
        paths.extend(path.iterdir() if path.is_dir() else map(Path,glob.glob(str(pattern))))
    return sorted({p.resolve() for p in paths if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff','.webp')})


def resize_gray(image,width):
    h,w=image.shape[:2];scale=min(1.,width/w);ow,oh=max(1,round(w*scale)),max(1,round(h*scale))
    gray=cv2.cvtColor(image[:,:,:3],cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(ow,oh),interpolation=cv2.INTER_AREA)
    matrix=np.array([[ow/w,0,(ow/w-1)/2],[0,oh/h,(oh/h-1)/2],[0,0,1.]])
    return gray,matrix


def project(H,points):
    points=np.asarray(points,float).reshape(-1,2);q=np.c_[points,np.ones(len(points))]@H.T
    if not np.isfinite(q).all() or np.any(np.abs(q[:,2])<1e-9):raise ValueError('Homography crosses the horizon')
    return q[:,:2]/q[:,2:]


def quad(shape,H):
    h,w=shape[:2];return project(H,[[0,0],[w-1,0],[w-1,h-1],[0,h-1]])


def valid_homography(H,source_shape,target_shape):
    if H is None or not np.isfinite(H).all():return False
    try:
        h,w=source_shape[:2];corners=np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]])
        depth=corners@H[2];points=quad(source_shape,H).astype(np.float32)
        if np.any(depth*depth[0]<=0) or not cv2.isContourConvex(points):return False
        area=cv2.contourArea(points,oriented=True)
        if not 25<area<4*np.prod(target_shape[:2]):return False
        center=np.array([[(w-1)/2,(h-1)/2],[(w-1)/2+1,(h-1)/2],[(w-1)/2,(h-1)/2+1]])
        q=project(H,center);values=np.linalg.svd((q[1:]-q[0]).T,compute_uv=False)
        return bool(values[-1]>.02 and values[0]<30 and values[0]/values[-1]<8)
    except (ValueError,np.linalg.LinAlgError,cv2.error):return False


def detect(gray,mask,count,method):
    feature=cv2.SIFT_create(nfeatures=count,contrastThreshold=.025) if method=='SIFT' else cv2.ORB_create(nfeatures=count,nlevels=10)
    keypoints,descriptors=feature.detectAndCompute(gray,mask)
    return np.array([k.pt for k in keypoints],np.float32).reshape(-1,2),descriptors


def fit_alignment(src,src_desc,dst,dst_desc,source_shape,cfg,target_scale=1.):
    if src_desc is None or dst_desc is None or min(len(src),len(dst))<8:return None
    if src_desc.dtype==np.uint8:matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    else:matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=64))
    pairs=matcher.knnMatch(src_desc,dst_desc,k=2)
    good=sorted([m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<cfg.ratio*n.distance],key=lambda m:m.distance)
    unique=[];used=set()
    for m in good:
        if m.trainIdx not in used:unique.append(m);used.add(m.trainIdx)
    if len(unique)<12:return None
    p=src[[m.queryIdx for m in unique]];q=dst[[m.trainIdx for m in unique]]
    # Hold out every fifth match: they never participate in fitting H.
    hold=np.arange(len(p))%5==0;threshold=max(.7,cfg.max_reprojection_px*target_scale)
    cv2.setRNGSeed(cfg.seed)
    H,mask=cv2.findHomography(p[~hold],q[~hold],cv2.RANSAC,threshold,maxIters=4000,confidence=.999)
    if H is None:return None
    try:errors=np.linalg.norm(project(H,p)-q,axis=1)/target_scale
    except ValueError:return None
    inlier=errors<=cfg.max_reprojection_px;count=int(inlier.sum())
    if count<4:return None
    hull=cv2.convexHull(p[inlier]);support=float(cv2.contourArea(hull)/np.prod(source_shape[:2]))
    validation=errors[hold];vratio=float(np.mean(validation<=cfg.max_reprojection_px))
    reliable=(count>=cfg.min_inliers and count/len(p)>=cfg.min_inlier_ratio and support>=cfg.min_support
              and np.count_nonzero(validation<=cfg.max_reprojection_px)>=4 and vratio>=.25)
    return dict(H=H,matched=len(p),inliers=count,inlier_ratio=count/len(p),support_fraction=support,
        held_out_matches=int(hold.sum()),held_out_inlier_ratio=vratio,
        mean_reprojection_px=float(errors[inlier].mean()),p95_reprojection_px=float(np.quantile(errors[inlier],.95)),
        reliable=bool(reliable),src=p[inlier],dst=q[inlier])


def find_alignment(map_gray,photo_gray,nfeatures=8000,ratio=.75,ransac_thresh=5.):
    """Small compatibility entry point for the original ORB-based checker."""
    cfg=Settings(feature_method='ORB',ratio=ratio,max_reprojection_px=ransac_thresh)
    p,pd=detect(photo_gray,None,nfeatures,'ORB');q,qd=detect(map_gray,None,nfeatures,'ORB')
    return fit_alignment(p,pd,q,qd,photo_gray.shape,cfg)


def edge_weights(shape,feather):
    # A nonzero image alone has no distance-transform boundary: pad with zero.
    h,w=shape[:2];mask=np.pad(np.ones((h,w),np.uint8),(1,1))
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)[1:-1,1:-1]-1
    return np.clip(distance/max(float(feather),1.),0,1)


def render_repairs(image,owners,patches,analysis_shape,tile,cfg,progress=print):
    """Render validated selections, measuring appearance against the ORIGINAL map.

    owners uses one-based source IDs (zero = untouched). Connected components
    bound scratch memory and prevent distant patches sharing one color fit.
    """
    mh,mw=analysis_shape;h,w=image.shape[:2];result=image.copy();accepted=[]
    gy=np.minimum((np.arange(h,dtype=np.int64)*mh//h)//tile,owners.shape[0]-1)
    gx=np.minimum((np.arange(w,dtype=np.int64)*mw//w)//tile,owners.shape[1]-1)
    pad=max(cfg.appearance_context_px,2*cfg.feather_px)
    for patch in patches:
        source_id=patch['source_index'];H=np.asarray(patch['H'],float)
        photo=cv2.imread(str(patch['file']),cv2.IMREAD_UNCHANGED)
        if photo is None:raise OSError(f'Photo disappeared: {patch["file"]}')
        n,labels,stats,_=cv2.connectedComponentsWithStats((owners==source_id).astype(np.uint8),8)
        source_alpha=photo[:,:,3].astype(np.float32)/255 if photo.shape[2]==4 else np.ones(photo.shape[:2],np.float32)
        source_color=photo[:,:,:3].astype(np.float32)*source_alpha[...,None]
        source_weight=edge_weights(photo.shape,8)*source_alpha
        components=[]
        for c in range(1,n):
            x,y,ww,hh,_=stats[c]
            l=max(0,int(np.searchsorted(gx,x))-pad);r=min(w,int(np.searchsorted(gx,x+ww))+pad)
            t=max(0,int(np.searchsorted(gy,y))-pad);b=min(h,int(np.searchsorted(gy,y+hh))+pad)
            local_owner=owners[gy[t:b,None],gx[None,l:r]]
            selected=(labels[gy[t:b,None],gx[None,l:r]]==c)&(image[t:b,l:r,3]>0)
            if not selected.any():continue
            T=np.array([[1,0,-l],[0,1,-t],[0,0,1.]])
            alpha=cv2.warpPerspective(source_alpha,T@H,(r-l,b-t))
            warped=cv2.warpPerspective(source_color,T@H,(r-l,b-t))/np.maximum(alpha[...,None],1e-8)
            weight=cv2.warpPerspective(source_weight,T@H,(r-l,b-t))
            weight*=image[t:b,l:r,3].astype(np.float32)/255
            context=(local_owner==0)&(image[t:b,l:r,3]==255)
            blended,metrics=blend_patch(image[t:b,l:r,:3],warped,selected,weight,context,
                feather=cfg.feather_px,match_detail=cfg.match_patch_detail)
            # Only this component writes; all context comes from the immutable
            # original, so the result does not depend on source iteration order.
            roi=result[t:b,l:r,:3];roi[selected]=blended[selected]
            metrics['bounds_xyxy']=[l,t,r,b];components.append(metrics)
        accepted.append({**patch,'affected_pixels':sum(c['affected_pixels'] for c in components),'blending':components})
        progress(f'[repair] {Path(patch["file"]).name}: {patch["tiles"]} verified tiles, appearance matched')
    return result,accepted


class Checker:
    def __init__(self,map_image,geo,cfg,progress=print):
        self.image=map_image;self.geo=geo;self.cfg=cfg;self.progress=progress
        self.gray,self.S=resize_gray(map_image,cfg.map_width);self.scale=float(np.sqrt(np.linalg.det(self.S[:2,:2])))
        self.alpha=cv2.resize(map_image[:,:,3],(self.gray.shape[1],self.gray.shape[0]),interpolation=cv2.INTER_NEAREST)
        mask=cv2.erode(self.alpha,np.ones((5,5),np.uint8))
        self.points,self.descriptors=detect(self.gray,mask,cfg.map_features,cfg.feature_method)
        self.frames=[];self.entries=[];self.pairs=[]

    def scan(self,paths):
        paths=list(paths)
        def prepare(path):
            start=time.perf_counter();image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            gray=S=mask=p=pd=gps=None
            if image is not None and image.ndim==3 and image.shape[2] in (3,4) and image.dtype==np.uint8:
                gray,S=resize_gray(image,self.cfg.photo_width)
                mask=None if image.shape[2]==3 else cv2.resize(image[:,:,3],(gray.shape[1],gray.shape[0]),interpolation=cv2.INTER_NEAREST)
                p,pd=detect(gray,mask,self.cfg.photo_features,self.cfg.feature_method);gps=read_gps(path)
            return path,start,image,gray,S,mask,p,pd,gps
        def estimate(path):
            try:
                with Image.open(path) as im:w,h=im.size
            except OSError:return 64*1024**2
            # Decode plus detector pyramid/scratch and a retained future result.
            return w*h*16 + self.cfg.photo_width**2*96 + 32*1024**2
        previous=cv2.getNumThreads();cv2.setNumThreads(1)
        try:
            with closing(bounded_map(prepare,paths,workers=worker_count(),bytes_per_item=estimate)) as prepared:
                return self._scan_prepared(paths,prepared)
        finally:cv2.setNumThreads(previous)

    def _scan_prepared(self,paths,prepared):
        for index,(path,start,image,gray,S,mask,p,pd,gps) in enumerate(prepared):
            entry=dict(file=str(path),status='UNRESOLVED',patched_tiles=0);self.entries.append(entry)
            if image is None or image.ndim!=3 or image.shape[2] not in (3,4):entry['reason']='Unreadable color photo';continue
            if image.dtype!=np.uint8:entry['reason']='Only 8-bit source photos are supported';continue
            q,qd=self.points,self.descriptors;predicted=None;search='full map'
            if self.geo and gps:
                predicted=self.geo.pixel(gps['lat'],gps['lon'])
                J=self.geo.metric_jacobian(predicted)
                try:radius=self.cfg.gps_search_m*np.linalg.norm(np.linalg.inv(J),axis=1)
                except np.linalg.LinAlgError:radius=np.array(self.image.shape[1::-1])
                bounds=project(self.S,[predicted-radius,predicted+radius]);inside=np.all((q>=bounds[0])&(q<=bounds[1]),axis=1)
                if inside.sum()>=30:q,qd=q[inside],qd[inside];search='GPS ROI (explicit georeference)'
            fit=fit_alignment(p,pd,q,qd,gray.shape,self.cfg,self.scale)
            if (fit is None or not fit['reliable']) and search!='full map':
                fallback=fit_alignment(p,pd,self.points,self.descriptors,gray.shape,self.cfg,self.scale)
                if fallback and (fit is None or fallback['inliers']>fit['inliers']):fit=fallback;search='full-map fallback'
            entry.update(gps=gps,search=search)
            if fit is None:entry['reason']='Not enough independently validated feature matches'
            else:
                H=np.linalg.inv(self.S)@fit['H']@S
                fit['reliable']=fit['reliable'] and valid_homography(H,image.shape,self.image.shape)
                entry.update({k:v for k,v in fit.items() if k not in ('H','src','dst')});entry['H_photo_to_map']=H.tolist()
                if predicted is not None:
                    center=project(H,[[(image.shape[1]-1)/2,(image.shape[0]-1)/2]])[0]
                    offset=self.geo.error_m(center,predicted)
                    entry.update(gps_center_offset_m=offset,gps_position_flag=bool(gps['trusted'] and offset>self.cfg.max_gps_discrepancy_m),
                        gps_note='Image-center versus camera GPS is a near-nadir proxy, not surveyed ground truth')
                if fit['reliable']:
                    frame=dict(path=path,gray=gray,alpha=mask,S=S,H=H,shape=image.shape,points=p,descriptors=pd,fit=fit,entry=entry,
                        footprint=quad(image.shape,H),neighbors=[],witnesses={})
                    frame['tiles']=self.local_discrepancies(frame)
                    entry['local_discrepancy_tiles']=len(frame['tiles']);entry['status']='FLAGGED' if frame['tiles'] or entry.get('gps_position_flag') else 'CONSISTENT'
                    self.frames.append(frame)
                else:entry['reason']='Weak validation/spatial support or implausible homography; no automatic patch'
            entry['seconds']=time.perf_counter()-start
            self.progress(f'[scan {index+1}/{len(paths)}] {path.name}: {entry["status"]}; {entry.get("inliers",0)} inliers; {entry.get("local_discrepancy_tiles",0)} suspect tiles')
        self.check_pairs()
        return self.entries

    def local_discrepancies(self,frame):
        H=self.S@frame['H']@np.linalg.inv(frame['S']);corners=quad(frame['gray'].shape,H)
        mh,mw=self.gray.shape;l,t=np.maximum(np.floor(corners.min(0)),0).astype(int);r,b=np.minimum(np.ceil(corners.max(0))+1,[mw,mh]).astype(int)
        if r-l<32 or b-t<32:return []
        T=np.array([[1,0,-l],[0,1,-t],[0,0,1.]])
        warped=cv2.warpPerspective(frame['gray'],T@H,(r-l,b-t))
        source_mask=np.full(frame['gray'].shape,255,np.uint8) if frame['alpha'] is None else frame['alpha'].copy()
        source_mask[:8]=source_mask[-8:]=0;source_mask[:,:8]=source_mask[:,-8:]=0
        valid=cv2.warpPerspective(source_mask,T@H,(r-l,b-t),flags=cv2.INTER_NEAREST)&self.alpha[t:b,l:r]
        pts=cv2.goodFeaturesToTrack(warped,maxCorners=2200,qualityLevel=.02,minDistance=5,mask=valid)
        if pts is None or len(pts)<12:return []
        target=self.gray[t:b,l:r]
        forward,status,_=cv2.calcOpticalFlowPyrLK(warped,target,pts,None,winSize=(21,21),maxLevel=3)
        if forward is None:return []
        backward,back_status,_=cv2.calcOpticalFlowPyrLK(target,warped,forward,None,winSize=(21,21),maxLevel=3)
        if backward is None:return []
        xy=pts[:,0];q=forward[:,0];fb=np.linalg.norm(backward[:,0]-xy,axis=1)
        keep=status.ravel().astype(bool)&back_status.ravel().astype(bool)&(fb<.65)
        keep&=np.all(q>=6,axis=1)&(q[:,0]<r-l-6)&(q[:,1]<b-t-6)
        # NCC rejects lighting-only differences and unstable optical-flow tracks.
        groups={};tile=self.cfg.tile_px
        for i in np.flatnonzero(keep):
            a=cv2.getRectSubPix(warped,(11,11),tuple(xy[i])).astype(float);z=cv2.getRectSubPix(target,(11,11),tuple(q[i])).astype(float)
            a-=a.mean();z-=z.mean();den=np.linalg.norm(a)*np.linalg.norm(z)
            if den<50 or np.sum(a*z)/den<.82:continue
            position=xy[i]+[l,t];shift=(q[i]-xy[i])/np.array([self.S[0,0],self.S[1,1]])
            gx,gy=(position//tile).astype(int);groups.setdefault((gx,gy),[]).append((position,shift))
        result=[];hull=cv2.convexHull(frame['fit']['dst'].astype(np.float32))
        for (gx,gy),items in groups.items():
            if len(items)<6:continue
            positions=np.array([p for p,s in items]);shifts=np.array([s for p,s in items]);median=np.median(shifts,axis=0)
            magnitude=float(np.linalg.norm(median));scatter=float(np.median(np.linalg.norm(shifts-median,axis=1)))
            center=np.array([(gx+.5)*tile,(gy+.5)*tile])
            area=cv2.contourArea(cv2.convexHull(positions.astype(np.float32)))/(tile*tile)
            if not self.cfg.max_reprojection_px<magnitude<=self.cfg.max_local_shift_px or scatter>max(1.5,magnitude*.4) or area<.12:continue
            if cv2.pointPolygonTest(hull,tuple(center),False)<0:continue
            result.append(dict(x=gx,y=gy,tracks=len(items),shift_px=magnitude,scatter_px=scatter,delta_native=median.tolist()))
        return result

    def check_pairs(self):
        frames=self.frames;candidates=set()
        for i,a in enumerate(frames):
            neighbors=[]
            for j,b in enumerate(frames):
                if i==j:continue
                area,_=cv2.intersectConvexConvex(a['footprint'].astype(np.float32),b['footprint'].astype(np.float32))
                if area>200:neighbors.append((np.linalg.norm(a['footprint'].mean(0)-b['footprint'].mean(0)),j))
            for _,j in sorted(neighbors)[:self.cfg.pair_neighbors]:candidates.add(tuple(sorted((i,j))))
        for index,(i,j) in enumerate(sorted(candidates)):
            a,b=frames[i],frames[j]
            direct=fit_alignment(a['points'],a['descriptors'],b['points'],b['descriptors'],a['gray'].shape,self.cfg,
                float(np.sqrt(np.linalg.det(b['S'][:2,:2]))))
            record=dict(photo_a=str(a['path']),photo_b=str(b['path']),status='UNCONFIRMED')
            if direct and direct['reliable']:
                pa=project(a['H']@np.linalg.inv(a['S']),direct['src']);pb=project(b['H']@np.linalg.inv(b['S']),direct['dst'])
                error=np.linalg.norm(pa-pb,axis=1);median=float(np.median(error));p90=float(np.quantile(error,.9))
                agrees=median<=self.cfg.max_pairwise_px and p90<=2*self.cfg.max_pairwise_px
                record.update(status='CONSISTENT' if agrees else 'DISAGREEMENT',median_map_px=median,p90_map_px=p90,direct_inliers=direct['inliers'])
                if agrees:
                    a['neighbors'].append(j);b['neighbors'].append(i)
                    bridge=b['H']@np.linalg.inv(b['S'])@direct['H']@a['S']@np.linalg.inv(a['H'])
                    a['witnesses'][j]=(bridge,cv2.convexHull(pa.astype(np.float32)))
                    b['witnesses'][i]=(np.linalg.inv(bridge),cv2.convexHull(pb.astype(np.float32)))
                else:
                    for frame in (a,b):
                        frame['entry']['status']='FLAGGED'
                        frame['entry']['pairwise_disagreement']=True
            self.pairs.append(record)
            if (index+1)%10==0 or index+1==len(candidates):self.progress(f'[cross-check] {index+1}/{len(candidates)} overlapping source pairs')

    def repair(self):
        tile=self.cfg.tile_px;mh,mw=self.gray.shape;gh,gw=math.ceil(mh/tile),math.ceil(mw/tile)
        owner=np.full((gh,gw),-1,np.int32);score=np.zeros((gh,gw),np.float32);accepted=[]
        for i,frame in enumerate(self.frames):
            fit=frame['fit'];sharp=float(cv2.Laplacian(frame['gray'],cv2.CV_32F).var())
            quality=fit['inlier_ratio']*min(1,fit['inliers']/100)*max(fit['support_fraction'],.001)**.25*math.log1p(sharp)
            for cell in frame['tiles']:
                x,y=cell['x'],cell['y'];center=project(np.linalg.inv(self.S),[[(x+.5)*tile,(y+.5)*tile]])[0]
                witnesses=0
                for bridge,hull in frame['witnesses'].values():
                    if cv2.pointPolygonTest(hull,tuple(center),False)>=0 and np.linalg.norm(project(bridge,[center])[0]-center)<=self.cfg.max_pairwise_px:
                        witnesses+=1
                if witnesses<self.cfg.min_confirming_neighbors:continue
                value=quality*min(1,cell['tracks']/15)/(1+cell['scatter_px'])
                if value>score[y,x]:score[y,x]=value;owner[y,x]=i
        changed=np.zeros((gh,gw),np.uint16);patches=[]
        for i in np.unique(owner[owner>=0]):
            frame=self.frames[i]
            count=int(np.count_nonzero(owner==i));frame['entry']['patched_tiles']=count
            frame['entry']['status']='PATCHED';changed[owner==i]=i+1
            patches.append(dict(file=str(frame['path']),source_index=int(i+1),tiles=count,H=frame['H'].tolist()))
        result,accepted=render_repairs(self.image,changed,patches,(mh,mw),tile,self.cfg,self.progress)
        np.testing.assert_array_equal(result[:,:,3],self.image[:,:,3])
        return result,dict(photos=self.entries,pairwise=self.pairs,patches=accepted,
            source_index_to_file={str(i+1):str(frame['path']) for i,frame in enumerate(self.frames)},
            grid_shape=[gw,gh],analysis_shape=[mw,mh],analysis_pixel_transform=self.S.tolist(),tile_analysis_pixels=tile,
            repaired_tiles=int(np.count_nonzero(owner>=0)),method='Validated map homographies + independent source-pair checks + local LK/NCC discrepancy tiles; one source per tile, intact-context color/detail matching and two-band smooth perimeter blending',
            absolute_accuracy_verified=False,unresolved_policy='Weak fits, unsupported local changes and GPS-only offsets are reported without auto-patching'),changed
