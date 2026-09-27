"""Repair disconnected visual match graphs with bounded, higher-resolution retries."""
from collections import OrderedDict
import time
import cv2
import numpy as np
from .geometry import project
from .io import write_json


def valid_transform(matrix,width,height,target_height):
    """Reject singular fits and transforms whose horizon crosses either image."""
    if matrix is None or not np.isfinite(matrix).all():return False
    try:inverse=np.linalg.inv(matrix)
    except np.linalg.LinAlgError:return False
    for H,h in ((matrix,height),(inverse,target_height)):
        corners=np.array([[0,0,1],[width,0,1],[width,h,1],[0,h,1]],np.float64)
        q=corners@H.T;z=q[:,2]
        if not (np.all(z>1e-10) or np.all(z< -1e-10)):return False
        points=q[:,:2]/z[:,None]
        area=abs(cv2.contourArea(points.astype(np.float32)))
        if not np.isfinite(area) or not .02*width*h<area<50*width*h:return False
    return True


def components(count, edges):
    parent=list(range(count))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for edge in edges:
        a,b=root(edge['i']),root(edge['j'])
        if a!=b:parent[b]=a
    groups={}
    for i in range(count):groups.setdefault(root(i),[]).append(i)
    return sorted(groups.values(),key=lambda g:(-len(g),g[0]))


def recover(source,names,metadata,pairs,edges,cfg,out,gpu=None,geo=None):
    """Only run on disconnected graphs; every added edge has real SIFT inliers.

    Retry cross-component neighbors at higher resolution, then express their
    coordinates in the original registration grid. Never synthesize GPS edges.
    """
    started=time.perf_counter();groups=components(len(names),edges)
    if len(groups)<=1:return edges,{'attempted':False,'connected':True,'seconds':0.}
    initial=[[names[i] for i in group] for group in groups]
    width=max(1280,cfg.registration_width)
    report=dict(attempted=True,connected=False,initial_components=initial,
        registration_width=cfg.registration_width,retry_width=width,
        sift_contrast_threshold=.01,
        attempts=[],added_edges=[],seconds=0.)
    def save():
        remaining=components(len(names),edges)
        report.update(connected=len(remaining)<=1,components=[[names[i] for i in g] for g in remaining],seconds=time.perf_counter()-started)
        write_json(out/'match_graph_report.json',report)
        return remaining
    save()
    print(f'Match recovery: {len(groups)} groups, sizes {[len(g) for g in groups]}; retry width {width}',flush=True)
    if not cfg.visual_fallback:return edges,report
    # Original geographic candidates plus short sequence links. Crossing GPS
    # search boundaries is allowed only when the photographs verify the link.
    candidates=set(tuple(sorted(pair)) for pair in pairs)
    candidates.update((i,j) for i in range(len(names)) for j in range(i+1,min(len(names),i+cfg.sequence_neighbors+1)))
    owners={i:k for k,g in enumerate(groups) for i in g}
    candidates=sorted((p for p in candidates if owners[p[0]]!=owners[p[1]]),
        key=lambda p:(abs(p[1]-p[0]),p[0],p[1]))
    # Faint grass texture is rejected by the normal .025 detector threshold.
    # Detect more candidates here; descriptor ratio/RANSAC acceptance stay strict.
    sift=cv2.SIFT_create(nfeatures=cfg.feature_count,contrastThreshold=.01)
    cache=OrderedDict()
    def features(i):
        if i in cache:cache.move_to_end(i);return cache[i]
        image=cv2.imread(str(source/names[i]))
        if image is None:raise ValueError(f'Cannot read {names[i]} during match recovery')
        h=round(metadata[i].height*width/metadata[i].width)
        small=cv2.resize(image,(width,h),interpolation=cv2.INTER_AREA)
        keys,desc=sift.detectAndCompute(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),None)
        result=(np.array([k.pt for k in keys],dtype=np.float64).reshape(-1,2),desc,
            None if gpu is None or desc is None else gpu.descriptors(desc))
        cache[i]=result
        if len(cache)>16:cache.popitem(last=False)
        return result
    for i,j in candidates:
        if len(report['attempts'])>=256:break
        if owners[i]==owners[j]:continue
        ka,da,ga=features(i);kb,db,gb=features(j)
        item=dict(source=names[i],target=names[j],features=[len(ka),len(kb)],accepted=False)
        report['attempts'].append(item)
        if da is None or db is None or min(len(da),len(db))<35:continue
        if gpu is not None:
            nearest=gpu.match_descriptors(ga,gb)
            query=np.flatnonzero(nearest[:,2]<.70**2*nearest[:,3]);target=nearest[query,0].astype(np.intp)
        else:
            matches=cv2.BFMatcher(cv2.NORM_L2).knnMatch(da,db,k=2)
            good=[v[0] for v in matches if len(v)==2 and v[0].distance<.70*v[1].distance]
            query=np.array([v.queryIdx for v in good],np.intp);target=np.array([v.trainIdx for v in good],np.intp)
        item['ratio_matches']=len(query)
        if len(query)<35:continue
        a,b=ka[query],kb[target]
        cv2.setRNGSeed((cfg.seed+i*1009+j*9176)&0x7fffffff)
        H,mask=cv2.findHomography(a,b,cv2.RANSAC,2.5)
        if H is None:continue
        valid=mask.ravel().astype(bool);a,b=a[valid],b[valid];item['inliers']=len(a)
        if len(a)<30 or len(a)/len(query)<.25:continue
        if any(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))<width*round(metadata[k].height*width/metadata[k].width)*.035 for points,k in ((a,i),(b,j))):continue
        # Per-axis conversion accounts for rounded resized image heights.
        converted=[]
        for points,k in ((a,i),(b,j)):
            m=metadata[k]
            converted.append(points*np.array([cfg.registration_width/width,
                round(m.height*cfg.registration_width/m.width)/round(m.height*width/m.width)]))
        a,b=converted
        if geo is not None:
            native=[]
            for points,k in ((a,i),(b,j)):
                m=metadata[k];native.append(points*np.array([m.width/cfg.registration_width,m.height/round(m.height*cfg.registration_width/m.width)]))
            if np.median(np.linalg.norm(project(geo[i],native[0])-project(geo[j],native[1]),axis=1))>cfg.gps_tolerance_m:continue
        H,_=cv2.findHomography(a,b,0)
        if not valid_transform(H,cfg.registration_width,
            round(metadata[i].height*cfg.registration_width/metadata[i].width),
            round(metadata[j].height*cfg.registration_width/metadata[j].width)):continue
        error=np.linalg.norm(project(H,a)-b,axis=1)
        if not np.isfinite(error).all() or np.median(error)>2.5:continue
        ix=np.linspace(0,len(a)-1,min(160,len(a))).astype(int)
        edges.append(dict(i=i,j=j,H=H,a=a[ix],b=b[ix],n=len(a),median=float(np.median(error)),method='SIFT recovery',matching_width=width))
        item['accepted']=True;report['added_edges'].append(dict(source=names[i],target=names[j],inliers=len(a),median_registration_px=float(np.median(error))))
        groups=save();owners={k:g for g,group in enumerate(groups) for k in group}
        print(f'Match recovery: {names[i]} -> {names[j]}, {len(a)} inliers; {len(groups)} groups remain',flush=True)
        if len(groups)<=1:break
    save()
    return edges,report


def require_connected(names,edges,out):
    groups=components(len(names),edges)
    if len(groups)<=1:return
    path=out/'match_graph_report.json'
    if not path.exists():write_json(path,dict(connected=False,components=[[names[i] for i in g] for g in groups]))
    examples=', '.join(names[g[0]] for g in groups[:6])
    raise RuntimeError(f'Disconnected visual match graph: {len(groups)} groups remain (sizes {[len(g) for g in groups]}; examples: {examples}). See {path} for exact photos and recovery attempts. No frames were dropped.')
