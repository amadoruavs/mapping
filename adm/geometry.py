import numpy as np
from scipy.spatial import cKDTree

def project(H, points):
    p=np.asarray(points,dtype=np.float64); q=np.c_[p,np.ones(len(p))]@H.T
    if np.any(np.abs(q[:,2])<1e-10): raise ValueError('Projection intersects horizon')
    return q[:,:2]/q[:,2:]

class LocalEarth:
    """Local WGS84 tangent approximation, appropriate for a small drone site."""
    def __init__(self,lat,lon):
        self.lat,self.lon=float(lat),float(lon); phi=np.deg2rad(lat); e2=6.69437999014e-3
        den=np.sqrt(1-e2*np.sin(phi)**2); self.n=6378137/den; self.m=6378137*(1-e2)/den**3
    def xy(self,lat,lon):
        dlon=(np.asarray(lon)-self.lon+180)%360-180
        return np.stack([np.deg2rad(dlon)*self.n*np.cos(np.deg2rad(self.lat)),np.deg2rad(np.asarray(lat)-self.lat)*self.m],axis=-1)
    def latlon(self,xy):
        p=np.asarray(xy); lat=self.lat+np.rad2deg(p[...,1]/self.m); lon=self.lon+np.rad2deg(p[...,0]/(self.n*np.cos(np.deg2rad(self.lat))))
        return np.stack([lat,(lon+180)%360-180],axis=-1)

def intrinsics(meta,cfg):
    fx=meta.width/(2*np.tan(np.deg2rad(cfg.hfov_deg)/2))*cfg.digital_zoom
    fy=fx if cfg.vfov_deg is None else meta.height/(2*np.tan(np.deg2rad(cfg.vfov_deg)/2))*cfg.digital_zoom
    return np.array([[fx,0,(meta.width-1)/2],[0,fy,(meta.height-1)/2],[0,0,1.]])

def camera_rotation(yaw,pitch,roll):
    """Camera right/down/forward to east/north/up. Yaw clockwise from true north."""
    y,p,r=np.deg2rad([yaw,pitch,roll])
    forward=np.array([np.sin(y)*np.cos(p),np.cos(y)*np.cos(p),np.sin(p)])
    right=np.array([np.cos(y),-np.sin(y),0.]); down=np.cross(forward,right)
    return np.column_stack([right*np.cos(r)+down*np.sin(r),-right*np.sin(r)+down*np.cos(r),forward])

def ground_homography(meta,cfg,earth):
    if not meta.geographic: raise ValueError('GPS and true camera yaw required')
    K=intrinsics(meta,cfg); rays=camera_rotation(meta.yaw_deg,meta.pitch_deg,meta.roll_deg)@np.linalg.inv(K)
    centre=earth.xy(meta.latitude,meta.longitude); h=meta.altitude_agl_m
    if meta.gps_reference=='image_center':
        central=rays@np.array([(meta.width-1)/2,(meta.height-1)/2,1.])
        if central[2]>=-1e-4: raise ValueError('Camera centre ray points above horizon')
        centre=centre+h*central[:2]/central[2]
    H=np.vstack([centre[0]*rays[2]-h*rays[0],centre[1]*rays[2]-h*rays[1],rays[2]])
    corners=np.array([[0,0],[meta.width-1,0],[meta.width-1,meta.height-1],[0,meta.height-1]])
    if np.any((np.c_[corners,np.ones(4)]@rays[2])>=-1e-4): raise ValueError(f'{meta.name}: footprint crosses horizon')
    return H/H[2,2]

def neighbor_pairs(metadata,cfg):
    """Footprint candidates from a spatial index; no all-pairs geographic scan."""
    earth=LocalEarth(metadata[0].latitude,metadata[0].longitude)
    hs=[ground_homography(m,cfg,earth) for m in metadata]
    footprints=[project(H,[[0,0],[m.width-1,0],[m.width-1,m.height-1],[0,m.height-1]]) for H,m in zip(hs,metadata)]
    centers=np.array([p.mean(0) for p in footprints]); radii=np.array([np.linalg.norm(p-c,axis=1).max() for p,c in zip(footprints,centers)])
    tree=cKDTree(centers); pairs=[]
    for i,p in enumerate(footprints):
        for j in tree.query_ball_point(centers[i],radii[i]+radii.max()+cfg.gps_tolerance_m):
            if j<=i: continue
            q=footprints[j]; tol=cfg.gps_tolerance_m
            if np.all(p.max(0)+tol>=q.min(0)) and np.all(q.max(0)+tol>=p.min(0)): pairs.append((i,j))
    return pairs,hs,earth

def gps_center_neighbors(metadata,cfg):
    """Conservative circular footprints when absolute camera heading is unreliable."""
    earth=LocalEarth(metadata[0].latitude,metadata[0].longitude)
    centers=np.array([earth.xy(m.latitude,m.longitude) for m in metadata])
    radii=[]
    for m in metadata:
        K=intrinsics(m,cfg)
        radii.append(m.altitude_agl_m*np.hypot(m.width/(2*K[0,0]),m.height/(2*K[1,1])))
    radii=np.array(radii); tree=cKDTree(centers)
    pairs=[(i,j) for i,c in enumerate(centers) for j in tree.query_ball_point(c,radii[i]+radii.max()+cfg.gps_tolerance_m) if j>i and np.linalg.norm(c-centers[j])<=radii[i]+radii[j]+cfg.gps_tolerance_m]
    return sorted(pairs),earth
