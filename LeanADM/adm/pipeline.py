from .io import write_json
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import cv2
from scipy.optimize import least_squares
from .config import Config,CAMERA_SPEC
from .metadata import read_metadata
from .selection import select_frames
from .geometry import neighbor_pairs,gps_center_neighbors,project
from .tiles import build_index
from .registration import collect_matches,solve
from .composite import compose
from .lighting import render_lighting

def align_to_geography(hs,metadata,geo,cfg,out):
    """Fit a robust similarity between the visual mosaic and geographic priors."""
    source=[]; target=[]
    for H,G,m in zip(hs,geo,metadata):
        corners=np.array([[0,0],[m.width-1,0],[m.width-1,m.height-1],[0,m.height-1]])
        source.extend(project(H,corners)); target.extend(project(G,corners))
    source=np.array(source); target=np.array(target); source[:,1]*=-1
    x,y=source.T
    A=np.zeros((len(x)*2,4)); A[0::2]=np.c_[x,-y,np.ones(len(x)),np.zeros(len(x))]; A[1::2]=np.c_[y,x,np.zeros(len(x)),np.ones(len(x))]
    initial=np.linalg.lstsq(A,target.ravel(),rcond=None)[0]
    fit=least_squares(lambda p:A@p-target.ravel(),initial,loss='soft_l1',f_scale=cfg.gps_tolerance_m)
    a,b,tx,ty=fit.x; transform=np.array([[a,b,tx],[b,-a,ty],[0,0,1.]])
    error=np.linalg.norm((A@fit.x).reshape(-1,2)-target,axis=1)
    report={'median_footprint_prior_residual_m':float(np.median(error)),'p95_footprint_prior_residual_m':float(np.percentile(error,95)),'survey_accuracy_verified':False,'assumed_altitude_m':cfg.altitude_agl_m,'method':'visual planar registration aligned to nominal camera footprints'}
    write_json(out/'geographic_alignment.json',report)
    if np.median(error)>cfg.gps_tolerance_m: raise ValueError('Visual mosaic contradicts camera footprint geometry; correct altitude / yaw / calibration')
    return [transform@H for H in hs]

def align_centers_to_geography(hs,metadata,earth,cfg,out):
    """GPS supplies world orientation and scale; image matches supply relative pose."""
    if len(hs)<3: raise ValueError('GPS-center alignment needs at least three images')
    if any(abs(m.pitch_deg+90)>5 for m in metadata): raise ValueError('GPS-center alignment currently requires near-nadir imagery')
    source=np.array([project(H,[[(m.width-1)/2,(m.height-1)/2]])[0] for H,m in zip(hs,metadata)])
    source[:,1]*=-1
    target=np.array([earth.xy(m.latitude,m.longitude) for m in metadata])
    if np.linalg.norm(np.ptp(target,axis=0))<3: raise ValueError('GPS camera positions have insufficient separation to establish scale and heading')
    x,y=source.T; A=np.zeros((len(x)*2,4))
    A[0::2]=np.c_[x,-y,np.ones(len(x)),np.zeros(len(x))]
    A[1::2]=np.c_[y,x,np.zeros(len(x)),np.ones(len(x))]
    def fit(mask):
        aa=A[np.repeat(mask,2)]; bb=target[mask].ravel()
        initial=np.linalg.lstsq(aa,bb,rcond=None)[0]
        return least_squares(lambda p:aa@p-bb,initial,loss='soft_l1',f_scale=.75).x
    parameters=fit(np.ones(len(x),bool)); a,b,tx,ty=parameters
    transform=np.array([[a,b,tx],[b,-a,ty],[0,0,1.]])
    errors=np.linalg.norm((A@parameters).reshape(-1,2)-target,axis=1)
    heldout=[]
    if cfg.export_profile=='diagnostic':
        for i in range(len(x)):
            mask=np.arange(len(x))!=i; prediction=A[2*i:2*i+2]@fit(mask)
            heldout.append(float(np.linalg.norm(prediction-target[i])))
    if np.median(errors)>cfg.gps_tolerance_m: raise ValueError('GPS camera positions contradict the visual mosaic')
    corrected=[transform@H for H in hs]; headings=[]
    for H,m,error in zip(corrected,metadata,errors) if cfg.export_profile=='diagnostic' else []:
        cx,cy=(m.width-1)/2,(m.height-1)/2
        top,centre=project(H,[[cx,cy-100],[cx,cy]])
        delta=top-centre; yaw=float(np.rad2deg(np.arctan2(delta[0],delta[1]))%360)
        disagreement=None if m.yaw_deg is None else float((yaw-m.yaw_deg+180)%360-180)
        headings.append({'name':m.name,'recorded_yaw_deg':m.yaw_deg,'image_and_gps_estimated_yaw_deg':yaw,'disagreement_deg':disagreement,'gps_center_residual_m':float(error)})
    report={'method':'visual planar registration aligned to GPS camera centers; recorded gimbal yaw excluded','median_center_residual_m':float(np.median(errors)),'p95_center_residual_m':float(np.percentile(errors,95)),'leave_one_out_median_m':float(np.median(heldout)) if heldout else None,'leave_one_out_p95_m':float(np.percentile(heldout,95)) if heldout else None,'survey_accuracy_verified':False,'independent_pixel_ground_truth':False,'nominal_altitude_for_neighbor_search_m':cfg.altitude_agl_m,'note':'Scale and north orientation come from GPS center motion; near-nadir flat-ground assumption. Tile coordinates share the fitted image geometry.'}
    write_json(out/'geographic_alignment.json',report)
    if headings:write_json(out/'heading_validation.json',headings)
    print(json.dumps(report,indent=2),flush=True)
    return corrected

def run(args):
    profile=getattr(args,'export_profile',None)
    if profile is None:
        return _run(args)
    from .storage import output_budget
    limit=getattr(args,'max_output_mb',None) or (32 if profile=='vectors' else 256)
    with output_budget(args.output,limit*1024**2):
        return _run(args)


def _run(args):
    started=time.perf_counter(); cfg=Config()
    if args.config:
        overrides=json.loads(args.config.read_text())
        cfg=Config(**{**cfg.to_dict(),**overrides})
    if args.altitude is not None: cfg.altitude_agl_m=args.altitude
    if args.max_dim is not None: cfg.max_output_dimension=args.max_dim
    if args.strict_corners: cfg.visual_fallback=False
    if getattr(args,'export_profile',None) is not None:cfg.export_profile=args.export_profile
    if cfg.export_profile!='diagnostic':
        cfg.compact_outputs=True;cfg.metal_recompute_warps=True
    if getattr(args,'gpu_optimized',False):
        if args.backend!='metal':raise ValueError('--gpu-optimized requires --backend metal')
        cfg.gpu_alignment=True;cfg.coherent_composite=True;cfg.compact_outputs=True;cfg.metal_recompute_warps=True
    if getattr(args,'stylized_shadows',False):
        if args.backend!='metal':raise ValueError('--stylized-shadows requires --backend metal')
        cfg.stylized_shadows=True
    if getattr(args,'lasso_crop',False): cfg.lasso_crop=True
    if getattr(args,'recompute_warps',False):
        if args.backend!='metal': raise ValueError('--recompute-warps requires --backend metal')
        cfg.metal_recompute_warps=True
    if args.backend!='metal' and (cfg.gpu_alignment or cfg.coherent_composite or cfg.stylized_shadows or cfg.lasso_crop):
        raise ValueError('GPU alignment, coherent compositing and stylized shadows require --backend metal')
    cfg.validate(); cv2.setNumThreads(6); cv2.setRNGSeed(cfg.seed)
    gpu=None
    if args.backend=='metal':
        from .metal import get_backend
        gpu=get_backend(); gpu_start=gpu.stats(); print('Metal + MPS:',gpu.device,flush=True)
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True); source=args.source.resolve()
    exclusions=json.loads(args.exclusions.read_text()) if args.exclusions else {}
    names,selection=select_frames(source,cfg,exclusions)
    write_json(out/'selection.json',selection)
    overrides=json.loads(args.metadata.read_text()) if args.metadata else {}; metadata=[]; audit=[]
    for row in selection:
        m,raw=read_metadata(source/row['name'],cfg,overrides.get(row['name']))
        audit.append({**m.to_dict(),'raw':raw})
        if row['name'] in names: metadata.append(m)
    write_json(out/'metadata.json',audit)
    write_json(out/'resolved_config.json',{**cfg.to_dict(),'camera_spec':CAMERA_SPEC})
    gps_centers=getattr(args,'gps_centers',False)
    missing=[m.name for m in metadata if (m.latitude is None or m.longitude is None) or (not gps_centers and not m.geographic)]; geo=None; earth=None; index=None
    if missing:
        if not args.allow_local: raise ValueError(f'{len(missing)} retained frames lack GPS or camera yaw. Supply --metadata sidecar or explicitly use --allow-local.')
        pairs=[(i,j) for i in range(len(names)) for j in range(i+1,min(len(names),i+cfg.sequence_neighbors+1))]
        print('LOCAL MODE: geographic hash validation unavailable',flush=True)
    elif gps_centers:
        pairs,earth=gps_center_neighbors(metadata,cfg)
        print('GPS CENTER MODE: image rotation estimated visually; raw gimbal yaw not used for placement',flush=True)
    else:
        pairs,geo,earth=neighbor_pairs(metadata,cfg)
        if cfg.compact_outputs:
            from .vector_index import FootprintIndex
            index=FootprintIndex(source,names,metadata,geo,earth)
            if cfg.export_profile!='png':index.save(out)
        else:index=build_index(source,names,geo,cfg,out,earth,gpu=gpu)
    prepared=time.perf_counter()
    save_cache=bool(args.resume or getattr(args,'cache_matches',False) or cfg.export_profile=='diagnostic')
    fingerprint=hashlib.sha256(json.dumps({'selection':selection,'metadata':audit,'config':cfg.to_dict(),'gps_center_alignment':gps_centers,'matching_backend':gpu.fingerprint if gpu else 'cpu-flann-v1','registration_implementation':hashlib.sha256(Path(__file__).with_name('registration.py').read_bytes()).hexdigest(),'opencv':cv2.__version__},sort_keys=True).encode()).hexdigest() if save_cache else None
    cached=out/'cache_fingerprint.txt'
    if args.resume and cached.exists() and cached.read_text()==fingerprint and (out/'registration.npz').exists():
        with np.load(out/'registration.npz',allow_pickle=True) as data:edges=data['edges'].tolist()
        match_report=json.loads((out/'matching_report.json').read_text())
        match_report['cache_reused']=True
    else:
        edges=collect_matches(source,names,metadata,pairs,cfg,out,geo,index,gpu=gpu,save_cache=save_cache)
        if save_cache:cached.write_text(fingerprint)
        match_report=json.loads((out/'matching_report.json').read_text());match_report['cache_reused']=False
    registered=time.perf_counter()
    if len(names)==1:
        fitted={0:np.eye(3)}
        write_json(out/'registration_report.json',{'images':1,'edges':0,'solver_success':True,'note':'single image; no alignment'})
    else: fitted=solve(names,edges,out,cfg.registration_width,gpu=gpu if cfg.gpu_alignment else None)
    native=[]
    for i,m in enumerate(metadata):
        sh=round(m.height*cfg.registration_width/m.width)
        native.append(fitted[i]@np.diag([cfg.registration_width/m.width,sh/m.height,1.]))
    if geo is not None: native=align_to_geography(native,metadata,geo,cfg,out)
    elif gps_centers and earth is not None: native=align_centers_to_geography(native,metadata,earth,cfg,out)
    solved=time.perf_counter()
    rgba,map_hs,geometry=compose(source,names,metadata,native,cfg,out,earth is not None,gpu=gpu)
    if earth:
        geometry['origin_latlon']=[earth.lat,earth.lon]
        write_json(out/'map_geometry.json',geometry)
        if gps_centers and cfg.export_profile!='png':
            if cfg.compact_outputs:
                from .vector_index import build_index as build_vectors
                index=build_vectors(source,names,metadata,native,out,earth)
            else:index=build_index(source,names,native,cfg,out,earth,independent_prior=False,gpu=gpu)
    elif cfg.export_profile!='png':
        if cfg.compact_outputs:
            from .vector_index import build_index as build_vectors
            index=build_vectors(source,names,metadata,map_hs,out)
        else:index=build_index(source,names,map_hs,cfg,out,gpu=gpu)
    if index: index.close()
    if cfg.export_profile=='vectors':
        from .project import write_recipe
        write_recipe(source,names,map_hs,geometry,cfg,out)
    composited=time.perf_counter()
    spacing=args.heightmap_spacing if args.heightmap_spacing is not None else geometry['metres_per_pixel']
    if cfg.export_profile in ('vectors','png'):
        light={'mode':'lighting deferred in project.json' if cfg.export_profile=='vectors' else 'neutral map only; lighting not requested by export profile','exports':[]}
    else:light=render_lighting(rgba,cfg,out,args.heightmap,spacing,args.backend,gpu=gpu)
    ended=time.perf_counter()
    summary={'input_images':len(selection),'retained_images':len(names),'excluded_images':len(selection)-len(names),'georeferenced':earth is not None,'geographic_alignment_mode':'GPS camera centers' if gps_centers and earth is not None else 'full camera pose' if geo is not None else 'local','candidate_pairs':len(pairs),'registered_pairs':len(edges),'corner_patches_evaluated':match_report['corner_patches_evaluated'],'corner_patches_accepted':match_report['corner_patches_accepted'],'visual_fallback_edges':sum(e.get('method')=='SIFT fallback' for e in edges),'timings_seconds':{'metadata_and_geographic_index':prepared-started,'pair_matching':registered-prepared,'global_alignment':solved-registered,'composite_and_local_index':composited-solved,'lighting':ended-composited,'total':ended-started},'lighting':light,'output_dimensions':[geometry['width'],geometry['height']], 'matching':match_report,'export_profile':cfg.export_profile}
    if gpu:
        summary['acceleration']=gpu.stats()
        if cfg.gpu_alignment:
            summary['acceleration']['cpu_stages']=['SIFT extraction and RANSAC match verification','metadata, selection, topology packing and double-precision geographic fit','image codecs and vector metadata export']
        for key in ('gpu_seconds','command_submissions','msl_dispatches','mps_operations'):
            summary['acceleration'][key]-=gpu_start[key]
        write_json(out/'gpu_report.json',summary['acceleration'])
    write_json(out/'report.json',summary); print(json.dumps(summary,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description='Aerial Delta Mapping: geospatial-prior and visual mosaic prototype')
    p.add_argument('source',type=Path); p.add_argument('--output',type=Path,default=Path('output/project'))
    p.add_argument('--export-profile',choices=['vectors','png','diagnostic'],default='png',help='Default png: neutral map and geometry; vectors: defer rendering; diagnostic: inspection images, lighting and match cache')
    p.add_argument('--max-output-mb',type=int,help='Hard output cap; defaults to 32 MiB for vector projects or 256 MiB for raster exports')
    p.add_argument('--metadata',type=Path,help='JSON mapping filenames to canonical metadata fields')
    p.add_argument('--config',type=Path); p.add_argument('--exclusions',type=Path)
    p.add_argument('--allow-local',action='store_true',help='Explicitly permit missing geographic metadata')
    p.add_argument('--gps-centers',action='store_true',help='Use GPS positions to orient the visual mosaic; exclude unreliable recorded yaw (near-nadir imagery only)')
    p.add_argument('--strict-corners',action='store_true',help='Disable SIFT fallback to measure the four-corner hypothesis')
    p.add_argument('--resume',action='store_true',help='Reuse matches only if image and config fingerprint agrees')
    p.add_argument('--cache-matches',action='store_true',help='Save matches for subsequent --resume runs; otherwise omitted outside diagnostic exports')
    p.add_argument('--recompute-warps',action='store_true',help='Metal: recompute warps in the selection pass to avoid intermediate disk files')
    p.add_argument('--gpu-optimized',action='store_true',help='GPU projective alignment, coherent source regions and compact vector outputs; keeps feature matching')
    p.add_argument('--stylized-shadows',action='store_true',help='Additional artistic screen-space shadow PNG; does not modify the neutral map')
    p.add_argument('--lasso-crop',action='store_true',help='GPU inward polygon crop to trim jagged perimeter and black edge strips')
    p.add_argument('--altitude',type=float); p.add_argument('--max-dim',type=int)
    p.add_argument('--heightmap',type=Path,help='Aligned finite .npy elevation array, metres, north-up')
    p.add_argument('--heightmap-spacing',type=float,help='Metres per height-map pixel, required for local mode')
    p.add_argument('--backend',choices=['cpu','metal'],default='cpu',help='Metal uses MPS and MSL across matching, compositing, tile packing, and lighting')
    run(p.parse_args())

if __name__=='__main__': main()
