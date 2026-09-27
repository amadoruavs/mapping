#!/usr/bin/env python3
"""ISM with a live, saved gallery of the actual processing stages."""
from pathlib import Path

# --------------------- EDIT THESE VALUES ---------------------
VISUAL_PROCESS = True
INPUT_REPO = Path('/Users/avneh/Downloads/killerofmaps')
OUTPUT_DESTINATION = Path(__file__).resolve().parent / 'output_visual'
PREVIEW_WIDTH = 1400
DETAIL_ROI = (1470, 2700, 570, 385)  # Native map pixels; None = center crop.
# ------------------------------------------------------------

import argparse
import datetime
import html
import json
import resource
import subprocess
import sys
import time
import run as base


class VisualProcess:
    def __init__(self, output):
        import cv2
        import numpy as np
        self.cv2=cv2;self.np=np;self.folder=Path(output)/'visual_process';self.folder.mkdir(parents=True,exist_ok=True)
        self.entries=[];self.seen=set();self.started=time.perf_counter();self.preview_seconds=0.;self.native_shape=None;self.restores=[]
        self.refresh('running')

    def refresh(self,status):
        rows='\n'.join(f'<section><h2>{html.escape(e["stage"])}</h2><p>{html.escape(e["description"])}</p>'
            f'<img src="{e["image"]}" loading="lazy">'+(f'<h3>Same map detail</h3><img src="{e["detail"]}" loading="lazy">' if e.get('detail') else '')+'</section>' for e in self.entries)
        page=f'''<!doctype html><html><head><meta charset="utf-8"><title>ISM process</title>
        {'<meta http-equiv="refresh" content="4">' if status=='running' else ''}
        <style>body{{font:16px system-ui;background:#eff1f2;color:#192127;max-width:1400px;margin:32px auto;padding:0 20px}}section{{background:white;padding:20px;margin:24px 0}}img{{max-width:100%;height:auto}}p{{line-height:1.5;color:#47535e}}</style></head>
        <body><h1>ISM · {html.escape(status)}</h1><p>Actual intermediate outputs. Representative photos and the first FSR frame show repeated operations. Full-map previews retain the same coordinates.</p>{rows}</body></html>'''
        temporary=self.folder/'process.tmp.html';temporary.write_text(page);temporary.replace(self.folder/'process.html')
        report={'status':status,'preview_seconds':self.preview_seconds,'elapsed_seconds':time.perf_counter()-self.started,'stages':self.entries,
            'scope':'Every major pipeline stage; first frame of each FSR shader pass; first three generated views. Previews are not full-resolution caches.',
            'timing_note':'Visual work adds synchronization, decoding and PNG export; this is not the base performance benchmark.'}
        (self.folder/'stages.json').write_text(json.dumps(report,indent=2)+'\n')

    def image(self,name,array,description='',detail=False):
        if name in self.seen:return
        self.seen.add(name);tick=time.perf_counter();np=self.np;cv2=self.cv2
        if hasattr(array,'numpy'):array=array.numpy(False)
        if array is None:return
        if array.ndim==2:array=cv2.cvtColor(array,cv2.COLOR_GRAY2BGR)
        h,w=array.shape[:2];scale=min(1,PREVIEW_WIDTH/w)
        preview=cv2.resize(array,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
        filename=f'{len(self.entries):02d}_stage.png'
        if not cv2.imwrite(str(self.folder/filename),preview):raise OSError('Could not save visual stage')
        entry={'stage':name,'description':description,'image':filename,'source_dimensions':[w,h]}
        if detail and self.native_shape:
            nh,nw=self.native_shape;sx=w/nw;sy=h/nh
            roi=DETAIL_ROI
            if roi is None or roi[0]+roi[2]>nw or roi[1]+roi[3]>nh:
                cw,ch=min(570,nw),min(385,nh);roi=((nw-cw)//2,(nh-ch)//2,cw,ch)
            x,y,cw,ch=roi;crop=array[round(y*sy):round((y+ch)*sy),round(x*sx):round((x+cw)*sx)]
            crop=cv2.resize(crop,(cw,ch),interpolation=cv2.INTER_AREA)
            entry['detail']=filename.replace('.png','_detail.png')
            if not cv2.imwrite(str(self.folder/entry['detail']),crop):raise OSError('Could not save visual detail')
        self.preview_seconds+=time.perf_counter()-tick;self.entries.append(entry);self.refresh('running')
        print(f'Visual step: {name}',flush=True)

    def contact(self,name,images,labels,description=''):
        cv2=self.cv2;np=self.np;cellw,cellh=360,240;columns=min(3,len(images))
        if not columns:return
        canvas=np.full((((len(images)+columns-1)//columns)*(cellh+30),columns*cellw,3),246,np.uint8)
        for i,(image,label) in enumerate(zip(images,labels)):
            if image is None:continue
            image=image[:,:,:3] if image.ndim==3 else cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            s=min(cellw/image.shape[1],cellh/image.shape[0]);small=cv2.resize(image,(round(image.shape[1]*s),round(image.shape[0]*s)),interpolation=cv2.INTER_AREA)
            x=(i%columns)*cellw;y=(i//columns)*(cellh+30)
            canvas[y+30:y+30+small.shape[0],x:x+small.shape[1]]=small
            cv2.putText(canvas,str(label)[:42],(x+7,y+21),cv2.FONT_HERSHEY_SIMPLEX,.48,(35,40,45),1,cv2.LINE_AA)
        self.image(name,canvas,description)

    def patch(self,obj,name,replacement):
        self.restores.append((obj,name,getattr(obj,name)));setattr(obj,name,replacement)

    def install(self):
        import adm.pipeline as pipeline
        import adm.presentation as presentation
        import adm.raytracing as raytracing
        import adm.lasso as lasso
        from adm.presentation_metal import PresentationMetal
        from adm.fsr_framegen import FrameGenerator
        from adm.geometry import project
        cv2=self.cv2;np=self.np;record=self
        select=pipeline.select_frames
        def selected(source,cfg,exclusions):
            files=sorted(p.name for p in source.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff'))
            sample=[files[i] for i in np.linspace(0,len(files)-1,min(9,len(files))).astype(int)] if files else []
            record.contact('Input photographs',[cv2.imread(str(source/name)) for name in sample],sample,f'{len(files)} input files; representative examples.')
            names,rows=select(source,cfg,exclusions)
            chosen=[names[i] for i in np.linspace(0,len(names)-1,min(9,len(names))).astype(int)]
            record.contact('Selected photographs',[cv2.imread(str(source/name)) for name in chosen],chosen,f'{len(names)} retained; {len(rows)-len(names)} excluded.')
            record.source=source;record.names=names;return names,rows
        self.patch(pipeline,'select_frames',selected)

        sift_create=cv2.SIFT_create
        class SIFTPreview:
            def __init__(self,detector):self.detector=detector
            def detectAndCompute(self,image,mask):
                keys,descriptors=self.detector.detectAndCompute(image,mask)
                if 'SIFT features' not in record.seen:
                    view=cv2.drawKeypoints(image,list(keys[:250]),None,color=(40,190,245),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    record.image('SIFT features',view,f'{len(keys)} detected features; first 250 shown on the first registration image.')
                return keys,descriptors
            def __getattr__(self,name):return getattr(self.detector,name)
        self.patch(cv2,'SIFT_create',lambda *a,**k:SIFTPreview(sift_create(*a,**k)))

        matching=pipeline.collect_matches
        def matched(source,names,metadata,pairs,cfg,*args,**kwargs):
            edges=matching(source,names,metadata,pairs,cfg,*args,**kwargs)
            if edges:
                edge=edges[0];images=[]
                for index in (edge['i'],edge['j']):
                    im=cv2.imread(str(source/names[index]));images.append(cv2.resize(im,(cfg.registration_width,round(im.shape[0]*cfg.registration_width/im.shape[1]))))
                view=np.zeros((max(im.shape[0] for im in images),sum(im.shape[1] for im in images),3),np.uint8)
                view[:images[0].shape[0],:images[0].shape[1]]=images[0];view[:images[1].shape[0],images[0].shape[1]:]=images[1]
                for i,(a,b) in enumerate(zip(edge['a'][:70],edge['b'][:70])):
                    cv2.line(view,tuple(np.rint(a).astype(int)),tuple(np.rint(b+[images[0].shape[1],0]).astype(int)),(60+(i*43)%190,80+(i*71)%170,220),1,cv2.LINE_AA)
                record.image('Verified feature matches',view,f'{len(edges)} accepted image pairs from {len(pairs)} candidates. Actual inlier correspondences from the first accepted pair.')
            return edges
        self.patch(pipeline,'collect_matches',matched)

        solve=pipeline.solve
        def solved(names,edges,out,width,*args,**kwargs):
            hs=solve(names,edges,out,width,*args,**kwargs);polygons=[]
            for index,name in enumerate(names):
                im=cv2.imread(str(record.source/name));height=im.shape[0]*width/im.shape[1]
                polygons.append(project(hs[index],np.array([[0,0],[width,0],[width,height],[0,height]])))
            points=np.concatenate(polygons);lo=points.min(0);span=np.maximum(points.max(0)-lo,1);s=1100/span.max()
            view=np.full((round(span[1]*s)+60,round(span[0]*s)+60,3),247,np.uint8)
            for poly in polygons:cv2.polylines(view,[np.rint((poly-lo)*s+30).astype(np.int32)],True,(65,135,190),1,cv2.LINE_AA)
            record.image('Aligned image footprints',view,'Actual solved homographies before geographic orientation; outlines show overlapping registered images.')
            return hs
        self.patch(pipeline,'solve',solved)

        dispatch=FrameGenerator.dispatch
        resources={'reconstruct_and_dilate':'dilated_motion_vectors','setup':'disocclusion_mask',
            'reconstruct_previous_depth':'reconstructed_depth_interpolated_frame','game_motion_vector_field':'game_motion_vector_field_x',
            'compute_game_vector_field_inpainting_pyramid':'inpainting_pyramid','disocclusion_mask':'disocclusion_mask',
            'interpolation':'output','compute_inpainting_pyramid':'inpainting_pyramid','inpainting':'output'}
        def dispatched(engine,name):
            result=dispatch(engine,name);label='FSR · '+name.replace('_',' ')
            if engine.frames==0 and name in resources and label not in record.seen:
                a=engine.resources[resources[name]].read()
                if name in ('interpolation','inpainting'):
                    view=cv2.cvtColor(a,cv2.COLOR_RGBA2BGR);description='Actual FSR color output for the first generated frame, before ADM applies its observed-footprint alpha.'
                elif name=='game_motion_vector_field':
                    view=np.where(a!=0,255,0).astype(np.uint8);description='Nonzero packed motion-field occupancy; white means populated, not terrain height.'
                else:
                    if a.dtype==np.uint32:a=a.view(np.float32)
                    a=a.astype(np.float32)
                    if a.ndim==3:a=np.linalg.norm(a,axis=2) if a.shape[2]==2 else a[:,:,0]
                    a=np.nan_to_num(a);low,high=np.percentile(a,[1,99]);scaled=np.clip((a-low)/max(float(high-low),1e-6),0,1)
                    view=cv2.applyColorMap((scaled*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS)
                    description=f'Diagnostic resource, normalized for display (1st–99th percentile {low:.5g}–{high:.5g}); not photographic color or measured elevation.'
                record.image(label,view,description)
            return result
        self.patch(FrameGenerator,'dispatch',dispatched)
        interpolate=FrameGenerator.interpolate
        def interpolated(engine,a,b,ha,hb):
            result=interpolate(engine,a,b,ha,hb)
            if engine.frames<=3:record.contact(f'Generated view {engine.frames}',[a,result[0],b],['Parent A','FSR generated midpoint','Parent B'],'Actual first three recursive generation outputs, including quarter-frame recursion.')
            return result
        self.patch(FrameGenerator,'interpolate',interpolated)

        write_presentation=presentation.write_presentation
        def presented(image,*args,**kwargs):
            record.native_shape=image.shape[:2];record.image('Neutral mapped image',image,'Mapping complete; presentation has not yet run.',True)
            return write_presentation(image,*args,**kwargs)
        self.patch(presentation,'write_presentation',presented)
        for method,label in [('chroma_clean','Chroma noise cleanup'),('lanczos','Lanczos upscale'),('clahe','Blended CLAHE'),('enhance','Final gentle contrast')]:
            original=getattr(PresentationMetal,method)
            def captured(engine,*args,_original=original,_label=label,**kwargs):
                result=_original(engine,*args,**kwargs);record.image(_label,result,'Actual full-map stage; detail crop is shown at a common native scale.',True);return result
            self.patch(PresentationMetal,method,captured)
        detail=PresentationMetal.detail
        def detailed(engine,source,fine=False):
            result=detail(engine,source,fine);record.image('Fine sharpening' if fine else 'Positive detail',result,f'Actual shader output; amount {engine.params["fine" if fine else "detail"]}.',True);return result
        self.patch(PresentationMetal,'detail',detailed)
        relight=raytracing.relight_map
        def relit(image,*args,**kwargs):
            result,report=relight(image,*args,**kwargs);record.image('Ray tracing · flat plane',result,'The selected flat plane preserves existing shadows and adds no fabricated relief.',True);return result,report
        self.patch(raytracing,'relight_map',relit)
        grade=PresentationMetal.grade
        def graded(engine,*args,**kwargs):
            result=grade(engine,*args,**kwargs);record.image('Cinematic color grade',result,'Supplied preset: tone zones, split tone, saturation, vignette, optional grain, clarity and sharpening.',True);return result
        self.patch(PresentationMetal,'grade',graded)
        crop=lasso.crop_in_place
        def cropped(image,*args,**kwargs):
            result=crop(image,*args,**kwargs)
            record.image('Final lasso crop',result.image,f'Perimeter trimmed; {result.report["retained_fraction"]:.1%} retained. Surviving pixels and geographic grid are unchanged.',True)
            return result
        self.patch(lasso,'crop_in_place',cropped)

    def finish(self,success):
        for obj,name,original in reversed(self.restores):setattr(obj,name,original)
        self.refresh('complete' if success else 'failed — see run.log')


def visual_worker(source,output,config):
    base.load_runtime();recorder=VisualProcess(output);recorder.install();success=False
    try:
        base.worker(source,output,config);success=True
    finally:recorder.finish(success)


def run(source,output,enabled=VISUAL_PROCESS):
    if not enabled:return base.run(source,output)
    source=Path(source).expanduser().resolve();output=Path(output).expanduser().resolve()
    if not source.is_dir():raise ValueError(f'Input directory does not exist: {source}')
    if output==source or output in source.parents or source in output.parents:raise ValueError('Keep input and output directories separate')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise ValueError('Choose a new empty visual-output directory')
    output.mkdir(parents=True,exist_ok=True);config=output/'input_config.json';config.write_text(json.dumps(base.SETTINGS,indent=2)+'\n')
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--input',str(source),'--output',str(output),'--config',str(config)]
    times={};start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f'Live gallery: {output / "visual_process/process.html"}\nProgress log: {output / "run.log"}',flush=True)
    with (output/'run.log').open('w') as log,base.timer('wall_seconds',times):
        result=subprocess.run(command,cwd=base.ROOT,stdout=log,stderr=subprocess.STDOUT)
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    (output/'timing.json').write_text(json.dumps({**times,'start_utc':start_utc,'exit_code':result.returncode,'visual_process':True,
        'input':str(source),'output':str(output),'peak_child_rss_bytes':usage.ru_maxrss,
        'timing_note':'Includes visual previews and GPU readback; compare base speed with run.py instead.'},indent=2)+'\n')
    if result.returncode:print((output/'run.log').read_text()[-8000:],file=sys.stderr)
    else:print(f'Final image: {output / "map_graded.png"}\nGallery: {output / "visual_process/process.html"}')
    return result.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=INPUT_REPO);parser.add_argument('--output',type=Path,default=OUTPUT_DESTINATION)
    parser.add_argument('--no-visual',action='store_true');parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--config',type=Path,help=argparse.SUPPRESS);args=parser.parse_args()
    if args.worker:
        if args.config is None:parser.error('Worker requires --config')
        visual_worker(args.input,args.output,args.config);return 0
    return run(args.input,args.output,VISUAL_PROCESS and not args.no_visual)


if __name__=='__main__':raise SystemExit(main())
