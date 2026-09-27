"""FSR ablations on all benchmark photos; frozen-pose renders isolate appearance.

No full-process timing is inferred from these composition-only experiments.
The separate benchmark_gpu_revision.py command measures fresh complete runs.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from adm.config import Config
from adm.enhancement import easu,rcas,upscale
from adm.geometry import project
from adm.coherent_composite import compose_pixels
from adm.metal import get_backend
from adm.virtual_views import VirtualViews

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/killerofmaps'
SOURCE=Path('/Users/avneh/Downloads/killerofmaps')
OUT=BASE/'fsr_experiments'


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def metrics(reference,candidate,mask):
    """SSIM on luminance, RGB PSNR, and gradient error on identical support."""
    a=reference[:,:,:3].astype(np.float32);b=candidate[:,:,:3].astype(np.float32)
    error=a-b;mse=float(np.square(error[mask]).mean())
    gray=lambda x:cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
    a,b=gray(a),gray(b)
    blur=lambda x:cv2.GaussianBlur(x,(11,11),1.5)
    ma,mb=blur(a),blur(b);va=blur(a*a)-ma*ma;vb=blur(b*b)-mb*mb;cov=blur(a*b)-ma*mb
    ssim=((2*ma*mb+6.5025)*(2*cov+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225))
    grad=lambda x:cv2.magnitude(cv2.Sobel(x,cv2.CV_32F,1,0),cv2.Sobel(x,cv2.CV_32F,0,1))
    ga,gb=grad(a),grad(b)
    return dict(psnr_db=float(10*np.log10(255**2/max(mse,1e-12))),ssim=float(ssim[mask].mean()),
                gradient_mae=float(abs(ga-gb)[mask].mean()),gradient_ratio=float(gb[mask].mean()/max(ga[mask].mean(),1e-8)),pixels=int(mask.sum()))


def interior(image):
    mask=(image[:,:,:3].max(2)>12).astype(np.uint8)
    mask=cv2.erode(mask,np.ones((13,13),np.uint8));mask[:48]=0;mask[-48:]=0;mask[:,:48]=0;mask[:,-48:]=0
    return mask.astype(bool)


def average(rows):
    keys=('psnr_db','ssim','gradient_mae','gradient_ratio')
    return {k:float(np.mean([r[k] for r in rows])) for k in keys}


def quality(gpu):
    rows=[];started=time.perf_counter()
    files=sorted(SOURCE.glob('*.jpg'))
    for i,path in enumerate(files):
        image=cv2.imread(str(path));h,w=image.shape[:2];low=cv2.resize(image,(w//2,h//2),interpolation=cv2.INTER_AREA)
        mask=interior(image);spatial=easu(low,(h,w),gpu)
        candidates={'bilinear':cv2.resize(low,(w,h),interpolation=cv2.INTER_LINEAR),
            'bicubic':cv2.resize(low,(w,h),interpolation=cv2.INTER_CUBIC),
            'lanczos4':cv2.resize(low,(w,h),interpolation=cv2.INTER_LANCZOS4),'easu':spatial}
        for strength in (.25,.5,1.):candidates[f'fsr_{strength:g}']=rcas(spatial,strength,gpu)
        for passes in (2,3):
            big,_=upscale(low,passes,.5,gpu)
            candidates[f'recursive_{passes}']=cv2.resize(big,(w,h),interpolation=cv2.INTER_AREA)
        values={key:metrics(image,value,mask) for key,value in candidates.items()}
        rows.append(dict(name=path.name,split='calibration' if i%3==0 else 'validation',values=values))
        if i%30==0:print('resolution recovery',i+1,'/',len(files),flush=True)
    summaries={split:{key:average([r['values'][key] for r in rows if r['split']==split]) for key in rows[0]['values']} for split in ('calibration','validation')}
    write(OUT/'resolution_recovery.json',dict(images=len(rows),seconds=time.perf_counter()-started,
        protocol='640px originals withheld as references; area downsample to 320 then recover 640. Fixed interior masks. Recursive candidates reduced back to 640 for equal-size comparison. Every third image calibrates; remainder validates; views share terrain.',
        limitation='Simulated resolution loss on existing JPEGs; not proof of new ground detail or native-camera super-resolution.',summary=summaries,images_detail=rows))
    print(json.dumps(summaries,indent=2),flush=True)



def render(gpu,mode,holdout=False):
    baseline=BASE/'metal_lean';geometry=json.loads((baseline/'map_geometry.json').read_text())
    all_names=geometry['input_names'];all_hs=np.load(baseline/'image_to_map.npy')
    held=np.arange(len(all_names))%6==3
    indices=np.flatnonzero(~held) if holdout else np.arange(len(all_names))
    names=[all_names[i] for i in indices];hs=all_hs[indices]
    modes={'baseline':{},'fsr3':{},'fsr':dict(fsr=True),'pre2x':dict(fsr=True,input_upscale_passes=1),
        'recursive4x':dict(fsr=True,input_upscale_passes=2,output_sharpen_passes=3),
        'interpolate':dict(fsr=True),'interpolate_recursive':dict(fsr=True,input_upscale_passes=1,output_sharpen_passes=3)}
    cfg=Config(coherent_composite=True,coherent_budget_mb=128,export_profile='png',**modes[mode])
    folder=OUT/(mode+('_heldout' if holdout else '_render'));folder.mkdir(exist_ok=True)
    start=time.perf_counter()
    if mode=='fsr3':
        from adm.fsr_views import FSRViews
        views=FSRViews(SOURCE,names,hs,levels=2)
    else:views=VirtualViews(SOURCE,names,hs,gpu,levels=2 if mode=='interpolate_recursive' else int(mode=='interpolate'))
    labels=names+[f'virtual_{k}' for k in range(len(views.records)-len(names))]
    result,*_=compose_pixels(SOURCE,labels,[r[3] for r in views.records],geometry['width'],geometry['height'],cfg,folder,gpu,frame_provider=views)
    composited=time.perf_counter();evaluations=[]
    if holdout:
        for i in np.flatnonzero(held):
            original=cv2.imread(str(SOURCE/all_names[i]));h,w=original.shape[:2]
            # Area-filter map minification to avoid sharpening scores from aliasing.
            scale=np.sqrt(abs(np.linalg.det(np.linalg.inv(all_hs[i])[:2,:2])))
            from compare_lean import resample
            corners=project(all_hs[i],[[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
            x0,y0=np.maximum(0,np.floor(corners.min(0)-12).astype(int))
            x1,y1=np.minimum([result.shape[1],result.shape[0]],np.ceil(corners.max(0)+13).astype(int))
            crop=result[y0:y1,x0:x1]
            transform=np.linalg.inv(all_hs[i])@np.array([[1,0,x0],[0,1,y0],[0,0,1.]])
            reprojected=resample(crop,transform,(w,h),min(scale,1))
            mask=interior(original)&(cv2.erode(reprojected[:,:,3],np.ones((13,13),np.uint8))==255)
            if mask.sum()>1000:evaluations.append(dict(name=all_names[i],**metrics(original,reprojected,mask)))
    else:
        cv2.imwrite(str(folder/'map_neutral.png'),result)
        np.save(folder/'image_to_map.npy',all_hs);write(folder/'map_geometry.json',geometry)
    report=dict(mode=mode,real_photos=len(names),virtual_views=len(views.records)-len(names),
        compose_seconds=composited-start,render_and_export_seconds=time.perf_counter()-start,
        virtual_generation_seconds=views.report()['generation_seconds'],virtual_generation_calls=views.report()['generation_calls'],
        fresh_pipeline=False,registration='Frozen original 306-photo geometry; virtual views contribute no matching constraints',
        interpolation=views.report()['algorithm'],
        heldout_appearance_images=len(evaluations),quality=average(evaluations) if evaluations else None,
        limitation='Held-out pixels excluded from compositor; their geometry still comes from the original joint alignment. Neighbouring validation views share terrain.',images=evaluations)
    write(folder/'interpolation_report.json',views.report())
    write(folder/'experiment.json',report);print(json.dumps({k:v for k,v in report.items() if k!='images'},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',choices=['quality','render','holdouts','presentation','frame_profile','presentation_map','stage_audit'])
    p.add_argument('--config',type=Path);p.add_argument('--output',type=Path);p.add_argument('--source-run',type=Path)
    p.add_argument('--mode',default='fsr',choices=['baseline','fsr3','fsr','pre2x','recursive4x','interpolate','interpolate_recursive'])
    p.add_argument('--holdout',action='store_true');args=p.parse_args();OUT.mkdir(exist_ok=True)
    cv2.setNumThreads(6);gpu=get_backend()
    if args.task=='stage_audit':
        from audit_presentation import run
        run(gpu,args.output,args.source_run)
    elif args.task=='presentation_map':
        from adm.presentation import write_presentation
        if not args.config or not args.output:p.error('presentation_map requires --config and --output')
        if args.output.exists() and any(args.output.iterdir()):raise RuntimeError('Choose a new output directory')
        args.output.mkdir(parents=True,exist_ok=True)
        cfg=Config(**{**json.loads(args.config.read_text()),'coherent_composite':True,'export_profile':'png'}).validate()
        source=args.source_run or BASE/'metal_fsr3_full';started=time.perf_counter()
        neutral=cv2.imread(str(source/'map_neutral.png'),cv2.IMREAD_UNCHANGED)
        if neutral is None:raise ValueError('Saved neutral FSR map is missing')
        geometry=json.loads((source/'map_geometry.json').read_text());hs=np.load(source/'image_to_map.npy')
        write_presentation(neutral,geometry,hs,cfg,args.output,gpu)
        report=dict(source_neutral=str(source/'map_neutral.png'),original_real_photos=len(hs),virtual_views=915,
            reuses_completed_mapping=True,fresh_pipeline=False,presentation_and_export_seconds=time.perf_counter()-started)
        write(args.output/'resolved_config.json',cfg.to_dict());write(args.output/'rerender_report.json',report)
        print(json.dumps(report,indent=2))
    elif args.task=='frame_profile':
        from adm.fsr_framegen import FrameGenerator
        engine=FrameGenerator();baseline=BASE/'metal_lean';g=json.loads((baseline/'map_geometry.json').read_text());hs=np.load(baseline/'image_to_map.npy')
        for i in range(30):
            a=cv2.imread(str(SOURCE/g['input_names'][i]));b=cv2.imread(str(SOURCE/g['input_names'][i+1]))
            engine.interpolate(a,b,hs[i],hs[i+1])
        write(OUT/'frame_profile.json',engine.report());print(json.dumps(engine.report(),indent=2))
    elif args.task=='presentation':
        from adm.presentation import enhance_reference
        from adm.presentation_metal import PresentationMetal
        from adm.cinematic_grade import grade_bgra
        files=sorted(SOURCE.glob('*.jpg'));photo=cv2.imread(str(files[14]));photo=cv2.cvtColor(photo,cv2.COLOR_BGR2BGRA)
        photo=easu(photo,(1280,1280),gpu);engine=PresentationMetal(gpu)
        t=time.perf_counter();reference=enhance_reference(photo,1);reference_grade=grade_bgra(reference);cpu=time.perf_counter()-t
        t=time.perf_counter();actual=engine.enhance(photo,1);sharp=time.perf_counter()-t
        t=time.perf_counter();graded=engine.grade(actual);grade=time.perf_counter()-t
        diff=np.abs(actual[:,:,:3].astype(float)-reference[:,:,:3].astype(float));gd=np.abs(graded[:,:,:3].astype(float)-reference_grade[:,:,:3].astype(float))
        cv2.imwrite(str(OUT/'presentation_metal.png'),graded);cv2.imwrite(str(OUT/'presentation_reference.png'),reference_grade)
        report=dict(source=files[14].name,shape=actual.shape,cpu_seconds=cpu,metal_sharpen_seconds=sharp,metal_grade_seconds=grade,stages=engine.timings,
            difference=dict(mean=float(diff.mean()),maximum=float(diff.max()),p99=float(np.percentile(diff,99))),
            graded_difference=dict(mean=float(gd.mean()),maximum=float(gd.max()),p99=float(np.percentile(gd,99))))
        write(OUT/'presentation_parity.json',report);print(json.dumps(report,indent=2))
    elif args.task=='holdouts':
        for mode in ('baseline','fsr','pre2x','recursive4x','interpolate','interpolate_recursive'):render(gpu,mode,True)
    elif args.task=='quality':quality(gpu)
    else:render(gpu,args.mode,args.holdout)


if __name__=='__main__':main()
