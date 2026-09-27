"""Replay presentation after the timed run and inspect identical road pixels.

Each operator sees the full map, preserving global CLAHE tiles and vignette
coordinates. Only small diagnostic crops are retained between stages.
"""
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from adm.presentation_metal import PresentationMetal

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'output/killerofmaps/metal_fsr3_sunset_e2e'


def run(gpu,output=None,source_run=None):
    source_run=source_run or RUN
    out=output or source_run/'quality_audit';out.mkdir(parents=True,exist_ok=True)
    neutral=cv2.imread(str(source_run/'map_neutral.png'),cv2.IMREAD_UNCHANGED)
    if neutral is None:raise RuntimeError('The full benchmark has not produced its neutral map')
    cfg=json.loads((source_run/'resolved_config.json').read_text());engine=PresentationMetal(gpu,cfg.get('presentation_profile','standard'))
    roi=(1470,2700,570,385);x,y,w,h=roi;rows=[];images={};mask=None
    def capture(name,buffer):
        nonlocal mask
        array=buffer.numpy(False) if hasattr(buffer,'numpy') else buffer
        s=array.shape[1]/neutral.shape[1]
        crop=array[round(y*s):round((y+h)*s),round(x*s):round((x+w)*s)].copy()
        aligned=cv2.resize(crop,(w,h),interpolation=cv2.INTER_AREA) if s!=1 else crop
        gray=cv2.cvtColor(aligned[:,:,:3],cv2.COLOR_BGR2GRAY).astype(np.float32)
        if mask is None:
            mask=cv2.erode(((gray>160)&(aligned[:,:,3]==255)).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
            mask[:8]=False;mask[-8:]=False;mask[:,:8]=False;mask[:,-8:]=False
        detail=gray-cv2.GaussianBlur(gray,(0,0),1.5)
        grad=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
        rows.append(dict(stage=name,pixels=int(mask.sum()),mean_luma=float(gray[mask].mean()),
            texture_rms=float(np.sqrt(np.mean(detail[mask]**2))),mean_gradient=float(grad[mask].mean()),
            bright_clipped_fraction=float(np.mean(np.max(aligned[:,:,:3],axis=2)[mask]>=254))))
        images[name]=aligned
    started=time.perf_counter();current=gpu.upload(neutral);capture('FSR map before presentation',current)
    for name,fn in [('Lanczos 2x',lambda s:engine.lanczos(s,(neutral.shape[0]*2,neutral.shape[1]*2))),
        ('Positive detail 1.3',engine.detail),('CLAHE',engine.clahe),('Fine sharpen',lambda s:engine.detail(s,True))]:
        current=fn(current);capture(name,current)
    count=current.shape[0]*current.shape[1];gpu.dispatch('post_contrast',[current],count,[count],[engine.params['contrast'],engine.params['offset']]);capture('Global contrast',current)
    presentation=current.numpy(False)
    if cfg.get('presentation_raytrace'):
        from adm.raytracing import relight_map
        presentation,ray_report=relight_map(presentation,gpu);capture('Ray tracing: flat plane',presentation)
    overrides={k:cfg['grade_'+k] for k in ('white_reduction','vibrance_reduction') if cfg.get('grade_'+k)}
    toned=engine.grade(presentation,cfg['color_grade'],overrides);capture('Cinematic pro_camera',toned)
    actual=cv2.imread(str(source_run/'map_graded.png'),cv2.IMREAD_UNCHANGED)
    replay_identical=np.array_equal(actual,toned)
    output_hash=hashlib.sha256(memoryview(np.ascontiguousarray(toned))).hexdigest()
    del actual,presentation,current,toned
    for i,row in enumerate(rows):
        row['texture_ratio_to_previous']=row['texture_rms']/rows[max(0,i-1)]['texture_rms']
        row['gradient_ratio_to_previous']=row['mean_gradient']/rows[max(0,i-1)]['mean_gradient']
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
    small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
    cw,ch=w,h;top=66;gap=14;canvas=Image.new('RGB',(cw*3+gap*2,(ch+top)*3+gap*2),(245,247,248));draw=ImageDraw.Draw(canvas)
    for i,(name,crop) in enumerate(images.items()):
        dx=(i%3)*(cw+gap);dy=(i//3)*(ch+top+gap)
        draw.text((dx+9,dy+8),name,fill=(20,25,30),font=font)
        draw.text((dx+9,dy+36),f"Texture RMS: {rows[i]['texture_rms']:.2f} | mean: {rows[i]['mean_luma']:.1f}",fill=(65,72,79),font=small)
        canvas.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGRA2RGB)),(dx,dy+top))
    canvas.save(out/'road_stages.png')
    report=dict(protocol='Post-benchmark replay, every operator on the full map; fixed bright-road mask from neutral crop; all metrics at native map resolution after area-filtering 2x output.',
        limitation='Local contrast/texture proxies in one crop, not independent ground-truth detail or overall map quality.',
        roi_native=list(roi),seconds=time.perf_counter()-started,replay_pixel_identical_to_benchmark=replay_identical,
        final_bgra_sha256=output_hash,stages=rows)
    (out/'stage_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not replay_identical:raise AssertionError('Stage replay did not reproduce benchmark output')
