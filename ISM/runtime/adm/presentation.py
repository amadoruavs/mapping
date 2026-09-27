"""User's post-map OpenCV recipe, retained as the numerical reference."""
import cv2
import numpy as np

PROFILES={
    'standard':dict(detail=1.3,fine=.35,clahe=.75,contrast=1.02,offset=-1.,chroma=0.),
    'siyi_low_light':dict(detail=.65,fine=.15,clahe=.35,contrast=1.,offset=0.,chroma=.7),
}


def chroma_clean_reference(image,strength=.7):
    """3x3 edge-aware chroma smoothing; preserve each pixel's luma and alpha."""
    from .cinematic_grade import luma
    rgb=image[:,:,2::-1].astype(np.float32)/255;light=luma(rgb);chroma=rgb-light[:,:,None]
    padded=np.pad(rgb,((1,1),(1,1),(0,0)),mode='reflect');alpha=np.pad(image[:,:,3],1,mode='reflect')
    total=np.zeros_like(rgb);weights=np.zeros_like(light);h,w=light.shape
    for dy in range(3):
        for dx in range(3):
            other=padded[dy:dy+h,dx:dx+w];lum=luma(other)
            weight=np.exp(-((lum-light)/.08)**2)*np.exp(-((dx-1)**2+(dy-1)**2)*.5)*(alpha[dy:dy+h,dx:dx+w]>0)
            total+=(other-lum[:,:,None])*weight[:,:,None];weights+=weight
    color=chroma*(1-strength)+total/np.maximum(weights[:,:,None],1e-6)*strength
    # Compress chroma when needed to preserve the original luminance in gamut.
    pos=np.maximum(color.max(2),1e-6);neg=np.maximum(-color.min(2),1e-6)
    scale=np.minimum(1,np.minimum((1-light)/pos,light/neg))
    result=image.copy();result[:,:,:3]=(np.clip((light[:,:,None]+color*scale[:,:,None])[:,:,::-1],0,1)*255+.5).astype(np.uint8)
    result[result[:,:,3]==0,:3]=0;return result


def dimensions(image,scale):
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2] not in (3,4):
        raise ValueError('Presentation expects uint8 BGR or BGRA')
    if not np.isfinite(scale) or not 1<=scale<=4:raise ValueError('Presentation scale must be from 1 to 4')
    h,w=image.shape[:2];ow,oh=int(w*scale),int(h*scale)
    if ow*oh>160_000_000:raise ValueError('Presentation exceeds 160 million output pixels')
    return ow,oh


def enhance_reference(image,scale=2.,stages=None,profile='standard'):
    """Revised recipe: no denoise, detail 1.3; alpha resized separately."""
    ow,oh=dimensions(image,scale)
    params=PROFILES[profile]
    def record(name,im):
        if stages is not None:stages[name]=im.copy()
        return im
    working=image
    if params['chroma']:
        if working.shape[2]==3:working=cv2.cvtColor(working,cv2.COLOR_BGR2BGRA)
        working=chroma_clean_reference(working,params['chroma']);record('chroma',working[:,:,:3])
    im=record('lanczos',cv2.resize(working[:,:,:3],(ow,oh),interpolation=cv2.INTER_LANCZOS4))
    low=cv2.GaussianBlur(im,(0,0),1.)
    detail=cv2.subtract(im,low)
    im=record('detail',cv2.addWeighted(im,1.,detail,params['detail'],0))
    before_clahe=im;lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB)
    lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0])
    im=record('clahe',cv2.addWeighted(before_clahe,1-params['clahe'],cv2.cvtColor(lab,cv2.COLOR_LAB2BGR),params['clahe'],0))
    fine=cv2.GaussianBlur(im,(0,0),.45)
    im=record('fine',cv2.addWeighted(im,1+params['fine'],fine,-params['fine'],0))
    im=record('contrast',cv2.convertScaleAbs(im,alpha=params['contrast'],beta=params['offset']))
    if image.shape[2]==4:
        im=cv2.cvtColor(im,cv2.COLOR_BGR2BGRA)
        im[:,:,3]=cv2.resize(image[:,:,3],(ow,oh),interpolation=cv2.INTER_NEAREST)
        im[im[:,:,3]==0,:3]=0
    return im


def pixel_transform(input_shape,output_shape):
    sy=output_shape[0]/input_shape[0];sx=output_shape[1]/input_shape[1]
    return np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])


def _final_lasso(image,cfg,out,gpu,filename,geometry):
    """Trim the finished image before its single export, retaining its pixel grid."""
    if not cfg.lasso_crop:return image
    if gpu is None:raise ValueError('Lasso cropping requires the Metal backend')
    import time
    from .lasso import crop_in_place
    from .io import write_json
    print('[lasso] Cropping final image after presentation',flush=True)
    started=time.perf_counter();cropped=crop_in_place(image,gpu,inset_ratio=cfg.lasso_inset_ratio,
        opening_ratio=cfg.lasso_opening_ratio,angular_radius=cfg.lasso_angular_radius)
    report={**cropped.report,'seconds':time.perf_counter()-started,
        'applies_to':[filename],'stage':'final_image_operation',
        'after_color_grade':cfg.color_grade!='none',
        'map_to_world':geometry['map_to_world']}
    write_json(out/'lasso_crop.json',report)
    final_geometry={**geometry,'width':image.shape[1],'height':image.shape[0],
        'applies_to':[filename],'covered_pixels':report['retained_pixels'],'lasso_crop':'lasso_crop.json'}
    write_json(out/'presentation_geometry.json',final_geometry)
    print(f'[lasso] Retained {report["retained_fraction"]:.1%} of coverage in {report["seconds"]:.3f}s',flush=True)
    return cropped.image


def write_presentation(rgba,geometry,hs,cfg,out,gpu):
    """Reuse a completed neutral map for presentation-only iterations."""
    import time
    from .io import write_json,write_array,write_image
    geographic=geometry.get("georeferenced",False)
    presentation_image=rgba
    pg=geometry
    if cfg.presentation:
        from .enhancement import easu
        from .presentation import pixel_transform,dimensions
        from .presentation_metal import PresentationMetal
        started=time.perf_counter();original_shape=rgba.shape
        # Validate the final pixel budget before allocating any enlarged image.
        total_scale=2**cfg.presentation_fsr_passes*cfg.presentation_scale
        if int(rgba.shape[0]*total_scale)*int(rgba.shape[1]*total_scale)>160_000_000:
            raise ValueError('Final presentation exceeds 160 million pixels; reduce scale or map dimensions')
        for _ in range(cfg.presentation_fsr_passes):
            presentation_image=easu(presentation_image,(presentation_image.shape[0]*2,presentation_image.shape[1]*2),gpu)
        upscaled=time.perf_counter();processor=PresentationMetal(gpu,getattr(cfg,'presentation_profile','standard'))
        presentation_image=processor.enhance(presentation_image,cfg.presentation_scale)
        enhanced=time.perf_counter()
        S=pixel_transform(original_shape,presentation_image.shape)
        pg={**geometry,'width':int(presentation_image.shape[1]),'height':int(presentation_image.shape[0]),
            'map_to_world':(np.array(geometry['map_to_world'])@np.linalg.inv(S)).tolist(),
            'metres_per_pixel':geometry['metres_per_pixel']/S[0,0] if geographic else None,
            'applies_to':['map_graded.png' if cfg.color_grade!='none' else 'map_enhanced.png'],
            'sampling':'Final presentation enlargement; no new measured ground detail'}
        write_json(out/'presentation_geometry.json',pg)
        write_array(out/'image_to_presentation.npy',np.array([S@H for H in hs]))
        write_json(out/'presentation_report.json',dict(backend='Metal',fsr='AMD FSR1 EASU' if cfg.presentation_fsr_passes else None,fsr_passes=cfg.presentation_fsr_passes,
            lanczos_scale=cfg.presentation_scale,profile=processor.profile,parameters=processor.params,
            order=(['FSR upscale'] if cfg.presentation_fsr_passes else [])+(['edge-aware chroma cleanup'] if processor.params['chroma'] else [])+
                ['Lanczos4','positive detail','Lab CLAHE clip=2 grid=8x8 blended','fine sharpen sigma=.45','gentle contrast'],
            fsr_seconds=upscaled-started,sharpen_seconds=enhanced-upscaled,stage_seconds=processor.timings,
            enhance_and_export_seconds=time.perf_counter()-started,geometry='presentation_geometry.json',alpha_preserved=True))
        if cfg.presentation_raytrace:
            from .raytracing import relight_map
            presentation_image,ray_report=relight_map(presentation_image,gpu)
            write_json(out/'raytracing_report.json',ray_report)
    if cfg.color_grade!='none':
        from .cinematic_grade import grade_bgra,PRESETS
        started=time.perf_counter()
        overrides={}
        grade_backend='NumPy/Pillow'
        if gpu is not None:
            from .presentation_metal import PresentationMetal
            graded=PresentationMetal(gpu).grade(presentation_image,cfg.color_grade,overrides,
                region_preset=None if cfg.regional_grade=='none' else cfg.regional_grade,seed=cfg.seed);grade_backend='Metal'
        else:graded=grade_bgra(presentation_image,cfg.color_grade,None if cfg.regional_grade=='none' else cfg.regional_grade,cfg.seed,overrides)
        applied=time.perf_counter()
        graded=_final_lasso(graded,cfg,out,gpu,'map_graded.png',pg)
        write_image(out/'map_graded.png',graded)
        write_json(out/'grading_report.json',dict(preset=cfg.color_grade,regional_preset=cfg.regional_grade,
            parameters={**PRESETS[cfg.color_grade],**overrides},after_compositing=True,alpha_preserved=True,dimensions_preserved=True,
            adjustments_inside_preset=True,post_grade_tone_pass=False,final_lasso_crop=cfg.lasso_crop,
            grade_seconds=applied-started,grade_and_export_seconds=time.perf_counter()-started,
            implementation=f'User-supplied cinematic recipes with Pillow-compatible clarity/sharpening; {grade_backend}',
            grain_rng='NumPy default_rng, uploaded only for grain presets',backend=grade_backend,seed=cfg.seed))
    elif cfg.presentation:
        presentation_image=_final_lasso(presentation_image,cfg,out,gpu,'map_enhanced.png',pg)
        write_image(out/'map_enhanced.png',presentation_image)
