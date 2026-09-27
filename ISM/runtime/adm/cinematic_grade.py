"""User-supplied cinematic grading recipes, adapted for bounded map processing.

Color equations and PIL unsharp-mask settings follow the supplied script.
Changes: preserve alpha, process color in strips, clamp split-tone luminance
before fractional powers, seed grain, and protect transparency from sharpening.
Grading changes presentation; it cannot recover clipped sensor information.
"""
import numpy as np
from PIL import Image, ImageFilter


def preset(shadow,highlight,lift,gamma,gain,contrast,saturation,vignette,grain,warmth,**extra):
    return dict(shadow_tint=shadow,highlight_tint=highlight,lift=lift,gamma=gamma,gain=gain,
        contrast=contrast,saturation=saturation,vignette_strength=vignette,grain=grain,warmth=warmth,**extra)


PRESETS={
    'teal_orange':preset((-.03,.01,.05),(.06,.02,-.05),(0,0,.01),(1,1,1),(1.05,1,.97),.45,1.15,.35,.012,0),
    'golden_hour':preset((.02,0,-.02),(.08,.04,-.08),(.03,.02,0),(1,1,1.05),(1.08,1,.9),.25,1.1,.25,.01,.08),
    'noir':preset((-.02,-.01,.02),(0,0,.02),(0,0,0),(1,1,1),(1,1,1.02),.6,.15,.55,.025,-.05),
    'moody_blue':preset((-.04,-.01,.05),(-.02,0,.03),(.03,.03,.05),(1,1,1),(.97,1,1.05),.3,.7,.4,.015,-.06),
    'drone_cinematic':preset((-.02,0,.03),(.04,.01,-.03),(0,0,.01),(1,1,1),(1.03,1,.99),.3,1.05,.3,.008,.02),
    'pro_camera':preset((-.01,0,.02),(.02,.01,-.01),(0,0,0),(1,1,1),(1,1,1),.35,.95,.08,0,0,
        exposure=0,blacks=.05,whites=-.06,shadows=.02,highlights=.06,sharpness=90,clarity=35),
    'aerial_documentary':preset((-.025,0,.035),(.02,.008,-.015),(0,0,0),(1,1,1),(1.02,1.02,1.02),.38,1.03,.15,.012,-.02,
        exposure=-.08,blacks=.015,shadows=.015,highlights=-.01,whites=-.035,sharpness=25,clarity=15),
}


def region(shadow,highlight,gain,contrast,saturation,warmth,lift=(0,0,0),**extra):
    return preset(shadow,highlight,lift,(1,1,1),gain,contrast,saturation,0,0,warmth,**extra)


REGION_PRESETS={
    'cool_regional':{
        'vegetation':region((-.05,.02,.06),(-.02,.03,.02),(.97,1.03,1.02),.3,1.1,-.03,lift=(0,0,.01)),
        'water':region((-.06,0,.09),(-.03,.02,.05),(.9,1,1.12),.35,1.2,-.08,lift=(0,0,.02)),
        'urban':region((-.02,0,.03),(0,0,.02),(.98,1,1.05),.35,.55,-.04,lift=(0,0,.01)),
        'bare_earth':region((.02,0,-.02),(.05,.02,-.05),(1.06,1,.93),.3,1.05,.06,lift=(.02,0,0)),
        'default':region((-.02,0,.03),(.01,0,0),(1,1,1.03),.3,.95,-.02,lift=(0,0,.01)),
    },
    'vibrant_regional':{
        'vegetation':region((0,.02,0),(.02,.03,-.02),(.95,1.1,.95),.35,1.4,0),
        'water':region((-.03,.01,.08),(-.04,.04,.05),(.85,1.02,1.18),.4,1.45,-.05,lift=(0,0,.01)),
        'urban':region((0,0,.01),(0,0,.01),(1,1,1.01),.3,.8,0),
        'bare_earth':region((.02,0,-.02),(.05,.02,-.05),(1.08,1,.9),.3,1.2,.08,lift=(.02,0,0)),
        'default':region((0,0,.01),(.01,.01,0),(1,1,1),.3,1.15,0),
    },
    'aerial_documentary_regional':{
        'vegetation':region((-.02,0,.025),(.01,.005,-.01),(.98,1.02,.99),.32,1,-.01,highlights=-.015,hue_shift_degrees=-4),
        'bare_earth':region((0,0,0),(.02,.01,-.015),(1.03,1,.97),.32,1.05,.02,highlights=.015,hue_shift_degrees=2),
        'urban':region((-.01,0,.015),(0,0,.01),(.99,1,1.02),.32,.85,-.01,highlights=.01),
        'water':region((-.025,0,.035),(0,0,.015),(.98,1,1.03),.32,.95,-.02),
        'default':region((-.025,0,.035),(.02,.008,-.015),(1,1,1),.32,.95,-.02),
    },
}
REGION_DETECT={
    'vegetation':(85,32,35,25,15,20), 'water':(150,22,30,25,10,20),
    'bare_earth':(20,16,25,25,25,25), 'urban':(None,None,45,25,55,25),
}


def to_pil(arr):
    return Image.fromarray((np.clip(arr,0,1)*255+.5).astype(np.uint8))


def luma(arr):
    return arr[...,0]*.299+arr[...,1]*.587+arr[...,2]*.114


def grade_array(arr,params):
    out=np.empty_like(arr)
    for c in range(3):out[...,c]=np.clip(arr[...,c]*params['gain'][c]+params['lift'][c],1e-6,1)**(1/params['gamma'][c])
    warmth=params.get('warmth',0);out[...,0]+=warmth*.5;out[...,2]-=warmth*.5
    hue=params.get('hue_shift_degrees',0)
    if hue:
        hsv=np.array(to_pil(out).convert('HSV'),dtype=np.float32)
        hsv[...,0]=np.mod(hsv[...,0]+hue/360*255,255)
        out=np.asarray(Image.fromarray(hsv.astype(np.uint8),'HSV').convert('RGB'),dtype=np.float32)/255
    # Warmth can move dark colors below zero; fractional powers need this bound.
    light=np.clip(luma(out),0,1);sw=(1-light)**1.5;hw=light**1.5
    for c in range(3):out[...,c]+=params['shadow_tint'][c]*sw+params['highlight_tint'][c]*hw
    light=luma(out)
    shift=(params.get('blacks',0)*np.clip(1-light/.3,0,1)+params.get('shadows',0)*np.exp(-((light-.25)/.25)**2)
        +params.get('highlights',0)*np.exp(-((light-.75)/.25)**2)+params.get('whites',0)*np.clip((light-.7)/.3,0,1))
    out+=shift[...,None]
    strength=params['contrast']
    if strength>0:out=out+strength*(out-.5)*(1-np.abs(2*out-1))
    light=luma(out)[...,None]
    return light+(out-light)*params['saturation']


def pro_camera_grade(arr,params=None):
    """Supplied pro_camera color equations; exposure/sharpening are applied outside."""
    return grade_array(arr,PRESETS['pro_camera'] if params is None else params)


def classify_regions(arr,blur_radius=6):
    hsv=np.asarray(Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8)).convert('HSV'),dtype=np.float32)
    h,s,v=hsv.transpose(2,0,1);scores={}
    def smooth(x,lo,hi):
        t=np.clip((x-lo)/max(hi-lo,1e-6),0,1);return t*t*(3-2*t)
    for name,(center,sigma,sat,soft,val,valsoft) in REGION_DETECT.items():
        if center is None:score=1-smooth(s,sat,sat+soft)
        else:
            d=abs(h-center);d=np.minimum(d,256-d)
            score=np.exp(-(d/sigma)**2)*smooth(s,sat-soft,sat)
        score*=smooth(v,val-valsoft,val)
        mask=Image.fromarray((np.clip(score,0,1)*255).astype(np.uint8))
        if blur_radius>0:mask=mask.filter(ImageFilter.GaussianBlur(blur_radius))
        scores[name]=np.asarray(mask,dtype=np.float32)/255
    total=sum(scores.values())+.12
    return {**{name:s/total for name,s in scores.items()},'default':.12/total}


def grade_bgra(image,preset_name='pro_camera',region_preset=None,seed=2026,overrides=None,strip_rows=128,region_blur=6.):
    """Final-output grading only. RGB working strips keep memory bounded."""
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=4:raise ValueError('Grading expects uint8 BGRA')
    params={**PRESETS[preset_name],**(overrides or {})};regions=REGION_PRESETS[region_preset] if region_preset else None
    h,w=image.shape[:2];rgb=np.empty((h,w,3),np.uint8);rng=np.random.default_rng(seed)
    halo=int(np.ceil(region_blur*4))+2 if regions else 0
    xx=(np.arange(w,dtype=np.float32)-w/2)/(w/2)
    for y in range(0,h,strip_rows):
        end=min(y+strip_rows,h);lo=max(0,y-halo);hi=min(h,end+halo)
        arr=image[lo:hi,:,2::-1].astype(np.float32)/255
        arr*=2**params.get('exposure',0)
        if regions:
            weights=classify_regions(arr,region_blur);out=np.zeros_like(arr)
            for name,weight in weights.items():out+=grade_array(arr,regions[name])*weight[...,None]
        else:out=grade_array(arr,params)
        out=out[y-lo:end-lo]
        vignette=params['vignette_strength']
        if vignette>0:
            yy=(np.arange(y,end,dtype=np.float32)-h/2)/(h/2)
            dist=np.sqrt(yy[:,None]**2+xx[None,:]**2)
            out*=np.clip(1-vignette*np.clip(dist-.3,0,None)**1.8,0,1)[...,None]
        if params['grain']>0:out+=rng.normal(0,params['grain'],(end-y,w,1)).astype(np.float32)
        rgb[y:end]=(np.clip(out,0,1)*255+.5).astype(np.uint8)
    before=Image.fromarray(rgb)
    graded=before
    if params.get('clarity',0)>0:graded=graded.filter(ImageFilter.UnsharpMask(radius=30,percent=int(params['clarity']),threshold=2))
    if params.get('sharpness',0)>0:graded=graded.filter(ImageFilter.UnsharpMask(radius=1.5,percent=int(params['sharpness']),threshold=2))
    final=np.asarray(graded).copy()
    # A transparent map boundary is not a black object to sharpen against.
    if np.any(image[:,:,3]!=255) and (params.get('clarity',0)>0 or params.get('sharpness',0)>0):
        radius=30 if params.get('clarity',0)>0 else 1.5
        mask=Image.fromarray(image[:,:,3]).filter(ImageFilter.GaussianBlur(radius))
        boundary=np.asarray(mask)<255;final[boundary]=rgb[boundary]
    result=np.empty_like(image);result[:,:,:3]=final[:,:,::-1];result[:,:,3]=image[:,:,3]
    result[image[:,:,3]==0,:3]=0
    return result


def finish_tone(image,white_reduction=0.,vibrance_reduction=0.,strip_rows=128):
    """Monotone white compression plus a saturation-aware vibrance reduction."""
    if not np.isfinite(white_reduction) or not 0<=white_reduction<.5 or not np.isfinite(vibrance_reduction) or not 0<=vibrance_reduction<=1:
        raise ValueError('Invalid white or vibrance reduction')
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=4:raise ValueError('Tone adjustment expects BGRA8')
    result=image.copy()
    for y in range(0,len(image),strip_rows):
        src=image[y:y+strip_rows];rgb=src[:,:,2::-1].astype(np.float32)/255
        light=luma(rgb);t=np.maximum((light-.5)*2,0);k=white_reduction/(.5-white_reduction)
        target=np.where(light>.5,.5+.5*t/(1+k*t),light)
        rgb*=np.divide(target,light,out=np.ones_like(light),where=light>0)[...,None]
        mx=rgb.max(2);sat=(mx-rgb.min(2))/np.maximum(mx,1e-6);light=luma(rgb)[...,None]
        rgb=light+(rgb-light)*(1-vibrance_reduction*(1-sat))[...,None]
        result[y:y+len(src),:,:3]=(np.clip(rgb[:,:,::-1],0,1)*255+.5).astype(np.uint8)
    result[result[:,:,3]==0,:3]=0
    return result
