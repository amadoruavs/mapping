"""Gentle virtual-exposure blending and aerial cinematic finishing.

Exposures are derived from the same photograph. This redistributes existing
tones; it cannot recover clipped sensor values or create additional HDR data.
All spatial fields are alpha-normalized, and fine image structure is retained.
"""
import cv2
import numpy as np

LOOKS={
    'natural_cinema':dict(exposure_strength=.45,contrast=.12,saturation=.98,warmth=.012,
        shadow_lift=.008,highlight_compression=.16,green_reduction=.045,diffusion=.018,halation=.006),
    'warm_documentary':dict(exposure_strength=.55,contrast=.16,saturation=.97,warmth=.024,
        shadow_lift=.010,highlight_compression=.20,green_reduction=.07,diffusion=.025,halation=.01),
}


def linear(rgb):return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def srgb(rgb):return np.where(rgb<=.0031308,rgb*12.92,1.055*np.maximum(rgb,0)**(1/2.4)-.055)


def luma(rgb):return rgb[...,0]*.2126+rgb[...,1]*.7152+rgb[...,2]*.0722


def blur_masked(values,alpha,sigma):
    mask=cv2.GaussianBlur(alpha,(0,0),sigma)
    return cv2.GaussianBlur(values*alpha[...,None],(0,0),sigma)/np.maximum(mask[...,None],1e-6)


def finish_cinematic(image,look='natural_cinema',strip_rows=128):
    if look not in LOOKS:raise ValueError(f'Unknown cinematic look: {look}')
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=4:raise ValueError('Cinematic finish expects BGRA8')
    if strip_rows<1:raise ValueError('strip_rows must be positive')
    params=LOOKS[look];h,w=image.shape[:2]
    # Premultiplied downsampling prevents hidden pixels outside a TIFF mask
    # from contaminating the exposure/white-balance/glow measurements.
    sw=min(768,w);sh=max(1,round(h*sw/w))
    small=np.empty((sh,sw,3),np.float32);alpha=image[:,:,3].astype(np.float32)/255
    small_alpha=cv2.resize(alpha,(sw,sh),interpolation=cv2.INTER_AREA)
    for c in range(3):
        small[:,:,c]=cv2.resize(image[:,:,2-c].astype(np.float32)/255*alpha,(sw,sh),interpolation=cv2.INTER_AREA)/np.maximum(small_alpha,1e-6)
    del alpha
    y=luma(small);sat=small.max(2)-small.min(2)
    # Restrict neutral inference to bright, low-chroma surfaces; muted grass
    # should not be mistaken for a gray card and stripped of its green color.
    neutral=(sat<.10)&(y>.60)&(y<.88)&(small_alpha>.99)
    gain=np.ones(3,np.float32)
    if neutral.sum()>64:
        med=np.median(small[neutral],axis=0);gray=float(luma(med))
        gain=np.clip(gray/np.maximum(med,.01),.96,1.04).astype(np.float32)
    small=np.clip(small*gain,0,1)
    stops=np.array([-.65,0.,.65],np.float32)
    # Weight each virtual exposure for useful midtones, with a neutral bias.
    ls=linear(small)
    weights=np.stack([np.exp(-.5*((luma(np.clip(srgb(ls*2**s),0,1))-.5)/.23)**2) for s in stops],axis=2)
    weights[:,:,1]*=1.8
    weights=blur_masked(weights,small_alpha,5.)
    weights/=np.maximum(weights.sum(2,keepdims=True),1e-6)
    # Only a blurred highlight field is diffused; never blur the base detail.
    bright=np.clip((luma(small)-.65)/.35,0,1)**2
    glow=blur_masked(small*bright[...,None],small_alpha,2.)
    halo=blur_masked(bright[...,None].repeat(3,2),small_alpha,5.)
    result=image.copy()
    # Remap constant-size fields per strip, avoiding multiple full-map float RGB
    # scratch arrays. Pixel-center mapping is the same for every strip size.
    xx=(np.arange(w,dtype=np.float32)+.5)*sw/w-.5
    max_exposure_delta=0.
    for row in range(0,h,strip_rows):
        end=min(row+strip_rows,h);yy=(np.arange(row,end,dtype=np.float32)+.5)*sh/h-.5
        mx,my=np.meshgrid(xx,yy)
        def field(a):return cv2.remap(a,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        rgb=np.clip(image[row:end,:,2::-1].astype(np.float32)/255*gain,0,1)
        source_linear=linear(rgb);fusion=np.zeros_like(rgb);wt=field(weights)
        for i,s in enumerate(stops):fusion+=np.clip(srgb(source_linear*2**s),0,1)*wt[:,:,i,None]
        delta=(fusion-rgb)*params['exposure_strength']
        delta=np.clip(delta,-.055,.055);rgb+=delta
        valid=image[row:end,:,3]>0
        if valid.any():max_exposure_delta=max(max_exposure_delta,float(np.abs(delta[valid]).max()))
        light=luma(rgb)
        target=light+params['contrast']*(light-.5)*(1-np.abs(2*light-1))
        target+=params['shadow_lift']*np.clip(1-light/.3,0,1)**2
        upper=np.maximum((target-.78)/.22,0)
        target=np.where(target>.78,.78+.22*upper/(1+params['highlight_compression']*upper),target)
        rgb*=np.divide(target,np.maximum(light,1e-6))[...,None]
        # Restrained warm upper tones / cool shadows. Shared photographic color
        # adjustments, not a simulated claim about actual sun position.
        warm=np.clip((light-.25)/.65,0,1)**1.3*params['warmth']
        rgb+=warm[...,None]*np.array([1.,.35,-.65],np.float32)
        cool=np.clip((.3-light)/.3,0,1)**2*.004
        rgb+=cool[...,None]*np.array([-.3,0,1],np.float32)
        hsv=cv2.cvtColor(np.clip(rgb,0,1),cv2.COLOR_RGB2HSV)
        vegetation=np.exp(-((hsv[:,:,0]-105)/42)**2)*np.clip(hsv[:,:,1]/.25,0,1)
        factor=params['saturation']-params['green_reduction']*vegetation
        light=luma(rgb)[...,None];rgb=light+(rgb-light)*factor[...,None]
        rgb+=params['diffusion']*field(glow)*(1-np.clip(rgb,0,1))
        rgb+=params['halation']*field(halo)*np.array([1.,.3,.08],np.float32)*(1-np.clip(rgb,0,1))
        result[row:end,:,:3]=np.rint(np.clip(rgb[:,:,::-1],0,1)*255).astype(np.uint8)
    result[image[:,:,3]==0,:3]=0
    report=dict(look=look,parameters=params,virtual_exposure_stops=stops.tolist(),white_balance_rgb_gain=gain.tolist(),
        neutral_reference_pixels=int(neutral.sum()),max_virtual_exposure_adjustment_codes=255*max_exposure_delta,
        alpha_preserved=bool(np.array_equal(image[:,:,3],result[:,:,3])),
        operations=['masked neutral balance','virtual exposure blend','filmic tone curve','warm highlights/cool shadows',
                    'selective green saturation','highlight diffusion','subtle warm halation'],
        source_information='All exposures derived from one image; no additional measured HDR or scene geometry')
    return result,report
