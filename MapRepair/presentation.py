"""User's post-map OpenCV recipe, retained as the numerical reference."""
import cv2
import numpy as np

PROFILES={
    'natural_detail':dict(detail=.25,fine=.05,clahe=.10,contrast=1.,offset=0.,chroma=.7),
    'standard':dict(detail=1.3,fine=.35,clahe=.75,contrast=1.02,offset=-1.,chroma=0.),
    'siyi_low_light':dict(detail=.65,fine=.15,clahe=.35,contrast=1.,offset=0.,chroma=.7),
}


def chroma_clean_reference(image,strength=.7):
    """3x3 edge-aware chroma smoothing; preserve each pixel's luma and alpha."""
    from cinematic_grade import luma
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


_chroma_full=chroma_clean_reference


def chroma_clean_reference(image,strength=.7):
    """Exact one-pixel-halo strips bound chroma scratch RAM on large maps."""
    result=image.copy();h=len(image)
    for y in range(0,h,128):
        end=min(h,y+128);lo=max(0,y-1);hi=min(h,end+1)
        result[y:end]=_chroma_full(image[lo:hi],strength)[y-lo:end-lo]
    return result


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
