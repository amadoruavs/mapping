"""Metal implementation of the user's final presentation pass.

OpenCV 5 fixed-point Lab/NLM/CLAHE conventions and Pillow 12 box/unsharp
conventions are retained. Small CPU-generated coefficient tables are uploaded
once. All image-sized arithmetic runs in Metal.
"""
import time
import cv2
import numpy as np
from .presentation import dimensions,PROFILES
from .cinematic_grade import PRESETS,REGION_PRESETS,REGION_DETECT


def grade_values(params):
    return [*params['lift'],*params['gamma'],*params['gain'],*params['shadow_tint'],*params['highlight_tint'],
        params['contrast'],params['saturation'],params.get('warmth',0),params['vignette_strength'],
        params.get('hue_shift_degrees',0),0,params.get('exposure',0),params.get('blacks',0),
        params.get('shadows',0),params.get('highlights',0),params.get('whites',0),0,0,0,0,0,0]


def color_tables(srgb,inverse):
    white=np.array([.950456,1.,1.088754])
    if not inverse:
        x=np.arange(256,dtype=np.float32)/255
        gamma=np.rint(2040*np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)) if srgb else np.arange(256)*8
        x=np.arange(3072,dtype=np.float32)/2040
        cbrt=np.rint(32768*np.where(x<(6/29)**3,x*(29/3)**3/116+16/116,np.cbrt(x)))
        coeff=np.rint(4096*np.array([[.412453,.357580,.180423],[.212671,.715160,.072169],[.019334,.119193,.950227]])/white[:,None])
        return np.concatenate((gamma,cbrt,coeff.ravel())).astype(np.int32)
    l=np.arange(256,dtype=np.float32);fy=l*100*16384/(255*116)+16*16384/116
    y=np.where(l<=20,np.rint(l*16384*20*9/(17*29**3)),np.rint(fy*fy*fy/16384**2))
    fy=np.where(l<=20,np.rint(16384*(16/116+l*5/(3*17*29))),np.rint(fy))
    yf=np.stack((y,fy),axis=1).ravel()
    i=np.arange(-8145,36864-8145,dtype=np.int64)
    xz=np.where(i<=3390,np.trunc(i*108/841)-int(16384*16/116)*108//841,(i*i//16384)*i//16384)
    x=np.arange(4096,dtype=np.float32)/4096
    gamma=np.rint(255*np.where(x<=.0031308,x*12.92,1.055*x**(1/2.4)-.055)) if srgb else np.floor(x*255)
    coeff=np.rint(4096*np.array([[3.240479,-1.53715,-.498535],[-.969256,1.875991,.041556],[.055648,-.204043,1.057311]])*white)
    return np.concatenate((yf,xz,gamma,coeff.ravel())).astype(np.int32)


def lanczos_coefficients(n,out):
    x=(np.arange(out)+.5)*n/out-.5;base=np.floor(x).astype(np.int32)
    offsets=np.arange(-3,5);d=x[:,None]-(base[:,None]+offsets)
    weights=np.sinc(d)*np.sinc(d/4);weights/=weights.sum(1)[:,None]
    return np.clip(base[:,None]+offsets,0,n-1).astype(np.int32),np.rint(weights*2048).astype(np.int32)


class PresentationMetal:
    def __init__(self,gpu,profile='standard'):
        self.gpu=gpu;self.tables={};self.timings={};self.profile=profile;self.params=PROFILES[profile]

    def chroma_clean(self,source):
        h,w=source.shape[:2];out=self.gpu.buffer(source.shape,np.uint8)
        self.gpu.dispatch('post_chroma_clean',[source,out],h*w,[w,h],[self.params['chroma']]);self.gpu.sync();return out

    def finish_tone(self,image,white_reduction=0.,vibrance_reduction=0.):
        if not np.isfinite(white_reduction) or not 0<=white_reduction<.5 or not np.isfinite(vibrance_reduction) or not 0<=vibrance_reduction<=1:
            raise ValueError('Invalid white or vibrance reduction')
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=4:raise ValueError('Tone adjustment expects BGRA8')
        if not white_reduction and not vibrance_reduction:return image
        source=self.gpu.shared_input(image);out=self.gpu.buffer(image.shape,np.uint8);count=np.prod(image.shape[:2])
        self.gpu.dispatch('post_finish_tone',[source,out],count,[count],[white_reduction,vibrance_reduction])
        return out.numpy(False)

    def color(self,source,srgb,inverse=False):
        key=(srgb,inverse)
        if key not in self.tables:self.tables[key]=self.gpu.upload(color_tables(*key))
        out=self.gpu.buffer(source.shape,np.uint8)
        self.gpu.dispatch('post_lab',[source,out,self.tables[key]],np.prod(source.shape[:2]),[np.prod(source.shape[:2]),int(inverse)])
        return out

    def denoise(self,source,optimized=True):
        h,w=source.shape[:2];lab=self.color(source,False)
        multiplier=(2**31-1)//(225*255)
        weights=np.rint(multiplier*np.exp(-np.arange(64)[None,:]*(64/49)/(4*np.array([[1],[2]]))))
        weights[weights<.001*multiplier]=0;weights=weights.astype(np.uint32)
        cutoffs=[(np.flatnonzero(row)[-1]+1)*64 for row in weights]
        filtered=self.gpu.buffer(source.shape,np.uint8)
        lut=self.gpu.upload(weights)
        if optimized:
            rows=self.gpu.buffer((h+6,w,2),np.uint16)
            sums=self.gpu.buffer(source.shape,np.uint32,zero=True);chroma=self.gpu.buffer((h,w),np.uint32,zero=True)
            for dy in range(-7,8):
                for dx in range(-7,8):
                    self.gpu.dispatch('post_nlm_rows',[lab,rows],w*(h+6),[w,h,dx,dy,*cutoffs])
                    self.gpu.dispatch('post_nlm_accumulate',[lab,rows,sums,chroma,lut],w*h,[w,h,dx,dy])
                self.gpu.sync()
            self.gpu.dispatch('post_nlm_finish',[lab,sums,chroma,filtered],w*h,[w*h])
        else:self.gpu.dispatch('post_nlm',[lab,filtered,lut],w*h,[w,h,*cutoffs])
        result=self.color(filtered,False,True);self.gpu.sync();return result

    def lanczos(self,source,shape):
        h,w=source.shape[:2];oh,ow=shape
        if (h,w)==(oh,ow):return source
        xi,xc=lanczos_coefficients(w,ow);yi,yc=lanczos_coefficients(h,oh)
        temp=self.gpu.buffer((h,ow,4),np.int32);out=self.gpu.buffer((oh,ow,4),np.uint8)
        self.gpu.dispatch('post_lanczos_x',[source,temp,self.gpu.upload(xi),self.gpu.upload(xc)],h*ow,[w,h,ow,oh])
        self.gpu.dispatch('post_lanczos_y',[temp,out,self.gpu.upload(yi),self.gpu.upload(yc),source],oh*ow,[w,h,ow,oh])
        self.gpu.sync();return out

    def gaussian(self,source,sigma):
        h,w=source.shape[:2];n=int(round(sigma*6+1))|1
        # OpenCV's uint8 Gaussian uses symmetric 8-bit fixed-point weights,
        # with rounding error carried toward the center (smooth.dispatch.cpp).
        raw=cv2.getGaussianKernel(n,sigma,cv2.CV_64F).ravel()*256
        fixed=np.zeros(n,np.uint32);error=0.
        for i in range(n//2):
            adjusted=raw[i]+error;fixed[i]=fixed[n-1-i]=round(adjusted);error=adjusted-fixed[i]
        fixed[n//2]=256-2*int(fixed[:n//2].sum());weights=self.gpu.upload(fixed)
        temp=self.gpu.buffer(source.shape,np.uint16);out=self.gpu.buffer(source.shape,np.uint8)
        self.gpu.dispatch('post_gauss_x',[source,temp,weights],h*w,[w,h,n//2])
        self.gpu.dispatch('post_gauss_y',[temp,out,weights],h*w,[w,h,n//2]);self.gpu.sync();return out

    def detail(self,source,fine=False):
        blur=self.gaussian(source,.45 if fine else 1.)
        out=self.gpu.buffer(source.shape,np.uint8);count=np.prod(source.shape[:2])
        self.gpu.dispatch('post_detail',[source,blur,out],count,[count,int(fine)],[self.params['fine' if fine else 'detail']]);self.gpu.sync();return out

    def clahe(self,source):
        h,w=source.shape[:2];lab=self.color(source,True);out=self.gpu.buffer(source.shape,np.uint8)
        # OpenCV extends BOTH axes when either dimension isn't divisible by 8.
        tw,th=(w//8,h//8) if w%8==0 and h%8==0 else ((w+8-w%8)//8,(h+8-h%8)//8)
        lut=self.gpu.buffer((64,256),np.uint8)
        self.gpu.dispatch('post_clahe_lut',[lab,lut],64*256,[w,h,tw,th],group=256)
        self.gpu.dispatch('post_clahe_apply',[lab,out,lut],w*h,[w,h,tw,th])
        graded=self.color(out,True,True);result=self.gpu.buffer(source.shape,np.uint8)
        self.gpu.dispatch('post_clahe_blend',[source,graded,result],w*h,[w*h],[self.params['clahe']]);self.gpu.sync();return result

    def enhance(self,image,scale=1.,stages=None):
        ow,oh=dimensions(image,scale);gpu=self.gpu
        if image.shape[2]==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
        src=gpu.shared_input(image);self.timings={}
        for name,fn in ([('chroma',self.chroma_clean)] if self.params['chroma'] else [])+[('lanczos',lambda s:self.lanczos(s,(oh,ow))),
                        ('detail',self.detail),('clahe',self.clahe),('fine',lambda s:self.detail(s,True))]:
            started=time.perf_counter();src=fn(src);gpu.sync();self.timings[name]=time.perf_counter()-started
            if stages is not None:stages[name]=src.numpy()[:,:,:3]
        started=time.perf_counter();gpu.dispatch('post_contrast',[src],ow*oh,[ow*oh],[self.params['contrast'],self.params['offset']]);gpu.sync()
        self.timings['contrast']=time.perf_counter()-started
        if stages is not None:stages['contrast']=src.numpy()[:,:,:3]
        return src.numpy(False)

    def pillow_blur(self,source,radius):
        if radius<=0:return source
        sigma2=np.float32(radius*radius/3);length=np.float32(np.sqrt(12.*sigma2+1));l=np.float32(np.floor((length-1)/2))
        a=np.float32((2*l+1)*(l*(l+1)-3*sigma2));a=np.float32(a/(6*(sigma2-(l+1)*(l+1))))
        r=np.float32(l+a);n=int(r);ww=int(np.float32(1<<24)/np.float32(r*2+1));fw=((1<<24)-(n*2+1)*ww)//2
        h,w=source.shape[:2];src=source
        # Ping-pong two allocations, encode all six dependent passes together.
        targets=[self.gpu.buffer(source.shape,np.uint8) for _ in range(2)];step=0
        for axis in (0,1):
            for _ in range(3):
                out=targets[step%2];step+=1;count=((h+127)//128)*w if axis else ((w+127)//128)*h
                self.gpu.dispatch('post_box',[src,out],count,[w,h,axis,n,ww,fw]);src=out
        return src

    def grade(self,image,preset_name='pro_camera',overrides=None,*,region_preset=None,seed=2026,region_blur=6.):
        """Supplied recipes on Metal, with the original seeded NumPy grain stream.

        Classification, mask feathering, per-region recipes, vignette, grain
        addition and both Pillow unsharp passes stay on GPU. Only random-number
        generation (when requested) and small coefficient tables use the CPU.
        """
        params={**PRESETS[preset_name],**(overrides or {})}
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=4:raise ValueError('Metal grade expects BGRA8')
        source=self.gpu.shared_input(image);h,w=image.shape[:2];count=w*h
        if not np.isfinite(region_blur) or not 0<=region_blur<=128:raise ValueError('Region blur must be 0..128 pixels')
        values=grade_values(params);grain=params['grain']>0
        noise=self.gpu.buffer((h,w) if grain else 1,np.float32)
        if grain:
            rng=np.random.default_rng(seed);target=noise.view()
            for y in range(0,h,128):target[y:y+128]=rng.normal(0,params['grain'],target[y:y+128].shape)
        before=self.gpu.buffer(image.shape,np.uint8)
        if region_preset:
            regions=REGION_PRESETS[region_preset];names=list(REGION_DETECT)
            detect=self.gpu.upload(np.array([[(-1 if v is None else v) for v in REGION_DETECT[name]] for name in names],np.float32))
            recipes=self.gpu.upload(np.array([grade_values(regions[name]) for name in names+['default']],np.float32))
            scores=self.gpu.buffer(image.shape,np.uint8)
            self.gpu.dispatch('post_region_scores',[source,scores,detect],count,[w,h],values)
            scores=self.pillow_blur(scores,region_blur)
            self.gpu.dispatch('post_grade_regional',[source,before,noise,scores,recipes],count,[w,h,grain],values)
        else:self.gpu.dispatch('post_grade',[source,before,noise],count,[w,h,grain],values)
        current=before;boundary=None
        for radius,key in ((30,'clarity'),(1.5,'sharpness')):
            if params.get(key,0)<=0:continue
            blur=self.pillow_blur(current,radius)
            if boundary is None:boundary=blur
            out=self.gpu.buffer(image.shape,np.uint8)
            self.gpu.dispatch('post_unsharp',[current,blur,out],count,[count,int(params[key])]);current=out
        self.gpu.dispatch('post_grade_alpha',[before,boundary or before,current],count,[count]);self.gpu.sync()
        return current.numpy(False)
