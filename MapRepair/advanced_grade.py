"""Non-destructive, strip-based grading of the completed lossless BGRA master."""
import io
import json
import threading
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from parallel_runtime import bounded_map, worker_count
from parallel_runtime.pixels import composite_bgra

LOCK = threading.Lock()
PREVIEW_LOCK = threading.Lock()
CACHE_LOCK = threading.Lock()
PREVIEW_CACHE = OrderedDict()
RANGES = {'exposure': (-3, 3, 0), 'temperature': (-100, 100, 0),
          'tint': (-100, 100, 0), 'contrast': (-100, 100, 0),
          'highlights': (-100, 100, 0), 'shadows': (-100, 100, 0),
          'whites': (-100, 100, 0), 'blacks': (-100, 100, 0),
          'saturation': (-100, 100, 0), 'vibrance': (-100, 100, 0),
          'sharpen': (0, 150, 0), 'radius': (.3, 3, 1), 'threshold': (0, 20, 3)}

RANGES.update(gamma=(.2,3,1), clarity=(-100,100,0), texture=(-100,100,0),
              hue=(-180,180,0), fade=(0,100,0), detail_protection=(0,100,100))
BANDS={'red':0,'orange':30,'yellow':60,'green':120,'aqua':180,'blue':240,'purple':270,'magenta':300}
for band in BANDS:
    for control in ('hue','saturation','luminance'):
        RANGES[f'{band}_{control}']=(-60,60,0) if control=='hue' else (-100,100,0)
for zone in ('shadow','midtone','highlight'):
    for color in ('red','green','blue'):RANGES[f'{zone}_{color}']=(-100,100,0)
IDENTITY=[[0.,0.],[1.,1.]]

def curves(values):
    if not isinstance(values,dict) or set(values)-{'master','red','green','blue'}:
        raise ValueError('Unknown curve channel')
    result={}
    for channel in ('master','red','green','blue'):
        try:a=np.asarray(values.get(channel,IDENTITY),dtype=np.float64)
        except (TypeError,ValueError):raise ValueError('Invalid curve points')
        if (a.ndim!=2 or a.shape[1]!=2 or not 2<=len(a)<=16 or not np.isfinite(a).all()
            or np.any(a<0) or np.any(a>1) or a[0,0]!=0 or a[-1,0]!=1
            or np.any(np.diff(a[:,0])<1/1024)):
            raise ValueError('Curves need 2–16 ordered points, including input endpoints 0 and 1')
        result[channel]=a.tolist()
    return result

def curve_lut(points):
    # Shape-preserving cubic Hermite interpolation; no overshooting around knots.
    a=np.asarray(points);x,y=a.T;h=np.diff(x);d=np.diff(y)/h;m=np.zeros(len(x))
    m[0]=d[0];m[-1]=d[-1]
    for i in range(1,len(x)-1):
        if d[i-1]*d[i]>0:
            w1=2*h[i]+h[i-1];w2=h[i]+2*h[i-1]
            m[i]=(w1+w2)/(w1/d[i-1]+w2/d[i])
    grid=np.linspace(0,1,4097);i=np.clip(np.searchsorted(x,grid,side='right')-1,0,len(x)-2)
    t=(grid-x[i])/h[i];t2=t*t;t3=t2*t
    return np.clip((2*t3-3*t2+1)*y[i]+(t3-2*t2+t)*h[i]*m[i]+(-2*t3+3*t2)*y[i+1]+(t3-t2)*h[i]*m[i+1],0,1).astype(np.float32)

def apply_curve(rgb,points):
    if all(x==y for x,y in points):return rgb
    lut=curve_lut(points);q=np.clip(rgb,0,1)*4096;i=np.minimum(q.astype(np.int32),4095)
    return lut[i]+(lut[i+1]-lut[i])*np.subtract(q,i,dtype=np.float32)

def halo(s,scale=1):
    sigma=max(s['radius'] if s['sharpen'] else 0,12 if s['clarity'] else 0,1.4 if s['texture'] else 0)*scale
    return max(16,int(np.ceil(sigma*4))+2)

def settings(values):
    if not isinstance(values, dict) or set(values) - (set(RANGES)|{'curves'}):
        raise ValueError('Unknown grading setting')
    result = {}
    for key, (low, high, default) in RANGES.items():
        try:value = float(values.get(key, default))
        except (TypeError,ValueError):raise ValueError('Invalid '+key)
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid ' + key)
        result[key] = value
    result['curves']=curves(values.get('curves',{}))
    return result

def grade(image, values, scale=1.):
    s = settings(values)
    if all(s[k] == v[2] for k, v in RANGES.items() if k not in ('radius', 'threshold', 'detail_protection')) and all(all(x==y for x,y in points) for points in s['curves'].values()):
        return image.copy()
    # White balance/exposure LUT avoids millions of per-pixel powers in previews.
    codes=np.arange(256,dtype=np.float32)[:,None]/255
    linear=np.where(codes<=.04045,codes/12.92,((codes+.055)/1.055)**2.4)
    gains=np.array([2**(s['temperature']/250),2**(-s['tint']/300),2**(-s['temperature']/250)],np.float32)
    linear=linear*gains*2**s['exposure']
    table=np.where(linear<=.0031308,linear*12.92,1.055*np.maximum(linear,0)**(1/2.4)-.055).astype(np.float32)
    rgb=cv2.LUT(np.ascontiguousarray(image[:,:,2::-1]),table[:,None,:])
    y = np.clip(rgb @ np.array([.2126,.7152,.0722], np.float32), 0, 1)
    shadow = (1-y) ** 2; highlight = y ** 2
    delta = (s['shadows'] * shadow + s['highlights'] * highlight) / 400
    delta += (s['whites'] * y ** 5 + s['blacks'] * (1-y) ** 5) / 500
    rgb += delta[:, :, None]
    rgb = (rgb - .5) * 2 ** (s['contrast']/100) + .5
    if s['gamma']!=1:rgb=np.maximum(rgb,0)**(1/s['gamma'])
    rgb=apply_curve(rgb,s['curves']['master']).astype(np.float32)
    for i,key in enumerate(('red','green','blue')):rgb[:,:,i]=apply_curve(rgb[:,:,i],s['curves'][key])
    y=np.clip(rgb @ np.array([.2126,.7152,.0722],np.float32),0,1)
    for zone,weight in [('shadow',(1-y)**2),('midtone',4*y*(1-y)),('highlight',y*y)]:
        if not any(s[f'{zone}_{c}'] for c in ('red','green','blue')):continue
        balance=np.array([s[f'{zone}_{c}'] for c in ('red','green','blue')],np.float32)/400
        rgb+=weight[:,:,None]*balance
    if s['hue'] or any(s[f'{band}_{key}'] for band in BANDS for key in ('hue','saturation','luminance')):
        hls=cv2.cvtColor(np.clip(rgb,0,1),cv2.COLOR_RGB2HLS);h,l,sat0=cv2.split(hls)
        changes=[np.zeros_like(h) for _ in range(3)];names=list(BANDS);centers=list(BANDS.values())
        for i,band in enumerate(names):
            if not any(s[f'{band}_{k}'] for k in ('hue','saturation','luminance')):continue
            distance=(h-centers[i]+180)%360-180
            left=(centers[i]-centers[i-1])%360;right=(centers[(i+1)%8]-centers[i])%360
            weight=np.clip(1-np.abs(distance)/np.where(distance<0,left,right),0,1)*np.minimum(sat0*4,1)
            for j,key in enumerate(('hue','saturation','luminance')):changes[j]+=weight*s[f'{band}_{key}']
        h=(h+s['hue']+changes[0])%360
        sat=np.clip(sat0*(1+changes[1]/100),0,1);delta=changes[2]/100
        l=np.clip(l+delta*np.where(delta>0,1-l,l),0,1)
        rgb=cv2.cvtColor(cv2.merge([h.astype(np.float32),l.astype(np.float32),sat.astype(np.float32)]),cv2.COLOR_HLS2RGB)
    if s['fade']:rgb=rgb*(1-s['fade']/200)+s['fade']/400
    y = rgb @ np.array([.2126,.7152,.0722], np.float32)
    chroma = rgb.max(2) - rgb.min(2)
    sat = (1 + s['saturation']/100) * (1 + s['vibrance']/100 * (1-np.clip(chroma, 0, 1)))
    rgb = y[:, :, None] + (rgb-y[:, :, None]) * sat[:, :, None]
    if s['sharpen'] or s['clarity'] or s['texture']:
        alpha=image[:,:,3].astype(np.float32)/255
        detail_delta=np.zeros_like(y)
        for amount,radius,threshold,limit in [(s['clarity'],12,0,.15),(s['texture'],1.4,0,.08),(s['sharpen'],s['radius'],s['threshold']/255,.08)]:
            if not amount:continue
            sigma=max(.15,radius*scale);size=2*int(np.ceil(sigma*4))+1
            weight=cv2.GaussianBlur(alpha,(size,size),sigma)
            low=cv2.GaussianBlur(y*alpha,(size,size),sigma)/np.maximum(weight,1e-6)
            detail=y-low;detail=np.sign(detail)*np.maximum(np.abs(detail)-threshold,0)
            detail_delta+=np.clip(detail*amount/100,-limit,limit)
        if s['detail_protection']:
            kernel=np.ones((5,5),np.uint8)
            low=cv2.erode(np.where(alpha>0,y,1).astype(np.float32),kernel)
            high=cv2.dilate(np.where(alpha>0,y,0).astype(np.float32),kernel)
            protected=np.clip(y+detail_delta,low,high)-y
            detail_delta+=(protected-detail_delta)*(s['detail_protection']/100)
        rgb+=detail_delta[:,:,None]
    out = image.copy()
    out[:, :, :3] = (np.clip(rgb[:, :, ::-1],0,1)*255+.5).astype(np.uint8)
    out[out[:, :, 3] == 0, :3] = 0
    return out

def preview_source(directory,source,w,h):
    paths=[directory/'grade_master.bgra',directory/'final_2x.jpg',directory/'final_2x_alpha.tif']
    key=(str(directory.resolve()),w,h,tuple((p.stat().st_size,p.stat().st_mtime_ns) if p.exists() else None for p in paths))
    with CACHE_LOCK:
        cached=PREVIEW_CACHE.get(key)
        if cached is not None:PREVIEW_CACHE.move_to_end(key);return cached
    if source is None:return None
    scale=min(1.,1920/w,1080/h);pw=max(1,round(w*scale));ph=max(1,round(h*scale))
    if (pw,ph)==(w,h):small=np.array(source)
    else:
        # Downsample in narrow source strips to bound working float memory.
        small=np.empty((ph,pw,4),np.uint8)
        for y in range(ph):
            lo=int(np.floor(y*h/ph));hi=int(np.ceil((y+1)*h/ph))
            rows=np.asarray(source[lo:hi],np.float32);weights=np.minimum(np.arange(lo,hi)+1,(y+1)*h/ph)-np.maximum(np.arange(lo,hi),y*h/ph)
            alpha=rows[:,:,3:4]/255;rows[:,:,:3]*=alpha
            row=np.sum(rows*weights[:,None,None],axis=0,dtype=np.float32)/(h/ph)
            reduced=cv2.resize(row[None],(pw,1),interpolation=cv2.INTER_AREA)
            reduced[:,:,:3]/=np.maximum(reduced[:,:,3:4]/255,1e-6)
            small[y]=np.clip(reduced[0]+.5,0,255).astype(np.uint8)
    small.flags.writeable=False
    result=(small,scale)
    with CACHE_LOCK:
        PREVIEW_CACHE[key]=result
        while len(PREVIEW_CACHE)>2:PREVIEW_CACHE.popitem(last=False)
    return result

def load_source(directory,w,h):
    master = directory/'grade_master.bgra'
    if master.exists():
        return np.memmap(master, mode='r', dtype=np.uint8, shape=(h,w,4))
    source = cv2.imread(str(directory/'final_2x.jpg'), cv2.IMREAD_COLOR)
    if source is None: raise ValueError('Cannot load finished image')
    source = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
    mask = directory/'final_2x_alpha.tif'
    if mask.exists():
        import rasterio
        with rasterio.open(mask) as ds: source[:,:,3] = ds.read(1)
    return source

def auto_grade(directory):
    from adaptive_finish import suggest_grade
    directory=Path(directory)
    geometry=json.loads((directory/'final_geometry.json').read_text())
    source=load_source(directory,geometry['width'],geometry['height'])
    # This is an additional editable pass on an already-finished map.
    result=suggest_grade(source,already_finished=True)
    result['settings']=settings(result['settings'])
    result['lossless_source']=(directory/'grade_master.bgra').exists()
    return result

def render(directory, values, preview=False, output_format='jpeg', crop=None):
    if output_format not in ('jpeg','png'):raise ValueError('Unknown export format')
    s = settings(values)
    directory = Path(directory)
    geometry = json.loads((directory/'final_geometry.json').read_text())
    w,h = geometry['width'],geometry['height']
    if crop is not None:
        if (not preview or not isinstance(crop,list) or len(crop)!=2
            or any(not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1 for v in crop)):
            raise ValueError('Detail crop requires a normalized [x,y] preview position')
        source=load_source(directory,w,h)
        cw,ch=min(640,w),min(480,h)
        x=max(0,min(w-cw,round(crop[0]*w-cw/2)));y=max(0,min(h-ch,round(crop[1]*h-ch/2)))
        pad=halo(s);left=max(0,x-pad);top=max(0,y-pad)
        block=grade(source[top:min(h,y+ch+pad),left:min(w,x+cw+pad)],s)
        return encode(block[y-top:y-top+ch,x-left:x-left+cw],geometry,False,'png')
    if preview:
        cached=preview_source(directory,None,w,h)
        if cached is not None:
            small,scale=cached
            return encode(grade(small,s,scale),geometry,True)
    source=load_source(directory,w,h)
    if preview:
        small,scale=preview_source(directory,source,w,h)
        output=grade(small,s,scale)
    else:
        output=np.empty((h,w,4),np.uint8)
        def render_strip(y):
            end=min(h,y+128); pad=halo(s); lo=max(0,y-pad); hi=min(h,end+pad)
            return y,grade(source[lo:hi],s)[y-lo:end-lo]
        # Bound retained results and temporary float arrays; writes stay ordered.
        for y,strip in bounded_map(render_strip,range(0,h,128),workers=worker_count(),bytes_per_item=max(1,w*160*256)):
            output[y:y+len(strip)]=strip
    return encode(output,geometry,preview,output_format)

def encode(output,geometry,preview,output_format='jpeg'):
    if output_format=='png' and not preview:
        ok,encoded=cv2.imencode('.png',output,[cv2.IMWRITE_PNG_COMPRESSION,3])
        if not ok:raise ValueError('PNG encoding failed')
        return encoded.tobytes()
    background=np.array(geometry.get('jpeg_background_rgb',[255,255,255]))[::-1]
    flat=composite_bgra(output,background,backend='cpu' if preview else None)
    stream=io.BytesIO()
    Image.fromarray(flat[:,:,::-1]).save(stream,format='JPEG',quality=90 if preview else 97,subsampling=0)
    return stream.getvalue()
