"""TIFF band/mask-aware input and JPEG export at the exact input raster extent."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def to_byte(values,value_range=None):
    if values.dtype==np.uint8 and value_range is None:return values
    if value_range is None:
        if not np.issubdtype(values.dtype,np.unsignedinteger):
            raise ValueError('Set TIFF_VALUE_RANGE=(minimum, maximum) for float/signed TIFF color samples')
        value_range=(0,np.iinfo(values.dtype).max)
    lo,hi=map(float,value_range)
    if not np.isfinite([lo,hi]).all() or hi<=lo:raise ValueError('Invalid TIFF_VALUE_RANGE')
    return np.rint(np.clip((np.nan_to_num(values.astype(np.float32),nan=lo)-lo)/(hi-lo),0,1)*255).astype(np.uint8)


def read_map(path,value_range=None):
    path=Path(path)
    if path.suffix.lower() not in ('.tif','.tiff'):
        image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim!=3 or image.dtype!=np.uint8 or image.shape[2] not in (3,4):
            raise ValueError('Map must be RGB/RGBA, or a TIFF with tagged color bands')
        if image.shape[2]==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
        return image,dict(reader='OpenCV',width=image.shape[1],height=image.shape[0],alpha='input alpha or fully opaque')
    import rasterio
    from rasterio.enums import ColorInterp
    with rasterio.open(path) as ds:
        roles=list(ds.colorinterp)
        if all(c in roles for c in (ColorInterp.red,ColorInterp.green,ColorInterp.blue)):
            rgb=[roles.index(c)+1 for c in (ColorInterp.red,ColorInterp.green,ColorInterp.blue)]
        elif ColorInterp.gray in roles:
            rgb=[roles.index(ColorInterp.gray)+1]*3
        elif ds.count==3 and all(c==ColorInterp.undefined for c in roles):rgb=[1,2,3]
        else:raise ValueError('TIFF must identify RGB/gray color bands; an untagged fourth band is not assumed to be alpha')
        image=np.empty((ds.height,ds.width,4),np.uint8)
        for c,band in enumerate(rgb):image[:,:,2-c]=to_byte(ds.read(band),value_range)
        # dataset_mask honors internal masks/nodata; alpha is combined with min
        # so a 50% alpha pixel stays 50%, rather than being multiplied twice.
        alpha=ds.dataset_mask()
        alpha_band=roles.index(ColorInterp.alpha)+1 if ColorInterp.alpha in roles else None
        if alpha_band:
            channel=ds.read(alpha_band)
            alpha=np.minimum(alpha,to_byte(channel,(0,1) if np.issubdtype(channel.dtype,np.floating) else None))
        image[:,:,3]=alpha;image[alpha==0,:3]=0
        return image,dict(reader='rasterio/GDAL',width=ds.width,height=ds.height,rgb_bands=rgb,
            alpha_band=alpha_band,alpha='dataset validity mask intersected with tagged alpha',
            color_interpretation=[c.name for c in roles],source_dtypes=list(ds.dtypes),
            value_range=value_range,crs=str(ds.crs) if ds.crs else None,
            pixel_corner_transform=list(ds.transform)[:6])


def flatten_original_extent(final,input_alpha,scale=2,background=(255,255,255)):
    """Crop to original raster bounds and composite using the ORIGINAL alpha.

    Returns BGR8. A map with internal holes keeps those holes; valid black
    pixels are not mistaken for empty areas. No automatic lasso/tight trim.
    """
    h,w=input_alpha.shape;oh,ow=h*scale,w*scale
    if final.shape[0]<oh or final.shape[1]<ow:raise ValueError('Final image is smaller than the original map extent')
    result=np.empty((oh,ow,3),np.uint8);bg=np.asarray(background,np.float32)[::-1]
    if bg.shape!=(3,) or not np.isfinite(bg).all() or np.any((bg<0)|(bg>255)):raise ValueError('Invalid JPEG background RGB')
    for y in range(0,oh,128):
        end=min(y+128,oh)
        alpha=np.repeat(input_alpha[np.arange(y,end)//scale],scale,axis=1).astype(np.float32)/255
        result[y:end]=np.rint(final[y:end,:ow,:3].astype(np.float32)*alpha[...,None]+bg*(1-alpha[...,None])).clip(0,255).astype(np.uint8)
    return result


def save_jpeg(path,bgr,quality=95):
    # 4:4:4 avoids chroma subsampling bleeding color across small map details.
    Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(path,quality=quality,subsampling=0)


def save_alpha_tiff(path,alpha,geo=None,scale=2):
    """Lossless sidecar: JPEG itself has no alpha channel. Write bounded strips."""
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window
    h,w=alpha.shape;kwargs={}
    if geo is not None:
        p=np.array([[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1.]])
        a=geo.matrix@p
        kwargs.update(crs=geo.crs,transform=Affine(a[0,0],a[0,1],a[0,2]-.5*(a[0,0]+a[0,1]),
            a[1,0],a[1,1],a[1,2]-.5*(a[1,0]+a[1,1])))
    with rasterio.open(path,'w',driver='GTiff',width=w*scale,height=h*scale,count=1,dtype='uint8',
                       compress='deflate',tiled=True,BIGTIFF='IF_SAFER',**kwargs) as ds:
        for y in range(0,h*scale,128):
            end=min(y+128,h*scale)
            block=np.repeat(alpha[np.arange(y,end)//scale],scale,axis=1)
            ds.write(block,1,window=Window(0,y,w*scale,end-y))
        ds.update_tags(description='Original input transparency/validity at output resolution. 0=transparent, 255=opaque.')
