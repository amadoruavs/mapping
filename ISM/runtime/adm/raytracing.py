"""Real Metal triangle intersections; measured geometry is never inferred from RGB.

The benchmark supplies only its flat map plane. That traces to a uniform light
ratio and preserves existing detail/shadows. A caller can supply registered
triangle soup and diffuse reflectance for actual occlusion / two-bounce shading.
Photographs already contain lighting, so applying an irradiance ratio to them
is an artistic relighting approximation, not recovered physical reflectance.
"""
import ctypes as C
import time
import numpy as np
from .metal import Params
from .lighting import sun_vector


def flat_plane(extent=1000.):
    return np.array([[[-extent,-extent,0],[extent,-extent,0],[extent,extent,0]],
                     [[-extent,-extent,0],[extent,extent,0],[-extent,extent,0]]],np.float32)


def trace(gpu,vertices,materials=None,shape=(86,128),extent=(1.,1.),camera_height=1.,
          elevation=3.,azimuth=255.,samples=8,bounces=2):
    """XYZ coordinates: X east, Y north, Z up; camera centered over the map."""
    vertices=np.asarray(vertices,np.float32);h,w=shape
    if vertices.ndim!=3 or vertices.shape[1:]!=(3,3) or not 0<len(vertices)<=1_000_000 or not np.isfinite(vertices).all():
        raise ValueError('Expected finite triangle vertices with shape (N,3,3)')
    if any(type(n) is not int or not 1<=n<=512 for n in shape):raise ValueError('Ray grid dimensions must be integers from 1 to 512')
    if type(samples) is not int or not 1<=samples<=64 or type(bounces) is not int or not 1<=bounces<=2:raise ValueError('Use 1..64 samples and 1..2 bounces')
    if len(extent)!=2 or not np.isfinite([*extent,camera_height,elevation,azimuth]).all() or min(extent)<=0 or not 1<=elevation<=89 or camera_height<=vertices[:,:,2].max():
        raise ValueError('Invalid camera extent/height or sun; camera must be above geometry')
    area=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)
    if np.any(area<1e-12):raise ValueError('Degenerate triangles')
    if materials is None:materials=np.full((len(vertices),3),.5,np.float32)
    materials=np.asarray(materials,np.float32)
    if materials.shape!=(len(vertices),3) or not np.isfinite(materials).all() or np.any((materials<0)|(materials>1)):
        raise ValueError('Diffuse reflectance must be finite RGB in [0,1] per triangle')
    packed=np.zeros((len(vertices)*3,4),np.float32);packed[:,:3]=vertices.reshape(-1,3)
    colors=np.zeros((len(vertices),4),np.float32);colors[:,:3]=materials
    gpu.thread_check();v=gpu.upload(packed);m=gpu.upload(colors);out=gpu.buffer((h,w,4))
    params=Params([w,h,samples,bounces],[*sun_vector(elevation,azimuth),*extent,camera_height])
    gpu.check(gpu.lib.adm_trace(gpu.handle,v.handle,m.handle,out.handle,len(vertices),C.byref(params)))
    return out


def relight_map(image,gpu,strength=.18):
    """Trace the only scene geometry available in a planar ADM map."""
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=4:raise ValueError('Expected BGRA8 map')
    if not np.isfinite(strength) or not 0<=strength<=1:raise ValueError('Ray light strength must be from 0 to 1')
    started=time.perf_counter();h,w=image.shape[:2];lh=max(1,round(128*h/w));lw=128
    if lh>512:lw=max(1,round(512*w/h));lh=512
    field=trace(gpu,flat_plane(),shape=(lh,lw),extent=(w/max(w,h),h/max(w,h)))
    irradiance=field.numpy(False);traced=time.perf_counter()
    source=gpu.shared_input(image);output=gpu.buffer(image.shape,np.uint8)
    gpu.dispatch('post_ray_light',[source,field,output],w*h,[w,h,lw,lh],[strength]);result=output.numpy(False)
    report=dict(backend='Metal hardware triangle ray tracing',kernel='cinematic_path_trace',geometry='flat map plane; 2 triangles',
        measured_elevation_available=False,invented_geometry=False,grid=[lw,lh],samples=8,max_diffuse_bounces=2,
        sun_elevation_deg=3,sun_azimuth_deg=255,sun_setting='Artistic low-sun assumption, not a timestamp-derived solar position',
        irradiance_ratio_min=float(irradiance[:,:,:3].min()),irradiance_ratio_max=float(irradiance[:,:,:3].max()),
        trace_and_build_seconds=traced-started,total_seconds=time.perf_counter()-started,strength=strength,
        limitation='No terrain or building shadows without a registered surface. A flat plane gives uniform light; original photographed shadows remain.')
    return result,report
