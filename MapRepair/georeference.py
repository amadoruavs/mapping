"""Explicit CRS and pixel-center georeferencing; GPS is a diagnostic/hint."""
import json
import math
from pathlib import Path
import re
import numpy as np
from PIL import Image


def read_gps(path):
    result=None
    try:
        with Image.open(path) as im:
            gps=im.getexif().get_ifd(34853)
            def angle(key,ref):
                d,m,s=map(float,gps[key]);value=d+m/60+s/3600
                return -value if str(gps.get(ref,'')).strip("b'\"") in ('S','W') else value
            if 2 in gps and 4 in gps:
                result=dict(lat=angle(2,1),lon=angle(4,3),source='EXIF',trusted=gps.get(9) not in ('V',b'V'))
        with open(path,'rb') as f:header=f.read(2_000_000).decode('latin1',errors='ignore')
        # SIYI's XMP often carries coordinates while the EXIF GPS IFD is empty.
        values={}
        for key in ('GpsLatitude','GpsLongitude','GPSLatitude','GPSLongitude'):
            m=re.search(r'(?:\w+:)?'+key+r'\s*=\s*["\']([-+\d.eE]+)',header)
            if not m:m=re.search(r'<(?:\w+:)?'+key+r'>([-+\d.eE]+)<',header)
            if m:values[key.lower()]=float(m.group(1))
        if 'gpslatitude' in values and 'gpslongitude' in values:
            result=dict(lat=values['gpslatitude'],lon=values['gpslongitude'],source='XMP',trusted=True)
        if result and 'NO GPS connected' in header:result['trusted']=False;result['note']='Camera reports NO GPS connected'
    except (OSError,ValueError,KeyError,ZeroDivisionError,TypeError):pass
    if result and (not np.isfinite([result['lat'],result['lon']]).all() or not -90<=result['lat']<=90 or not -180<=result['lon']<=180):return None
    return result


class GeoReference:
    def __init__(self,matrix,crs,source):
        self.matrix=np.asarray(matrix,float);self.crs=crs;self.source=source
        if self.matrix.shape!=(3,3) or not np.isfinite(self.matrix).all() or abs(np.linalg.det(self.matrix))<1e-25:
            raise ValueError('Invalid map geotransform')
        self.inverse=np.linalg.inv(self.matrix);self.to_world=self.to_ll=None
        if crs!='EPSG:4326':
            from pyproj import Transformer
            self.to_world=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
            self.to_ll=Transformer.from_crs(crs,'EPSG:4326',always_xy=True)

    def pixel(self,lat,lon):
        x,y=self.to_world.transform(lon,lat) if self.to_world else (lon,lat)
        p=self.inverse@np.array([x,y,1.]);return p[:2]/p[2]

    def latlon(self,pixel):
        v=self.matrix@np.r_[pixel,1.];x,y=v[:2]/v[2]
        lon,lat=self.to_ll.transform(x,y) if self.to_ll else (x,y)
        return np.array([lat,lon])

    def metric_jacobian(self,pixel):
        ll=self.latlon(pixel);basis=np.stack([self.latlon(np.asarray(pixel)+d)-ll for d in ((1,0),(0,1))],axis=1)
        # Rows become east, north metres; retains both rotated/sheared axes.
        return np.array([[0,111320*math.cos(math.radians(ll[0]))],[111320,0]])@basis

    def error_m(self,a,b):return float(np.linalg.norm(self.metric_jacobian(b)@(np.asarray(a)-b)))

    def export(self,directory,name,scale=2,world_suffix='.pgw'):
        pixel=np.array([[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1.]])
        matrix=self.matrix@pixel;directory=Path(directory)
        (directory/(name+world_suffix)).write_text('\n'.join(f'{v:.16g}' for v in [matrix[0,0],matrix[1,0],matrix[0,1],matrix[1,1],matrix[0,2],matrix[1,2]])+'\n')
        from pyproj import CRS
        (directory/(name+'.prj')).write_text(CRS.from_user_input(self.crs).to_wkt())
        return dict(pixel_to_world=matrix.tolist(),crs=self.crs,source=self.source,pixel_reference='centers')


def load_georeference(map_path,shape,bounds=None,crs=None):
    path=Path(map_path);h,w=shape[:2]
    if bounds is not None:
        a,b,c,d=map(float,bounds)
        if not (-180<=a<c<=180 and -90<b<d<90):raise ValueError('Bounds must be MIN_LON MIN_LAT MAX_LON MAX_LAT')
        sx=(c-a)/w;sy=-(d-b)/h
        return GeoReference([[sx,0,a+sx/2],[0,sy,d+sy/2],[0,0,1]],'EPSG:4326','explicit map bounds')
    if path.suffix.lower() in ('.tif','.tiff'):
        import rasterio
        with rasterio.open(path) as dataset:
            if dataset.crs is not None:
                a=dataset.transform
                return GeoReference([[a.a,a.b,a.c+.5*(a.a+a.b)],
                    [a.d,a.e,a.f+.5*(a.d+a.e)],[0,0,1]],crs or dataset.crs.to_wkt(),str(path)+' embedded GeoTIFF')
    # ISM's map geometry uses a local tangent approximation with an explicit origin.
    for name in ('presentation_geometry.json','map_geometry.json'):
        p=path.parent/name
        if not p.exists():continue
        if name=='map_geometry.json' and path.name not in ('map_neutral.png','map_graded.png','map_enhanced.png'):continue
        data=json.loads(p.read_text())
        if [data.get('width'),data.get('height')]!=[w,h] or not data.get('georeferenced'):continue
        if data.get('applies_to') and path.name not in data['applies_to']:continue
        origin=data.get('origin_latlon')
        if origin is None and (path.parent/'map_geometry.json').exists():
            origin=json.loads((path.parent/'map_geometry.json').read_text()).get('origin_latlon')
        if origin is None:continue
        lat,lon=origin;phi=math.radians(lat);e2=6.69437999014e-3;den=math.sqrt(1-e2*math.sin(phi)**2)
        n=6378137/den;m=6378137*(1-e2)/den**3
        ll=np.array([[180/math.pi/(n*math.cos(phi)),0,lon],[0,180/math.pi/m,lat],[0,0,1.]])
        return GeoReference(ll@np.array(data['map_to_world']),'EPSG:4326',str(p))
    for ext in ('.wld','.pgw','.pngw','.tfw','.tifw','.jgw','.jpgw'):
        wf=path.with_suffix(ext)
        if not wf.exists():continue
        values=[float(x) for x in wf.read_text().split()]
        if len(values)!=6:raise ValueError('World file must contain six values')
        projection=crs
        if projection is None and path.with_suffix('.prj').exists():projection=path.with_suffix('.prj').read_text()
        if projection is None:raise ValueError('World file CRS is unknown. Set MAP_CRS (e.g. EPSG:32615 or EPSG:4326).')
        a,d,b,e,c,f=values;return GeoReference([[a,b,c],[d,e,f],[0,0,1]],projection,str(wf))
    return None
