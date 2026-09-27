"""All EXIF / XMP assumptions live here; adapt this module for a flight log."""
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import xml.etree.ElementTree as ET
from PIL import Image, ExifTags

# Only absolute camera headings belong here. Flight yaw is not gimbal yaw.
XMP_FIELDS = {
    'latitude': ['GpsLatitude', 'GPSLatitude'],
    'longitude': ['GpsLongitude', 'GpsLongtitude', 'GPSLongitude'],
    'yaw_deg': ['GimbalYawDegree', 'CameraYawDegree'],
    'pitch_deg': ['GimbalPitchDegree', 'CameraPitchDegree'],
    'roll_deg': ['GimbalRollDegree', 'CameraRollDegree'],
}

@dataclass
class FrameMetadata:
    name: str
    width: int
    height: int
    latitude: float | None = None
    longitude: float | None = None
    yaw_deg: float | None = None
    pitch_deg: float = -90.0
    roll_deg: float = 0.0
    altitude_agl_m: float = 30.0
    gps_reference: str = 'camera'  # 'camera' or 'image_center'
    exif_altitude_m: float | None = None  # Logged only; not assumed AGL.
    gps_accuracy_m: float | None = None
    gps_status: str | None = None
    orientation: int = 1
    timestamp: str | None = None
    provenance: dict | None = None
    warnings: list | None = None

    @property
    def geographic(self): return all(v is not None and math.isfinite(v) for v in [self.latitude,self.longitude,self.yaw_deg])
    def to_dict(self): return {**asdict(self),'geographic':self.geographic}

def _float(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (ValueError,TypeError,ZeroDivisionError): return None

def _coordinate(value, ref):
    if value is None or ref is None: return None
    try:
        if isinstance(ref,bytes): ref=ref.decode()
        d,m,s=map(float,value); result=d+m/60+s/3600
        return -result if ref.upper() in ['S','W'] else result
    except (ValueError,TypeError,ZeroDivisionError): return None

def jpeg_metadata_segments(path):
    """Read APP/COM payloads without searching compressed pixel bytes for tags."""
    found=[]
    with open(path,'rb') as f:
        if f.read(2)!=b'\xff\xd8': return found
        while True:
            marker=f.read(2)
            if len(marker)!=2 or marker[0]!=255: break
            while marker[1]==255: marker=b'\xff'+f.read(1)
            code=marker[1]
            if code in [0xDA,0xD9]: break
            if code==1 or 0xD0<=code<=0xD7: continue
            size=int.from_bytes(f.read(2),'big')
            if size<2: break
            data=f.read(size-2)
            if 0xE0<=code<=0xEF or code==0xFE: found.append((code,data))
    return found

def read_metadata(path: Path, cfg, override=None):
    warnings=[]; provenance={'altitude_agl_m':'hardcoded flight config','pitch_deg':'configured nadir assumption','roll_deg':'configured roll assumption'}
    with Image.open(path) as im:
        width,height=im.size; exif=im.getexif(); gps=exif.get_ifd(34853); camera=exif.get_ifd(34665)
        meta=FrameMetadata(path.name,width,height,pitch_deg=cfg.default_pitch_deg,roll_deg=cfg.default_roll_deg,altitude_agl_m=cfg.altitude_agl_m,orientation=int(exif.get(274,1)),provenance=provenance,warnings=warnings)
        meta.latitude=_coordinate(gps.get(2),gps.get(1)); meta.longitude=_coordinate(gps.get(4),gps.get(3))
        for name in ['latitude','longitude']:
            if getattr(meta,name) is not None: provenance[name]='EXIF GPS camera position'
        heading_ref=gps.get(16,''); heading_ref=heading_ref.decode() if isinstance(heading_ref,bytes) else heading_ref
        if heading_ref=='T':
            meta.yaw_deg=_float(gps.get(17))
            if meta.yaw_deg is not None: provenance['yaw_deg']='EXIF GPSImgDirection, true north'
        elif gps.get(17) is not None: warnings.append('GPSImgDirection is not explicitly true north; yaw left missing')
        altitude=_float(gps.get(6)); ref=gps.get(5,0)
        if isinstance(ref,bytes): ref=int.from_bytes(ref,'big')
        meta.exif_altitude_m=None if altitude is None else (-altitude if ref==1 else altitude)
        meta.timestamp=camera.get(36867,exif.get(306))
    segments=jpeg_metadata_segments(path); xmp={}
    for _,data in segments:
        if b'http://ns.adobe.com/xap/1.0/' not in data: continue
        try:
            xml=data.split(b'\x00',1)[1]; root=ET.fromstring(xml)
            for element in root.iter():
                for key,val in element.attrib.items(): xmp[key.split('}')[-1]]=val
                if element.text and element.text.strip(): xmp[element.tag.split('}')[-1]]=element.text.strip()
        except (ET.ParseError,IndexError): warnings.append('Unparseable XMP packet')
    for target,aliases in XMP_FIELDS.items():
        for alias in aliases:
            value=_float(xmp.get(alias))
            if value is not None:
                setattr(meta,target,value); provenance[target]=f'XMP {alias}'; break
    meta.gps_status=xmp.get('GpsStatus',xmp.get('GPSStatus'))
    accuracy=[_float(xmp.get(k)) for k in ('RtkStdLat','RtkStdLon','GPSXYAccuracy','GpsXYAccuracy')]
    accuracy=[v for v in accuracy if v is not None and v>0]
    if accuracy:
        meta.gps_accuracy_m=max(accuracy);provenance['gps_accuracy_m']='XMP horizontal accuracy, conservative maximum'
    if meta.gps_status and 'NO GPS' in meta.gps_status.upper():
        warnings.append('GPS coordinates are present but camera reports NO GPS connected; accuracy unverified')
    allowed={'latitude','longitude','yaw_deg','pitch_deg','roll_deg','altitude_agl_m','gps_reference','timestamp','gps_accuracy_m'}
    for key,value in (override or {}).items():
        if key not in allowed: raise ValueError(f'{path.name}: unknown metadata override {key}')
        if key not in ['gps_reference','timestamp']:
            value=_float(value)
            if value is None: raise ValueError(f'{path.name}: nonfinite metadata override {key}')
        setattr(meta,key,value); provenance[key]='explicit sidecar override'
    if meta.latitude is not None and not -90<meta.latitude<90: raise ValueError('Latitude out of range / polar projection unsupported')
    if meta.longitude is not None and not -180<=meta.longitude<=180: raise ValueError('Longitude out of range')
    if meta.altitude_agl_m<=0: raise ValueError('Altitude must be above local ground')
    if meta.gps_accuracy_m is not None and meta.gps_accuracy_m<=0:raise ValueError('GPS accuracy must be positive')
    if meta.gps_reference not in ['camera','image_center']: raise ValueError('Invalid GPS reference')
    if meta.orientation!=1: raise ValueError(f'{path.name}: EXIF orientation {meta.orientation}; normalize pixels and camera axes first')
    if not meta.geographic: warnings.append('No complete geographic pose; local mode required')
    raw={'exif_tag_count':len(exif),'gps_tag_count':len(gps),'xmp_fields':xmp,'segments':[{'marker':hex(c),'bytes':len(d),'kind':'EXIF' if d.startswith(b'Exif\0\0') else 'XMP' if b'http://ns.adobe.com/xap/' in d else 'JFIF' if d.startswith(b'JFIF') else 'other'} for c,d in segments]}
    return meta,raw
