"""Summarize the measured 94-photo comparison and export matched-area crops."""
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ISM/runtime'))
from adm.geometry import LocalEarth,project
from adm.geographic_alignment import TopocentricEarth
OUT=ROOT/'ISM/benchmarks/odm_geo94'


def read(folder,name):return json.loads((folder/name).read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def image_center_proxy(folder,geographic):
    md=read(folder,'metadata.json');hs=np.load(folder/'transforms.npy');reg=read(folder,'resolved_config.json')['registration_width']
    centers=np.array([[reg/m['width']*(m['width']-1)/2,round(m['height']*reg/m['width'])/m['height']*(m['height']-1)/2] for m in md])
    pts=np.array([project(h,[c])[0] for h,c in zip(hs,centers)])
    earth=(TopocentricEarth if geographic else LocalEarth)(md[0]['latitude'],md[0]['longitude'])
    target=np.array([earth.xy(m['latitude'],m['longitude']) for m in md])
    if not geographic:
        pts[:,1]*=-1;x,y=pts.T;A=np.zeros((len(x)*2,4));A[0::2]=np.c_[x,-y,np.ones(len(x)),np.zeros(len(x))]
        A[1::2]=np.c_[y,x,np.zeros(len(x)),np.ones(len(x))]
        initial=np.linalg.lstsq(A,target.ravel(),rcond=None)[0]
        fit=least_squares(lambda v:A@v-target.ravel(),initial,loss='soft_l1',f_scale=.75)
        pts=(A@fit.x).reshape(-1,2)
    errors=np.linalg.norm(pts-target,axis=1)
    return dict(median_m=float(np.median(errors)),p95_m=float(np.percentile(errors,95)),
        definition='Projected image-centre distance to recorded GPS; shared near-nadir proxy, not camera-centre or surveyed accuracy')


def main():
    report={'input':'/Users/avneh/Downloads/mapv2','images':94,'variants':{},'reproducibility':{}}
    hashes=read(OUT,'input_hashes.json')
    for label,first,repeat in [('old','old','old_repeat'),('new','new_visual_init','new_repeat')]:
        folder=OUT/first;registration=read(folder,'registration_report.json');times=[read(OUT/n,'timing.json') for n in (first,repeat)]
        selection=read(folder,'selection.json');assert {r['name']:r['sha256'] for r in selection}==hashes
        assert all(r['excluded'] is None for r in selection)
        row=dict(wall_samples_s=[r['wall_seconds'] for r in times],median_wall_s=float(np.median([r['wall_seconds'] for r in times])),
            peak_rss_samples_bytes=[r['peak_child_rss_bytes'] for r in times],
            stages_first_s=read(folder,'report.json')['timings_seconds'],
            visual_residual_median_px=registration['median_registration_error_px'],
            visual_residual_p95_px=registration['p95_registration_error_px'],
            image_center_gps_proxy=image_center_proxy(folder,label=='new'),
            geographic_fit_report=read(folder,'geographic_alignment.json'),
            output_dimensions=read(folder,'report.json')['output_dimensions'])
        report['variants'][label]=row
        for name in ('map_neutral.png','map_graded.png','transforms.npy'):
            same=digest(folder/name)==digest(OUT/repeat/name)
            report['reproducibility'][f'{label}_{name}']=same
            if not same:raise AssertionError(f'Repeat differs: {label} {name}')
    spec=read(OUT,'source_old_hashes.json')
    assert all(digest(OUT/'source_old'/name)==value for name,value in spec.items())
    report['preserved_old_source_hashes_verified']=len(spec)
    report['interpretation']='No clear speed win across two runs. New physical model improves the shared image-centre GPS proxy but increases visual transfer residuals; not an overall quality or survey-accuracy win.'
    report['prior_parity']=read(OUT,'upstream_parity.json')
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')

    canvas=Image.new('RGB',(1800,935),(242,242,240));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',22)
    maps=[]
    for i,(folder,label) in enumerate([(OUT/'old','Verified visual ISM'),(OUT/'new_visual_init','ISM with OpenSfM geographic priors')]):
        source=Image.open(folder/'map_graded.png');im=source.copy();im.thumbnail((880,875))
        canvas.paste(im.convert('RGB'),(900*i+10,50));draw.text((900*i+10,14),label,font=font,fill=(15,20,22))
        maps.append((cv2.imread(str(folder/'map_graded.png'),cv2.IMREAD_UNCHANGED),read(folder,'map_geometry.json')))
    canvas.save(OUT/'overview_comparison.png')
    # Both maps use the same origin; local tangent vs ECEF differences here are
    # less than a millimetre. Sample fixed physical areas at the same 10 cm grid.
    cropcanvas=Image.new('RGB',(1800,850),(242,242,240));draw=ImageDraw.Draw(cropcanvas)
    for row,(name,left,top,width,height) in enumerate([('North road',-10,35,120,45),('South road',-10,-125,120,45)]):
        W,H=880,330;gsd=width/W
        yy,xx=np.mgrid[0:H,0:W];world=np.stack([left+(xx+.5)*gsd,top-(yy+.5)*gsd,np.ones_like(xx)],axis=-1)
        for col,(rgba,geometry) in enumerate(maps):
            xy=world@np.linalg.inv(np.array(geometry['map_to_world'])).T
            crop=cv2.remap(rgba,xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv2.INTER_LINEAR)
            cropcanvas.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGRA2RGB)),(col*900+10,row*415+50))
            draw.text((col*900+10,row*415+14),f'{name} / {"visual" if col==0 else "geographic"}',font=font,fill=(15,20,22))
    cropcanvas.save(OUT/'road_comparison.png')
    print(json.dumps({k:{kk:v[kk] for kk in ('wall_samples_s','visual_residual_median_px','image_center_gps_proxy')} for k,v in report['variants'].items()},indent=2))


if __name__=='__main__':main()
