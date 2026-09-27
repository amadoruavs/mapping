"""Verify the completed SIYI run and save a comparison with the previous finish."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil
import sys
import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'MapRepair'))
from native import Native
from map_io import flatten_original_extent,save_jpeg
from run import highlight_bloom

out=ROOT/'output/siyi_odm_openrt_golden_hour'
original=cv2.imread('/Users/avneh/Downloads/odm.png',-1)
repaired=cv2.imread(str(out/'01_repaired.png'),-1)
graded=cv2.imread(str(out/'03_graded.png'),-1)
lit=cv2.imread(str(out/'04_lit.png'),-1)
with rasterio.open(out/'final_2x_alpha.tif') as ds:mask=ds.read(1)
expected=np.repeat(np.repeat(original[:,:,3],2,0),2,1)
assert np.array_equal(mask,expected)
for name in ('01_repaired.png','02_sharpened.png','03_graded.png','04_lit.png'):
    assert np.array_equal(cv2.imread(str(out/name),-1)[:,:,3],original[:,:,3]),name
timing=json.loads((out/'timing.json').read_text())
geometry=json.loads((out/'final_geometry.json').read_text())
report=json.loads((out/'geometry_report.json').read_text())
assert geometry['crop_input_bounds_xyxy']==[0,0,original.shape[1],original.shape[0]]
assert not geometry['georeferenced']
assert hashlib.sha256(Path(timing['map']).read_bytes()).hexdigest()==timing['input_map_sha256']
with Image.open(out/'final_2x.jpg') as im:
    assert im.format=='JPEG' and im.size==(1754,1356)

# The previous native plane tracer is an exact no-op. Reuse the same repaired,
# sharpened and graded stages; vary only the added OpenRT lighting.
previous=highlight_bloom(graded,.02)
native=Native(8)
final=native.fsr2x(previous,out/'_comparison_work',.2)
save_jpeg(out/'previous_finish_2x.jpg',flatten_original_extent(final,original[:,:,3]),95)
final._mmap.close();del final
shutil.rmtree(out/'_comparison_work')

original_bgr=flatten_original_extent(original,original[:,:,3],scale=1)
save_jpeg(out/'input_preview.jpg',original_bgr,95)
panels=[('Input ODM map',out/'input_preview.jpg'),
        ('Previous finish',out/'previous_finish_2x.jpg'),
        ('Previous finish + OpenRT golden hour',out/'final_2x.jpg')]
width=700;height=round(width*original.shape[0]/original.shape[1]);gap=12;header=44
canvas=Image.new('RGB',(width*3+gap*4,height+header+gap),(245,246,247))
draw=ImageDraw.Draw(canvas)
try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',21)
except OSError:font=ImageFont.load_default()
for i,(label,path) in enumerate(panels):
    x=gap+i*(width+gap)
    draw.text((x,12),label,font=font,fill=(30,35,40))
    with Image.open(path) as im:canvas.paste(im.convert('RGB').resize((width,height),Image.Resampling.LANCZOS),(x,header))
canvas.save(out/'comparison.jpg',quality=95,subsampling=0)
valid=original[:,:,3]==255
delta=lit[:,:,:3].astype(float)-previous[:,:,:3]
summary=dict(input=timing['map'],photos_folder='/Users/avneh/Downloads/101SIYI_IMG',photos=40,
    total_seconds=timing['total_seconds'],stage_seconds=timing['stage_seconds'],
    photo_status_counts=dict(Counter(e['status'] for e in report['photos'])),
    repaired_tiles=report['repaired_tiles'],repair_pixels_unchanged=np.array_equal(repaired,original),
    final_size=[1754,1356],original_footprint_mask_exact=True,all_stage_alpha_exact=True,
    original_input_hash_unchanged=True,georeferenced=False,
    mean_lighting_delta_rgb_codes=delta[valid].mean(0)[::-1].tolist(),
    final_jpeg=str(out/'final_2x.jpg'),comparison=str(out/'comparison.jpg'),
    timing_scope='Existing PNG scan/repair and finish; initial C++ compilation excluded. Comparison export excluded.',
    lighting='Actual OpenRT warm sun/cool sky on a flat plane; artistic 5 pm look, original shadows retained')
(Path(__file__).parent/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
