"""Render saved alignment without rerunning photo matching or alignment."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ISM/runtime'))
import cv2
import numpy as np
from adm.config import Config
from adm.coherent_composite import compose_pixels
from adm.metal import get_backend

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--saved',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fsr',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise ValueError('Use a new render output directory')
    args.output.mkdir(parents=True)
    raw=json.loads((args.saved/'resolved_config.json').read_text())
    cfg=Config(**{k:v for k,v in raw.items() if k in Config.__dataclass_fields__})
    geometry=json.loads((args.saved/'map_geometry.json').read_text());names=geometry['input_names']
    hs=np.load(args.saved/'image_to_map.npy');width,height=geometry['width'],geometry['height']
    cfg.export_profile='diagnostic';cfg.presentation=False;cfg.color_grade='none'
    cv2.setNumThreads(6);gpu=get_backend();start=time.perf_counter();views=None
    if args.fsr:
        from adm.fsr_views import FSRViews
        views=FSRViews(args.input,names,hs,cfg.interpolation_levels,cfg.altitude_agl_m,cfg.hfov_deg,cfg.fsr_frame_budget_mb)
        names=names+[f'virtual_{i}' for i in range(len(views.records)-len(names))]
        hs=[r[3] for r in views.records]
    result,coverage,ids,valid=compose_pixels(args.input,names,hs,width,height,cfg,args.output,gpu,frame_provider=views)
    if not cv2.imwrite(str(args.output/'map_neutral.png'),result):raise RuntimeError('Could not save render')
    cv2.imwrite(str(args.output/'source_ids.png'),ids)
    preview=cv2.resize(result,(1300,round(height*1300/width)),interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(args.output/'preview.png'),preview)
    report={'seconds':time.perf_counter()-start,'fsr':args.fsr,'images':len(names),'covered_pixels':int(valid.sum())}
    if views:report['interpolation']=views.report()
    (args.output/'timing.json').write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k!='interpolation'},flush=True)

if __name__=='__main__':main()
