#!/usr/bin/env python3
"""Apply the supplied cinematic recipes to FINAL maps or presentation photos."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from adm.cinematic_grade import PRESETS,REGION_PRESETS,grade_bgra


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input',type=Path);group.add_argument('--input-dir',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--output-dir',type=Path)
    p.add_argument('--preset',choices=PRESETS,default='pro_camera')
    p.add_argument('--regional',action='store_true');p.add_argument('--region-preset',choices=REGION_PRESETS,default='cool_regional')
    p.add_argument('--region-blur',type=float,default=6);p.add_argument('--seed',type=int,default=2026)
    options=['contrast','saturation','vignette_strength','grain','warmth','exposure','blacks','whites','shadows','highlights','sharpness','clarity','white_reduction','vibrance_reduction']
    for name in options:p.add_argument('--'+name.replace('_','-'),type=float)
    args=p.parse_args()
    if args.input and not args.output:p.error('--output is required with --input')
    if args.input_dir and not args.output_dir:p.error('--output-dir is required with --input-dir')
    pairs=[(args.input,args.output)] if args.input else [(f,args.output_dir/f.name) for f in sorted(args.input_dir.iterdir()) if f.suffix.lower() in ('.jpg','.jpeg','.png')]
    overrides={k:getattr(args,k) for k in options if getattr(args,k) is not None}
    for i,(source,dest) in enumerate(pairs):
        if source.resolve()==dest.resolve():raise ValueError('Choose a separate graded output')
        start=time.perf_counter();image=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)
        if image is None:raise ValueError(f'Cannot decode {source}')
        if image.ndim!=3:raise ValueError('Expected an RGB/RGBA image')
        if image.shape[2]==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
        result=grade_bgra(image,args.preset,args.region_preset if args.regional else None,args.seed+i,overrides,region_blur=args.region_blur)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(dest),result):raise OSError(dest)
        print(f'{dest}: {time.perf_counter()-start:.3f}s',flush=True)


if __name__=='__main__':main()
