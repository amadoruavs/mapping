"""Apply the GPU lasso to a saved map; retain its original canvas/geography."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from adm.io import write_image
from adm.lasso import crop_in_place
from adm.metal import get_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--inset', type=float, default=.01)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        raise ValueError('Choose a new output path to preserve existing maps')
    start = time.perf_counter()
    image = cv2.imread(str(args.input), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('Input image could not be decoded')
    gpu = get_backend()
    before = gpu.stats()
    crop_start = time.perf_counter()
    result = crop_in_place(image, gpu, inset_ratio=args.inset)
    crop_seconds = time.perf_counter()-crop_start
    after = gpu.stats()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_image(args.output, result.image)
    total = time.perf_counter()-start
    preview = cv2.resize(result.image, (1200, round(image.shape[0]*1200/image.shape[1])))
    write_image(args.output.with_name(args.output.stem+'_preview.png'), preview)
    report = dict(result.report, source=str(args.input.resolve()), output=str(args.output.resolve()),
                  crop_wall_seconds=crop_seconds, total_with_decode_and_png_seconds=total,
                  gpu_seconds=after['gpu_seconds']-before['gpu_seconds'],
                  command_submissions=after['command_submissions']-before['command_submissions'])
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='vertices_pixels'}, indent=2))


if __name__ == '__main__':
    main()
