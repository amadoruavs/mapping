"""Separate process: browser refreshes never interrupt a running image job."""
import argparse
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--map', type=Path, required=True)
    p.add_argument('--photos', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print('[preflight] starting', flush=True)
    try:
        import run
        from native import Native
        from PIL import Image
        import rasterio
    except ImportError as exc:
        raise RuntimeError(f'Missing processing dependency: {exc}. Install requirements.txt using this Python: {sys.executable}') from exc
    if args.map.suffix.lower() in ('.tif', '.tiff'):
        with rasterio.open(args.map) as src:
            width, height = src.width, src.height
    else:
        with Image.open(args.map) as src:
            width, height = src.size
            src.verify()
    if width * height * 4 > run.MAX_OUTPUT_PIXELS:
        raise ValueError(f'This {width} × {height} map exceeds the configured output size limit. Use a smaller map or adjust MAX_OUTPUT_PIXELS in run.py.')
    print(f'[input] {width} × {height}; checking processor build', flush=True)
    Native(run.CPU_THREADS)
    if run.RAYTRACE and run.RAYTRACE_BACKEND == 'openrt':
        from openrt_backend import OpenRT
        OpenRT(run.CPU_THREADS, lighting=run.OPENRT_LIGHTING)
    print('[preflight] ready', flush=True)
    run.pipeline(args.map, args.photos, args.output, map_crs=run.MAP_CRS,
                 map_bounds=run.MAP_BOUNDS, save_stages=False)


if __name__ == '__main__':
    main()
