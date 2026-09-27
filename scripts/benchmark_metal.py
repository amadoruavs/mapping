"""Measure CPU and Metal stage latency, including host input/output exchange.

Results are local measurements, not throughput claims for other devices. Run
after validate_metal.py; do not run concurrently with other GPU benchmarks.
"""
import argparse
import gc
import json
from pathlib import Path
import resource
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from adm.config import Config
from adm.lighting import _choose_cpu, linear_to_srgb
from adm.metal import get_backend


def measure(fn, repeats):
    fn()  # Warm pipelines, MPS specializations and allocators.
    times = []
    for _ in range(repeats):
        gc.collect(); started = time.perf_counter(); result = fn()
        times.append(time.perf_counter()-started); del result
    return dict(median_seconds=statistics.median(times), samples_seconds=times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=1536)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('output/metal_benchmark.json'))
    args = parser.parse_args()
    if args.size < 64 or args.repeats < 1: parser.error('size >= 64 and repeats >= 1 required')
    cv2.setNumThreads(6); gpu = get_backend(); rng = np.random.default_rng(2026)
    h, w = args.size*3//4, args.size
    rgba = rng.integers(0, 256, (h, w, 4), dtype=np.uint8); rgba[:, :, 3] = 255
    rgba[:10, :, 3] = 0
    cfg = Config(); rows = []

    def compare(name, cpu, metal, shape):
        a = measure(cpu, args.repeats); b = measure(metal, args.repeats)
        row = dict(stage=name, shape=shape, cpu=a, metal=b,
                   speedup=a['median_seconds']/b['median_seconds'])
        rows.append(row); print(json.dumps(row), flush=True)

    def cpu_light():
        chosen = _choose_cpu(rgba, cfg)
        return chosen, np.round(np.clip(linear_to_srgb(chosen), 0, 1)*255).astype(np.uint8)
    compare('flat lighting and SDR encoding, five environments', cpu_light,
            lambda: gpu.lighting(rgba, cfg), [h, w])
    image = rng.normal(size=(h, w, 4)).astype(np.float32)
    compare('Gaussian sigma 7, four channels', lambda: cv2.GaussianBlur(image, (0, 0), 7),
            lambda: gpu.gaussian(gpu.upload(image), 7).numpy(False), list(image.shape))
    bgr = np.ascontiguousarray(rgba[:, :, :3]); H = np.array([[.92, .07, 12], [-.06, .99, 7], [.00001, -.00002, 1.]])
    def cpu_warp():
        yy, xx = np.mgrid[:h, :w]
        d = np.minimum.reduce([xx/(w/2), (w-1-xx)/(w/2), yy/(h/2), (h-1-yy)/(h/2)]).astype(np.float32)
        e = np.clip(d*8, 0, 1); e[d < .02] = 0
        return cv2.warpPerspective(bgr, H, (w, h)), cv2.warpPerspective(e, H, (w, h))
    def metal_warp():
        a, b = gpu.warp(gpu.upload(bgr), H, (h, w)); gpu.sync(); return a.numpy(False), b.numpy(False)
    compare('warp plus edge confidence', cpu_warp, metal_warp, [h, w])
    a = rng.integers(0, 150, (7000, 128)).astype(np.float32)
    b = rng.integers(0, 150, (7000, 128)).astype(np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    # Feature upload is timed too; the actual pipeline caches these per image.
    compare('exact descriptor two-nearest, includes upload', lambda: matcher.knnMatch(a, b, k=2),
            lambda: gpu.match_descriptors(gpu.descriptors(a), gpu.descriptors(b)), [7000, 7000, 128])
    th, tw = min(h, 384), min(w, 512)
    yy, xx = np.mgrid[:th, :tw]; terrain = (3*np.sin(xx/21)+.03*yy).astype(np.float32)
    terrain[:, tw//2:tw//2+3] += 8
    terrain_rgba = rgba[:th, :tw]; terrain_cfg = Config(lighting_candidates=5, shadow_steps=48)
    compare('terrain lighting, five environments and 48 shadow steps',
            lambda: _choose_cpu(terrain_rgba, terrain_cfg, terrain, .7),
            lambda: gpu.lighting(terrain_rgba, terrain_cfg, terrain, .7), [th, tw])
    report = dict(acceleration=gpu.stats(), opencv=cv2.__version__, numpy=np.__version__,
                  cpu_threads=6, includes_host_exchange=True, excludes_image_codecs=True,
                  peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  measurements=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__': main()
