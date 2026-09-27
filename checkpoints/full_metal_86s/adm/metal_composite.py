"""Resident map reductions and disk-backed observations for the Metal backend."""
import cv2
import numpy as np
from .geometry import project


def compose_pixels(source, names, hs, width, height, cfg, out, gpu):
    if len(names) > 65535:
        raise ValueError('16-bit source IDs support at most 65535 images')
    shape = (height, width)
    mean = gpu.buffer((*shape, 4), zero=True)
    coverage = gpu.buffer(shape, np.uint16, zero=True)
    cache = out / 'warps'
    if not cfg.metal_recompute_warps: cache.mkdir(exist_ok=True)
    records = []
    for i, (name, H) in enumerate(zip(names, hs)):
        image = cv2.imread(str(source / name)); h, w = image.shape[:2]
        p = project(H, [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        x0, y0 = np.maximum(0, np.floor(p.min(0)).astype(int))
        x1, y1 = np.minimum([width, height], np.ceil(p.max(0)).astype(int)+1)
        rw, rh = int(x1-x0), int(y1-y0)
        transform = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.]]) @ H
        warped, weight = gpu.warp(gpu.upload(image), transform, (rh, rw))
        gpu.dispatch('accumulate', [warped, weight, mean, coverage], rw*rh,
                     [rw, rh, x0, y0, width])
        if cfg.metal_recompute_warps: gpu.sync()
        else:
            np.save(cache / f'{i:04}_rgb.npy', warped.numpy(False))
            np.save(cache / f'{i:04}_weight.npy', weight.numpy(False).astype(np.float16))
        records.append((int(x0), int(y0), rw, rh))
        print('composite warp', name, flush=True)
    best = gpu.buffer(shape); selected = gpu.buffer((*shape, 4), np.uint8)
    ids = gpu.buffer(shape, np.uint16); count = width*height
    gpu.dispatch('finish_mean', [mean, best, selected, ids], count, [count])
    rng = np.random.default_rng(cfg.seed)
    gh, gw = max(2, (height+95)//96), max(2, (width+95)//96)
    for i, (x0, y0, rw, rh) in enumerate(records):
        if cfg.metal_recompute_warps:
            image = cv2.imread(str(source/names[i]))
            transform = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.]]) @ hs[i]
            rgb, weight = gpu.warp(gpu.upload(image), transform, (rh, rw))
            confidence = gpu.buffer((rh, rw), np.float16)
            gpu.dispatch('confidence_half', [weight, confidence], rw*rh, [rw*rh])
        else:
            rgb = gpu.upload(np.load(cache / f'{i:04}_rgb.npy', mmap_mode='r'))
            confidence = gpu.upload(np.load(cache / f'{i:04}_weight.npy', mmap_mode='r'))
        coarse = gpu.upload(rng.standard_normal((gh, gw)).astype(np.float32))
        gpu.dispatch('select_source', [rgb, confidence, mean, coarse, best, selected, ids], rw*rh,
                     [rw, rh, x0, y0, width, height, gw, gh, i], [cfg.selection_sigma_rgb])
        gpu.sync()  # Bound live observations to one ROI, independent of image count.
    del best
    diff = gpu.buffer((*shape, 4))
    gpu.dispatch('seam_input', [mean, selected, diff], count, [count])
    # Later tiles need original neighbors, so filtering uses a distinct output.
    blurred = gpu.gaussian(diff, 7)
    gpu.dispatch('seam_output', [blurred, mean, selected], count, [count])
    gpu.sync()
    valid = mean.view()[:, :, 3] > .001
    return selected.numpy(False), coverage.numpy(False), ids.numpy(False), valid
