"""Conservative GPU perimeter cropping without resampling retained image data."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LassoResult:
    image: np.ndarray
    report: dict


def crop_in_place(image, gpu, *, coverage=None, source_ids=None, inset_ratio=.01):
    """Crop a shared GPU image in place, or upload a CPU image without modifying it.

    The original canvas and pixel-to-world transform remain valid. The polygon
    changes opacity and support indices only; surviving colors are exact copies.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError('Lasso cropping requires a uint8 four-channel image')
    if not 0 <= inset_ratio <= .1:
        raise ValueError('Lasso inset must be between zero and ten percent')
    height, width = image.shape[:2]
    if min(width, height) < 16:
        raise ValueError('Image too small for a stable lasso')
    for value in (coverage, source_ids):
        if value is not None and (value.shape != (height, width) or value.dtype != np.uint16):
            raise ValueError('Support arrays must match the image and use uint16')

    stride = max(1, min(8, max(width, height)//512))
    ray_count, anchor_count = 512, 64
    gw, gh = (width+stride-1)//stride, (height+stride-1)//stride
    source = gpu.shared_input(image)
    mask = gpu.buffer((gh, gw), np.uint8)
    mask_scratch = gpu.buffer((gh, gw), np.uint8)
    rows = gpu.buffer((gh, 4))
    center = gpu.buffer(4)
    rays = gpu.buffer(ray_count)
    ray_scratch = gpu.buffer(ray_count)
    anchors = gpu.buffer(anchor_count)
    scratch = gpu.buffer(anchor_count)
    counts = gpu.buffer(2, np.uint32, zero=True)
    coverage_buffer = gpu.shared_input(coverage) if coverage is not None else gpu.buffer(1, np.uint16)
    ids_buffer = gpu.shared_input(source_ids) if source_ids is not None else gpu.buffer(1, np.uint16)

    gpu.dispatch('lasso_mask', [source, mask], gw*gh, [width, height, stride, gw, gh])
    opening_radius = max(1, round(max(width, height)*.09/stride))
    for dilate, axis in ((0, 0), (0, 1), (1, 0), (1, 1)):
        gpu.dispatch('lasso_morph', [mask, mask_scratch], gw*gh,
                     [gw, gh, axis, dilate, opening_radius])
        mask, mask_scratch = mask_scratch, mask
    gpu.dispatch('lasso_rows', [mask, rows], gh, [gw, gh])
    gpu.dispatch('lasso_center', [rows, mask, center], 1, [gw, gh])
    inset = max(1., max(width, height)*inset_ratio/stride)
    gpu.dispatch('lasso_rays', [mask, center, rays], ray_count, [gw, gh, ray_count], [inset])
    gpu.dispatch('lasso_open', [rays, ray_scratch], ray_count, [ray_count, 0])
    gpu.dispatch('lasso_open', [ray_scratch, rays], ray_count, [ray_count, 1])
    gpu.dispatch('lasso_anchors', [rays, anchors], anchor_count, [ray_count, anchor_count])
    for _ in range(8):
        gpu.dispatch('lasso_refine', [mask, center, anchors, scratch], anchor_count, [gw, gh, anchor_count])
        anchors, scratch = scratch, anchors
    gpu.dispatch('lasso_apply', [source, center, anchors, coverage_buffer, ids_buffer, counts],
                 width*height, [width, height, stride, anchor_count, coverage is not None, source_ids is not None], group=256)
    gpu.sync()

    c, radii = center.view(), anchors.view()
    if not c[3]:
        raise ValueError('No opaque central support for a connected lasso crop')
    before, after = map(int, counts.view())
    if after < .5*before:
        raise ValueError('Lasso would discard over half the covered image; inspect disconnected coverage')
    # Export only the small polygon/diagnostic vectors; image analysis stays GPU.
    angles = np.arange(anchor_count)*2*np.pi/anchor_count
    polygon = (c[:2]+radii[:, None]*np.c_[np.cos(angles), np.sin(angles)])*stride-.5
    report = dict(method='GPU inward polygon lasso', vertices_pixels=polygon.tolist(),
                  original_covered_pixels=before, retained_pixels=after,
                  retained_fraction=after/max(1, before), inset_ratio=inset_ratio,
                  mask_grid=[gw, gh], scratch_bytes=gw*gh*2+gh*16+ray_count*8+anchor_count*8+24,
                  resampled=False, retained_colors_changed=False, canvas_transform_changed=False,
                  limitation='Trims the outer footprint and perimeter black strips; cannot repair internal misalignment.')
    return LassoResult(source.view(), report)
