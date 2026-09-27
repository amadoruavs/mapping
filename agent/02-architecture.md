# Architecture

## ISM / LeanADM / root ADM

```
photos
  │
  ├─ selection (sort, hash, skip, dedup, optional label filter)
  ├─ metadata  (EXIF/XMP; AGL from config, not EXIF altitude)
  ├─ neighbors (GPS-center KD-tree, or sequence window)
  ├─ SIFT @ registration_width   (CPU OpenCV)
  ├─ descriptor match            (Metal/MPS)
  ├─ RANSAC homography + gates   (CPU)
  ├─ optional disconnected-graph recovery (verified visual bridges only)
  ├─ GPU alignment               (8 DOF / image, anchor fixed)
  ├─ geographic similarity       (CPU float64; scale + north from GPS)
  │     or opensfm_planar joint refine (ISM only)
  ├─ coherent compositor         (one source per pixel + low-frequency exposure)
  ├─ optional presentation       (sharpen / grade / lasso)
  └─ PNG + geometry + reports
```

CPU still owns codecs, SIFT, RANSAC, metadata, graph packing, and the final geographic conversion. Metal owns descriptor matching, the alignment solve, compositing, and (in ISM) presentation kernels.

## Alignment in one paragraph

Each image has eight projective parameters. The middle image is the anchor. A spanning tree initializes the rest. Metal builds a sparse-ish normal system (actually a dense `(8N)×(8N)` float32 matrix) from pairwise correspondences, applies coordinate-wise soft-L1, and iterates a damped Gauss–Newton / Levenberg-style loop with MPS Cholesky. Convergence is checked every four encoded iterations. After that, a 4-parameter similarity (`s, θ, tx, ty` with a Y flip) maps mosaic space onto a local ENU tangent plane using GPS camera centers.

The objective that only looks at *mapped* pixel error can cheat by shrinking every photo toward a point. That is why ISM defaults to **symmetric transfer error in source-image pixels**.

## Compositor in one paragraph

Footprints are packed into a canvas whose longest side is `max_output_dimension` (default 6500). A coarse grid votes for a source label with unary cost (edge confidence + optional photometric) and 32 red/black Potts sweeps. Full-resolution pixels are sampled from the winning source only. Exposure bias is a blurred, clamped low-frequency offset. This hides high-frequency flicker. It does **not** fix 3D parallax. Elevated objects can still be cut.

## MapRepair

```
existing map (TIFF/PNG/JPEG)
  │
  ├─ map_io          (rasterio/GDAL; real alpha; CRS/affine)
  ├─ georeference    (world file / .prj / optional bounds)
  ├─ SIFT vs photos  (cached; GPS ROI if the map is georeferenced)
  ├─ RANSAC + held-out matches + local LK/NCC tiles
  ├─ independent overlapping pairs must agree at a tile
  ├─ conservative patch + patch_blend (slow RGB offset, feather)
  ├─ adaptive or legacy finish
  ├─ OpenRT or compact C++ plane tracer (optional)
  ├─ FSR1 2× (EASU + RCAS)
  └─ JPEG/PNG + alpha TIFF + world file
```

Web path: `run.sh` → private `.runtime` venv → `web.py` (stdlib HTTP, loopback only) → `web_worker.py` → `run.py`.

## GPU / CPU split that people get wrong

| Work | Where |
|---|---|
| JPEG decode, EXIF, SIFT, RANSAC | CPU |
| Descriptor L2 / ratio test | Metal/MPS |
| Homography GN + Cholesky | Metal/MPS |
| GPS similarity / WGS84 | CPU float64 |
| Coherent label + warp | Metal |
| MapRepair geometry + OpenRT + FSR | CPU C++ / Python |

There is no NPU implementation. HIP / XDNA2 kernels in `acceleration/` are host-math certified only.

## Memory

- ISM alignment workspace default 1024 MiB (`alignment_budget_mb`).
- Coarse compositor default 128 MiB in ISM.
- Python Metal allocations are checked against 60% of RAM unless `ADM_METAL_MEMORY_LIMIT_MB` is set.
- Planned solver scratch is `observations*156 + edges*1088 + (8N)^2*8 + (8N)*12` bytes. It is **not** constant in image count.
- MapRepair bounds output pixels (`MAX_OUTPUT_PIXELS = 200_000_000`) because 2× is 4× the pixels.
