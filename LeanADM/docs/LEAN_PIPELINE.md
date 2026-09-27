# Lean ADM: removed work, measured speed, and crispness

Measured 2026-09-14. Fresh runs use the same input photographs, registration settings, output dimensions, and coherent-selection budgets as their saved optimized baselines. Every neutral BGRA pixel and every image-to-map transform is identical across all three comparisons.

## What changed

- The normal CLI export is now `png`: one neutral PNG, geometry/transforms, and small run reports. No match cache, corner vocabulary dumps, support PNGs, lighting images, or post-registration footprint index is written by default.
- `--cache-matches` explicitly saves matches; `--resume` reuses a compatible cache or creates one. Diagnostic exports retain the cache and inspection artifacts. Cache identity now includes the Objective-C bridge and registration implementation.
- The GPU solver checks its convergence flag between four-iteration batches instead of always submitting 60 iterations. Analytical derivatives, robust weighting, Cholesky solves, damping, acceptance and updates remain on the GPU. The CPU only observes the completion flag. At most three frozen iterations execute at the end of a batch.
- GPU matcher results are indexed directly as arrays. Per-match Python DMatch construction/square roots and native-coordinate projections that were unused in GPS-center mode are removed. SIFT settings, pair candidates, ratio threshold, RANSAC gates and retained-point limits are unchanged.
- Per-pair log output is throttled outside diagnostic mode. Detailed corner reports and GPS leave-one-out fits/heading diagnostics only run in diagnostic mode. The actual geographic fit and acceptance gate remain active.
- Normal coherent rendering omits full-map coverage/source-ID buffers and their writes. Primary full-resolution buffers fall from 12 to 8 bytes per pixel. Selection and exposure correction are unchanged.
- `vectors` now stops before pixel rendering and saves `project.json` plus geometry/source references. It no longer renders a map while claiming vector-only output.

Selection checks, SIFT/RANSAC, relative alignment, geographic fitting, coherent region selection, and exposure correction remain. Their absence of isolated quality-percentage measurements is not evidence that they are unnecessary.

## Fresh measurements

Full process wall time, including startup and exports:

| Dataset | Previous ADM | Lean ADM | Saved ODM | Previous / lean | Lean output |
|---|---:|---:|---:|---:|---:|
| SIYI, 29 images | 30.000 s | 9.560 s | 279.824 s | 3.14x | 64.31 MiB |
| Archive, 63 images | 33.635 s | 9.112 s | 352.776 s | 3.69x | 47.37 MiB |
| killerofmaps, 306 images | 119.488 s | 83.206 s | Not measured | 1.44x | 52.73 MiB |

Previous output sizes were 197.07, 152.68 and 80.20 MiB respectively. Baseline SIYI/Archive timings include corner diagnostics and optional golden/shadow exports; lean timings include only the neutral map. The 306-image baseline already skipped lighting. These are single-run comparisons with filesystem cache, desktop load and thermal conditions uncontrolled. Saved ODM runs perform 3D reconstruction on CPU in ARM64 Docker; ADM runs a planar pipeline natively with Metal.

Internal stage timings (process startup/final report overhead is outside these totals):

| Stage | SIYI | Archive | 306 images |
|---|---:|---:|---:|
| Input selection, metadata, pair setup | 1.450 s | .891 s | 15.604 s |
| SIFT and GPU descriptor upload | 2.203 s | 1.238 s | 6.996 s |
| Descriptor matching and ratio filtering | 1.974 s | 3.942 s | 43.173 s |
| Pair verification and bookkeeping | .186 s | 1.006 s | 7.175 s |
| Global alignment and geographic fit | .119 s | .423 s | 7.398 s |
| Composition, map export and geometry | 3.206 s | 1.195 s | 2.395 s |
| Total internal pipeline | 9.140 s | 8.699 s | 82.771 s |

The 306-image alignment stage fell from 35.222 to 7.398 s. It converged after 9 iterations and encoded 12 instead of 60. SIYI encoded 16 (13 active), Archive 8 (7 active). Matching remains the largest 306-image cost. Solver residuals and map transforms match the saved results exactly.

## Crispness against ODM

The comparison first applies an area filter when reducing image size, then warps both neutral maps onto the same physical grid. Both statistics use exactly the same covered interior pixels, eroded by five pixels to exclude transparency boundaries. This avoids treating the larger ADM output dimensions or downsampling aliasing as extra sharpness.

| Dataset | Physical sampling | Mean edge strength: ADM / ODM | Fine-edge variance: ADM / ODM |
|---|---:|---:|---:|
| SIYI | 5.00 cm/pixel | .784x (21.6% lower) | .448x (55.2% lower) |
| Archive | 21.64 cm/pixel | 1.066x (6.6% higher) | 1.340x (34.0% higher) |

Mean edge strength is mean Sobel gradient magnitude. Fine-edge variance is Laplacian variance. These are edge/texture response statistics, not “times sharper,” resolved-detail measurements, or survey-accuracy scores. Noise, remaining seams and geographic differences affect them. The result is mixed: ODM has stronger edge/texture response on SIYI; ADM has somewhat stronger response on Archive. No corresponding ODM raster exists for the 306-image comparison.

Older reports applied direct bilinear minification and measured each map's separate valid footprint. Their much higher ADM sharpness ratios are not interchangeable with this area-filtered, shared-coverage comparison.

The crop figures use fixed spatial quantiles of shared coverage, rather than selecting locations for favorable sharpness. They are shown at 2x pixel size on the common geographic grid; geographic misalignment is visible and is not silently fitted away.

- [SIYI crispness crops](../output/comparison/siyi/evaluation_lean/crispness_crops.png)
- [Archive crispness crops](../output/comparison/archive/evaluation_lean/crispness_crops.png)
- [SIYI full measurement](../output/comparison/siyi/evaluation_lean/comparison.json)
- [Archive full measurement](../output/comparison/archive/evaluation_lean/comparison.json)
- [306-image full measurement](../output/killerofmaps/evaluation_lean/comparison.json)

## Run and inspect

```sh
.venv/bin/python -m adm.pipeline /path/to/images \
  --output output/my_map --config /path/to/flight_config.json \
  --gps-centers --backend metal --gpu-optimized
```

This uses the PNG profile with a default 256 MiB output allowance. Use `--export-profile vectors` to defer all rendering (32 MiB default allowance). Use `--export-profile diagnostic --max-output-mb 512` for support images and extra diagnostics; add `--stylized-shadows` if that diagnostic image is wanted. Source photographs remain necessary for vector projects. Existing output directories are not automatically purged.

```sh
.venv/bin/python scripts/validate_metal.py
.venv/bin/python scripts/benchmark_gpu_revision.py siyi --output output/comparison/siyi/another_lean_run
.venv/bin/python scripts/compare_lean.py siyi
```

The benchmark refuses to overwrite a nonempty result. `compare_lean.py` evaluates the canonical `metal_lean` directory against preserved baselines and saved ODM rasters, outside the pipeline timer. It fails if neutral-map pixel parity is lost.

Validation covers GPU numerical behavior and convergence, diagnostic/PNG pixel equality, optional cache reuse, vector exports bypassing rendering, absence of lean cache/support artifacts, and comparison filtering/masking. All 40 tests pass.
