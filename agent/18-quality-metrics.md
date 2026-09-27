# How to read the numbers in MEMORY.md

## Timing

| Name | Includes |
|---|---|
| Full-process wall | Process start → final PNG/JPEG, including GPU init and export |
| Internal pipeline | Stage timers inside the worker, excluding import/startup |
| Presentation-only | Sharpen/grade/lasso/FSR on an *existing* neutral map |
| Alignment-only | Matching already done; just GN + geography |
| Browser run | Upload + worker + download |

Never mix them. A 9 s grade replay is not a 9 s mapper.

## Alignment residuals

| Number | Unit | What it is |
|---|---|---|
| Mapped-coordinate objective | mosaic px | Can fall while the map shrinks |
| Symmetric transfer median/p95 | source px | i→j and j→i, the honest visual number |
| Image-center / GPS | metres | Projected principal point vs GPS. Near-nadir proxy. GPS used in the fit |
| Camera-center / GPS | metres | Physical camera (opensfm_planar) vs GPS |
| ODM reprojection | px in SfM | Not comparable to the above |

## Compositor / "quality" proxies

| Number | What it is **not** |
|---|---|
| Source-transition fraction | Not a quality score. Lower means larger same-photo regions |
| Sobel mean / Laplacian variance vs ODM | Edge response on a shared grid, not "sharper detail" |
| SSIM vs ODM | Agreement with ODM, not truth |
| CLAHE/detail RGB mean error vs CPU | Implementation parity, not prettier maps |
| "19 consistent / 0 patched" | Evidence gates, not "the map is perfect" |

## Memory

| Number | What |
|---|---|
| Peak process RSS | CPU-visible resident set. Not total unified GPU memory |
| Planned scratch | Solver formula, capped by `alignment_budget_mb` |
| `coherent_budget_mb` | Coarse observation target, not a hard total alloc |
| `registration.npz` size | Saved matches/params, not RAM |

## Crispness comparison method that is valid

1. Area-filter when minifying.
2. Warp both maps onto the same physical grid.
3. Use the exact shared coverage mask, eroded (~5 px) so transparent borders do not win.
4. Then compute Sobel / Laplacian.

Older bilinear-only, separate-footprint ratios **overstated ADM** and must not be quoted.

## Parity tests vs flight quality

A test that says Metal grade matches CPU on a fixture, or OpenRT is a no-op on a plane, does not say the 94-photo mosaic is good. Keep those sentences in different paragraphs.
