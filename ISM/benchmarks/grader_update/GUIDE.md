# ISM — integrated Metal mapping

ISM keeps all custom GPU shaders in **ISM.metal**: 91 entry points compiled into
one Metal 4 library. MPS performs matrix operations and Cholesky solves. Keep
`run.py`, `runtime/`, and the shader together; the shader cannot load photos or
run the application by itself.

## Current default and geographic option

The default is the best verified visual alignment: bidirectional source-image
transfer error, real-photo rendering, and the gentle SIYI presentation settings.
The collapsing legacy objective, generated frames and GPS mismatch bypass are
off in `run.py`.

`lasso_crop=True` is now enabled in `run.py`. Lasso trims the finished map
**after sharpening and color grading**, immediately before the final PNG export.
It makes the discarded perimeter transparent without resampling, changing the
canvas, or changing retained pixels or their geographic coordinates. Its
polygon, retained coverage and time are saved in `lasso_crop.json`. The neutral
map remains available before presentation; `map_graded.png` is the final cropped
output. Set `lasso_crop=False` to disable it.

The new `opensfm_planar` option adds OpenSfM GPS and camera-orientation residuals
to a physical camera solve. Both methods are available in the same compiled
library. The geographic option is **not promoted to default**: it improves one
GPS proxy but slightly worsens visual residuals on the 94-photo test.

## Run it

Requirements: Apple silicon, macOS 26+, Xcode Command Line Tools, Python 3.12,
and the packages in `requirements.txt`. The tested machine is an Apple M4 Pro
with 24 GB unified memory. The existing parent `.venv` already has the packages.

From the ADM directory:

```sh
# Default visual method
.venv/bin/python ISM/run.py --input /Users/avneh/Downloads/mapv2 --output /Users/avneh/Code/ADM/ism_visual_next

# New geographic method
.venv/bin/python ISM/run.py --input /Users/avneh/Downloads/mapv2 --output /Users/avneh/Code/ADM/ism_geo_next --geographic-solver opensfm_planar

# Compile all kernels / run portable validation
.venv/bin/python ISM/run.py --compile-only
.venv/bin/python ISM/run.py --self-test
```

Or edit `INPUT_REPO`, `OUTPUT_DESTINATION`, and `SETTINGS` at the top of `run.py`,
then run it directly. Input is a photo directory. Output must be new or empty;
existing results are preserved. The launcher writes `run.log` and `timing.json`.
Its timer includes fresh worker startup, imports, GPU initialization, raw image
processing and final PNG export. The output cap is 768 MiB.

The two saved methods use these common settings:

| Setting | Value |
|---|---:|
| Actual batch | 94 images in `Downloads/mapv2` |
| Height above ground | 51.816 m |
| Assumed horizontal FOV | 81 degrees |
| Registration width | 1280 pixels |
| Alignment iteration cap | 600, early stop |
| Alignment workspace allowance | 1024 MiB |
| Coarse compositor allowance | 128 MiB |
| Maximum map dimension | 6500 pixels |
| Interpolation levels | 0 |
| Presentation scale | 1.0 |
| Spatial FSR / ray lighting | Off |
| Presentation / color grade | `siyi_low_light` |

## Exact pipeline

1. Select and hash originals; read EXIF/XMP and configured flight assumptions.
2. Use GPS centre distances to propose overlaps. Extract SIFT features; match
   descriptors with MPS; validate pairs with the existing ratio test and RANSAC.
   Recover disconnected graph components using verified image matches.
3. Solve the visual graph on Metal with mean bidirectional soft-L1 transfer
   error in source-image pixels. The anchor fixes its arbitrary coordinate
   system. Check convergence every four encoded iterations.
4. With `visual`, robustly align the finished mosaic to GPS centres.
   With `opensfm_planar`, use the visual result to initialize physical camera
   centres/rotations, then jointly optimize visual and geographic residuals on
   Metal/MPS. No second geographic fit changes that result after convergence.
5. Select coherent source regions and sample original detail, with bounded
   low-frequency exposure correction. Each output pixel retains one source.
6. Apply SIYI chroma cleanup, Lanczos at the configured scale, positive detail,
   blended CLAHE, fine sharpening and gentle contrast, then the SIYI grade.
7. Apply the final GPU lasso to the finished presentation image, then export
   its PNG. Save the neutral map, geographic geometry, diagnostics and timing.

The geographic method uses OpenSfM's `DataPriorError`, `UpVectorError` with
Cauchy loss, and WGS84/topocentric equations, adapted to fixed planar ground.
A second observed axis retains heading information at nadir. GPS uncertainty
is 10 m unless measured horizontal accuracy/sidecar data is available; altitude
uncertainty is 10 m and recorded orientation uncertainty is 10 degrees.
The geographic constraints participate in every joint refinement iteration.

This is a port of geographic residuals, **not all of ODM**. Dense structure from
motion, triangulated terrain, lens calibration, rolling-shutter correction,
GCP adjustment and mesh orthorectification are not implemented. Read
[runtime/OPENSFM_PORT.md](runtime/OPENSFM_PORT.md) for exact source links,
equations, adaptations and the GPU/CPU division.

## Current measured result: 94 photos

These mapping benchmarks predate the newly enabled final lasso and exclude its
cost. The new crop is separately checked on the saved 94-photo graded map under
`benchmarks/final_lasso_94/`; the full mapping benchmark was not rerun for this
change. The main wall timer includes the crop on future runs.

| Measurement | Preserved visual ISM | Geographic ISM |
|---|---:|---:|
| Full-process run 1 | 22.427 s | 20.855 s |
| Full-process run 2, reversed order | 20.742 s | 21.615 s |
| Median of two runs | 21.584 s | 21.235 s |
| Median source-pixel transfer residual | 1.099 px | 1.230 px |
| 95th-percentile transfer residual | 3.190 px | 3.638 px |
| Median image-centre/GPS proxy | 2.382 m | 2.141 m |
| 95th-percentile image-centre/GPS proxy | 6.829 m | 6.084 m |
| Final dimensions | 6500 × 5978 | 6500 × 6021 |

Both retain all 94 originals. The same raw inputs, matching settings and
presentation recipe are used. These are fresh processes without match-cache
reuse; OS/driver caches were not forcibly purged. Order reversal changes which
method is faster, so there is **no clear speed win**. Both complete under two
minutes on this batch. Neutral maps, graded maps and transforms are byte-identical
between repeats of each method.

The shared GPS proxy compares the projected image centre against recorded GPS,
a near-nadir approximation, and is not surveyed accuracy. The new physical
camera-centre GPS residual is separately 3.517 m median / 6.190 m p95; it must
not be conflated with the image-centre proxy. Some visible seams remain in both
maps. There is no demonstrated overall quality winner or added measured detail.

45 portable tests pass, including physical geometry/Jacobian/Hessian checks,
noisy-camera convergence, horizon rejection and existing presentation tests.
Actual unmodified OpenSfM C++ functors and WGS84 code were also compared over
128 fixtures; maximum Metal orientation-cost difference was 4.32e-7.
This is residual parity, not identical full Ceres/OpenSfM reconstruction results.

- [Full report and raw measurements](benchmarks/odm_geo94/RESULTS.md)
- [Overview comparison](benchmarks/odm_geo94/overview_comparison.png)
- [Same-area road comparisons](benchmarks/odm_geo94/road_comparison.png)
- [Visual map](benchmarks/odm_geo94/old/map_graded.png)
- [Geographic map](benchmarks/odm_geo94/new_visual_init/map_graded.png)
- [Preserved old package](benchmarks/odm_geo94/source_old/run.py)

No 306-photo dataset or old C dataset was run for this comparison.

## Appearance and optional visual runner

SIYI appearance remains tuned from IMG_0556.jpg: +.08 stops exposure, .035
shadow lift, subtle RGB balance 1.05/1/.94, .02 luma contrast and restrained warm
highlights. Chroma cleanup preserves luminance and is separate from the removed
NLM denoise pass. No new grading was introduced in the geographic comparison.

For the other standard appearance, select `presentation_profile='standard'`
and `color_grade='pro_camera'`. The pro-camera whites and vibrance controls live
inside the grade preset; no second compression pass is applied afterward.

`run_visual.py` remains the separate gallery runner with its own True/False
visual-process flag. Its preview/export overhead is outside the base benchmarks.
FSR generation and flat-plane ray tracing remain available as explicit
experimental settings in the integrated library. Prior tests found generated
views could introduce road fragments and substantially increase runtime; they
are off in the tested defaults. A flat ray-traced plane cannot create measured
building or terrain shadows.

## Files and source provenance

- `ISM.metal`: only custom shader source, including the new geographic kernels.
- `run.py`: settings, compile/test options, fresh-process launcher and timer.
- `runtime/adm/`: CPU orchestration, metadata, SIFT/RANSAC, codecs and packing.
- `runtime/ADMMetal.m`, `runtime/FSRMetal.m`: Metal/MPS host resource bridges.
- `runtime/bundle_manifest.json`: kernel inventory and source hashes.
- `runtime/OPENSFM_PORT.md`, `runtime/LICENSES.txt`: port scope and upstream notices.
- `build/`: cached native libraries, compilation and validation reports.

OpenSfM source: OpenDroneMap fork revision
`c5328439465e6ace011f39077d1077d7b1cdd65d`, pinned by the inspected ODM build.
AMD FSR SDK source: `c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`.
Historical results and rollback archives remain preserved; their settings and
measurements do not describe the current defaults.
