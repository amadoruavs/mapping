# ISM

Integrated Metal mapping. This is the custom mapper you should open first if someone says "our mapper" or "the Mac pipeline."

Guide written with the code: [ISM/GUIDE.md](../ISM/GUIDE.md).

## Required files

The whole `ISM/` folder. `ISM.metal` cannot load photos or run the app. `run.py` cannot map without `runtime/` and the shader.

| Path | What |
|---|---|
| `ISM/run.py` | Paths + SETTINGS + process timer |
| `ISM/run_visual.py` | Optional HTML gallery; shares SETTINGS, different path defaults |
| `ISM/ISM.metal` | 92 kernels, one Metal 4 library |
| `ISM/runtime/adm/` | Host Python (do not confuse with repo-root `adm/`) |
| `ISM/runtime/ADMMetal.m`, `FSRMetal.m` | ObjC bridge |
| `ISM/runtime/OPENSFM_PORT.md` | Geographic residual port |
| `ISM/runtime.zip` | Historical rollback snapshot |
| `ISM/tests/` | Portable tests |
| `ISM/requirements.txt` | Same three packages as the root mapper |

## Current SETTINGS in `run.py`

Copied from the archived file. If they have drifted, the file wins.

```
altitude_agl_m                 51.816
allow_gps_mismatch             False
skip_first                     0
registration_width             1280
coherent_budget_mb             128
alignment_budget_mb            1024
alignment_max_iterations       600   # early-stop still applies
alignment_symmetric_transfer   True  # do not turn off without a reason
geographic_solver              visual
geographic_*_sigma             10 m / 10 m / 10 deg
interpolation_levels           0     # real photos only
fsr                            False
presentation                   True
presentation_fsr_passes        0
presentation_scale             1.0
presentation_raytrace          False
presentation_profile           siyi_low_light
color_grade                    pro_camera
lasso_crop                     True  # after grade, before PNG
lasso_inset_ratio              0.005
lasso_opening_ratio            0.06
lasso_angular_radius           12
```

Hardcoded paths still say `/Users/avneh/Downloads/mapv2` and `/Users/avneh/Code/ADM/final_test`. Always override.

## Two geographic methods

| `geographic_solver` | Behavior |
|---|---|
| `visual` (default) | Align mosaics visually, then a robust similarity to GPS centers. |
| `opensfm_planar` | After a visual init, jointly refine 6-DOF cameras on a plane using OpenSfM-style GPS/up-vector residuals. |

`opensfm_planar` is a tradeoff, not a proven upgrade. 94-photo comparison (historical, pre-grader/lasso):

| | visual | opensfm_planar |
|---|---:|---:|
| Median wall of two runs | 21.58 s | 21.24 s |
| Median source transfer | 1.099 px | 1.230 px |
| Median image-center/GPS proxy | 2.382 m | 2.141 m |
| Output size | 6500×5978 | 6500×6021 |

No clear speed or quality winner. Do not compare the *physical camera-center* residual (3.52 m) of the new method to the *image-center* residual of the old method.

## SIYI presentation

`siyi_low_light`: chroma-only edge-aware cleanup 0.7 (luma preserved), Lanczos 2× if scale says so (ISM default scale is 1.0), detail 0.65, CLAHE blend 0.35, fine 0.15.

`pro_camera` is the user-supplied cinematic preset (contrast 0.35, sat 0.95, vignette 0.08, grain 0, blacks 0.05, shadows 0.02, highlights 0.06, whites −0.06, clarity 35, sharpness 90). When presentation sharpening already ran, do not also stack the grader's unsharp (`GRADE_SHARPEN` analogue is off in the ISM path used for MapRepair; in ISM the grade's own clarity/sharpness follow the presentation pass).

## Commands

```sh
.venv/bin/python ISM/run.py --input PHOTOS --output EMPTY
.venv/bin/python ISM/run.py --geographic-solver opensfm_planar --input PHOTOS --output EMPTY
.venv/bin/python ISM/run.py --compile-only
.venv/bin/python ISM/run.py --self-test
```

Needs Apple silicon, macOS 26+ (coherent textures / Metal 4), Xcode CLT, Python 3.12.

## 564-photo flight

A 564-image 4K set (`Downloads/maps`) disconnected until recovery SIFT contrast was lowered to 0.01 for *recovery attempts only*. That produced 11 verified bridges, then hit the 512 MiB alignment workspace guard. Budget was raised to 1024 MiB. **Full 564 E2E was never verified.** Do not claim it works.

A 600-iteration attempt on that set collapsed in scale. See [12-known-failures.md](12-known-failures.md).

## Tests

GUIDE.md last recorded 50 portable tests including Jacobians, noisy-camera convergence, horizon rejection, and OpenSfM C++ residual parity (max Metal orientation-cost error ~4.3e-7). That is residual parity, not Ceres/OpenSfM solution parity.

## Visual runner

`run_visual.py` is a separate launcher. `VISUAL_PROCESS` toggles a saved/live HTML gallery. Observation hooks were pixel-identical in tests; a full visual flight smoke was still pending when this was archived.
