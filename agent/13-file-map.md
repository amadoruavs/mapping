# File map

Short index of files a future agent will open. Not every file.

## MapRepair

| File | Role |
|---|---|
| `run.py` | CLI + hardcoded paths + finish settings |
| `run.sh` | OS deps, private venv, compile, launch `web.py` |
| `web.py` | Loopback HTTP server |
| `web_worker.py` | Isolated job process |
| `web_ui/` | Static HTML/CSS/JS, including grading UI |
| `map_geo_checker.py` | SIFT/RANSAC/tile evidence |
| `patch_blend.py` | Conservative repair blend |
| `map_io.py` | TIFF/PNG/JPEG + alpha + CRS |
| `georeference.py` | World file / bounds |
| `adaptive_finish.py` | Default offline cinema |
| `cinematic_finish.py` | Older virtual-exposure looks |
| `cinematic_grade.py` | `pro_camera` and other presets |
| `advanced_grade.py` | Interactive 52-control grader |
| `presentation.py` | Sharpen profiles |
| `native.cpp` / `native.py` | Compact C++ FSR + plane tracer |
| `openrt/` + `openrt_backend.py` | Real OpenRT worker |
| `third_party/OpenRT/` | Pinned upstream |
| `parallel_runtime/` | Shared scheduler copy |
| `tests/` | Unit tests |
| `results/siyi_odm_openrt_golden_hour/` | Small saved example finish |
| `AMD_KERNELS.md`, `TARGET_250.md`, `UPDATE_LAPTOP.md` | Laptop / AMD notes |

## ISM

| File | Role |
|---|---|
| `run.py` | Launcher |
| `run_visual.py` | Visual process |
| `ISM.metal` | All kernels |
| `runtime/adm/*.py` | Host pipeline |
| `runtime/adm/gpu_alignment.py` | Metal GN |
| `runtime/adm/pipeline.py` | Orchestration + GPS gate |
| `runtime/adm/match_graph.py` | Disconnected recovery |
| `runtime/adm/fsr_geometry.py` | Generated-view gates |
| `runtime/adm/geographic_alignment.py` | Similarity + reports |
| `GUIDE.md` | Operator guide |
| `runtime/OPENSFM_PORT.md` | Residual port |
| `runtime.zip` | Rollback bytes |

## Root mapper

| File | Role |
|---|---|
| `adm/pipeline.py` | CLI orchestration |
| `adm/config.py` | Defaults including 30 m AGL placeholder |
| `adm/metadata.py` | EXIF/XMP adapter |
| `adm/registration.py` | Matching + graph |
| `adm/gpu_alignment.py` | Metal alignment |
| `adm/coherent_composite.py` | Source labels |
| `adm/metal.py` | dylib build + buffers |
| `shaders/*.metal` | Split shader files (pre-ISM) |
| `shaders/ADMMetal.m` | Bridge |

## Docs worth opening

| File | Role |
|---|---|
| `MEMORY.md` | Dated log — recheck before quoting |
| `docs/LEAN_PIPELINE.md` | Lean trim + 83 s measurements |
| `docs/FSR_PIPELINE.md` | How FSR was actually wired |
| `docs/GPU_ALIGNMENT.md` | Alignment revision |
| `docs/METAL_BACKEND.md` | Memory / numerical notes |
| `docs/OPENSFM_PORT.md` | Geographic residuals |
| `docs/README.md` | Index of the 2026-09-13 research set |
| `docs/research-log.md` | Verified vs estimated |
| `OPTIMIZATION.md` | Deferred ideas, not current behavior |
| `RESULTS.md` | Early result notes |

## Acceleration

`acceleration/parallel_runtime/` is the canonical copy; MapRepair and the ODM port received syncs. `acceleration/docs/upstream/` has downloaded AMD/MLIR-AIE notes (small here; huge validation rasters were dropped).

## Experiments / history

| Path | Role |
|---|---|
| `experiments/gps_chunk_stack/` | 10-photo GPS tile stack, no visual graph |
| `checkpoints/full_metal_86s/` | Old source checkpoint |
| `backups/ism_before_rollback_*` | Rejected geo-priors |
| `notes/odm-fedora/` | Native ODM port write-ups |
