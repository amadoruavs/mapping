# ADM project context (as of 2026-09-13)

## What is in the repo

`/Users/avneh/Code/ADM/` currently contains one thing: `ODM/`, a fresh, unmodified clone of
[OpenDroneMap/ODM](https://github.com/OpenDroneMap/ODM) at commit `1a71341` (merge of PR #2076
"improve-ci", 2026-09-10), version **3.6.2** (`ODM/VERSION`). Nothing has been built yet:
there is no `ODM/SuperBuild/install/`, no `.pixi/` env, no datasets.

The remote is the upstream `https://github.com/OpenDroneMap/ODM.git`. There is no ADM-specific
fork or branch yet. A separate agent ("Codex") is expected to land code changes; until that
happens **these docs are research only and no code is to be touched**.

## What ODM is (one paragraph)

ODM is a command-line photogrammetry pipeline. Input: a folder of drone JPEG/TIFF/DNG images
(EXIF GPS + timestamps). Output: sparse + dense point cloud (LAZ), a 3D textured mesh
(OBJ, optional GLB and Cesium 3D Tiles), a 2.5D mesh, a georeferenced orthophoto (GeoTIFF,
optional COG/PNG/KMZ/XYZ tiles), DSM/DTM rasters, and an HTML/PDF report. It is orchestrated in
Python (`run.py` -> `stages/odm_app.py`) and shells out to compiled binaries built by a CMake
SuperBuild under a pixi/conda-forge environment.

## Stage graph (from `ODM/stages/odm_app.py`)

```
dataset -> split -> merge -> opensfm -> [openmvs] -> odm_filterpoints -> odm_meshing
        -> mvs_texturing -> odm_georeferencing -> odm_dem -> odm_orthophoto -> odm_report
        -> odm_postprocess
```

`openmvs` is skipped when `--fast-orthophoto` is set (orthophoto from the sparse cloud).

Binaries resolved in `ODM/opendm/context.py` (all under `SuperBuild/install/bin`):

| Stage | Binary / library | Source repo (pinned in SuperBuild) |
|---|---|---|
| opensfm | `opensfm/bin/opensfm` (Python + pybind C++) | OpenDroneMap/OpenSfM `c5328439` |
| opensfm (GPU SIFT) | `opensfm/opensfm/pypopsift*.so` | OpenDroneMap/pypopsift `fe2d1cc` (CUDA only) |
| openmvs | `DensifyPointCloud`, `ReconstructMesh` | OpenDroneMap/openMVS tag `355` |
| odm_meshing | `PoissonRecon`, `dem2mesh`, `dem2points` | OpenDroneMap/PoissonRecon `290`, dem2mesh `334` |
| mvs_texturing | `texrecon` | OpenDroneMap/mvs-texturing `c5a4d0c` (+ mve `356`) |
| odm_filterpoints | `FPCFilter` | OpenDroneMap/FPCFilter |
| odm_dem | PDAL (conda) + `RenderDEM` | conda pdal 2.8.4 |
| odm_orthophoto | `odm_orthophoto` | OpenDroneMap/odm_orthophoto `355` |
| 3D tiles | `Obj2Tiles`, draco | OpenDroneMap/Obj2Tiles, google/draco |
| point cloud classify | OpenPointClass | OpenDroneMap/OpenPointClass |
| EPT tiles | Entwine | connormanning/entwine |
| sky/bg removal | onnxruntime (conda, CPU build on macOS) | models downloaded at runtime |

## Build system

* `pixi.toml` defines environments `default` (build + cpu), `prod`, `gpu`, `gpu-prod`.
  Platforms: linux-64, win-64, linux-aarch64, **osx-arm64 (macOS >= 14.0)**. The `gpu`
  feature (CUDA 12) exists only for linux-64 and win-64.
* `pixi run build` = `cmake -S SuperBuild -B SuperBuild/build && cmake --build SuperBuild/build`.
* `SuperBuild/CMakeLists.txt` reads `ODM_ENABLE_CUDA` from the environment (set by the pixi
  `gpu` feature) and forwards it as `-DOpenMVS_USE_CUDA` and gates `pypopsift`.
* macOS-specific handling already present: RPATH baking for conda libs, PoissonRecon
  `Makefile.macos` + patches, a CGAL header patch for clang 19, `OpenMVS_USE_SSE=OFF` on arm64,
  `DYLD_LIBRARY_PATH`/`DYLD_FALLBACK_LIBRARY_PATH` activation env.

## The three asks driving this research

1. **Custom golden hour shaders** — render/relight ODM outputs (textured mesh, orthophoto+DSM)
   to look like golden hour. See `docs/golden-hour-shaders/`.
2. **Metal on this laptop** — make the pipeline compile for and use Apple Metal on an M4 Pro.
   See `docs/gpu-acceleration/metal/`.
3. **Integrated Radeon** — same for an AMD APU laptop (Linux/Windows). See
   `docs/gpu-acceleration/radeon/`.

Both GPU asks are bounded by one fact: **every GPU path in ODM today is CUDA-only**
(see `03-gpu-touchpoints-today.md`), so on this Mac ODM currently runs 100% on CPU.

## Related sibling projects (for context only)

* `~/Code/UAVImaging2627` — SUAS 2026 detection work (ANetV1, synthetic data generator with
  sun-consistent shadows). Same owner; the "golden hour" interest likely overlaps with
  rendering realistic lighting for aerial imagery.
* `~/Code/UAVS25_26` — YOLO training pipeline.
* `~/Code/website-v5` — AmadorUAVs site (three.js / GLSL experience).
