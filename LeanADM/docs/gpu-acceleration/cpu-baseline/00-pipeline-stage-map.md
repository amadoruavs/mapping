# ODM 3.6.2 pipeline stage map: what runs, how it parallelizes, what it costs

Scope: ODM at `/Users/avneh/Code/ADM/ODM` (commit `1a71341`, version 3.6.2, pixi/conda-forge
SuperBuild). Every number below was read from the code at that commit unless a URL says
otherwise. Target machine: MacBook Pro, Apple M4 Pro, 8 P-cores + 4 E-cores (`sysctl
hw.perflevel0.physicalcpu`=8, `hw.perflevel1.physicalcpu`=4), 24 GB unified memory, macOS 26.6.
On this machine `opendm/gpu.py::has_gpu()` always returns False (it looks for `nvidia-smi`),
so **every stage runs on the CPU today**.

Research status: documentation only; nothing has been built or run in this checkout.

## 1. Stage order and wiring

`stages/odm_app.py` builds a linked list of `ODM_Stage` objects (`opendm/types.py:414`):

```
dataset -> split -> merge -> opensfm -> openmvs -> odm_filterpoints -> odm_meshing
        -> mvs_texturing -> odm_georeferencing -> odm_dem -> odm_orthophoto
        -> odm_report -> odm_postprocess
```

* With `--fast-orthophoto`, `opensfm` connects straight to `odm_filterpoints`
  (`odm_app.py:69-73`); `openmvs` never runs and `--skip-3dmodel` is forced on
  (`opendm/config.py:936-938`).
* `--end-with <stage>` stops after that stage; `--rerun <stage>` runs only that stage;
  `--rerun-from <stage>` reruns that stage and all later ones (`config.py:193-218`,
  `RerunFrom` action slices `processopts`).
* Each stage's wall time is appended to `<project>/benchmark.txt` as
  `"<stage> runtime: <seconds> seconds"` (`opendm/system.py::benchmark`, called from
  `ODM_Stage.run`, `types.py:451`) and recorded in `<project>/log.json` (`opendm/log.py`:
  `stages[].startTime`, per-subprocess `processes[]`, overall `totalTime`). See
  `04-benchmark-and-profiling-plan.md`.
* Progress weights in `odm_app.py` (5, 25, 50, 52, 60, 70, 80, 90, 98, 99, 100) are UI
  guesses, not measured time shares.

## 2. Per-stage table

Legend for "Parallelism": OMP = OpenMP via conda `llvm-openmp` (libomp); Py-threads = Python
threads (GIL-bound unless the work is in C); Py-procs = Python processes (joblib "threading"
backend in OpenSfM is threads; see OpenSfM notes). "Bound" is the expected bottleneck on this
laptop.

| # | Stage (`processopts` name) | Binary / library | Parallelism | Bound | Key cost drivers (flags) | GPU path today |
|---|---|---|---|---|---|---|
| 1 | `dataset` | Python (`stages/dataset.py`): EXIF via exifread/piexif, `ODM_Photo`, optional video frame extraction (OpenCV), optional sky/bg masks (onnxruntime) | `parallel_map` Py-threads = `--max-concurrency` for masks only | IO / single-core Python | image count; `--sky-removal`, `--bg-removal` (ONNX, CPU EP on macOS), `--video-*` | ONNX CUDA EP only (not on macOS) |
| 2 | `split` / `merge` | Python + OpenSfM `create_submodels`, `LocalRemoteExecutor` (`stages/splitmerge.py`) | sequential submodels locally; parallel only with `--sm-cluster` (ClusterODM) | as opensfm | `--split`, `--split-overlap`, `--sm-no-align`, `--merge` | none |
| 3 | `opensfm` | `SuperBuild/install/bin/opensfm/bin/opensfm` (Python + pybind11 C++; Ceres 2.2.0, OpenCV 4.12 from conda) | see section 3 | mixed: extraction/matching multi-core; reconstruction mostly **single-core** with Ceres threads in BA | `--feature-quality`, `--min-num-features`, `--feature-type`, `--matcher-type`, `--matcher-neighbors`, `--matcher-order`, `--sfm-algorithm`, `--use-hybrid-bundle-adjustment`, `--rolling-shutter`, `--orthophoto-resolution` (undistort size) | PopSift SIFT_GPU (CUDA only, `pypopsift` not built without `ODM_ENABLE_CUDA`) |
| 4 | `openmvs` | `DensifyPointCloud` (OpenDroneMap/openMVS tag `355`) | OMP, `--max-threads <max-concurrency>` | **CPU + memory** (depth-map working set) | `--pc-quality`, `--pc-skip-geometric`, `--pc-filter`, `--orthophoto-resolution`/`--ignore-gsd` (via `undist_image_max_size`), masks | OpenMVS CUDA (`-DOpenMVS_USE_CUDA=OFF` on macOS, `--cuda-device -2` passed) |
| 5 | `odm_filterpoints` | `FPCFilter` (OpenDroneMap/FPCFilter `331`) | OMP, `--concurrency <max-concurrency>` | CPU, memory ~ point count | `--pc-filter` (std-dev, meanK=16), `--pc-sample`, `--boundary`/`--auto-boundary` | none |
| 6 | `odm_meshing` | `PoissonRecon` (3D mesh) + `ReconstructMesh` (OpenMVS, decimation/cleanup); `renderdem` + `dem2mesh` + `ReconstructMesh` (2.5D mesh); `dem2points`+`PoissonRecon` when `--fast-orthophoto` | OMP: PoissonRecon `--threads max(1, max-concurrency-1)`; dem2mesh `-maxConcurrency`; renderdem `OMP_NUM_THREADS` | CPU + memory (octree depth) | `--mesh-octree-depth`, `--mesh-size`, `--skip-3dmodel`, `--use-3dmesh`, `--fast-orthophoto` | none |
| 7 | `mvs_texturing` | `texrecon` (OpenDroneMap/mvs-texturing `c5a4d0c`, MVE `356`) | OMP, `--num_threads=<max-concurrency>` | CPU (view selection / MRF) + IO (reads all undistorted TIFs) | image count and undistorted size, mesh size, `--texturing-skip-global-seam-leveling`, `--texturing-single-material`, `--gltf`, `--use-3dmesh` (skips 2.5D run), `--skip-3dmodel` (skips 3D run) | none |
| 8 | `odm_georeferencing` | `pdal translate` (conda PDAL 2.8.4), Python cropper (PDAL + GDAL), optional `pcclassify` (OpenPointClass, `OMP_NUM_THREADS`), `entwine`, `untwine` (COPC) | PDAL single-threaded translate; classify OMP | IO + single-core | `--crop`, `--pc-classify`, `--pc-rectify`, `--pc-las/--pc-csv/--pc-ept/--pc-copc`, `--align` (CODEM) | none |
| 9 | `odm_dem` | `renderdem` (OpenDroneMap/RenderDEM) + GDAL (`gdalbuildvrt`, `gdal_translate NUM_THREADS`, `gdal_fillnodata`) + `fastrasterfilter` (median smoothing) | OMP (`OMP_NUM_THREADS=max-concurrency`), GDAL `NUM_THREADS` | CPU for rasterization, IO for GeoTIFF writes | `--dsm`, `--dtm`, `--dem-resolution`, `--dem-gapfill-steps`, `--dem-decimation`, `--tiles`, `--cog` | none |
| 10 | `odm_orthophoto` | `odm_orthophoto` (OpenDroneMap/odm_orthophoto `355`; OpenCV + GDAL + Eigen, OpenMP) + GDAL post-steps | OMP (`OMP_NUM_THREADS=max-concurrency` set in `stages/odm_orthophoto.py:110`); GDAL `NUM_THREADS` | memory (`GDAL_CACHEMAX` = `get_max_memory_mb()`), CPU | `--orthophoto-resolution`, `--orthophoto-compression`, `--orthophoto-cutline`, `--tiles`, `--cog`, `--build-overviews`, `--orthophoto-png/kmz` | none |
| 11 | `odm_report` | Python (OpenSfM stats, `pdal translate` overlap raster, `gdaldem`, fpdf2 PDF) | single-core | IO | `--skip-report` | none |
| 12 | `odm_postprocess` | `Obj2Tiles` (prebuilt .NET binary) for `--3d-tiles`; copy of outputs | single | IO | `--3d-tiles`, `--copy-to` | none |

Binary paths are resolved in `opendm/context.py` (all under `SuperBuild/install/bin`).

## 3. `opensfm` stage in detail (`stages/run_opensfm.py`, `opendm/osfm.py`)

Sub-steps in order (each an `opensfm <command>` subprocess via `OSFMContext.run`, so each is
its own Python process that imports the pybind extension):

1. `setup` (Python, in-process): writes `opensfm/config.yaml`. Computes
   `feature_process_size` and matcher settings (numbers below).
2. `extract_metadata` -> `photos_to_metadata` (Python).
3. `detect_features` (`opensfm/features_processing.py`): producer/consumer.
   Readers (`read_processes` config, capped at half of `processes`) push images into a bounded
   queue; `parallel_map(process, arguments, processes, 1)` runs detection with `processes` =
   `--max-concurrency` (`osfm.py:254 "processes: %s" % args.max_concurrency`). Per-image work is
   OpenCV C++ (DSP-SIFT default) so threads do scale. **On macOS the memory cap is disabled**:
   `opensfm/context.py::memory_available()` shells out to `free -t -m` (Linux only) and returns
   `None` on Darwin, so `processes` is never reduced for RAM. Watch this with 24 GB
   (`02-apple-silicon-cpu-tuning.md`).
4. `match_features` (`opensfm/matching.py`): `parallel_map(match_unwrap_args, args, processes,
   jobs_per_process=2)`, `processes = processes_that_fit_in_memory(processes, 512 MB)` (again a
   no-op on macOS). Pair selection (`opensfm/pairs_selection.py`): union of GPS-distance /
   GPS-neighbors, Delaunay-graph rounds (`matching_graph_rounds`), time, filename order, BoW,
   VLAD candidates. With no GPS and every strategy disabled it falls back to all pairs (O(n^2)).
5. `create_tracks` (mostly single-core Python/C++).
6. `reconstruct` (incremental SfM): `_compute_pair_reconstructability` is
   `parallel_map`'d; growing the reconstruction is sequential. Bundle adjustment
   (`opensfm/src/sfm/src/ba_helpers.cc:262-264`):
   `ba.SetNumThreads(config["processes"])`, `ba.SetMaxNumIterations(10)`,
   `ba.SetLinearSolverType("DENSE_SCHUR")`. Ceres 2.2 uses its own C++ thread pool (OpenMP/TBB
   backends were removed in 2.2.0, see
   https://raw.githubusercontent.com/ceres-solver/ceres-solver/2.2.0/docs/source/version_history.rst).
   `DENSE_SCHUR` means the Schur complement is solved with Eigen dense Cholesky by default
   (`dense_linear_algebra_library_type` is not set by OpenSfM), so the conda BLAS only matters
   if Ceres was built to route dense ops to LAPACK. Community profiling of large sets found this
   sub-step dominant and largely single-core
   (https://community.opendronemap.org/t/profiling-opensfm/9179).
7. `export_stats` (unless `--skip-report`), `export_geocoords`.
8. `undistort` (`opensfm/undistort.py`): `parallel_map(undistort_image_and_masks, arguments,
   processes)`; writes `undistorted/images/*.tif` at `undistorted_image_max_size =
   max(gsd.image_max_size(...), get_depthmap_resolution(...))` (`run_opensfm.py:80-84`). These
   TIFs feed OpenMVS and texrecon, so `--orthophoto-resolution` indirectly sizes MVS input.
9. `export_visualsfm --points` (NVM for texrecon); `export_ply` when `--fast-orthophoto`;
   `export_openmvs` runs at the start of the `openmvs` stage.

### 3.1 `--feature-quality` -> `feature_process_size` (exact, `osfm.py:202-232`)

```
scale = {ultra: 1, high: 0.5, medium: 0.25, low: 0.125, lowest: 0.0675}
multiplier = 2 if megapixels < 2 else (0.5 if megapixels > 42 else 1)
factor = min(1, scale[q] * multiplier)
feature_process_size = clamp(int(max_dim * factor), 320, 4480)     # default 2048 if dims unknown
```

Worked examples (max_dim = longest side of the largest image):

| Image | MP | ultra | high (default) | medium | low | lowest |
|---|---|---|---|---|---|---|
| 4000x3000 (DJI FC300X, Toledo) | 12 | 4000 | 2000 | 1000 | 500 | 320 (270 clamped) |
| 4896x3672 (Sony WX220, Aukerman/Zoo/Waterbury) | 18 | 4480 (clamped) | 2448 | 1224 | 612 | 330 |
| 5472x3648 (Autel XT705, Helenenschacht) | 20 | 4480 | 2736 | 1368 | 684 | 369 |

Other OpenSfM config written by ODM (`osfm.py:249-266`): `flann_algorithm: KDTREE`,
`feature_min_frames: <--min-num-features>` (default 10000; OpenSfM lowers the SIFT peak
threshold until that many features are found, so higher values cost more per image),
`sift_peak_threshold: 0.066`, `matching_gps_distance: 0`, `bundle_outlier_filtering_type: AUTO`,
`triangulation_type: ROBUST`, `retriangulation_ratio: 2`, `undistorted_image_format: tif`.

### 3.2 Matching parameters (`osfm.py:237-296`)

* `--matcher-neighbors N > 0` -> `matching_gps_neighbors: N`, `matching_graph_rounds: 0`.
* `--matcher-neighbors 0` (default) -> `matching_gps_neighbors: 0`, `matching_graph_rounds: 50`
  (Delaunay triangulation of GPS positions plus 49 jittered rounds; roughly O(n) pairs).
* <= 3 photos -> `matching_gps_neighbors: 3`.
* `--matcher-order N` only applies to non-georeferenced sets (`matching_order_neighbors`).
* `--matcher-type`: `flann` (default) -> `FLANN`; `bow` -> `WORDS` and forces HAHOG features;
  `bruteforce` -> `BRUTEFORCE`. No GPS and no explicit `--matcher-type` -> BOW.
* `--use-hybrid-bundle-adjustment` -> `bundle_interval: 100`, `bundle_new_points_ratio: 1.2`,
  `local_bundle_radius: 1` (otherwise `local_bundle_radius: 0`).

## 4. `openmvs` stage in detail (`stages/openmvs.py`)

Command (single-scene path, `openmvs.py:64-93`):

```
DensifyPointCloud scene.mvs --resolution-level L --dense-config-file Densify.ini
  --max-resolution <undist_image_max_size> --min-resolution {640 | 320 for low/lowest}
  --max-threads <max-concurrency> --number-views-fuse 2 --sub-resolution-levels 2
  --archive-type 3 -w depthmaps -v 0 --cuda-device -2 [--geometric-iters 0] [--ignore-mask-label 0]
Densify.ini: "Optimize = 7\nMin Views Filter = 1\n"
```

then `DensifyPointCloud --filter-point-cloud -20 -i scene_dense.mvs` unless `--pc-filter 0`.
OOM (exit 137/143) triggers a retry with `--fusion-mode 1` and sub-scene tiling
(`--sub-scene-area 660000`), which is slower but bounded in memory.

### 4.1 `--pc-quality` -> depth-map resolution (exact, `opendm/utils.py:20-46`)

```
scale = {ultra: 0.5, high: 0.25, medium: 0.125, low: 0.0675, lowest: 0.03375}
multiplier = 2 if megapixels < 6 else (0.5 if megapixels > 42 else 1)
depthmap_resolution = max(320, int(max_dim * scale[q] * multiplier))   # 640 if dims unknown
resolution_level = 0 if undist_image_max_size <= depthmap_resolution
                   else round(log2(undist_image_max_size / depthmap_resolution))
```

| Image | ultra | high | medium (default) | low | lowest |
|---|---|---|---|---|---|
| 4000x3000 | 2000 | 1000 | 500 | 320 | 320 |
| 4896x3672 | 2448 | 1224 | 612 | 330 | 320 |
| 5472x3648 | 2736 | 1368 | 684 | 369 | 320 |

Each step up roughly quadruples pixels per depth map, matching the docs' "increases processing
time roughly by a factor of 4x" (https://docs.opendronemap.org/arguments/). Memory follows the
same 4x. `--pc-skip-geometric` drops the geometric-consistency pass (`--geometric-iters 0`), a
large saving on big sets per the docs and community threads
(https://community.opendronemap.org/t/processing-time-differences/15255).

## 5. Meshing, texturing, DEM, orthophoto cost drivers

* PoissonRecon: `--depth <mesh-octree-depth, clamped 1..14>` (default 11), `--pointWeight 4`,
  `--samplesPerNode 1`, `--bType 2`, `--linearFit`, `--threads max(1, max-concurrency-1)`
  ("poissonrecon can get stuck on some machines if --threads == all cores",
  `stages/odm_meshing.py:31`). Memory and time grow ~8x per +1 depth in 3D (octree), so
  depth 11->12 is the single most expensive knob after `--pc-quality`. Output is then decimated
  by `ReconstructMesh --target-face-num <2 * mesh-size>` (`opendm/mesh.py:191-202`).
* 2.5D mesh (always built unless `--use-3dmesh`): `renderdem` DSM at resolution derived from
  point spacing (`get_dem_radius_steps(..., 3, --orthophoto-resolution, pi/2)`; x8 coarser with
  `--fast-orthophoto`), `dem2mesh -maxVertexCount <mesh-size> -maxTileLength 2000
  -edgeSwapThreshold 0.15 -maxConcurrency <max-concurrency>`, then `ReconstructMesh`.
* texrecon: `-d gmi -o gauss_clamping -t none --max_texture_size 8192 (x3 if any image side
  > 8000 px) --num_threads=<max-concurrency> [--nadir_mode for 2.5D]`. Two runs by default
  (3D + 2.5D); `--skip-3dmodel` leaves only the 2.5D run, `--use-3dmesh` only the 3D run.
* DEM: `renderdem --resolution <dem-resolution/100 m> --radiuses r,r*sqrt2,... (gapfill
  steps) --tile-size 4096 --decimation <dem-decimation>` then GDAL VRT/translate with
  `NUM_THREADS=<max-concurrency>` and `GDAL_CACHEMAX <get_max_memory()>%`, then
  `fastrasterfilter --window-size 512 --radius 4` (median smoothing).
* Orthophoto: `odm_orthophoto -resolution <px per m> -inpaintThreshold 1.0 -co TILED=YES
  -co COMPRESS=DEFLATE -co PREDICTOR=2 -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 -co
  NUM_THREADS=<max-concurrency> --config GDAL_CACHEMAX <get_max_memory_mb()*1024*1024>`
  (`stages/odm_orthophoto.py:34-110`, `opendm/orthophoto.py:21-30`). Output pixel count is
  (area / resolution^2); DEFLATE with predictor is CPU-heavy on write.
* Both `--orthophoto-resolution` and `--dem-resolution` are floored by the GSD estimate
  (`gsd.cap_resolution`: `gsd * (1 - 0.1)`) unless `--ignore-gsd`; asking for 1 cm on a
  3 cm/px flight changes nothing except the undistort size logic (`gsd.image_max_size` with
  `gsd_error_estimate=0.5`, power-of-two downscale).

## 6. Where wall time goes (community evidence; no native macOS numbers exist yet)

No published run gives a per-stage split for ODM 3.x on Apple silicon; `benchmark.txt` from a
local run is the authoritative source (plan in `04-benchmark-and-profiling-plan.md`). Evidence
for the shape of the distribution:

* Small/medium georeferenced sets at defaults (feature high, pc medium): total times such as
  Aukerman (77 img) 15m52s, Toledo (87) 14m05s, Shitan (493) 1h12m on a Ryzen 9 5900X / 64 GB
  (odm-benchmarks CSV, 2025-03-22,
  https://raw.githubusercontent.com/OpenDroneMap/odm-benchmarks/master/data/benchmarks.csv).
  With defaults the expected split is roughly: opensfm 30-45 %, openmvs 25-40 %, meshing +
  texturing 15-25 %, DEM + orthophoto + report 5-15 % (ESTIMATE, to be measured).
* A laptop (Acer Aspire 5) on Aukerman: "Fast Ortho 5 min / Default 9 min / High Res 30 min"
  (https://community.opendronemap.org/t/odm-benchmarks-update/10367), i.e. dense
  reconstruction + 3D model is ~45 % of default time and the "high" preset is ~3x default.
* Large sets (1,500+ images): OpenSfM `reconstruct` "took a huge amount of time - ~10 hours",
  mostly single-core, ~120 GB RAM (https://community.opendronemap.org/t/profiling-opensfm/9179);
  bundle adjustment grows super-linearly with image count
  (https://community.opendronemap.org/t/opensfm-processing-time/11252). Above ~1,000 images
  SfM, not MVS, becomes the bottleneck unless `--split` is used.
* Quality presets move the balance: "High Resolution" (feature ultra, pc high) took 36m15s vs
  a "3D" preset (octree 12, mesh 300000, pc high, use-3dmesh) 5h57m on the same i7-10700K
  (https://community.opendronemap.org/t/odm-benchmarks-update/10367), i.e. octree depth 12 +
  large mesh can dwarf everything else.
* Apple silicon rows in odm-benchmarks are Docker-only (M1 Pro 5 cores/7.9 GB: Toledo 22m12s
  on ODM 2.9.2; M1 Max 8 cores/32 GB: Tuniu 1 18m with resize 2048) and therefore not
  representative of a native arm64 build.

## 7. Memory sizing helpers that every stage shares (`opendm/concurrency.py`)

* `get_max_memory(minimum=5, use_at_most=0.5)` returns a **percentage**:
  `max(5, (100 - virtual_memory().percent) * 0.5)`, i.e. half of the currently free share.
  Used as `GDAL_CACHEMAX <n>%` in DEM/report steps.
* `get_max_memory_mb(minimum=100, use_at_most=0.5)` returns **MB**:
  `max(100, available_MB * 0.5)`. Used for `odm_orthophoto`'s `GDAL_CACHEMAX` (bytes).
* Both use the `vmem` pypi package (`virtual_memory()`), evaluated when the stage starts, so
  a machine already under memory pressure gets a smaller cache.
* `parallel_map` (Python threads, PriorityQueue) retries single-threaded on any exception,
  which is ODM's generic OOM fallback for mask generation etc.
* `--max-concurrency` defaults to `multiprocessing.cpu_count()` = 12 here (`opendm/context.py:52`).
  The help text's "~1GB per thread and 2 megapixel image resolution" rule
  (https://docs.opendronemap.org/arguments/) gives 12 threads x (18 MP / 2 MP) ~ 100 GB for
  Aukerman-size images if taken literally; in practice OpenMVS's depth maps, not feature
  extraction, set the peak. On 24 GB expect to cap `--max-concurrency` at 8 (P-cores) for
  `--pc-quality high` and above (see `03-fast-settings-playbook.md`).

## 8. GPU touchpoints (for the Metal track)

Exactly three: PopSift SIFT (`osfm.py:298-317`, CUDA, `--feature-type sift` only),
OpenMVS `DensifyPointCloud` (`openmvs.py:82-86`, CUDA), onnxruntime EP selection
(`opendm/skyremoval/skyfilter.py`, `opendm/bgfilter.py`). Everything else is CPU. Detailed in
`docs/00-context/03-gpu-touchpoints-today.md`.

## Sources

* Code: `stages/odm_app.py`, `stages/run_opensfm.py`, `stages/openmvs.py`,
  `stages/odm_filterpoints.py`, `stages/odm_meshing.py`, `stages/mvstex.py`,
  `stages/odm_georeferencing.py`, `stages/odm_dem.py`, `stages/odm_orthophoto.py`,
  `stages/odm_report.py`, `stages/odm_postprocess.py`, `stages/splitmerge.py`,
  `opendm/osfm.py`, `opendm/utils.py`, `opendm/gsd.py`, `opendm/mesh.py`,
  `opendm/dem/commands.py`, `opendm/orthophoto.py`, `opendm/point_cloud.py`,
  `opendm/concurrency.py`, `opendm/context.py`, `opendm/config.py`, `opendm/system.py`,
  `opendm/log.py`, `opendm/types.py` (all at commit 1a71341).
* OpenSfM (ODM fork, commit c5328439): `opensfm/context.py`, `opensfm/features_processing.py`,
  `opensfm/matching.py`, `opensfm/pairs_selection.py`, `opensfm/reconstruction.py`,
  `opensfm/undistort.py`, `opensfm/src/sfm/src/ba_helpers.cc`,
  `opensfm/src/bundle/src/bundle_adjuster.cc`, `opensfm/config.py` -
  https://github.com/OpenDroneMap/OpenSfM/tree/c5328439465e6ace011f39077d1077d7b1cdd65d
* OpenMVS fork CMake: https://github.com/OpenDroneMap/openMVS/blob/355/CMakeLists.txt
* odm_orthophoto CMake (OpenMP, OpenCV, GDAL, Eigen):
  https://github.com/OpenDroneMap/odm_orthophoto/blob/355/CMakeLists.txt
* FPCFilter CMake (OpenMP): https://github.com/OpenDroneMap/FPCFilter/blob/331/CMakeLists.txt
* dem2mesh CMake (OpenMP, Apple clang branch): https://github.com/OpenDroneMap/dem2mesh/blob/334/CMakeLists.txt
* RenderDEM CMake (OpenMP, PDAL): https://github.com/OpenDroneMap/RenderDEM/blob/main/CMakeLists.txt
* PoissonRecon macOS Makefile: https://github.com/OpenDroneMap/PoissonRecon/blob/290/Makefile.macos
* Ceres 2.2 release notes (C++ threads only):
  https://raw.githubusercontent.com/ceres-solver/ceres-solver/2.2.0/docs/source/version_history.rst
* ODM docs, arguments: https://docs.opendronemap.org/arguments/
* ODM docs, hardware: https://docs.opendronemap.org/installation/
* odm-benchmarks: https://github.com/OpenDroneMap/odm-benchmarks and
  https://raw.githubusercontent.com/OpenDroneMap/odm-benchmarks/master/data/benchmarks.csv
* Community: https://community.opendronemap.org/t/odm-benchmarks-update/10367 ,
  https://community.opendronemap.org/t/profiling-opensfm/9179 ,
  https://community.opendronemap.org/t/opensfm-processing-time/11252 ,
  https://community.opendronemap.org/t/processing-time-differences/15255
