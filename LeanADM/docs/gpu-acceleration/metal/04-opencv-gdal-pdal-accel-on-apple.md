# The rest of the pipeline on Apple silicon: OpenCV, GDAL, PDAL, PoissonRecon, texturing, orthophoto

Research date: 2026-09-13. Scope: every heavy ODM step that is *not* OpenSfM features/BA
(`02`), OpenMVS densify (`01`) or the ONNX filters (`03`). For each: does it have any GPU
path, what Apple-specific CPU acceleration applies, and what a Metal port would look like.

Summary: **none of these components has a GPU code path that works on macOS**, and only one
(`odm_orthophoto`) is an obvious candidate for a Metal rewrite. The others are memory/IO-bound
or already scale on 12 cores through OpenMP/TBB/multiprocessing — provided OpenMP is actually
enabled in the macOS SuperBuild (§7).

---

## 1. OpenCV (undistortion, resizing, image I/O, masks, SIFT)

* conda-forge OpenCV 4.12 osx-arm64 is built `-DWITH_OPENCL=0` (and no CUDA), so `cv::UMat`
  silently runs on the CPU and `cv::ocl::haveOpenCL()` is false in ODM's env
  ([opencv-feedstock build.sh](https://github.com/conda-forge/opencv-feedstock/blob/main/recipe/build.sh)).
  The runtime switches `cv::ocl::setUseOpenCL()`, `OPENCV_OPENCL_DEVICE=":GPU:0"`,
  `OPENCV_OPENCL_RUNTIME` ([OpenCL optimizations wiki](https://github.com/opencv/opencv/wiki/OpenCL-optimizations))
  therefore do nothing. Apple's OpenCL 1.2 still enumerates the M4 Pro GPU on macOS 26.6
  (`00-*.md` §8), so a *custom* OpenCV built with `WITH_OPENCL=ON` would get T-API kernels for
  `resize`, `remap`, `cvtColor`, `GaussianBlur`, `warpAffine/Perspective`, morphology, etc.
  Reports from M1 users: it works but gains are inconsistent and the DNN OpenCL target is
  unreliable ([forum.opencv 2517](https://forum.opencv.org/t/using-opencl-with-dnn-on-m1-mac/2517)).
  Given deprecation and the conda build, not recommended.
* OpenCV has **no Metal backend**. What ODM gets for free on arm64: NEON universal intrinsics
  in all `*.simd.hpp` kernels, the Arm KleidiCV HAL (OpenCV 4.11+; the feedstock only disables
  its SVE2 path on macOS), `parallel_for_` (pthreads) and LAPACK via conda `libblas`
  (`WITH_LAPACK=1`), which becomes Accelerate if the BLAS variant is switched (`02-*.md` §7).
* Apple-native alternative for the undistort/resize/remap steps in `opendm/` (Python `cv2`):
  **vImage** in Accelerate (`vImageWarp_*`, `vImageScale_*`, `vImageConvolve_*`) — multi-core,
  NEON, no GPU ([vImage docs](https://developer.apple.com/documentation/accelerate/vimage));
  or a 20-line Metal `remap` kernel with a linear sampler. Undistortion is a few % of a job;
  low priority. Both would need a small pybind11/ctypes shim.
* OpenSfM's undistortion (`opensfm/undistort.py`) uses `cv2.remap` per image in a process pool
  — already parallel.

## 2. GDAL 3.11 (orthophoto GeoTIFF write, DEM warping/merging, overviews, COG)

* Multithreading exists (`-multi`, `-wo NUM_THREADS=ALL_CPUS`, `-wm` as a % of RAM, GDAL ≥ 3.10)
  but "computation is not multithreaded itself" for the warper apart from `NUM_THREADS`
  ([gdalwarp docs](https://gdal.org/en/stable/programs/gdalwarp.html),
  [GDALWarpOptions](https://gdal.org/en/stable/api/gdalwarp_cpp.html)).
* GDAL has had an **OpenCL warper** (`USE_OPENCL` warp option, `GDAL_USE_OPENCL` CMake
  option) since 1.x; historically flaky and rarely built ([gdal-dev 2020](https://lists.osgeo.org/pipermail/gdal-dev/2020-November/052935.html),
  [NVIDIA forum](https://forums.developer.nvidia.com/t/gdal-wont-run-on-gpu-with-cuda-or-opencl-forced-to-cpu/36986)).
  The conda-forge recipe (`build_core.sh`) has **no OpenCL mention** and the current
  `build_hints` page does not document the option in the fetched text
  **[uncertain: whether `GDAL_USE_OPENCL` still exists in 3.11 CMake]**. Not available in ODM's
  env; no Metal/CUDA path exists.
* ODM's GDAL-heavy steps (`opendm/dem/`, `opendm/orthophoto.py`, `opendm/cogeo.py`) are IO
  and compression bound (DEFLATE/ZSTD/JPEG via libtiff). Apple-specific wins: fast NVMe, and
  making sure `GDAL_NUM_THREADS=ALL_CPUS` and `GDAL_CACHEMAX` are set (ODM already passes
  `--threads` where it can). No GPU story.

## 3. PDAL 2.8 (point-cloud filtering, SMRF ground classification, LAS/LAZ I/O, EPT)

* PDAL has no GPU code path (no CUDA/OpenCL/Metal anywhere in the project; searches found
  nothing) **[uncertain: absence of evidence]**. `filters.smrf` and `filters.outlier` are
  single-threaded per pipeline; ODM parallelises by tiling (`opendm/dem/` splits into tiles and
  runs a process pool) and by `--pc-quality`/`--dem-resolution` choices.
* LAZ decompression (LASzip/laz-perf) is scalar C++; `untwine`/Entwine (EPT build for the
  viewer) are multithreaded on their own.
* Apple-specific: nothing beyond the 12 cores. A Metal SMRF is conceivable (morphological
  opening on a raster is a separable min/max filter) but SMRF is not a dominant cost.

## 4. PoissonRecon (mesh reconstruction, `odm_meshing`)

* ODM builds `OpenDroneMap/PoissonRecon` tag `290` with a hand-written Makefile; on macOS via
  `Makefile.macos` patched by `poissonrecon-macos-makefile.patch` to use
  `-fopenmp=libomp` and `-lomp -L$(CONDA_PREFIX)/lib` (conda `llvm-openmp`) instead of
  Homebrew `libgomp` (patch read locally). So **PoissonRecon is OpenMP-parallel on the Mac**;
  ODM passes `--threads` (`opendm/mesh.py`). It is memory-bandwidth bound (octree + multigrid);
  the M4 Pro's ~273 GB/s **[uncertain: exact figure]** unified memory is favourable. No GPU
  version exists upstream (Kazhdan's code is CPU only).

## 5. mvs-texturing (`odm_texturing`)

* `OpenDroneMap/mvs-texturing` (`c5a4d0c9…`): `FIND_PACKAGE(OpenMP)` → `OpenMP_C_FLAGS`
  appended, `FIND_PACKAGE(TBB REQUIRED)`, and `-march=native` (fetched CMakeLists). The
  pixi build feature includes `tbb-devel 2022.3.*` "for mvstexturing"; runtime `tbb 2022.3.0`
  is locked for osx-arm64. View selection (graph cut via MRF/MAPMAP), seam levelling and
  texture atlas generation are CPU; no GPU variant exists. `-march=native` on Apple clang
  targets the M4 correctly but produces binaries that are not portable to older M-series
  **[uncertain: whether conda clang honours `-march=native` for apple-m4]** — harmless for a
  local build.
* OpenMVS' own texturing (`TextureMesh`) is being funded by ODM as a scalable replacement
  ([community thread](https://community.opendronemap.org/t/adding-scalable-texturing-to-openmvs-and-removing-a-memory-bottleneck/25571/3));
  upstream v2.4.0 also lists "GPU-based selection" for the viewer only. No GPU texturing.

## 6. odm_orthophoto — the one good Metal candidate

`OpenDroneMap/odm_orthophoto` `src/OdmOrthoPhoto.cpp` (48.7 KB, fetched): loads the textured
OBJ (PCL), then for every triangle runs `drawTexturedTriangle<T>()` — sorts vertices by y,
scanline-fills the two half-triangles, `getBarycentricCoordinates()` per pixel, a z-buffer test
(`depth_`; keep if `z >= depthValue`), and `renderPixel<T>()` with **bilinear texture
sampling** into the output raster; writes GeoTIFF through GDAL (`GDALCreate`/`GDALRasterIO`).
**No threading at all** (no `omp`/`thread` in the file) — it is a single-core software
rasteriser. On big surveys (tens of millions of triangles, 10 000×10 000+ px orthophotos) it
runs for many minutes on one core.

Metal sketch (this is exactly what a GPU is for):

* Upload vertex positions (already in the ortho plane), UVs and per-face texture index as
  `Shared` buffers; each texture atlas as an `MTLTexture` (RGBA8/RGB16 as needed; ODM atlases
  are 8 K JPEG/PNG tiles, well under the 16 K texture limit).
* Render pass, not compute: orthographic projection = identity on xy, depth from z; a trivial
  vertex shader passes UV, the fragment shader samples the atlas with a linear sampler and
  writes colour + alpha; the depth attachment replaces `depth_`. Apple's TBDR handles the
  z-test in tile memory for free. Output to a `Shared` RGBA texture, read back into the
  `cv::Mat` GDAL writer. Tile the output in e.g. 8 K × 8 K chunks to stay under
  `maxBufferLength` and texture limits; render each tile with a shifted ortho matrix.
* Edge cases to preserve: ODM's `-outputDsm/-outputDtm`-style extra bands (depth as a band),
  alpha/ nodata handling, and exact pixel-centre conventions (half-pixel offset) so results
  match the CPU rasteriser bit-for-bit where possible.
* Effort: ~1 week for a Metal path with a CPU fallback; the code is small and self-contained.
  Value: this stage is often the second-longest CPU-serial step after densify on large
  datasets **[uncertain: no profile on the target machine yet]**. The Radeon/Linux track could
  share the design via Vulkan or even OpenGL-ES compute-free rendering.

## 7. dem2mesh, FPCFilter, OpenPointClass, RenderDEM, Obj2Tiles, draco

* `dem2mesh` (`odm_25dmeshing`): serial C++ over a DEM raster + jmeshlib edge swaps; ODM patches
  a NULL-vertex-ring crash. Small cost; no GPU.
* `FPCFilter` (fast point-cloud statistical filter, replaces PDAL outlier): C++17 with `std::thread`/
  `std::async`-style parallelism **[uncertain: not inspected]**; built without `CONDA_CMAKE_ARGS`
  (see `External-FPCFilter.cmake`), so it uses whatever compiler CMake finds.
* `OpenPointClass` (`--pc-classify`): LightGBM/random-forest point classification; LightGBM is
  OpenMP-parallel (ODM patches it: `openpointclass-lightgbm.patch`); no GPU on macOS (LightGBM's
  GPU build is CUDA/OpenCL-for-Linux).
* `RenderDEM`, `Obj2Tiles` (C#/.NET), `draco` — modest cost; no GPU.

## 8. Making sure OpenMP is on in the macOS SuperBuild

Every parallel component above (PoissonRecon, mvs-texturing, OpenMVS CPU PatchMatch,
LightGBM) depends on OpenMP being found. On macOS the situation is:

* Apple's clang (CLT) ships **no** OpenMP runtime; conda-forge clang does not include it in the
  compiler package either. The `llvm-openmp` conda package provides `libomp.dylib` and
  `omp.h`, and "clang on all platforms ... default to the llvm-openmp runtime"
  ([conda-forge knowledge base](https://conda-forge.org/docs/maintainer/knowledge_base/),
  [openmp-feedstock](https://github.com/conda-forge/openmp-feedstock), [llvm-openmp on prefix.dev](https://prefix.dev/channels/conda-forge/packages/llvm-openmp)).
  `pixi.lock` already resolves `llvm-openmp-22.1.8` for osx-arm64 (as a transitive dependency
  of OpenBLAS `openmp_*` builds), but it is **not** an explicit dependency in `pixi.toml`.
* CMake's `FindOpenMP` with conda clang works when `omp.h`/`libomp.dylib` are on the prefix
  path (CMake passes `-fopenmp` and links `libomp`); the SuperBuild's `CONDA_CMAKE_ARGS` set
  `CMAKE_PREFIX_PATH=$CONDA_PREFIX`, which is sufficient in practice
  ([CMake Discourse](https://discourse.cmake.org/t/how-to-find-openmp-with-clang-on-macos/8860))
  **[uncertain: not confirmed on this machine — no build has been run]**.
* Known trap: upstream openMVS hard-codes `OpenMP_ROOT "/opt/homebrew/opt/libomp"` on Apple
  arm64 (`01-*.md` §1). If no Homebrew libomp exists, `FindOpenMP` may still succeed via the
  conda prefix; if a Homebrew copy exists it may be picked instead, mixing two OpenMP runtimes
  (a classic source of "multiple OpenMP runtimes loaded" warnings and slowdowns). Passing
  `-DOpenMP_ROOT=$CONDA_PREFIX` from `External-OpenMVS.cmake` on APPLE is the fix (document only).
* Recommended `pixi.toml` change (document only): add `llvm-openmp = "*"` to
  `[feature.build.target.osx-arm64.dependencies]` (and to the runtime `cpu` feature for
  osx-arm64 so `libomp.dylib` is present in `prod`), and grep each ExternalProject's configure
  log for `Found OpenMP`.
* A second trap: `-fopenmp` with the *Apple* clang from CLT will fail at link time
  (`-lomp` missing) unless `-L$CONDA_PREFIX/lib` is given — this is precisely why the
  PoissonRecon macOS Makefile patch adds `-fopenmp=libomp -lomp -L$(CONDA_PREFIX)/lib`.

## 9. Priorities

| Component | Apple-GPU potential | Recommended action |
|---|---|---|
| odm_orthophoto | high (render pass) | Metal rasteriser with CPU fallback, after densify |
| OpenCV undistort/resize | low–medium | leave; optionally vImage |
| PoissonRecon, texturing, OpenMVS CPU | none (GPU), high (OpenMP) | verify OpenMP is found; set `OpenMP_ROOT` |
| GDAL, PDAL, dem2mesh, FPCFilter, OPC | none | tune threads/cache; nothing else |
| BLAS consumers (Ceres, OpenCV, numpy) | none (GPU) | switch to Accelerate (`02-*.md` §7) |

## Sources

- ODM (read locally): `/Users/avneh/Code/ADM/ODM/SuperBuild/CMakeLists.txt`, `/Users/avneh/Code/ADM/ODM/SuperBuild/cmake/poissonrecon-macos-makefile.patch`, `/Users/avneh/Code/ADM/ODM/SuperBuild/cmake/External-MvsTexturing.cmake`, `/Users/avneh/Code/ADM/ODM/SuperBuild/cmake/External-FPCFilter.cmake`, `/Users/avneh/Code/ADM/ODM/pixi.toml`, `/Users/avneh/Code/ADM/ODM/pixi.lock`
- conda-forge opencv-feedstock build.sh: https://github.com/conda-forge/opencv-feedstock/blob/main/recipe/build.sh
- OpenCV OpenCL optimizations wiki: https://github.com/opencv/opencv/wiki/OpenCL-optimizations ; OpenCV forum (OpenCL on M1): https://forum.opencv.org/t/using-opencl-with-dnn-on-m1-mac/2517
- Apple vImage: https://developer.apple.com/documentation/accelerate/vimage
- GDAL gdalwarp: https://gdal.org/en/stable/programs/gdalwarp.html ; GDALWarpOptions: https://gdal.org/en/stable/api/gdalwarp_cpp.html ; build hints: https://gdal.org/en/stable/build_hints.html
- gdal-dev on GPU GDAL (2020): https://lists.osgeo.org/pipermail/gdal-dev/2020-November/052935.html ; NVIDIA forum thread: https://forums.developer.nvidia.com/t/gdal-wont-run-on-gpu-with-cuda-or-opencl-forced-to-cpu/36986
- conda-forge gdal-feedstock build_core.sh: https://raw.githubusercontent.com/conda-forge/gdal-feedstock/main/recipe/build_core.sh
- OpenDroneMap/odm_orthophoto src listing: https://api.github.com/repos/OpenDroneMap/odm_orthophoto/contents/src ; OdmOrthoPhoto.cpp: https://raw.githubusercontent.com/OpenDroneMap/odm_orthophoto/main/src/OdmOrthoPhoto.cpp
- OpenDroneMap/mvs-texturing CMakeLists (c5a4d0c9): https://raw.githubusercontent.com/OpenDroneMap/mvs-texturing/c5a4d0c9a434553533c6e39d426e349fcfa5f48d/CMakeLists.txt
- OpenDroneMap/PoissonRecon: https://github.com/OpenDroneMap/PoissonRecon
- ODM community, scalable texturing: https://community.opendronemap.org/t/adding-scalable-texturing-to-openmvs-and-removing-a-memory-bottleneck/25571/3
- conda-forge knowledge base (OpenMP): https://conda-forge.org/docs/maintainer/knowledge_base/ ; openmp-feedstock: https://github.com/conda-forge/openmp-feedstock ; llvm-openmp: https://prefix.dev/channels/conda-forge/packages/llvm-openmp
- CMake Discourse, FindOpenMP with clang on macOS: https://discourse.cmake.org/t/how-to-find-openmp-with-clang-on-macos/8860
- R-project, OpenMP on macOS with Xcode tools: https://mac.r-project.org/openmp/
- cdcseacave/openMVS root CMakeLists (OpenMP_ROOT on Apple arm64): https://raw.githubusercontent.com/cdcseacave/openMVS/develop/CMakeLists.txt
