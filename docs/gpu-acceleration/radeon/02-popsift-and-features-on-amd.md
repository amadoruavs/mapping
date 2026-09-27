# Feature extraction and bundle adjustment on AMD iGPUs

ODM's GPU feature path is PopSift (CUDA) wrapped by `pypopsift` and selected by OpenSfM's
`feature_type: SIFT_GPU`. This document covers what exists for AMD, the alternatives, how
OpenSfM's feature dispatch works so a new backend can be documented, and Ceres/BLAS on the CPU.

---

## 0. First, a gating fact most people miss

`opendm/osfm.py` only considers the GPU when the user's feature type is exactly `sift`:

```python
feature_type = args.feature_type.upper()
...
if feature_type == "SIFT":
    if has_gpu(args) and max_dims is not None:
        ...
        if has_popsift_and_can_handle_texsize(w, h):
            feature_type = "SIFT_GPU"
```

and ODM 3.6.2's default is `--feature-type dspsift` (`opendm/config.py`, choices
`akaze, dspsift, hahog, orb, sift`). **With default settings ODM never uses PopSift**, even on
an NVIDIA box. Whatever AMD feature backend is added inherits this: it only matters for users
who pass `--feature-type sift`, or the gating is extended to DSP-SIFT (PopSift does not
implement DSP-SIFT).

## 1. PopSift on AMD: there is now a HIP port (unmerged)

* **[alicevision/popsift PR #186 "[ROCm] HIP port for AMD GPUs"](https://github.com/alicevision/popsift/pull/186)**,
  opened 2026-06-03 by an AMD engineer (jeffdaily), head `AMD-Ecosystem/popsift:moat-port`,
  +922/−239 in 40 files, still **open** as of 2026-08-20 with the maintainer (griwodz) engaged
  ("I'm really thankful that you have made the HIP port!").
* Design: a single `src/popsift/cuda_to_hip.h` aliases CUDA spellings to HIP under `USE_HIP`;
  the `.cu` sources build as HIP; the CUDA build path is "byte-for-byte unchanged".
* Five fault classes it had to solve (all relevant to any texture-heavy CUDA port):
  1. HIP rejects linear-filtered float element-read textures → software bilinear.
  2. Layered `cudaArray` writes collapse to the last layer on ROCm (ROCm/clr#275) → 3D arrays.
  3. Wavefront-generic warp collectives (wave32 on RDNA3, wave64 on CDNA/GCN).
  4. RootSift/L2 NaN guards.
  5. Link `hip::host` publicly so plain C++ consumers do not need the HIP compiler.
* Tested: gfx90a (MI250X, Linux, ROCm 7.2.1), gfx1100 (W7800, Linux, wave32), gfx1201
  (RDNA4), and **gfx1151 (Radeon 8060S APU, Windows, TheRock ROCm)** — the last one is the
  closest to a Strix Point/Strix Halo laptop. Output validated against VLFeat: 14397 vs 14509
  descriptors on Oxford boat img1, 13914 mutual matches, 99.6% within 0.5 px.
* Not tested: gfx1103 (780M), gfx1150 (890M), gfx90c. Expect gfx1150 to behave like gfx1151;
  gfx1103 should compile natively (see doc 00 §1.2) — mark as unverified.

What ODM would need on top of the PR:

* `pypopsift` pins PopSift by `FetchContent` to commit `a19d394` and builds a pybind11 module
  (`CMakeLists.txt` in [OpenDroneMap/pypopsift](https://github.com/OpenDroneMap/pypopsift));
  `src/popsift.cpp` calls `cudaGetDevice`, `cudaDeviceReset`, `popsift::cuda::device_prop_t`
  — it needs the same compat header and a `USE_HIP` option passed through
  `SuperBuild/cmake/External-PyPopsift.cmake`, which is currently wrapped in
  `if(SB_ENABLE_CUDA)`.
* `opendm/gpu.py::has_popsift_and_can_handle_texsize` loads `libcuda.so`/`nvcuda.dll` with
  ctypes to check compute capability ≥ 3.5 before `pypopsift.fits_texture()`; an AMD path
  needs a `libamdhip64` probe (or just `try: import pypopsift`).
* The PopSift maintainer also keeps a **`sycl` branch** (May 2026) that "can support CPU only
  including Mac via SYCL" but "is stripped down to one or at most two algorithms in every step";
  he notes Intel's compiler does poorly and AdaptiveCpp lacks `sycl::image` hardware
  interpolation. Not a production path.

Historical context: AliceVision's 2018 discussion explains why texture hardware made CUDA
sticky ("HIP does not help in porting textures because OpenCL doesn't have any" was the
concern then); the 2026 PR shows it is solvable with software interpolation.
[AliceVision#439](https://github.com/alicevision/AliceVision/issues/439),
[Meshroom#595](https://github.com/alicevision/meshroom/issues/595)

## 2. Alternatives to PopSift

### 2.1 OpenCV OpenCL (T-API / UMat)

* OpenCV has **no OpenCL kernel for SIFT** ("Only SURF feature has a proper OpenCL kernel
  implementation"). [OpenCV forum](https://forum.opencv.org/t/currently-is-there-no-gpu-implementation-of-sift/20866)
* conda-forge's OpenCV — the one in ODM's pixi env (`py-opencv==4.12.0`) — is built with
  `-DWITH_OPENCL=0`, so UMat silently runs on the CPU anyway.
  [opencv-feedstock build.sh](https://github.com/conda-forge/opencv-feedstock/blob/main/recipe/build.sh)
* OpenSfM's `SIFT`/`DSPSIFT` use OpenCV's CPU SIFT; `HAHOG` is OpenSfM's own C++ (VLFeat-based).
  Verdict: no.

### 2.2 SiftGPU (Changchang Wu, GLSL/CUDA)

* GLSL path runs on any OpenGL GPU in principle, but the project itself says "The GLSL version
  may not work on ATI now. They did compile successfully with ATI Catalyst 8.9, but not any
  more with 9.x versions" (2009-era). Untouched for a decade. ODM's pre-2021 "OpenCL-based"
  SIFT GPU (referenced in a [May 2021 forum thread](https://community.opendronemap.org/t/odm-gpu-on-radeon/7622))
  was dropped in favour of PopSift. [SiftGPU](https://github.com/pitzer/SiftGPU), [ccwu.me](http://ccwu.me/siftgpu/)
  Verdict: archaeology only.

### 2.3 Vulkan SIFT

* **VulkanSift** (MIT, C library, loads Vulkan at runtime, no deps): SIFT detection + 2-NN
  brute-force matching "on (almost) any GPU"; tested on NVIDIA RTX/GTX, Adreno 615 (Android)
  and Apple M2 — **not on AMD** per its README, but Vulkan on RADV/Adrenalin should work.
  Has optional perf comparison against SiftGPU/PopSift/OpenCV. 94 commits; activity date not
  captured — check before adopting. [maelaubert/VulkanSift](https://github.com/maelaubert/VulkanSift)
* Integration cost: a small pybind11 wrapper returning `(points Nx4, descriptors)` in
  OpenSfM's format (section 3), plus a `feature_type: SIFT_VULKAN` branch. Works identically on
  Linux (RADV) and Windows and on gfx90c. Best cross-vendor option if PopSift-HIP stalls.

### 2.4 OpenCL SIFT repos

Assorted academic OpenCL SIFT ports exist but none is maintained or packaged; combined with
`WITH_OPENCL=0` in conda OpenCV, skip.

### 2.5 Learned features via ONNX Runtime

Exports that already exist:

* SuperPoint / DISK / ALIKED / SIFT + LightGlue: [fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)
  (CLI `lightglue-onnx`, TensorRT/OpenVINO), [colmap/LightGlue-ONNX](https://github.com/colmap/LightGlue-ONNX),
  upstream [cvg/LightGlue](https://github.com/cvg/LightGlue).
* XFeat + LighterGlue ONNX: [noahzhy/xfeat_lightglue_onnx](https://github.com/noahzhy/xfeat_lightglue_onnx)
  (LighterGlue ≈3× faster than LightGlue).

Runtime on AMD: MIGraphX EP (Linux, gfx1150 listed; gfx1103 crashes), DirectML EP (Windows,
any DX12 GPU), WebGPU EP (both). See doc 03. Caveats: OpenSfM's matching stage expects
descriptor arrays and runs FLANN/BoW/brute-force; LightGlue is a *matcher* and would need a
matcher plugin too. XFeat descriptors (64-d float) drop into brute-force matching directly.
This is a research-grade change to ODM's SfM behaviour, not an acceleration of the existing one.

## 3. How OpenSfM feature types are dispatched (for documenting a new backend)

ODM builds `OpenDroneMap/OpenSfM` at commit `c532843…` (`External-OpenSfM.cmake`).
`opensfm/features.py::extract_features()` converts to grayscale, resizes to
`feature_process_size`, then branches on `config["feature_type"]`:

```
SIFT     -> extract_features_sift      (OpenCV)
SURF     -> extract_features_surf
AKAZE    -> extract_features_akaze
HAHOG    -> extract_features_hahog     (OpenSfM C++)
ORB      -> extract_features_orb
SIFT_GPU -> extract_features_popsift   (pypopsift)
DSPSIFT  -> extract_features_dspsift
```

The GPU branch is literally:

```python
elif feature_type == 'SIFT_GPU':
    points, desc = extract_features_popsift(image_gray, config, features_count)
```

and `extract_features_popsift` calls pypopsift with `peak_threshold`, `edge_threshold`,
`target_num_features`, `use_root`. A backend must return `points` as an N×4 float array
(x, y, size, angle), `descriptors` N×D, and colours are sampled by the caller; when
`config["feature_root"]` is set the caller applies `root_feature()`.
[features.py at ODM's commit](https://github.com/OpenDroneMap/OpenSfM/blob/c5328439465e6ace011f39077d1077d7b1cdd65d/opensfm/features.py)

Documenting a new type therefore means: (1) a Python-callable extractor with that contract,
(2) an `elif feature_type == 'SIFT_HIP'`/`'SIFT_VULKAN'` branch (or reuse `SIFT_GPU` and let
pypopsift be the HIP build), (3) `opensfm/config.py` defaults, (4) ODM `osfm.py` gating and
`gpu.py` detection. Descriptor dimensionality and normalisation must match what the matcher
expects (128-d uint8/float for SIFT-like).

## 4. Ceres bundle adjustment on AMD (CPU)

* ODM uses conda-forge `ceres-solver==2.2.0` (`pixi.toml`), which OpenSfM links for BA.
  Ceres 2.2 offers `dense_linear_algebra_library_type = EIGEN | LAPACK | CUDA` and
  `sparse_linear_algebra_library_type = SUITE_SPARSE | EIGEN_SPARSE | ACCELERATE_SPARSE |
  CUDA_SPARSE`; the CUDA variants need an NVIDIA GPU and there is no HIP/ROCm backend, so
  on AMD the solver is CPU-only. (Ceres docs at ceres-solver.org; the site refused
  connections during this research — statement from the 2.2 documentation.)
* Threads: OpenSfM sets `processes: <max_concurrency>` for extraction/matching; Ceres'
  `num_threads` inside OpenSfM's BA follows OpenSfM's config. Nothing AMD-specific.
* BLAS: conda-forge routes `libblas`/`libcblas`/`liblapack` through a variant selector
  (`libblas=*=*openblas`, `*mkl`, `*blis`, `*netlib`, `accelerate` on macOS).
  [blas-feedstock](https://github.com/conda-forge/blas-feedstock/blob/master/README.md)
  AMD's **AOCL-BLAS** ("AMD's fork of BLIS optimized for AMD processors") is on conda-forge as
  `aocl-blas` 5.2 for linux-64 / win-64 / osx-64, but I found **no evidence it is wired into the
  `libblas` selector**; the practical Zen-optimised choice inside pixi is `libblas=*=*blis`
  (BLIS is what AOCL-BLAS forks) or plain OpenBLAS with `OPENBLAS_CORETYPE`/threads tuned.
  [aocl-blas on conda-forge](https://anaconda.org/conda-forge/aocl-blas)
  Expect small gains: BA time is dominated by sparse Schur/SuiteSparse and Jacobian evaluation,
  not dense BLAS.
* Thread oversubscription is the real BA/matching knob on an 8-core laptop: OpenSfM forks
  `max_concurrency` processes and each may spin OpenBLAS/OpenMP threads; setting
  `OPENBLAS_NUM_THREADS=1` / `OMP_NUM_THREADS=1` for the OpenSfM stage is a common fix
  (see doc 05).

## 5. Recommendation

1. Track PopSift PR #186; when merged (or by pinning the `AMD-Ecosystem/popsift:moat-port`
   branch in `pypopsift`), build pypopsift with `USE_HIP` on Linux for gfx1150/1151 first,
   then gfx1103. Windows only after the OpenMVS Windows toolchain question (doc 01 §3.4) is settled.
2. Keep DSP-SIFT on CPU as the default; GPU SIFT only helps `--feature-type sift` users and
   ODM's own 2021 note calls the feature-extraction GPU gain "modest" on consumer hardware.
   [ODM blog](https://opendronemap.org/2021/11/gpu-point-cloud-densification-lands-in-odm-and-its-fast/)
3. VulkanSift is the fallback that covers every AMD iGPU on both OSes without ROCm.

## Sources

- https://github.com/alicevision/popsift/pull/186
- https://github.com/alicevision/popsift (README: CUDA ≥ 8.0, compute capability ≥ 3.0; branches `sycl`, `wip/prep-for-sycl`)
- https://github.com/OpenDroneMap/pypopsift
- https://github.com/OpenDroneMap/OpenSfM/blob/c5328439465e6ace011f39077d1077d7b1cdd65d/opensfm/features.py
- https://github.com/alicevision/AliceVision/issues/439 , https://github.com/alicevision/meshroom/issues/595
- https://forum.opencv.org/t/currently-is-there-no-gpu-implementation-of-sift/20866
- https://github.com/conda-forge/opencv-feedstock/blob/main/recipe/build.sh
- https://github.com/pitzer/SiftGPU , http://ccwu.me/siftgpu/ , https://community.opendronemap.org/t/odm-gpu-on-radeon/7622
- https://github.com/maelaubert/VulkanSift
- https://github.com/fabio-sim/LightGlue-ONNX , https://github.com/colmap/LightGlue-ONNX , https://github.com/cvg/LightGlue , https://github.com/noahzhy/xfeat_lightglue_onnx
- https://github.com/conda-forge/blas-feedstock/blob/master/README.md , https://anaconda.org/conda-forge/aocl-blas
- https://opendronemap.org/2021/11/gpu-point-cloud-densification-lands-in-odm-and-its-fast/
- ODM sources: `opendm/osfm.py`, `opendm/gpu.py`, `opendm/config.py`, `SuperBuild/cmake/External-PyPopsift.cmake`, `pixi.toml`
