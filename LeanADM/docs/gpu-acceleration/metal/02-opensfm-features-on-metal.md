# OpenSfM feature extraction, matching and bundle adjustment on Apple silicon

Research date: 2026-09-13. Scope: the `opensfm` ExternalProject
(`SuperBuild/cmake/External-OpenSfM.cmake`, ODM fork commit `c5328439`), `pypopsift`
(built only when `ODM_ENABLE_CUDA=ON`), and `opendm/osfm.py`'s feature/matcher selection.

---

## 1. How ODM picks a feature extractor today

`opendm/osfm.py` (read locally): `--feature-type` (default `sift`) is upper-cased; if
`--matcher-type bow` (auto-selected when no GPS) the feature type is forced to `HAHOG`. If the
type is `SIFT` **and** `has_gpu(args)` (nvidia-smi / nvcuda.dll) **and**
`has_popsift_and_can_handle_texsize(w, h)` (`opendm/gpu.py`: needs CUDA compute ≥ 3.5 via
`libcuda.so`/`nvcuda.dll` ctypes and `pypopsift.fits_texture(w*1.02, h*1.02)`), the type
becomes `SIFT_GPU`. Otherwise CPU. On macOS every GPU check returns false, so the Mac runs
`SIFT` (OpenCV) or `DSPSIFT`/`HAHOG` (OpenSfM's C++).

`opensfm/features.py` at the pinned commit dispatches on `config["feature_type"]`:

```python
if   feature_type == "SIFT":     extract_features_sift(...)      # cv2.SIFT_create
elif feature_type == "SURF":     extract_features_surf(...)      # cv2.xfeatures2d.SURF_create
elif feature_type == "AKAZE":    extract_features_akaze(...)     # pyfeatures.akaze
elif feature_type == "HAHOG":    extract_features_hahog(...)     # pyfeatures.hahog
elif feature_type == "ORB":      extract_features_orb(...)       # cv2.ORB_create
elif feature_type == "SIFT_GPU": extract_features_popsift(...)   # pypopsift (CUDA)
elif feature_type == "DSPSIFT":  extract_features_dspsift(...)   # pyfeatures.dspsift
```

Each returns `(points, desc)`; `extract_features` then samples colours at the keypoints and
normalises coordinates, returning `(points, descriptors, colors)`. Config keys:
`feature_process_size`, `feature_min_frames`, `sift_peak_threshold`,
`surf_hessian_threshold`, `akaze_dthreshold`, `hahog_peak_threshold`, `feature_root`.

**Adding a new backend later (document only):** add
`extract_features_<name>(image_gray, config, features_count) -> (points[N,4] (x,y,size,angle), desc[N,D])`
and an `elif` branch; expose it in `opendm/config.py --feature-type` choices and in the
`osfm.py` selection block (mirroring the `SIFT_GPU` switch with a Metal availability probe,
see `05-*.md` §6). Descriptor dtype/dimension must be compatible with the matcher
(`float32` 128-D for FLANN/WORDS paths; binary for Hamming brute force).

---

**History worth knowing.** ODM already shipped a *non-CUDA* GPU SIFT once: PR #1239 "GPU
Support" (created 2021-03-05, merged 2021-03-15) added "support for GPU SIFT (OpenCL based)"
through silx/pyopencl (`SIFT_GPU` when a GPU was detected; GPU matching was optional and the
author found CPU FLANN matching better). It was replaced on 2021-12-16 by CUDA PopSift
(commit `0ca7d1c` "Pypopsift support", alongside "Handle large images with GPU_SIFT" after
[issue #1365](https://github.com/OpenDroneMap/ODM/issues/1365) on >11 000 px images). So the
`SIFT_GPU` plumbing in `osfm.py`/`gpu.py` predates PopSift and was designed to be
backend-agnostic; an OpenCL path *would* still run on the M4 Pro (Apple OpenCL 1.2 enumerates
the GPU, `00-*.md` §8) but silx's kernels are pyopencl/Python-driven and the platform is
deprecated — a Metal extractor is the better re-introduction.

## 2. PopSift is CUDA-only (verified)

[alicevision/popsift](https://github.com/alicevision/popsift): "an implementation of the SIFT
algorithm in CUDA"; README requires "CUDA >= 8.0", compute capability ≥ 3.0 (3.5 in ODM's
check), MPL-2.0, and mentions no other backend. AliceVision maintainers explain why porting is
hard: PopSift and the DepthMap stage "use textures everywhere, and a lot of their speed gain
comes from the hardware interpolation engine" of NVIDIA texture units
([AliceVision group thread](https://groups.google.com/g/alicevision/c/_pW-BO9aPrU/m/ZxdyKytrEQAJ)).
ODM's `External-PyPopsift.cmake` skips the project unless `SB_ENABLE_CUDA`. Metal does have
hardware bilinear texture sampling (`filter::linear`, `supports32BitFloatFiltering`), so a
Metal SIFT is feasible — but it is a rewrite, not a port (PopSift is ~15 k lines of CUDA).

---

## 3. OpenCV SIFT: CPU (NEON) only — no OpenCL path

* `modules/features2d/src/sift.dispatch.cpp` (OpenCV 4.x, fetched): **no** `ocl`, `UMat`,
  `CV_OCL_RUN` or `HAVE_OPENCL` anywhere. `detectAndCompute` does `Mat image = _image.getMat();`
  immediately, so a `UMat` input is downloaded and processed on the CPU. The hot loops are
  `CV_CPU_DISPATCH(calcSIFTDescriptor, …)` and `CV_CPU_DISPATCH(findScaleSpaceExtrema, …)`
  from `sift.simd.hpp` (universal intrinsics → NEON on arm64) plus `parallel_for_` over
  octaves/keypoints. Community answers confirm only SURF ever had OCL kernels in features2d
  ([answers.opencv 139464](https://answers.opencv.org/question/139464/opencv-featuredetection-with-opencl/)).
* OpenCV's OpenCL T-API in general: enabled via `cv::UMat`, `cv::ocl::setUseOpenCL(bool)`,
  env `OPENCV_OPENCL_DEVICE=":GPU:0"`, `OPENCV_OPENCL_RUNTIME=disabled`, kernel binary cache
  vars; modules with kernels are mainly `imgproc`, `core`, `video`, `dnn`, `photo`
  ([OpenCL optimizations wiki](https://github.com/opencv/opencv/wiki/OpenCL-optimizations)).
* **conda-forge OpenCV is built with `-DWITH_OPENCL=0`** (also `WITH_OPENCLAMDFFT=0`,
  `WITH_OPENCLAMDBLAS=0`, `WITH_CUDA=0`; `WITH_LAPACK=1`; on macOS arm64
  `-DKLEIDICV_ENABLE_SVE2=OFF`, i.e. the Arm KleidiCV HAL is otherwise on)
  ([opencv-feedstock build.sh](https://github.com/conda-forge/opencv-feedstock/blob/main/recipe/build.sh)).
  ODM's `pixi.lock` pins `libopencv-4.12.0-qt6_py312h1f2130c_604` / `py-opencv-4.12.0` on
  osx-arm64. So even for imgproc, `UMat` is a no-op in ODM's environment; enabling it would
  need a custom OpenCV build against Apple's deprecated OpenCL 1.2 (still enumerates the M4 Pro
  GPU on macOS 26.6 — `00-*.md` §8). Not worth it for SIFT, which has no kernels anyway.
* OpenCV has no Metal backend. Apple-side CPU acceleration is already present: NEON universal
  intrinsics, KleidiCV HAL (4.11+), and `WITH_LAPACK` pointing at conda BLAS (§6).

Practical CPU knobs on the M4 Pro: `feature_process_size` (ODM `--feature-quality`),
`--max-concurrency 12` (OpenSfM extracts per image in a process pool), and `DSPSIFT`
(domain-size pooled SIFT in OpenSfM's own C++; usually better matching than OpenCV SIFT at
similar cost).

---

## 4. Other GPU SIFT options

| Option | GPU API | Status | Fit |
|---|---|---|---|
| SiftGPU (Wu) | OpenGL GLSL / CUDA | 2007–2013; OpenGL is deprecated on macOS and its GLSL path relies on compatibility features; effectively dead on Apple silicon **[uncertain: nobody has reported a working build on macOS 26]** | No |
| [colmap-metal](https://github.com/lanxinger/colmap-metal) | Metal | "Metal SIFT extraction and matching run without CUDA or a Qt/OpenGL context"; BSD-3; requires Xcode + `xcrun metal` to build; README benchmark is "not a current performance ... guarantee" | Best existing reference for a Metal SIFT; its shaders could inform a `SIFT_METAL` extractor exposed to OpenSfM via pybind11. Needs license-compatible reuse (BSD-3 → fine). |
| PopSift | CUDA | see §2 | No |
| OpenCV SIFT + UMat | OpenCL | no kernels | No |
| MPS-assisted SIFT (own code) | Metal/MPS | `MPSImageGaussianBlur`/`MPSImagePyramid` for the scale space, custom kernels for DoG extrema, orientation and descriptor | A 1–3 week project for one engineer; the descriptor kernel is embarrassingly parallel (one thread per keypoint, 16 KB of threadgroup memory for the 4×4×8 histogram is plenty). |

---

## 5. Learned features (SuperPoint / DISK / ALIKED + LightGlue)

* [cvg/LightGlue](https://github.com/cvg/LightGlue) (ICCV 2023) with SuperPoint, DISK, ALIKED
  or SIFT features; [fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)
  exports extractor+matcher to ONNX (`lightglue-onnx export/infer`), supports dynamic batch and
  keypoint counts, fused attention on onnxruntime ≥ 1.16, and lists CUDA/TensorRT/OpenVINO
  EPs; CoreML is not documented there. [colmap/LightGlue-ONNX](https://github.com/colmap/LightGlue-ONNX)
  is a fork used by COLMAP; `kornia.feature.lightglue_onnx` wraps the same graphs.
* Running on the Mac: (a) **PyTorch MPS** — float32/float16/bfloat16 only, no float64, unknown
  ops fall back to CPU with `PYTORCH_ENABLE_MPS_FALLBACK=1` ([PyTorch MPS notes](https://docs.pytorch.org/docs/2.13/notes/mps.html));
  adds a ~200 MB torch dependency to the pixi env. (b) **onnxruntime CoreML EP** — see
  `03-*.md`; transformers with dynamic sequence length re-compile per shape and may run *slower*
  than CPU ("Kokoro 82M ... 280ms CPU vs 290ms CoreML" in the [Xybrid benchmark](https://www.xybrid.ai/blog/onnx-runtime-6x-faster-apple-silicon-coreml)),
  so LightGlue should be exported with a fixed keypoint count (e.g. 2048/4096, padded).
  The SuperPoint extractor itself is a small fixed-shape CNN and is a good CoreML/ANE target.
* Licences: LightGlue code Apache-2.0; **SuperPoint weights are non-commercial**
  (Magic Leap); DISK/ALIKED weights have their own terms — check before shipping in ODM
  (ODM is AGPL; SkyRemoval is AGPL-3.0 already).
* Integration: an `extract_features_onnx()` branch returning 256-D float descriptors plus a
  matcher branch in `opensfm/matching.py` (LightGlue is a *matcher*, replacing FLANN for the
  pair). Effort: ~1–2 weeks incl. export pipeline; scientific risk: SIFT/DSPSIFT is well tuned
  for nadir aerial overlap; learned matchers help most on low-texture/oblique sets.

---

## 6. Matching

`opensfm/matching.py` (fetched): `matcher_type ∈ {WORDS (BOW), FLANN, BRUTEFORCE}`;
WORDS uses `pyfeatures.match_using_words`, FLANN uses OpenCV's `FlannBasedMatcher` via
`feature_loader.load_features_index` (config `flann_checks`, `lowes_ratio`,
`symmetric_matching`), BRUTEFORCE uses `cv2.DescriptorMatcher_create` (L2 or Hamming). Pairs
come from `pairs_selection.match_candidates_from_metadata` (GPS neighbours / `--matcher-order`,
`--matcher-neighbors`). All CPU; ODM maps `--matcher-type {bow,flann,bruteforce}`.

**GPU brute-force sketch (Metal/MPS).** For descriptor sets `A[nA×128]`, `B[nB×128]`
(L2-normalised float or half): `S = A·Bᵀ` with `MPSMatrixMultiplication` (or a hand-written
tiled kernel using 32 KB threadgroup memory, 128×128 tiles), then a row-wise top-2 reduction
kernel using `simd_min`-style SIMD-group reductions for Lowe's ratio, and the symmetric check by
also reducing columns. For ODM sizes (≈ 4–20 k features/image, 128-D) `S` is ≤ 20k×20k floats
= 1.6 GB **per pair** at the extreme, so tile `B` in 4 k chunks. Given unified memory, descriptors
can be wrapped zero-copy. Realistic speed-up over FLANN on 12 cores: modest per pair, larger
when many pairs are batched; matching is rarely ODM's bottleneck compared to densify.
MPS caveats: rows/cols multiples of 8 and a reported 65535 dimension limit
([neurolabusc/Metal](https://github.com/neurolabusc/Metal)).

---

## 7. Bundle adjustment: Ceres 2.2 on the Mac

* Ceres' GPU story is CUDA-only: dense Cholesky/QR via cuBLAS/cuSOLVER (`USE_CUDA`), and since
  2.2/cuDSS "GPU acceleration ... for `SPARSE_NORMAL_CHOLESKY` and `SPARSE_SCHUR`"
  ([Ceres installation](https://ceres-solver.readthedocs.io/latest/installation.html)).
  **There is no Metal/MPS backend in Ceres** and none proposed
  ([COLMAP #2643](https://github.com/colmap/colmap/issues/2643) discusses CUDA BA only).
* Apple-specific CPU path Ceres *does* have: `ACCELERATESPARSE` ("Apple's Accelerate framework
  includes support for solving sparse linear systems", Xcode 9+, `WITH_ACCELERATESPARSE`
  default ON when detected) and Accelerate BLAS/LAPACK auto-detection.
* What ODM gets from conda-forge: `pixi.lock` (osx-arm64) →
  `ceres-solver-2.2.0-cpugplhc6e4162_210` (the *gpl* variant: `-DSUITESPARSE=ON`) with
  `suitesparse-7.10.1`, and `libblas/liblapack 3.11.0 *_openblas` +
  `libopenblas-0.3.33-openmp_*`. The feedstock's `build.sh` passes `-DBLA_VENDOR=Generic
  -DACCELERATESPARSE=OFF -DUSE_CUDA=OFF` ([ceres-solver-feedstock build.sh](https://raw.githubusercontent.com/conda-forge/ceres-solver-feedstock/main/recipe/build.sh)),
  so the conda Ceres uses SuiteSparse CHOLMOD + OpenBLAS, **not** Accelerate.
* Cheap win: switch the environment's BLAS to Apple Accelerate —
  `libblas=*=*accelerate` (classic) or `libblas=*=*_newaccelerate` for the macOS ≥ 13.3 LAPACK
  interface ([conda-forge news 2025-07-31](https://conda-forge.org/news/2025/07/31/new-accelerate-macos/),
  [blas-feedstock #103](https://github.com/conda-forge/blas-feedstock/issues/103)). Accelerate
  drives the M4's SME matrix unit and is much faster than OpenBLAS at small/medium GEMM
  ([dev.to](https://dev.to/frosnerd/comparing-openblas-and-accelerate-on-apple-silicon-for-blas-routines-2pb9),
  [Eclectic Light](https://eclecticlight.co/2024/11/27/inside-m4-chips-matrix-processing-and-power-modes/)).
  Because conda's Ceres, SuiteSparse, OpenCV (`WITH_LAPACK`), numpy and scipy all link the
  `libblas`/`liblapack` metapackages, this is a one-line change in `pixi.toml`
  (`[target.osx-arm64.dependencies] libblas = { version = "*", build = "*accelerate*" }`)
  **[uncertain: build-string wildcard syntax to prefer `_newaccelerate`; verify with
  `pixi list libblas`]**. Expected effect: BA and OpenMVS' Eigen/BLAS-heavy steps get faster
  dense kernels; CHOLMOD's supernodal factorisation calls BLAS too. Measure — it can also
  change OpenBLAS's threading behaviour (Accelerate "will appear to use one thread").
* Rebuilding Ceres with `ACCELERATESPARSE=ON` (Accelerate's sparse solver instead of CHOLMOD)
  is possible as a SuperBuild ExternalProject but Ceres' own docs and users usually find
  SuiteSparse faster for BA-sized problems **[uncertain: no Apple-silicon benchmark found]**.

---

## 8. Summary table

| Step | Today on Mac | Apple-GPU option | Effort | Value |
|---|---|---|---|---|
| SIFT/DSPSIFT extraction | CPU, NEON, 12 procs | Metal SIFT (colmap-metal as reference) or CoreML SuperPoint | 1–3 wk | medium (extraction is a few % of a large job; bigger on many-image sets) |
| Matching | FLANN/BOW CPU | Metal GEMM brute force or LightGlue-ONNX | 1–2 wk | low–medium |
| BA (Ceres) | SuiteSparse + OpenBLAS | none on GPU; Accelerate BLAS on CPU (SME) | hours | medium, free |
| Undistortion (OpenCV remap) | CPU | vImage / Metal remap | days | low |

Priority for the Mac: do the Accelerate switch first (measurable, no code), leave SIFT on CPU
until densify (`01-*.md`) is on Metal.

---

## Sources

- ODM: `/Users/avneh/Code/ADM/ODM/opendm/osfm.py`, `/Users/avneh/Code/ADM/ODM/opendm/gpu.py`, `/Users/avneh/Code/ADM/ODM/SuperBuild/cmake/External-PyPopsift.cmake`, `/Users/avneh/Code/ADM/ODM/SuperBuild/cmake/External-OpenSfM.cmake`, `/Users/avneh/Code/ADM/ODM/pixi.lock` (read locally)
- OpenSfM features.py (ODM fork c5328439): https://raw.githubusercontent.com/OpenDroneMap/OpenSfM/c5328439465e6ace011f39077d1077d7b1cdd65d/opensfm/features.py
- OpenSfM matching.py: https://raw.githubusercontent.com/OpenDroneMap/OpenSfM/c5328439465e6ace011f39077d1077d7b1cdd65d/opensfm/matching.py
- ODM PR #1239 "GPU Support" (OpenCL SIFT, 2021): https://api.github.com/search/issues?q=repo:OpenDroneMap/ODM+opencl+sift+is:pr ; `opendm/gpu.py` history (pypopsift commit 0ca7d1c, 2021-12-16): https://api.github.com/repos/OpenDroneMap/ODM/commits?path=opendm/gpu.py ; issue #1365: https://github.com/OpenDroneMap/ODM/issues/1365
- alicevision/popsift README: https://github.com/alicevision/popsift/blob/develop/README.md ; AliceVision group on textures: https://groups.google.com/g/alicevision/c/_pW-BO9aPrU/m/ZxdyKytrEQAJ
- OpenCV sift.dispatch.cpp: https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/sift.dispatch.cpp
- OpenCV OpenCL optimizations wiki: https://github.com/opencv/opencv/wiki/OpenCL-optimizations
- answers.opencv.org 139464: https://answers.opencv.org/question/139464/opencv-featuredetection-with-opencl/
- OpenCV forum, OpenCL DNN on M1: https://forum.opencv.org/t/using-opencl-with-dnn-on-m1-mac/2517
- conda-forge opencv-feedstock build.sh: https://github.com/conda-forge/opencv-feedstock/blob/main/recipe/build.sh
- colmap-metal: https://github.com/lanxinger/colmap-metal
- cvg/LightGlue: https://github.com/cvg/LightGlue ; fabio-sim/LightGlue-ONNX: https://github.com/fabio-sim/LightGlue-ONNX ; colmap/LightGlue-ONNX: https://github.com/colmap/LightGlue-ONNX ; kornia lightglue_onnx: https://kornia.readthedocs.io/en/v0.7.1/_modules/kornia/feature/lightglue_onnx/lightglue.html
- PyTorch MPS backend notes: https://docs.pytorch.org/docs/2.13/notes/mps.html
- Xybrid, ONNX Runtime + CoreML on Apple silicon: https://www.xybrid.ai/blog/onnx-runtime-6x-faster-apple-silicon-coreml
- Ceres installation docs: https://ceres-solver.readthedocs.io/latest/installation.html ; COLMAP #2643: https://github.com/colmap/colmap/issues/2643
- conda-forge ceres-solver-feedstock build.sh: https://raw.githubusercontent.com/conda-forge/ceres-solver-feedstock/main/recipe/build.sh ; recipe.yaml: https://raw.githubusercontent.com/conda-forge/ceres-solver-feedstock/main/recipe/recipe.yaml
- conda-forge Accelerate news: https://conda-forge.org/news/2025/07/31/new-accelerate-macos/ ; blas-feedstock #103: https://github.com/conda-forge/blas-feedstock/issues/103 ; numpy-feedstock #253: https://github.com/conda-forge/numpy-feedstock/issues/253
- OpenBLAS vs Accelerate: https://dev.to/frosnerd/comparing-openblas-and-accelerate-on-apple-silicon-for-blas-routines-2pb9 ; Eclectic Light M4 SME: https://eclecticlight.co/2024/11/27/inside-m4-chips-matrix-processing-and-power-modes/ ; OpenBLAS #3789 (AMX): https://github.com/xianyi/OpenBLAS/issues/3789
- neurolabusc/Metal (MPS matmul limits): https://github.com/neurolabusc/Metal
