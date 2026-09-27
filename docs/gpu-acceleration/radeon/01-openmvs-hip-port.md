# Porting OpenMVS densification (PatchMatchCUDA) to HIP for AMD iGPUs

ODM builds `https://github.com/OpenDroneMap/openMVS` at tag `355`
(`SuperBuild/cmake/External-OpenMVS.cmake`), passing `-DOpenMVS_USE_CUDA=${SB_ENABLE_CUDA}`
and `-DOpenMVS_MAX_CUDA_COMPATIBILITY=ON`. `stages/openmvs.py` runs `DensifyPointCloud` with
`--cuda-device -1` (best GPU) when `opendm.gpu.has_gpu()` is true, otherwise `--cuda-device -2`
(CPU), and retries on CPU if the GPU run exits 1. This document is about giving that binary an
AMD backend. Nothing here has been executed; it is a plan grounded in the actual sources.

---

## 1. Source layout at tag 355 (what has to move)

Listed with `gh api repos/OpenDroneMap/openMVS/contents/...?ref=355`:

| File | Lines | Role |
|---|---|---|
| `libs/MVS/PatchMatchCUDA.cu` | 664 | the 4 kernels (`InitializeScore`, `BlackPixelProcess`, `RedPixelProcess`, `FilterPlanes`, templated on `<bool GEOM>`) |
| `libs/MVS/PatchMatchCUDA.cpp` | 421 | host side: texture/array allocation, uploads, multi-resolution loop |
| `libs/MVS/PatchMatchCUDA.h` / `.inl` | 58 / 143 | class + device inline helpers (includes `SceneDensify.h`) |
| `libs/Common/UtilCUDA.h` / `.cpp` | 720 / 435 | `SEACAVE::CUDA` device enumeration/selection and driver-API wrappers (`Device`, `MemDevice`, `EventRT`, `StreamRT`, `ModuleRT`, `KernelRT`, `TArrayRT`, `TTextureRT`, `TSurfaceRT`) |
| `libs/MVS/SceneRefineCUDA.cpp` | 2867 | mesh refinement via driver-API `ModuleRT`/`KernelRT` (see 1.3) |
| `libs/MVS/SceneDensify.cpp` | — | `#ifdef _USE_CUDA` glue that instantiates `MVS::CUDA::PatchMatch` and checks `SEACAVE::CUDA::devices` |

### 1.1 Device code (`PatchMatchCUDA.cu`)

A grep of the tag-355 file finds:

* `curandState` per pixel, `curand_init`, `curand_uniform` (≈10 call sites) — random plane
  hypotheses and Monte-Carlo view selection.
* `tex2D<float>` at 4 sites — image and depth sampling.
* `__constant__` cameras/params uploaded with `cudaMemcpyToSymbolAsync`.
* `__launch_bounds__(256,2)` or `(512,1)` depending on `PATCHMATCHCUDA_LB_256_2`; blocks of
  32×8 / 32×16; checkerboard grid.
* Fast-math intrinsics `__sincosf`, `__fdividef`, `__expf`, `__ldg`.
* **No** `__shfl*`, atomics, cooperative groups, half precision, CUB/Thrust, dynamic
  parallelism found by grep. (Upstream `develop` added a fused confidence kernel,
  `RunConfidenceCUDA`; tag 355 predates it.)

### 1.2 Host code (`PatchMatchCUDA.cpp`)

* `cudaCreateTextureObject` with `cudaAddressModeWrap`, **`cudaFilterModeLinear`**,
  **`cudaReadModeElementType`**, non-normalised coordinates, over `cudaMallocArray` float1
  arrays filled by `cudaMemcpy2DToArray`; separate depth textures for the geometric pass.
* `cudaMalloc`/`cudaMemcpy`/`cudaFree`, `cudaGetLastError`. No streams, no
  `cudaGetDeviceProperties`, no `cudaMemGetInfo` in this file.

### 1.3 Device selection (`UtilCUDA.cpp`) and the driver API

* `initDevices`: `cuInit(0)`, `cuDeviceGetCount`, `cuDeviceComputeCapability` (rejects
  major < 3 with "compute capability 3.2 or greater required"), `cuDeviceGetAttribute` for
  compute mode and `MULTIPROCESSOR_COUNT`, a `_convertSMVer2Cores` SM→cores table used to
  rank devices by `multiProcessorCount * cores * clockRate`, `cuCtxCreate`,
  `cuDeviceTotalMem`. `desiredDeviceID` is `-1` (best) by default; `< -1` disables.
* `UtilCUDA.h` includes `cuda.h`, `cuda_runtime.h`, `cuda_texture_types.h`,
  `curand_kernel.h`; `ModuleRT`/`KernelRT` wrap `cuModuleGetTexRef`/`cuModuleGetSurfRef`
  and `cuLaunchGrid` (texture *references*, deprecated in CUDA 12).
* Those driver-API module classes are only consumed by `SceneRefineCUDA.cpp`. The upstream
  author says the original refine kernels were lost ("I lost the original cuda kernels for
  this" — [openMVS#1076](https://github.com/cdcseacave/openMVS/issues/1076)), and ODM does not
  run `RefineMesh`. **Scope the port to PatchMatch; compile `SceneRefineCUDA.cpp` out.**

### 1.4 Build system at tag 355

Root `CMakeLists.txt`: `OpenMVS_USE_CUDA` (default ON), `FIND_PACKAGE(CUDA)` (legacy
FindCUDA) then `ENABLE_LANGUAGE(CUDA)`, `CMAKE_CUDA_ARCHITECTURES` defaults to `75`,
`OpenMVS_MAX_CUDA_COMPATIBILITY` shells out to `nvcc --list-gpu-arch/--list-gpu-code` to
emit `-gencode` for every arch, `--expt-relaxed-constexpr`, sets `_USE_CUDA`.
`libs/MVS/CMakeLists.txt`: `FILE(GLOB LIBRARY_FILES_CUDA "*.cu")`, target property
`CUDA_ARCHITECTURES "50;72;75"`, links `${CUDA_CUDA_LIBRARY}` (libcuda, which is why ODM's
`External-OpenMVS.cmake` hunts for the `libcuda.so` stub on Linux and downloads
`nvcuda_dummy.dll` on Windows).

Upstream `develop` (not ODM's tag) has since moved to `CMP0146 NEW` + `find_package(CUDAToolkit)`
linking `CUDA::cuda_driver CUDA::cudart CUDA::curand`, and — relevant precedent — merged a
**Metal backend** for the same PatchMatch kernels (PR #1279, 2026-06-26, +1358/−83 in 27 files)
with a generic `--gpu-device` option, `SEACAVE::CUDA::isCpuRequested`, and a
`UtilCUDADevice.h` split so plain C++ TUs can hold GPU classes.
[openMVS#1279](https://github.com/cdcseacave/openMVS/pull/1279)

## 2. Prior art and community state

* **No OpenMVS HIP/ROCm fork exists** (GitHub repo and issue search, Sept 2026). The
  maintainer's current position (Sept 2026): "OpenMVS GPU acceleration specifically targets
  NVIDIA GPUs through CUDA (no native OpenCL/ROCm/Metal support)" plus a full CPU path.
  [openMVS#1085](https://github.com/cdcseacave/openMVS/issues/1085)
* Maintainer on OpenCL (2023): "the densification algorithm would be a perfect match for
  porting to OpenCL". [openMVS#1076](https://github.com/cdcseacave/openMVS/issues/1076)
* ODM forum: "Help requested: Navigate source code to HIPify for AMD GPU support" (Sept 2023)
  ended without a port; earlier threads confirm CUDA-only status and suggest SYCLomatic.
  [forum 17500](https://community.opendronemap.org/t/help-requested-navigate-source-code-to-hipify-for-amd-gpu-support/17500),
  [forum 16485](https://community.opendronemap.org/t/amd-radeon-support/16485),
  [forum 25189](https://community.opendronemap.org/t/which-gpu-amd-or-nvidia/25189)
* **Closest precedent: PopSift PR #186** (AMD engineer, June–Aug 2026), a `USE_HIP` port of a
  texture-heavy CUDA SIFT that documents the exact HIP pitfalls PatchMatch will hit (section 3).
  [popsift#186](https://github.com/alicevision/popsift/pull/186)
* The Metal port measured **"~5–6× faster wall-time"** than the CPU `DepthEstimator` on an
  Apple GPU, "geometric-consistency passes ~20× faster", with ~0.1% median depth difference.
  This is the best available datapoint for "PatchMatchCUDA on a shared-memory iGPU".
  [openMVS#1279](https://github.com/cdcseacave/openMVS/pull/1279)

## 3. What hipify handles, and what it will not

### 3.1 Mechanical mapping (all supported)

Per the HIPIFY tables every runtime call used is 1:1: `hipCreateTextureObject`,
`hipDestroyTextureObject`, `hipMallocArray`, `hipMemcpy2DToArray`, `hipFreeArray`,
`hipMemcpyToSymbolAsync`, `hipGetDeviceProperties`, `hipGetDeviceCount`, `hipSetDevice`,
`hipMemGetInfo`, `hipDeviceGetAttribute`, `hipStreamCreate`. cuRAND device API maps to
`hiprandState_t`, `hiprand_init`, `hiprand_uniform` (`hiprand_kernel.h`, rocRAND backend).
[CUDA Runtime API → HIP](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html),
[cuRAND → HIP](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CURAND_API_supported_by_HIP.html)

`hipify-perl` is string substitution ("unable to find a replacement for user-defined
functions", many warnings); `hipify-clang` is the semantic tool. Either handles this file
size. [HIPIFY docs](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/)

### 3.2 The hard parts (from the PopSift port, directly applicable)

1. **Linear-filtered float textures.** "HIP rejects a `cudaFilterModeLinear` + element-read
   float texture at creation, so the linear fetches do manual bilinear interpolation in
   software (point-sample the 4 neighbors and lerp, CUDA's -0.5 texel-center convention)."
   PatchMatchCUDA.cpp creates exactly that combination. Plan for a `tex2D` shim that does
   4 point fetches + lerp under `USE_HIP`. Cost: extra ALU/bandwidth; on an iGPU this is
   noticeable but not fatal. (Observed on ROCm 7.2.1; re-test on your ROCm.)
2. **Wavefront width.** RDNA3/3.5 iGPUs run wave32 by default, gfx90c wave64. PatchMatch
   has no explicit warp collectives (grep), so this is mostly moot; still audit anything
   that assumes `warpSize == 32` in the checkerboard indexing.
3. **`_convertSMVer2Cores` / compute-capability checks.** `hipDeviceComputeCapability`
   returns the gfx major/minor (11.0 for gfx1103), so the `major < 3` gate passes but the
   SM-to-cores table is meaningless; replace the ranking with `multiProcessorCount` (CUs)
   and `clockRate`, or simply pick device 0 on a laptop.
4. **Driver-API contexts.** `cuCtxCreate`, `cuModuleGetTexRef` etc. exist in HIP but are
   deprecated/limited; PatchMatch does not need them. Keep `UtilCUDA` driver-API classes
   under `#ifndef USE_HIP`.
5. **Intrinsics.** `__sincosf`, `__fdividef`, `__expf`, `__ldg` exist in HIP; verify
   precision-sensitive spots (the Metal port needed `curand → counter-based PCG` and had to
   tune nothing else numerically). `__launch_bounds__(256,2)` is accepted by hipcc (second
   argument is min blocks per multiprocessor; ROCm may interpret it differently — keep the
   one-argument form or benchmark).
6. **Legacy 2018 hipify result on AliceVision** (same code family): 713 refs converted, 323
   texture refs, with unconverted `cudaReadModeNormalizedFloat`, `cudaArraySurfaceLoadStore`,
   3D array/memcpy APIs — none of which PatchMatch uses.
   [AliceVision#439 comment](https://github.com/alicevision/AliceVision/issues/439#issuecomment-417422887)

### 3.3 Two implementation shapes

* **A. Compat header (PopSift style).** One `cuda_to_hip.h` that aliases CUDA spellings to
  HIP under `USE_HIP`; the `.cu` files compile as HIP; CUDA path byte-identical. Smallest
  diff, easiest to upstream, no duplicated kernels. Recommended.
* **B. hipify into `PatchMatchHIP.hip` + parallel class.** Cleaner separation, but two copies
  of the kernels to keep in sync (the Metal PR shows how a second backend gets wired into
  `SceneDensify` via a second worker pool; that is more work than A).

### 3.4 CMake changes (shape A)

```cmake
# root CMakeLists.txt
SET(OpenMVS_USE_HIP OFF CACHE BOOL "Enable HIP (AMD ROCm) backend")
if(OpenMVS_USE_HIP)
  find_package(hip REQUIRED)        # hip::device, hip::host
  find_package(hiprand REQUIRED)    # device headers only
  ENABLE_LANGUAGE(HIP)              # CMake >= 3.21; Linux
  set(CMAKE_HIP_ARCHITECTURES "gfx1103;gfx1150;gfx1151;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201" CACHE STRING "")
  add_compile_definitions(_USE_CUDA USE_HIP)   # keep the existing #ifdef _USE_CUDA glue
endif()
# libs/MVS/CMakeLists.txt
if(OpenMVS_USE_HIP)
  set_source_files_properties(${LIBRARY_FILES_CUDA} PROPERTIES LANGUAGE HIP)
  target_link_libraries(MVS PRIVATE hip::device hip::hiprand)
endif()
```

Notes: `CMAKE_HIP_ARCHITECTURES` replaces both the `75` default and the `--list-gpu-arch`
compatibility trick (there is no fat-binary auto-list; enumerate the gfx targets you ship).
Add `gfx90c` only if you will test it. On **Windows the HIP SDK does not support CMake's HIP
language** ("CMake HIP Language: Unsupported"); llama.cpp's recipe is to build the whole
target with the HIP SDK's `clang`/`clang++` under Ninja from an x64 Native Tools prompt and
link `hip::device`. That collides with ODM's `cl`+Ninja setup in `scripts/activate-msvc.bat`
and `pixi.toml`'s win-64 `build` task; expect the Windows build to be a separate ExternalProject
configuration with `CMAKE_C/CXX_COMPILER=clang(++)` from `%HIP_PATH%\bin`.
[HIP SDK components](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html),
[llama.cpp Windows HIP build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

## 4. ODM-side plumbing (describe only; see doc 04 for the full recipe)

* `SuperBuild/CMakeLists.txt` lines 62–67: mirror `SB_ENABLE_CUDA` with `SB_ENABLE_HIP`
  from `$ENV{ODM_ENABLE_HIP}`.
* `External-OpenMVS.cmake`: add `-DOpenMVS_USE_HIP=${SB_ENABLE_HIP}`,
  `-DCMAKE_HIP_ARCHITECTURES=...`, `-DCMAKE_PREFIX_PATH=/opt/rocm` (or the conda prefix);
  the `SB_LIBCUDA`/stub search and the `nvcuda_dummy.dll` download are CUDA-only and should be
  skipped when HIP is on.
* `stages/openmvs.py`: keep `--cuda-device -1/-2` if the HIP build keeps the option name
  (simplest), or follow upstream's `--gpu-device`. The exit-code-1 CPU fallback already exists.
* `opendm/gpu.py`: `has_gpu()` only knows `nvidia-smi`/`nvcuda.dll`; add an AMD probe (doc 04).

## 5. The ZLUDA path (run the CUDA binary unmodified)

What it would need: ZLUDA translating OpenMVS's PTX (kernels + inlined cuRAND device code)
and implementing `cudart` texture objects with linear filtering. ZLUDA's own status: texture
support "basic" (PhysX/Blender), cuRAND unmentioned, project unfunded since 2026 and "back to
being my weekend project"; it needs a working ROCm for the GPU (override on gfx1103; community
rocBLAS on Windows). [ZLUDA Q1&Q2 2026](https://vosen.github.io/ZLUDA/blog/zluda-update-q1q2-2026/)
Practicality on an iGPU: **low**. It is worth one afternoon of trying with the existing
`opendronemap/odm:gpu`-style binary (`LD_LIBRARY_PATH` to ZLUDA's `libcuda.so.1`) because
the cost is near zero, but do not plan on it. Also note the CUDA EULA §1.2(8) translation
clause. [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html)

## 6. Effort estimate

| Work item | Reference | Estimate |
|---|---|---|
| Compat header + `.cu` audit (curand→hiprand, tex2D shim, constant memory) | PopSift port: +922/−239 in 40 files, ~2.5 months of an AMD engineer's calendar time incl. review | 1–2 weeks for PatchMatch (≈1.3 k lines) |
| `UtilCUDA` device enumeration under HIP | 435 lines, driver API | 2–3 days |
| CMake (Linux) | section 3.4 | 1–2 days |
| CMake (Windows, clang toolchain inside ODM's MSVC-based SuperBuild) | llama.cpp precedent | 1–2 weeks, high uncertainty |
| Validation vs CUDA and CPU (depth agreement, fused cloud counts) | Metal PR method: median rel. depth diff, valid-pixel %, point counts | 1 week |
| ODM plumbing + pixi env + docs | doc 04 | 3–5 days |
| **Total** | | **4–7 person-weeks** for Linux; add 1–2 for Windows |

Assumes someone comfortable with CUDA/HIP. Upstreaming to `cdcseacave/openMVS` is plausible
given the Metal precedent, which reduces ODM's fork burden.

## 7. Expected speed-up on an 8-core Zen 4 laptop — be honest

Datapoints:

* NVIDIA dGPU: ODM's 2021 announcement, "Early benchmarks point to a ~10x speed improvement
  in point cloud densification times" (Brighton Beach, hardware unspecified).
  [ODM blog](https://opendronemap.org/2021/11/gpu-point-cloud-densification-lands-in-odm-and-its-fast/)
* Apple Silicon (shared memory, like an APU): 5–6× wall-time on densification vs the CPU path.
  [openMVS#1279](https://github.com/cdcseacave/openMVS/pull/1279)
* Whole-pipeline laptop numbers from the ODM benchmark CSV: i7-11800H Aukerman 13 min CPU vs
  11 min with an RTX 3050 Ti; i5-8250U Brighton Beach 16 min → 12 min with an MX150. Small
  laptop GPUs buy ~15–25% end-to-end. [odm-benchmarks](https://github.com/OpenDroneMap/odm-benchmarks/blob/master/data/benchmarks.csv)
* No public PatchMatch-on-Radeon-iGPU numbers exist. Analogues: Ollama on a 780M with the
  override ran 3× CPU on compute-bound decode ([ollama#3189](https://github.com/ollama/ollama/issues/3189));
  on a single-channel machine the CPU won memory-bound generation ([k8s.it](https://www.k8s.it/posts/running-a-local-llm-on-amd-radeon-780m-gfx1103-rocm-and-the-gpu-that-wasnt-supposed-to-work/)).

Arithmetic: a 780M is 12 CUs at up to ~2.7–3.0 GHz; sources quote 4.2 TFLOPS FP32
(single-issue) or 8.7 TFLOPS (dual-issue peak). An 8-core Zen 4 at ~4 GHz with 2×256-bit FMA
pipes is ≈1 TFLOPS FP32 peak. So the theoretical ratio is 4–8×, but: (a) PatchMatch is
texture/bandwidth heavy and both sides share ~80–100 GB/s DDR5/LPDDR5x; (b) the software
bilinear shim (3.2.1) adds fetches; (c) CPU and GPU share one 35–54 W envelope, so the CPU
stages slow when the iGPU is busy; (d) densification is one stage among feature extraction,
matching, BA, meshing, texturing, DEM/ortho that stay on the CPU.

**Realistic expectation: 2–4× on the densification stage, 1.2–1.6× end-to-end, on a dual-channel
laptop; less on single-channel or 16 GB machines. An 890M (16 CUs, RDNA3.5) sits at the upper
end; Vega 8 (`gfx90c`) is likely ≤2× and may not beat the CPU once the shim is included.**
Measure before committing engineering time: the Metal PR's methodology (4-image scene, valid-pixel
ratio, median depth error, wall time) is a good template.

## Sources

- https://github.com/OpenDroneMap/openMVS (tag 355; files listed via GitHub API)
- https://github.com/cdcseacave/openMVS/pull/1279
- https://github.com/cdcseacave/openMVS/issues/1076 , https://github.com/cdcseacave/openMVS/issues/1085
- https://github.com/alicevision/popsift/pull/186
- https://github.com/alicevision/AliceVision/issues/439#issuecomment-417422887
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/ , https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html , https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CURAND_API_supported_by_HIP.html
- https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- https://community.opendronemap.org/t/help-requested-navigate-source-code-to-hipify-for-amd-gpu-support/17500 , https://community.opendronemap.org/t/amd-radeon-support/16485 , https://community.opendronemap.org/t/which-gpu-amd-or-nvidia/25189
- https://opendronemap.org/2021/11/gpu-point-cloud-densification-lands-in-odm-and-its-fast/
- https://github.com/OpenDroneMap/odm-benchmarks/blob/master/data/benchmarks.csv
- https://vosen.github.io/ZLUDA/blog/zluda-update-q1q2-2026/ , https://docs.nvidia.com/cuda/eula/index.html
- https://github.com/ollama/ollama/issues/3189 , https://www.k8s.it/posts/running-a-local-llm-on-amd-radeon-780m-gfx1103-rocm-and-the-gpu-that-wasnt-supposed-to-work/
- https://www.cpu-monkey.com/en/igpu-amd_radeon_780m , https://www.notebookcheck.net/Radeon-780M-vs-Vega-7_11564_10098.247598.0.html
