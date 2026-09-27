# OpenMVS on Metal — GPU anatomy, the upstream Metal backend, and porting options

Research date: 2026-09-13. Scope: the `openmvs` ExternalProject in ODM 3.6.2
(`SuperBuild/cmake/External-OpenMVS.cmake`) and the dense stage `stages/openmvs.py`.

**Headline finding.** Upstream `cdcseacave/openMVS` merged a native **Metal compute backend
for PatchMatch densification** on 2026-06-26 (PR #1279, commit `4694ded`, "dense: add Metal
based PatchMatch densification", developed and benchmarked on an "Apple M4 Max, 48 GB").
ODM's fork `OpenDroneMap/openMVS` at tag `355` — the one the SuperBuild builds — predates it
and contains only the CUDA backend; so does the pending ODM PR #2074 "Openmvs 240" (open since
2026-08-31), which moves the pin to the fork's `omvs-v240` commit `d7081f4` = upstream
**v2.4.0 (2026-01-20)**, i.e. still five months before the Metal merge. The cheapest route to
GPU densification on the M4 Pro is therefore *not* a new port but **rebasing ODM's openMVS pin
onto upstream `develop` (≥ `4694ded`)**, or backporting the 4 Metal files + CMake glue on top
of PR #2074's branch, plus three ODM-side changes (CMake option plumbing, the
`--cuda-device` → `--gpu-device` flag rename in `stages/openmvs.py`, and a Metal probe in
`opendm/gpu.py`). Details in §3–§5.

Host facts that shape the plan **[verified locally]**: this Mac has only Command Line Tools —
`xcrun -sdk macosx metal` does not exist (no Xcode / Metal Toolchain), so a build-time
`.metallib` step is impossible today; upstream's backend needs none because it compiles the
MSL at runtime with `newLibraryWithSource:`. `pixi` is not installed either (see `05-*.md`).

---

## 1. What ODM builds today

`External-OpenMVS.cmake` (read locally):

* Source: `https://github.com/OpenDroneMap/openMVS`, `GIT_TAG 355`
  (branch `355` tip = merge commit `8e4c5f3`, 2026-05-16, "Allow disabling subdirectory on
  install", parent `482e98a`; the fork has 36 branches incl. `omvs-v240`=`d7081f4`,
  `master`=`5811720`, `develop`=`c410c9d` per the GitHub branches API). The `355` line is
  based on an upstream state from well before v2.4.0 — it lacks `ConfidenceCUDA`, the 2026
  ROI work and Metal — **[uncertain: exact upstream base commit/date; the coordinator's
  research puts it at 2023-08]**.
* Pending change: ODM PR #2074 "Openmvs 240" (branch `openmvs-240`, open, created
  2026-08-31) sets `GIT_TAG d7081f414e3b6d0bdb7ade59ac071d049fdee29e` (= fork branch
  `omvs-v240`, upstream v2.4.0), adds `-DCMAKE_CUDA_ARCHITECTURES=OFF`, moves CGAL 5.6 → 6,
  adds nanoflann, and touches `stages/openmvs.py`, `opendm/osfm.py`, `pixi.toml`/`pixi.lock`
  ([PR #2074 files](https://api.github.com/repos/OpenDroneMap/ODM/pulls/2074/files)). Any
  Metal work should be stacked on this branch, not on `355`.
* CMake args: `-DOpenMVS_USE_CUDA=${SB_ENABLE_CUDA}` (OFF unless `ODM_ENABLE_CUDA=ON`),
  `-DOpenMVS_MAX_CUDA_COMPATIBILITY=ON`, `-DOpenMVS_USE_SSE=OFF` on arm64,
  `-DOpenMVS_ENABLE_TESTS=OFF`, Eigen 3.4 and VCG from sibling ExternalProjects, conda OpenCV,
  and on macOS a pre-build patch of CGAL 5.6.1's `boost/graph/iterator.h` (CGAL #8313) because
  conda clang 19 rejects it.
* Upstream's own root `CMakeLists.txt` hard-disables CUDA on Apple:
  `IF(APPLE) SET(OpenMVS_USE_CUDA OFF) MESSAGE(STATUS "Disabling CUDA on MacOS")` — so on the
  Mac the ODM build is CPU-only regardless of flags.
* Upstream also sets `OpenMP_ROOT "/opt/homebrew/opt/libomp"` on Apple arm64, which is wrong
  for a pixi environment (`llvm-openmp` 22.1.8 is in `pixi.lock` for osx-arm64). Whether
  OpenMP is actually found in the current Mac build should be checked in the CMake configure
  log **[uncertain]** — see `04-*.md` §7.

`stages/openmvs.py` (read locally): runs `DensifyPointCloud` with
`--cuda-device -1` when `has_gpu(args)` (nvidia-smi / nvcuda.dll present) else
`--cuda-device -2`; on exit code 1 with GPU it retries with `-2`; on 137/143 it turns on
tiling (`--fusion-mode 1`, sub-scenes). On macOS `has_gpu()` is always false today.

---

## 2. Where the GPU code lives in OpenMVS (tag 355 vs upstream develop)

`libs/MVS/` at ODM tag 355 (GitHub contents API): `PatchMatchCUDA.{cpp,cu,h,inl}`,
`SceneRefineCUDA.cpp` (90,966 B), `SemiGlobalMatcher.{cpp,h}`, `SceneDensify.cpp`, no Metal.

`libs/MVS/` at upstream `develop` (contents API, sizes in bytes):

| File | Size | Role |
|---|---|---|
| `PatchMatchCUDA.cu` | 35,285 | 4 CUDA kernels (see §2.1) |
| `PatchMatchCUDA.cpp` | 22,485 | host orchestration, device pooling, multi-resolution loop (~850 lines) |
| `PatchMatchCUDA.h/.inl` | 3,764 / 4,492 | class + `Params` |
| **`PatchMatchMetal.metal`** | **26,747** | MSL port of the 4 kernels (~650 lines) |
| **`PatchMatchMetal.mm`** | **16,816** | Objective-C++ host side (~450 lines) |
| **`PatchMatchMetal.h`** | **2,213** | PIMPL header, `namespace MVS::METAL` |
| `ConfidenceCUDA.cu/.h` | 17,313 / 5,532 | new (2026) GPU depth-confidence recalibration; CUDA only |
| `SceneRefineCUDA.cu/.cpp/.inl` | 47,344 / 53,567 / 8,369 | mesh refinement (RefineMesh) on CUDA |
| `SemiGlobalMatcher.cpp` | 85,471 | SGM — CPU only, no CUDA/Metal variant |
| `SceneDensify.cpp` | 178,683 | CPU PatchMatch + dispatcher |

So the CUDA surface is three independent pieces: **PatchMatch depth maps** (the hot path ODM
uses), **ConfidenceCUDA** (new, optional, "~3 ms/map" per `AGENTS.md`), and **RefineMesh**
(ODM does not run `RefineMesh`; `stages/openmvs.py` only calls `DensifyPointCloud`). SGM
(`--optimize` bit in `Densify.ini`; ODM writes `Optimize = 7`) has no GPU path anywhere.

### 2.1 PatchMatchCUDA.cu anatomy (fetched from upstream)

Kernels (all take the same argument set):

```
__global__ void InitializeScore(const cudaTextureObject_t* textureImages,
    const cudaTextureObject_t* textureDepths, const CUDA::Camera* cameras, Point4* planes,
    const float* lowDepths, float* costs, curandState* randStates, unsigned* selectedViews,
    const PatchMatch::Params params)
__global__ void BlackPixelProcess(... same ..., const int iter)
__global__ void RedPixelProcess  (... same ..., const int iter)
__global__ void FilterPlanes(Point4* planes, float* costs, unsigned* selectedViews,
    int width, int height, const PatchMatch::Params params)
```

* Block size 32×16; red/black checkerboard propagation (Galliani-style), 8 directions with
  adaptive search distances, "asymmetric" propagate-then-refine, four refinement hypotheses
  (perturbed depth, perturbed normal, random normal, normal from neighbouring depths).
* View selection: per-pixel top-K (`nInitTopK = 3`) with cost→probability (PDF→CDF) sampling,
  bit-packed into `selectedViews`.
* Geometric consistency: optional forward-backward reprojection term using neighbour depth
  textures.
* CUDA features used: `cudaTextureObject_t` (bilinear sampling of R32F images and depths),
  `curandState` per pixel, `__constant__` camera memory; **no shared memory, no shuffles, no
  atomics, no half, no double on device**. This is why it ported cleanly to Metal.
* `Params` defaults (`.inl`): `nNumViews=5`, `nEstimationIters=3`, `fDepthMin=0`,
  `fDepthMax=100`, `nInitTopK=3`, `bGeomConsistency=false`, `bLowResProcessed=false`,
  `fThresholdKeepCost=0`. `DensifyPointCloud.cpp` raises defaults to 8 views / 4 iterations
  when `_USE_CUDA` is defined.
* Host side (`PatchMatchCUDA.cpp`, upstream): global mutex + CUDA event to serialise
  `__constant__` uploads between worker threads; multi-resolution loop over
  `OPTDENSE::nSubResolutionLevels` (ODM passes `--sub-resolution-levels 2`), upsampling with
  `cv::INTER_NEAREST`; `cudaMallocArray` + texture objects per scale; pinned staging above
  1.5 Mpx; optional fused confidence pass at scale 0.

### 2.2 `--cuda-device` semantics

Tag 355 (`apps/DensifyPointCloud/DensifyPointCloud.cpp`):

```
#ifdef _USE_CUDA
("cuda-device", value(&CUDA::desiredDeviceID)->default_value(-1),
 "CUDA device number to be used for depth-map estimation (-2 - CPU processing, -1 - best GPU, >=0 - device index)")
#else
("cuda-device", value(&OPT::mockCudaDevice)->default_value(-1), "Just a placeholder (not a CUDA build)")
#endif
```

Upstream `develop` renamed it:

```
("gpu-device", value<std::string>(&SEACAVE::CUDA::desiredDeviceIDs)->default_value("-1"),
 "GPU device(s) for depth-map estimation (-1 best GPU, -2/cpu/empty CPU, >=0 comma-separated IDs)")
```

compiled only when `_USE_CUDA || _USE_METAL`. **ODM's `stages/openmvs.py` passes
`--cuda-device`; a rebase onto upstream would need that string changed (or an alias patch).**
Upstream also documents known CUDA-vs-CPU artefact differences on aerial data
([issue #708](https://github.com/cdcseacave/openMVS/issues/708)) and a crash on image-size
changes ([#1253](https://github.com/cdcseacave/openMVS/issues/1253)) — the Metal path inherits
the algorithm, so aerial validation is needed either way.

---

## 3. The upstream Metal backend (PR #1279) in detail

Source: [PR #1279](https://github.com/cdcseacave/openMVS/pull/1279) "Add Metal compute backend
for PatchMatch densification (Apple Silicon)", author `leNeo` (Frédéric Siffert), opened
2026-06-08, merged 2026-06-26 into `develop` (merge sha `4694ded`). Developed and measured on
"Apple M4 Max, 48 GB". The PR text states the shader strategy explicitly: "MSL embedded from
the .metal at configure time and compiled at runtime via newLibraryWithSource" — no Xcode or
Metal Toolchain requirement is stated, and none is needed. Reported results on the bundled
4-image test scene: "~5–6× faster wall-clock time" than CPU; Metal and CPU "agree to
~0.10–0.11 % median relative difference"; geometric-consistency passes "~20× faster".
Maintainer review: cdcseacave asked for "full testing on various datasets and configurations
and compare to the existing CUDA implementation" with Tanks-and-Temples F1 tables; after
that he noted the CUDA version exceeded Metal by "1–2 % in F1 score" and asked the author to
"finetune a bit the metal implementation" — so expect the Metal path to be *slightly* less
accurate than CUDA until tuned, though on par with the CPU path. Author's own to-do list:
buffer caching across calls, prebuilt `.metallib` instead of runtime compile, "validation on
small scene only", "large drone-scale benchmarks" pending.

Design (from the fetched sources):

* **Header** `PatchMatchMetal.h`: guarded by `#ifdef _USE_METAL`; `namespace MVS::METAL`;
  `class PatchMatch` with the *same* `Params` struct as CUDA, `IsValid()`, `Init(bool geom)`,
  `Release()` (no-op — "the Metal backend keeps no persistent per-estimate GPU state"),
  `EstimateDepthMap(DepthData&)`; PIMPL so no Metal types leak into C++ TUs.
* **Host** `PatchMatchMetal.mm`: `#import <Metal/Metal.h>` (not metal-cpp), ARC
  (`-fobjc-arc`), `MTLCreateSystemDefaultDevice()`, library from
  `newLibraryWithSource:` of `kPatchMatchMSL` (a string generated into
  `PatchMatchMetal_msl.h` at configure time from the `.metal` file — "the MSL kernels are the
  single source of truth; embed them as a string"), seven pipeline states (Init/Black/Red/Filter
  × `GEOM` function constant false/true), buffers `MTLResourceStorageModeShared`, images and
  neighbour depths as `MTLPixelFormatR32Float` textures uploaded with `replaceRegion:`, a 1×1
  dummy texture for neighbours without depth maps, threadgroup `MTLSizeMake(32, 8, 1)`, grid
  `W × H` for init/filter and `W × (H+1)/2` for checkerboard passes, one
  `waitUntilCompleted` per resolution level, `@autoreleasepool` per scale to bound memory.
* **Kernels** `PatchMatchMetal.metal`: "faithful translation" of the CUDA file; signatures e.g.

  ```
  kernel void BlackPixelProcess(array<texture2d<float>, MAX_VIEWS> images [[texture(0)]],
      array<texture2d<float>, MAX_VIEWS> depthImages [[texture(MAX_VIEWS)]],
      constant Camera* cams [[buffer(0)]], device float4* planes [[buffer(1)]],
      device const float* lowDepths [[buffer(2)]], device float* costs [[buffer(3)]],
      device uint* rngStates [[buffer(4)]], device uint* selectedViews [[buffer(5)]],
      constant Params& prm [[buffer(6)]], constant int& iter [[buffer(7)]],
      uint2 gid [[thread_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
  ```

  `curand` replaced by a PCG hash (`static inline uint pcg(thread uint& s)`), sampling via
  `constexpr sampler(coord::pixel, filter::linear, address::clamp_to_edge)`, all `float32`,
  no threadgroup memory, no SIMD-group ops.
* **Dispatcher** `SceneDensify.cpp`: `DepthMapsData` holds `pmCUDAPool` and `pmMetalPool` with
  atomic slot counters; `EstimateDepthMap` tries CUDA pool → Metal pool → CPU
  `DepthEstimator` threads. `AllocateMetalPool` probes `IsValid()` and stops growing the pool
  on failure. Metal runs "both photometric and geometric-consistency passes"; the fused
  confidence pass (`ConfidenceCUDA`) has **no Metal counterpart** — confidence falls back to the
  CPU cost→confidence conversion.
* **CMake**: root `CMakeLists.txt` — `IF(APPLE) FIND_LIBRARY(METAL_FRAMEWORK Metal)
  FIND_LIBRARY(FOUNDATION_FRAMEWORK Foundation)`; if both found `ENABLE_LANGUAGE(OBJCXX)`,
  `SET(_USE_METAL TRUE)`, append frameworks to `OpenMVS_EXTRA_LIBS`, message "Enabling Metal
  compute backend on macOS"; else "Metal/Foundation frameworks not found; Metal backend
  disabled". There is **no `OpenMVS_USE_METAL` option** — it is automatic on Apple.
  `libs/MVS/CMakeLists.txt` appends `PatchMatchMetal.mm` with
  `SKIP_PRECOMPILE_HEADERS ON COMPILE_OPTIONS -fobjc-arc` and generates the MSL header.
  No `xcrun metal` is invoked anywhere → **builds with Command Line Tools only**, which matches
  the target machine (no Xcode.app; see `00-*.md` §7).

What is *not* on Metal upstream: SGM, `ConfidenceCUDA`, `SceneRefineCUDA` (RefineMesh),
the new `roma2` learned matching (`CreateStructure --roma2`, ONNX Runtime in-process, released
2026-09-09 as tag `roma2-model` — on macOS that could use the CoreML EP, see `03-*.md`).

---

## 4. Existing alternatives surveyed (none better than §3)

| Candidate | GPU API | Notes |
|---|---|---|
| OpenMVS issue [#1076](https://github.com/cdcseacave/openMVS/issues/1076) "opencl support?" (2023) | — | Maintainer: "the densification algorithm would be a perfect match for porting to openCL"; refine-mesh "should use plain openGL shaders". Closed 2026-09-12: "Pull requests for alternative compute backends are welcome." No OpenCL/HIP port ever landed; [#306](https://github.com/cdcseacave/openMVS/issues/306), [#558](https://github.com/cdcseacave/openMVS/issues/558) (CUDA on Mac) confirm CUDA-only history. |
| COLMAP PatchMatch stereo | CUDA only | "Dense reconstruction requires compilation with CUDA support" ([FAQ](https://colmap.github.io/faq.html), [#2390](https://github.com/colmap/colmap/issues/2390), [#192](https://github.com/colmap/colmap/issues/192)). |
| [lanxinger/colmap-metal](https://github.com/lanxinger/colmap-metal) / byplay-io fork | Metal | Metal **SIFT + matching** only; README: "Dense stereo/PatchMatch and GPU bundle adjustment remain CUDA-backed; this fork does not implement those stages in Metal." Relevant to `02-*.md`, not here. |
| ACMM/ACMP/ACMH ([GhiXu/ACMM](https://github.com/GhiXu/ACMM)), Gipuma ([kysucix/gipuma](https://github.com/kysucix/gipuma)), fixstars [cuda-multi-view-stereo](https://github.com/fixstars/cuda-multi-view-stereo), TAPA-MVS | CUDA | All CUDA; curated in [Awesome-PatchMatch-MVS](https://github.com/XYZ-qiyh/Awesome-PatchMatch-MVS). |
| Vulkan/WebGPU PatchMatch MVS | Vulkan | Searched "vulkan patchmatch stereo", "metal multi-view stereo": no open-source multi-view PatchMatch in Vulkan/WGSL found (only single-image-pair stereo demos and rendering samples) **[uncertain: absence of evidence]**. |
| OpenSfM's own dense (`opensfm/dense.py`, `pydense.DepthmapEstimator`) | CPU | `PATCH_MATCH` / `PATCH_MATCH_SAMPLE` / `BRUTE_FORCE`, multiprocessing pool. ODM replaced it with OpenMVS years ago; no GPU. |
| Learned MVS (MVSNet, CasMVSNet, [PatchmatchNet](https://github.com/FangjinhuaWang/PatchmatchNet)) | PyTorch | PatchmatchNet ships TorchScript export; no maintained ONNX export found. Would run via PyTorch-MPS or coremltools; see §6 (v). |
| Apple RealityKit `PhotogrammetrySession` | Metal (closed) | See §7. |

---

## 5. Realistic paths, effort, recommendation

### (i) Stay on CPU
Zero effort. On 12 performance/efficiency cores the CPU PatchMatch with `--max-threads 12` is
what runs today. Baseline numbers belong in `../cpu-baseline/`.

### (ii-a) Rebase ODM's openMVS pin onto upstream with the Metal backend — **primary recommendation**, 1–3 days
1. **Pin**: start from ODM PR #2074's `openmvs-240` branch (already on v2.4.0, CGAL 6,
   `-DCMAKE_CUDA_ARCHITECTURES=OFF`) and move `GIT_TAG` in
   `SuperBuild/cmake/External-OpenMVS.cmake` to a fork branch that tracks upstream `develop`
   ≥ `4694ded` (create e.g. `omvs-metal` in `OpenDroneMap/openMVS` from upstream develop, then
   re-apply the fork's ODM-specific commits: `INSTALL_USE_SUBDIR`, scalable texturing work
   funded by ODM — [community thread](https://community.opendronemap.org/t/adding-scalable-texturing-to-openmvs-and-removing-a-memory-bottleneck/25571/3)).
   Alternative if the rebase is too invasive: cherry-pick PR #1279 (`PatchMatchMetal.{h,mm,metal}`,
   the `SceneDensify.cpp` pool hooks, the `DensifyPointCloud.cpp` `--gpu-device` option, root
   and `libs/MVS` CMake changes — the PR is self-contained) onto `omvs-v240`
   **[uncertain: conflict size against v2.4.0's `SceneDensify.cpp`]**.
2. **CMake plumbing** (`External-OpenMVS.cmake`, `SuperBuild/CMakeLists.txt`): upstream
   auto-enables Metal on APPLE when `Metal.framework`/`Foundation.framework` are found (no
   `OpenMVS_USE_METAL` option exists — §3). ODM should add an `ODM_ENABLE_METAL` env → 
   `SB_ENABLE_METAL` switch mirroring `ODM_ENABLE_CUDA`, and on APPLE pass
   `-DOpenMP_ROOT=$CONDA_PREFIX` (upstream hard-codes `/opt/homebrew/opt/libomp`) and, if the
   conda clang cannot see the SDK, `-DCMAKE_OSX_SYSROOT=$SDKROOT`. To *force Metal off* a
   one-line fork patch adding `OPTION(OpenMVS_USE_METAL ...)` around upstream's detection block
   is needed (document only). Layout in `05-*.md` §4.
3. **`stages/openmvs.py`**: rename the flag — pass `--gpu-device -1` (GPU) / `-2` (CPU)
   instead of `--cuda-device`; upstream compiles the option only under `_USE_CUDA || _USE_METAL`,
   so a CPU-only build would reject it — keep the existing "not a GPU build" placeholder
   behaviour in mind or gate on a build-time capability check. Keep the exit-code-1 → CPU
   retry and the OOM → tiling retry unchanged.
4. **`opendm/gpu.py`**: add a `has_metal()` probe (`05-*.md` §6) and have the OpenMVS stage
   use `has_gpu() or has_metal()`; `has_popsift_and_can_handle_texsize` stays CUDA-only.
5. **Build verification**: conda clang must find `Metal.framework` through the SDK sysroot and
   compile OBJCXX; `-fobjc-arc` on the `.mm`; **no `xcrun metal` needed** (this Mac has none).
6. **Validate** on an aerial dataset (upstream only tested a 4-image scene; the maintainer
   measured CUDA 1–2 % better F1 than Metal on TnT): compare against CPU output (point count,
   `--pc-quality` levels, `--pc-geometric`), and memory on the shared 24 GB.

### (ii-b) Improve the Metal backend — 1–3 weeks
Author-listed gaps plus obvious wins: cache `MTLBuffer`/`MTLTexture` across calls (avoid
`replaceRegion` re-uploads; use `newBufferWithBytesNoCopy` on 16 KB-aligned `cv::Mat` data —
see `00-*.md` §4); use `half` textures for images (R16Float, halves bandwidth); port
`ConfidenceCUDA.cu` (17 KB, "per-pixel math shared verbatim with the CPU via
`ConfidenceRefine.h`") so `ADJUST_CONFIDENCE` is available on Metal; optional `.metallib`
prebuild when the Metal Toolchain exists; Metal 4 argument tables/residency sets to cut
per-dispatch binding cost. Potential SGM port is a separate, larger job (85 KB CPU file,
dynamic-programming path aggregation — GPU SGM designs exist in the literature but none in
OpenMVS).

### (iii) Route via MoltenVK / Vulkan
Would require writing Vulkan compute + GLSL from scratch (no existing Vulkan MVS to reuse),
then run it through SPIRV-Cross→MSL at runtime. Only justified if one shader source must also
serve the Radeon/Linux track. Effort: weeks; performance ≤ native Metal. Not recommended now
that a native backend exists upstream.

### (iv) Replace densification with a different MVS
No Metal/OpenCL MVS exists (§4). Apple's `PhotogrammetrySession` (§7) is a black box that does
SfM+MVS+mesh+texture end-to-end and cannot emit ODM's `scene_dense.mvs`/depth maps; it could
only replace the whole ODM 3D branch for small object-scale captures, not orthophoto workflows.
Not viable as a drop-in.

### (v) Learned MVS through onnxruntime CoreML / PyTorch MPS
PatchmatchNet is explicitly designed for "resource-limited devices" and has a TorchScript
export; CasMVSNet/MVSNet are cost-volume-heavy. Practical blockers: no maintained ONNX exports
with dynamic image sizes; CoreML EP op coverage for `grid_sample`-style homography warping and
3D convolutions is partial (`GridSample` is listed for MLProgram, but 3D conv is not —
`03-*.md`); accuracy on nadir aerial imagery vs PatchMatch is unproven for ODM's DEM/orthophoto
outputs. Effort: weeks of research; keep as an R&D track only.

**Recommendation:** (ii-a) immediately, then (ii-b) increments driven by profiling on real
drone datasets. Keep CPU fallback exactly as `stages/openmvs.py` already does.

---

## 6. Design sketch, for reference (if a from-scratch Metal port were needed)

This is what upstream implemented; kept here so the doc stands alone.

* **Buffers (Shared):** `float4 planes[W*H]` (normal.xyz, depth), `float costs[W*H]`,
  `uint rng[W*H]`, `uint selectedViews[W*H]` (bitmask over ≤ 32 neighbours → `MAX_VIEWS`),
  `float lowDepths[W*H]` (previous scale), `Camera cams[N]` (K, R, C, size) in `constant`,
  `Params` in `constant`.
* **Textures:** reference + neighbour grey images as `R32Float`/`R16Float` 2D textures with a
  linear clamp sampler; neighbour depth maps as `R32Float` textures for the geometric term.
* **Kernel 1 – InitializeScore:** per pixel, random plane from `[fDepthMin,fDepthMax]` or the
  upsampled low-res plane; compute NCC cost against every neighbour; keep top-K views.
* **Kernel 2/3 – Black/Red PixelProcess:** checkerboard half-grid dispatch; gather candidate
  planes from 8 directions at adaptive strides; evaluate; then 4 refinement hypotheses; update
  view-selection probabilities. `iter` bound as `constant int&`.
* **Kernel 4 – FilterPlanes:** zero out planes whose cost > `fThresholdKeepCost`.
* **Host loop:** for scale = coarse→fine: upload/resize images; Init; for `iter` in
  `nEstimationIters`: Black, Red; Filter; read back planes → `depthMap`/`normalMap`/`confMap`.
  One command buffer per scale; `waitUntilCompleted` once per scale.
* **Threads:** 32×8 threadgroups (256 threads = 8 SIMD-groups) is a good default on Apple GPUs;
  tune with `maxTotalThreadsPerThreadgroup` (1024) and occupancy tools in Xcode's Metal
  debugger (requires Xcode.app).

---

## 7. Apple's own photogrammetry engine (for completeness)

RealityKit `PhotogrammetrySession` (macOS 12+) reconstructs a textured mesh from 20–200 images
at `preview/reduced/medium/full/raw` detail, outputs USDZ/USDA/OBJ, runs locally, requires
"Silicon Macs or Intel Macs with 4GB AMD and 16GB RAM" (WWDC21 10076 notes), and is driven by
the `HelloPhotogrammetry` command-line sample
([Apple docs](https://developer.apple.com/documentation/RealityKit/creating-a-photogrammetry-command-line-app),
[WWDC21 10076](https://developer.apple.com/videos/play/wwdc2021/10076/),
[sfomuseum CLI wrapper](https://github.com/sfomuseum/swift-photogrammetry-render-cli)). It
does not expose cameras, sparse/dense point clouds, or georeferencing, so it cannot slot into
ODM's stage graph; it is a useful *speed reference* for what the M4 Pro GPU can do on a
comparable problem.

---

## Sources

- ODM SuperBuild: `/Users/avneh/Code/ADM/ODM/SuperBuild/cmake/External-OpenMVS.cmake`, `/Users/avneh/Code/ADM/ODM/stages/openmvs.py`, `/Users/avneh/Code/ADM/ODM/opendm/gpu.py` (read locally)
- OpenDroneMap/openMVS branches API: https://api.github.com/repos/OpenDroneMap/openMVS/branches ; branch 355 tip: https://api.github.com/repos/OpenDroneMap/openMVS/commits/355
- ODM PR #2074 "Openmvs 240": https://api.github.com/repos/OpenDroneMap/ODM/pulls/2074 ; files: https://api.github.com/repos/OpenDroneMap/ODM/pulls/2074/files
- PR #1279 page (M4 Max, runtime MSL, review comments): https://github.com/cdcseacave/openMVS/pull/1279
- OpenDroneMap/openMVS tag 355 `libs/MVS` listing: https://api.github.com/repos/OpenDroneMap/openMVS/contents/libs/MVS?ref=355
- OpenDroneMap/openMVS tag 355 DensifyPointCloud.cpp: https://raw.githubusercontent.com/OpenDroneMap/openMVS/355/apps/DensifyPointCloud/DensifyPointCloud.cpp
- cdcseacave/openMVS: https://github.com/cdcseacave/openMVS
- PR #1279 (Metal backend): https://github.com/cdcseacave/openMVS/pull/1279 ; commit history of `PatchMatchMetal.mm`: https://api.github.com/repos/cdcseacave/openMVS/commits?path=libs/MVS/PatchMatchMetal.mm
- Upstream sources (develop): `libs/MVS/PatchMatchMetal.h`, `.mm`, `.metal`, `PatchMatchCUDA.cu`, `.cpp`, `.inl`, `ConfidenceCUDA.h`, `SceneDensify.cpp`, `libs/MVS/CMakeLists.txt`, root `CMakeLists.txt`, `apps/DensifyPointCloud/DensifyPointCloud.cpp`, `libs/MVS/AGENTS.md` — https://raw.githubusercontent.com/cdcseacave/openMVS/develop/...
- Upstream `libs/MVS` listing: https://api.github.com/repos/cdcseacave/openMVS/contents/libs/MVS
- Upstream releases (v2.4.0 2026-01-20; roma2-model 2026-09-09): https://api.github.com/repos/cdcseacave/openMVS/releases
- Issues: #1076 https://github.com/cdcseacave/openMVS/issues/1076 ; #306 https://github.com/cdcseacave/openMVS/issues/306 ; #558 https://github.com/cdcseacave/openMVS/issues/558 ; #708 https://github.com/cdcseacave/openMVS/issues/708 ; #1253 https://github.com/cdcseacave/openMVS/issues/1253
- ODM community, scalable texturing in openMVS: https://community.opendronemap.org/t/adding-scalable-texturing-to-openmvs-and-removing-a-memory-bottleneck/25571/3
- COLMAP FAQ: https://colmap.github.io/faq.html ; issues #2390 https://github.com/colmap/colmap/issues/2390 ; #192 https://github.com/colmap/colmap/issues/192
- colmap-metal: https://github.com/lanxinger/colmap-metal ; https://github.com/byplay-io/colmap-metal
- ACMM: https://github.com/GhiXu/ACMM ; Gipuma: https://github.com/kysucix/gipuma ; fixstars CUMVS: https://github.com/fixstars/cuda-multi-view-stereo ; Awesome-PatchMatch-MVS: https://github.com/XYZ-qiyh/Awesome-PatchMatch-MVS
- PatchmatchNet: https://github.com/FangjinhuaWang/PatchmatchNet ; MVSNet: https://arxiv.org/pdf/1804.02505
- OpenSfM dense.py (ODM fork commit c5328439): https://raw.githubusercontent.com/OpenDroneMap/OpenSfM/c5328439465e6ace011f39077d1077d7b1cdd65d/opensfm/dense.py
- Apple, photogrammetry command-line app: https://developer.apple.com/documentation/RealityKit/creating-a-photogrammetry-command-line-app ; WWDC21 10076: https://developer.apple.com/videos/play/wwdc2021/10076/ ; WWDCNotes: https://github.com/WWDCNotes/Content/blob/main/content/notes/wwdc21/10076.md ; sfomuseum CLI: https://github.com/sfomuseum/swift-photogrammetry-render-cli
