# OpenMVS CUDA anatomy (ODM fork, tag `355`)

Scope: the exact GPU code that ODM 3.6.2 builds and runs when `stages/openmvs.py`
invokes `DensifyPointCloud`. Everything below was read from a clone of
`https://github.com/OpenDroneMap/openMVS` at tag `355` (commit `8e4c5f31a`,
2026-05-16, "Merge pull request #2 from sylveon/355"), and compared against
upstream `cdcseacave/openMVS` via the GitHub compare API. Line numbers refer to
that tag unless marked *upstream*.

Version identity: `CMakeLists.txt:56-58` sets `OpenMVS_VERSION 2.2.0`. The
fork's merge-base with upstream `develop` is `f57daf5b7` (2023-08-20,
"interface: add depth and mask importer to RTMV"); the fork is 94 commits ahead
and 148 behind `develop` (GitHub compare `develop...OpenDroneMap:openMVS:355`).

---

## 1. Inventory of CUDA source files

| File | LOC | Role |
|---|---|---|
| `libs/MVS/PatchMatchCUDA.cu` | 664 | All PatchMatch kernels + `RunCUDA` launcher |
| `libs/MVS/PatchMatchCUDA.cpp` | 421 | Host side: allocation, texture creation, upload/download, multi-resolution loop |
| `libs/MVS/PatchMatchCUDA.inl` | 143 | Class declaration `MVS::PatchMatchCUDA`, `Params`, `Camera`; includes `<curand_kernel.h>`, `<cuda_texture_types.h>`, Eigen |
| `libs/MVS/PatchMatchCUDA.h` | 58 | Thin wrapper that pushes/pops `EIGEN_DEFAULT_DENSE_INDEX_TYPE` around the `.inl` |
| `libs/Common/UtilCUDA.h` | 720 | Driver-API helpers: `Device`, `initDevice`, `checkCudaCall`, PTX JIT module/kernel wrappers (`KernelRT`, `cuLaunchGrid*`, `cuModuleGetTexRef`) |
| `libs/Common/UtilCUDA.cpp` | 435 | `desiredDeviceID` global (line 23), SM-to-cores table (27-64), best-GFLOPS device pick (110-160), `initDevice` (164-193), `ptxJIT` (196-257) |
| `libs/MVS/SceneRefineCUDA.cpp` | 2867 | Mesh refinement on GPU: 13 kernels stored as **PTX/C strings** and JIT-loaded via `cuModuleLoadData` (kernel list at lines 2027-2039; `Reset(module,"ProjectMesh")` etc. at 2124-2162). Only used by `RefineMesh --use-cuda`; **ODM never calls RefineMesh**, so this file is dead weight for ODM |
| **Total** | **5308** | |

Build wiring: `libs/MVS/CMakeLists.txt:30-33` globs `*.cu` into the MVS library
only when `_USE_CUDA`; `libs/MVS/CMakeLists.txt:51` pins
`CUDA_ARCHITECTURES "50;72;75"` on the `MVS` target (and line 62 on `pyOpenMVS`).
No `.cuh` files exist; no cub/thrust anywhere in `libs/MVS` or `libs/Common`.

---

## 2. How `--cuda-device` selects GPU vs CPU

`apps/DensifyPointCloud/DensifyPointCloud.cpp:103-107`:

```cpp
#ifdef _USE_CUDA
("cuda-device", value(&CUDA::desiredDeviceID)->default_value(-1),
   "CUDA device number to be used for depth-map estimation (-2 - CPU processing, -1 - best GPU, >=0 - device index)")
#else
("cuda-device", value(&OPT::mockCudaDevice)->default_value(-1), "Just a placeholder (not a CUDA build)")
#endif
```

The `#else` placeholder (and `int mockCudaDevice;` at line 73) is a fork addition
("add mock cuda-device opt", `3bbe78e08`, 2022-05-27) so that ODM can always pass
`--cuda-device -2` without the CPU build rejecting the flag.

Selection happens in `libs/MVS/SceneDensify.cpp:1823-1832`:

```cpp
#ifdef _USE_CUDA
if (CUDA::desiredDeviceID >= -1 && data.nFusionMode >= 0) {
    data.depthMaps.pmCUDA = new PatchMatchCUDA(CUDA::desiredDeviceID);
    if (CUDA::devices.IsEmpty())
        data.depthMaps.pmCUDA.Release();      // init failed -> CPU path
    else
        data.depthMaps.pmCUDA->Init(false);
}
#endif
```

- `-2` fails the `>= -1` test, so `pmCUDA` stays null and the CPU estimator is used.
- `-1` / `>=0` go through `PatchMatchCUDA::PatchMatchCUDA(int device)`
  (`PatchMatchCUDA.cpp:46-51`) → `CUDA::initDevice(device)`
  (`UtilCUDA.cpp:164-193`): `cuInit(0)`, then either `_gpuCheckDeviceId` or
  `_gpuGetMaxGflopsDeviceId`; **rejects compute capability major < 3**
  (line 178, message says "3.2 or greater required"); `cuCtxCreate` (line 183).
  If it fails, `devices` stays empty and the code silently falls back to CPU.
- Note the `nFusionMode >= 0` guard: ODM's OOM retry adds `--fusion-mode 1`
  (`stages/openmvs.py:118`), which is still `>= 0`, so GPU remains enabled for
  sub-scene splitting; the later per-sub-scene densify runs
  (`stages/openmvs.py:178`) also carry `gpu_config`.

Dispatch per image, `SceneDensify.cpp:616-623`:

```cpp
bool DepthMapsData::EstimateDepthMap(IIndex idxImage, int nGeometricIter)
{
    #ifdef _USE_CUDA
    if (pmCUDA) { pmCUDA->EstimateDepthMap(arrDepthData[idxImage]); return true; }
    #endif
    ... CPU DepthEstimator path ...
```

There is exactly **one** `PatchMatchCUDA` instance (`CAutoPtr<PatchMatchCUDA> pmCUDA`,
`SceneDensify.h:91`), and `DenseReconstructionEstimate` runs two worker threads
(`SceneDensify.cpp:1846`); estimation is serialized through the `sem(1)`
semaphore constructed at `SceneDensify.cpp:1605` (moderately confident; the
event loop is not reproduced here).

Geometric-consistency phase: `SceneDensify.cpp:1861-1867` re-initialises with
`pmCUDA->Release(); pmCUDA->Init(true)`; `Init(true)` forces
`nEstimationIters = 1` for geometric passes (`PatchMatchCUDA.cpp:97-106`).
`OPTDENSE::nEstimationGeometricIters` defaults to 2 (`DepthMap.cpp:105`);
ODM's `--geometric-iters 0` (`stages/openmvs.py:91`) disables the phase.

**Compile-time default divergence (important for benchmarking):**
`DensifyPointCloud.cpp:111-117`

```cpp
#ifdef _USE_CUDA
const unsigned nNumViewsDefault(8); const unsigned numIters(4);
#else
const unsigned nNumViewsDefault(5); const unsigned numIters(3);
#endif
```

feed `--number-views` (line 142) and `--iters` (line 146). ODM passes neither,
so a CUDA-enabled binary uses 8 views / 4 iterations **even when run with
`--cuda-device -2`**, while a CPU-only binary uses 5 / 3. Any CPU-vs-GPU
comparison on the same `odm:gpu` image is therefore not apples-to-apples with
the plain `odm` image.

---

## 3. Kernel anatomy (`libs/MVS/PatchMatchCUDA.cu`)

Constants (lines 34-44): `MAX_VIEWS 32`, `NUM_SAMPLES 32` (Monte-Carlo view
samples), `nSizeHalfWindow 4`, `nSizeStep 2` → the NCC patch is a 9×9 window
sampled every 2 px = **25 texels** per view per hypothesis. `ImagePixels` is
`cudaTextureObject_t`; `RandState` is `curandState` (= `curandStateXORWOW_t`).

Four `__global__` kernels, launched from `RunCUDA` (lines 631-663):

| Kernel | Lines | Grid | What it does |
|---|---|---|---|
| `InitializeScore` | 592-596 → `InitializePixelScore` 551-591 | full (`gridSizeFull`) | `curand_init(1234, p.y, p.x, state)` (line 564, deterministic seed); if `depth<=0` draw a random plane (uniform depth in `[fDepthMin,fDepthMax]`, random hemisphere normal facing the camera, lines 567-570); else if normal faces away, re-draw the normal (571-574); score against all views; bubble-sort costs (`Sort`, 107-120); keep the `nInitTopK` (=3, `.inl:87`) best as `selectedViews` bitmask; initial cost = mean of top-K |
| `BlackPixelProcess` | 599-604 | checkerboard | `p.y = p.y*2 + (threadIdx.x%2==0 ? 0 : 1)` then `ProcessPixel` |
| `RedPixelProcess` | 605-610 | checkerboard | complementary parity, same body |
| `FilterPlanes` | 613-627 | full | if `depth<=0` or `cost >= fThresholdKeepCost` zero the plane, cost and views |

`RunCUDA` sequence (lines 643-662): `InitializeScore` → `cudaDeviceSynchronize`
→ for `iter in [0,nEstimationIters)`: Black, sync, Red, sync → optional
`FilterPlanes` → three `cudaMemcpy` D2H (planes, cost map, views map) → sync.
Block is 32×16 (`BLOCK_W 32`, `BLOCK_H 16`, lines 636-641); everything runs on
the **default stream**; no streams, events, graphs, shared memory, shuffles,
atomics, constant memory or dynamic parallelism are used.

### 3.1 `ProcessPixel` (lines 389-548) — ACMH propagation + refinement

Cited paper (comment at 387-388): *"Multi-View Stereo with Asymmetric
Checkerboard Propagation and Multi-Hypothesis Joint View Selection"* (ACMH, 2018).

1. **Adaptive checkerboard sampling** (403-445): eight direction sets
   `dirs[8][11]` — four "near" V-shapes (7 offsets each) and four "far" lines
   (11 offsets each, stride 2 up to ±23 px). For each set, pick the neighbour
   with the lowest current cost, interpolate its plane to `p`
   (`MultiViewScoreNeighborPlane` 370-375 → `InterpolatePixel` 218-242), and
   score it against every view into `costArray[8][MAX_VIEWS]` (a 1 KB
   per-thread local array).
2. **Multi-hypothesis joint view selection** (447-491): priors from the 4-neighbour
   `selectedViews` bitmasks (0.9 if the neighbour used the view, else 0.1);
   per view count how many of the 8 hypotheses score below
   `thCost = 0.8*exp(-iter²/32)` (line 458) and above `thCostBad = 1.2`;
   convert to a PDF, `PDF2CDF` (136-147), then draw `NUM_SAMPLES=32`
   `curand_uniform` samples to build integer `viewWeights[]`.
3. **Propagation** (493-513): aggregate `finalCosts[8]` with those weights
   (`AggregateMultiViewScores` 378-385, divides by `NUM_SAMPLES`), compare with
   the current plane's aggregated cost, and adopt the best neighbour plane
   (normal copied, depth = interpolated depth).
4. **Refinement** (515-547): three or four candidate planes — perturbed depth
   (`±0.5 %`, `perturbationDepth 0.005`), perturbed normal (`0.01π` rad Euler
   rotation, `GeneratePerturbedNormal` 173-203), a fresh random normal, and, when
   all 4 direct neighbours are valid, a normal from the depth gradient
   (`ComputeDepthGradient` 245-258). Keep the cheapest.

### 3.2 Cost: bilateral-weighted ZNCC (`ScorePlane`, lines 299-358)

- `ComputeHomography` (261-268): `H = K_trg · R_trg · (R_refᵀ + t nᵀ) · K_ref⁻¹`
  with `t = (C_ref − C_trg)/(n·X)`; note `K.inverse()` is an Eigen 3×3 inverse
  evaluated in device code.
- Early-out `maxCost 1.2` if the centre projects outside the target (305-306).
- Walks the 5×5 stepped window incrementally (`X += H.col(0)`, `baseX += H.col(1)`,
  lines 318-337) with `tex2D<float>` bilinear fetches at `+0.5` pixel centres.
- `ComputeBilateralWeight` (271-278): `exp(spatial²·σs + Δcolour²·σc)` with
  `σs = −1/(2·(nSizeHalfWindow−1)²)`, `σc = −1/(2·(25/255)²)`.
- ZNCC → `ncc = 1 − cov/√(varRef·varTrg)`, then the **low-resolution depth prior**
  (349-356): in textureless patches (`varRef` small, `smoothSigmaDepth −1/0.02`)
  blend toward `|lowDepth−depth|/lowDepth` (clamped 0.5). Output clamped `[0,2]`.
- `MultiViewScorePlane` (361-368) scores all `nNumViews` targets and, in
  geometric mode, adds `0.1 · GeometricConsistencyWeight` (281-296): forward
  project, `tex2D` the neighbour's depth texture, back-project, reprojection
  distance `min(4, √(d(d+2)))`; a null depth texture contributes 0.

### 3.3 Host side (`PatchMatchCUDA.cpp`)

- `AllocatePatchMatchCUDA` (108-123): per reference-image size allocates
  `cudaTextureObject_t[numImages]`, `Camera[numImages]`, host `Point4[size]`
  plus device `Point4[size]` (depth+normal), `float[size]` costs,
  `unsigned[size]` selected views, `curandState[size]`.
- `AllocateImageCUDA` (125-172): `cudaMallocArray` of 32-bit float, texture
  descriptor `addressMode Wrap`, `filterMode Linear`, `readMode ElementType`,
  `normalizedCoords 0` (137-143); identical descriptor for per-neighbour depth
  textures in geometric mode (155-170).
- `EstimateDepthMap` (174-418): multi-resolution loop over
  `OPTDENSE::nSubResolutionLevels` (default 2, `DepthMap.cpp:72`; ODM passes
  `--sub-resolution-levels 2`), halving the scale per level via
  `DepthMapsData::ScaleDepthData` (`SceneDensify.cpp:578`); the previous level's
  depth/normal/views are `cv::resize`d up and uploaded as `cudaLowDepths`
  (213-216, 349-352). Textures are re-created only when image count or size
  changes (264-291); otherwise `cudaMemcpy2DToArray` reuploads. Threshold rule
  (235-252): `fThresholdKeepCost = fNCCThresholdKeep` (×1.2 at the smallest
  level or when geometric passes are pending; 0 = no filtering at middle levels).
  Post-processing (361-394): confidence = `1 − cost` (0 if cost ≥ 1), and the
  32-bit view bitmask is unpacked into up to 4 view indices (255 = none).
  Ignore-mask applied on CPU afterwards (405-411; fork commit
  `56ef831b7` "Fix ignore mask on GPU").

---

## 4. CPU fallback (`libs/MVS/DepthMap.cpp`, 2158 LOC; `SceneDensify.cpp:616-805`)

The CPU path is a **different algorithm**, not a port of the CUDA kernels:

- `DepthEstimator` (`DepthMap.h:277-281`): same window constants
  (`nSizeHalfWindow 4`, `nSizeStep 2`), `nTexels = 25`.
- Traversal is a zig-zag/row-sweep (`MapMatrix2ZigzagIdx`, `DepthMap.cpp:329`),
  alternating direction per iteration, one `DepthEstimator` per thread
  (`SceneDensify.cpp:700-716, 733-750`) sharing an atomic pixel counter
  (`idxPixel`).
- Propagation from **direct neighbours only** (`ProcessPixel` 630-914),
  smoothness bonus (`fRandomSmoothBonus 0.93`, `DepthMap.cpp:114`), then
  `nRandomIters=6` random-search steps (`DepthMap.cpp:107, 812-843`) with
  angle ranges 16°/10° and depth ratio 0.003 (109-111).
- Cost is weighted NCC too (`ScorePixelImage` 465-566, `DENSE_NCC_WEIGHTED`
  branch uses a precomputed `weightMap0`, `SceneDensify.cpp:645-646, 673-675`).
- View selection is fixed per pixel (top scores across `images`), not the
  Monte-Carlo scheme.
- Geometric iterations reuse the same loop with `iterBegin = nEstimationIters +
  nGeometricIter` (`SceneDensify.cpp:628-629`).

Upstream's own Metal PR measured the consequence: CUDA/Metal raw depth maps are
≈98 % dense vs ≈78 % for the CPU `DepthEstimator` on their test scene
(PR #1279 body) — i.e. the two backends do not produce equivalent outputs.

---

## 5. CUDA API surface actually used (PatchMatch path)

| Category | Used | Where |
|---|---|---|
| Runtime memory | `cudaMalloc`, `cudaFree`, `cudaMemcpy` (H2D/D2H), `cudaMallocArray`, `cudaFreeArray`, `cudaMemcpy2DToArray` | `PatchMatchCUDA.cpp:111-122, 130, 155, 292, 298, 327-351, 359`; `.cu:656-660` |
| Textures | `cudaCreateChannelDesc(32,0,0,0,Float)`, `cudaResourceTypeArray`, `cudaCreateTextureObject`, `cudaDestroyTextureObject`, device `tex2D<float>` (4 call sites: `.cu:288, 317, 322, 323`) | `PatchMatchCUDA.cpp:127-170` |
| RNG | `curand_init`, `curand_uniform` on `curandState` (XORWOW), one state per pixel | `.cu:156-157, 177-179, 212, 484, 564, 570` |
| Launch/sync | `<<<grid,block>>>`, `cudaDeviceSynchronize`, `cudaGetLastError` | `.cu:643-662`; `.cpp:357` |
| Driver API | `cuInit`, `cuDeviceComputeCapability`, `cuDeviceGetName`, `cuDeviceTotalMem`, `cuCtxCreate` | `UtilCUDA.cpp:89, 120, 169-190` |
| Device-side C++ | Eigen `Matrix<float,3,3>::inverse()`, `.hnormalized()`, `.normalized()`; `constexpr` device functions; `static constexpr int2 dirs[8][11]` | `.cu:58-104, 246, 267, 403` |
| nvcc flags | `--expt-relaxed-constexpr` (required for Eigen in device code) | `CMakeLists.txt:154` |
| **Not used** | shared memory, `__shfl_*`, warp vote, atomics, `__constant__`, streams, events, cub/thrust, cooperative groups, half precision, unified memory | grep of `.cu` |

Non-PatchMatch CUDA (not exercised by ODM): `SceneRefineCUDA.cpp` +
`UtilCUDA.h` use the driver API's PTX JIT (`cuModuleLoadData`,
`cuModuleGetFunction`, `cuLaunchGrid`, `cuModuleGetTexRef`, `cuModuleGetSurfRef`;
`UtilCUDA.cpp:196-262`, `UtilCUDA.h:363-388, 618, 686`).

---

## 6. Compute-capability and architecture flags

- Runtime gate: CC major ≥ 3 (`UtilCUDA.cpp:178`). `opendm/gpu.py:17` separately
  requires ≥ 3.5 before enabling PopSift, but **not** before enabling OpenMVS
  (`has_gpu` only checks for `nvidia-smi`/`nvcuda.dll`, `gpu.py:73-95`).
- `CMakeLists.txt:108-160`:
  - `FIND_PACKAGE(CUDA)` + `ENABLE_LANGUAGE(CUDA)` (109-111).
  - If `OpenMVS_MAX_CUDA_COMPATIBILITY` is OFF: `CMAKE_CUDA_ARCHITECTURES` defaults
    to `75` when not set (126-128).
  - If ON (**ODM sets `-DOpenMVS_MAX_CUDA_COMPATIBILITY=ON`**,
    `SuperBuild/cmake/External-OpenMVS.cmake`): runs `nvcc --list-gpu-arch` and
    `--list-gpu-code` (118-120, 137-139), then appends
    `-arch=<first code>` and one `-gencode=arch=compute_XX,code=sm_XX` pair per
    architecture the installed nvcc supports (144-150). With CUDA 12.x
    (`pixi.toml:80-81` pins `cuda-toolkit = "12.*"`) that is sm_50 … sm_90;
    CUDA 12.x dropped sm_35/37 (the fork's 2021 commit "Target sm_35",
    `4b85f74e0`, is superseded).
  - `libs/MVS/CMakeLists.txt:51` additionally sets the target property
    `CUDA_ARCHITECTURES "50;72;75"`. CMake emits `-gencode` for both the target
    property and the raw `CMAKE_CUDA_FLAGS`; the exact union/precedence on the
    final fatbin was **not verified** here — check `nvcc` command lines in a
    build log.
- Fork commit `7bd32db29` (2021-10-29) introduced `OpenMVS_MAX_CUDA_COMPATIBILITY`;
  upstream has since removed this mechanism in favour of standard
  `CMAKE_CUDA_ARCHITECTURES` (upstream `libs/MVS/CMakeLists.txt:95-98` comment).

---

## 7. Memory per reference image (derived from allocations)

Let `P = W×H` pixels of the depth map at the resolution level actually
estimated, `N = nNumViews` (8 in a CUDA build unless overridden), and
`I = N+1` images.

Per-pixel device buffers (`PatchMatchCUDA.cpp:118-122, 216`):

| Buffer | Bytes/px |
|---|---|
| `cudaDepthNormalEstimates` (`Point4` = 4 floats) | 16 |
| `cudaDepthNormalCosts` (float) | 4 |
| `cudaSelectedViews` (uint32) | 4 |
| `cudaRandStates` (`curandStateXORWOW_t`: 6 uint + 2 int flags + float + double ≈ 48 B) | ~48 (verify with `sizeof` on your toolkit) |
| `cudaLowDepths` (float, only at levels > smallest) | 4 |
| **Subtotal** | **~76** |

Plus one `cudaArray` of `float` per image (4 B/px each, at the level's
resolution) and, in geometric mode, one depth `cudaArray` per neighbour
(`4·N` B/px). Host mirror `depthNormalEstimates` is another 16 B/px of RAM.

Worked example — ODM default `--pc-quality medium` with 20 MP (5472×3648) input:
`stages/openmvs.py:57` computes `resolution_level` so the depth map is roughly
`depthmap_resolution` wide; assume 2736×1824 ≈ 5.0 MP at the full level:

- per-pixel buffers: 5.0 M × 76 B ≈ **380 MB**
- 9 image textures: 9 × 5.0 M × 4 B ≈ **180 MB**
- geometric pass, 8 depth textures: 8 × 20 MB ≈ **160 MB**
- total ≈ **0.7 GB** per `PatchMatchCUDA` instance, transient per image;
  sub-resolution levels are ¼ and ¹⁄₁₆ of this. `curandState` is the single
  largest per-pixel item (≈63 % of the pixel buffers); a counter-based RNG
  (as upstream's Metal port uses) reduces it to 4 B/px.

Register/local pressure per thread: `costArray[8][32]` floats (1024 B) +
`viewSelectionPriors[32]`, `samplingProbs[32]`, `costVector[32]`,
`viewWeights[32]` → ≥1.5 KB of local memory per thread, so 32×16 = 512 threads
per block spill heavily; `MAX_VIEWS 32` is a compile-time cap
(`ProcessPixel`, lines 423, 448, 456, 482, 504).

---

## 8. What the ODM fork changed vs upstream

Per the compare `develop...OpenDroneMap:openMVS:355` (94 commits ahead of the
2023-08-20 merge-base), the CUDA-relevant fork history is:

| Date | Commit | Change |
|---|---|---|
| 2021-10-25 | `8e24a58c8` | "Expose --cuda-device option in DensifyPointCloud" |
| 2021-10-28 | `4b85f74e0` | "Target sm_35, add check after RunCUDA" (`cudaGetLastError`, `PatchMatchCUDA.cpp:357`) |
| 2021-10-29 | `7bd32db29` | "Add OpenMVS_MAX_CUDA_COMPATIBILITY" |
| 2021-11-11 | `6f766dcc0` | "Fix bubble sort" (`.cu:107-120`) |
| 2021-11-15/22 | `4ba546384`, `607cc1f3c` | NCC keep-threshold tweaks for the GPU path |
| 2021-12-17 | `9de2e1d86` | "Expose cuda-device option regardless of CUDA compilation" |
| 2022-05-18…06-04 | `6a35b46ed` … `df873de38` | Multi-sub-resolution PatchMatch on CPU and GPU (cdcseacave's `PatchMatch_MultiSubResolutions[GPU]` branches merged into the fork; includes "fix CUDA error: cudaErrorMemoryAllocation = 2", `02f2d1ae0`) |
| 2022-05-27 | `3bbe78e08` | mock `--cuda-device` for non-CUDA builds |
| 2022-09-19 | `56ef831b7` | "Fix ignore mask on GPU, remove disk space info" |
| 2022-12-01 | `2c8a3544b` | Remove vcpkg manifest mode |
| 2026-05-16 | `482e98a06` | "Allow disabling subdirectory on install" (`INSTALL_USE_SUBDIR`) |

Net diff of the CUDA files against the merge-base is tiny: `PatchMatchCUDA.cu`
+2/-0 and `PatchMatchCUDA.cpp` +2/-0 (blank lines only);
`DensifyPointCloud.cpp` +4/-1 (mock option, log file disabled);
top-level `CMakeLists.txt` comments out `cmake_dependent_option(OpenMVS_USE_SSE …)`;
`libs/MVS/CMakeLists.txt` skips PCH on Windows. **The kernels in ODM are
byte-for-byte the upstream kernels as of 2023-08-20.** Multi-resolution
PatchMatch itself was upstreamed (v2.1.0, 2022-11-28: "Improve depth-map
estimation in textureless areas using multi-resolution").

ODM-side build glue (`SuperBuild/cmake/External-OpenMVS.cmake`): links against
the CUDA toolkit's `stubs/libcuda.so` when no driver is present; downloads a
dummy `nvcuda.dll` on Windows so the CUDA-linked binaries start on machines
without NVIDIA drivers; `-DOpenMVS_USE_CUDA=${SB_ENABLE_CUDA}` where
`SB_ENABLE_CUDA` comes from `ODM_ENABLE_CUDA=ON` (`SuperBuild/CMakeLists.txt:62-67`,
set by the pixi `gpu` feature, `pixi.toml:77-78`). The pixi `gpu` feature is
restricted to `linux-64-cuda-12-0` and `win-64-cuda-12-0` (`pixi.toml:14`);
`osx-arm64` only exists in the CPU environment.

---

## 9. Upstream status (cdcseacave/openMVS, 2025-2026)

- **v2.4.0 (2026-01-20)** release notes: "CUDA support: New GPU architectures
  (A100, etc.), CameraCUDA refactoring", depth-confidence work, GPU picker in
  the viewer. `develop` HEAD on 2026-09-13 is `596ed7ce5`.
- `develop` tree now has: `libs/MVS/PatchMatchCUDA.cu` (823 LOC),
  `libs/MVS/ConfidenceCUDA.cu` (359), `libs/MVS/SceneRefineCUDA.cu` (1109 —
  the PTX strings became real `.cu`), `libs/MVS/CUDA/Camera.h` (150),
  `libs/MVS/CUDA/Maths.h` (304), `libs/Common/UtilCUDADevice.h`.
- **Metal backend exists upstream**: PR #1279 "Add Metal compute backend for
  PatchMatch densification (Apple Silicon)" by `leNeo`, opened 2026-06-08,
  **merged 2026-06-26** into `develop`
  (https://github.com/cdcseacave/openMVS/pull/1279). Files:
  `libs/MVS/PatchMatchMetal.metal` (529 LOC), `PatchMatchMetal.mm` (403),
  `PatchMatchMetal.h` (90), `libs/Common/UtilMetal.{h,mm,cpp}` (21/83/23);
  `SceneDensify.cpp` +102. Design (PR body + files):
  - MSL port of the same 4 kernels; `curand` → counter-based PCG
    (`.metal:99-105`); `__constant__`/params → `constant` buffers;
    `<bool GEOM>` → Metal function constant (`.metal:40`);
    `texture2d<float>` arrays of `MAX_VIEWS 32` (`.metal:42, 289-291`);
    sampler `coord::pixel, filter::linear, address::clamp_to_edge`
    (`.metal:203`) — **CUDA uses `Wrap`**, a small border-behaviour difference.
  - Host `.mm`: `MTLCreateSystemDefaultDevice`, `newLibraryWithSource`
    (runtime MSL compile, `.mm:92`), `MTLPixelFormatR32Float` textures
    (`.mm:231-234`), `MTLResourceStorageModeShared` buffers (`.mm:241-246`),
    one command buffer per scale with `dispatchThreads` + single
    `waitUntilCompleted` (`.mm:319-353`).
  - CMake: `_USE_METAL` auto-enabled on Apple when Metal+Foundation are found
    (upstream `CMakeLists.txt:283-296`); the `.metal` source is embedded into a
    generated header at configure time (`libs/MVS/CMakeLists.txt:46-54`).
  - CLI changed to `--gpu-device <string>` ("-1 best GPU, -2/cpu/empty CPU,
    >=0 comma-separated IDs"; upstream `DensifyPointCloud.cpp:121`) with
    `SEACAVE::CUDA::isCpuRequested` (upstream `UtilCUDA.cpp:17-23`); backend
    pool selection in upstream `SceneDensify.cpp:3187-3213`. `--cuda-device`
    no longer exists upstream — ODM's `stages/openmvs.py` would need updating
    when it moves to a Metal-capable OpenMVS.
  - Validation: maintainer ran the CUDA backend on the author's TnT harness and
    initially found F1 "1-2% higher" than Metal; a geometric-consistency
    null-depth handling bug was fixed and the PR was squashed. Author tested
    on an M4 Max.
  - Known limitations listed in the PR: buffers re-allocated per scale per
    call; runtime MSL compile rather than `.metallib`; validated on a small
    scene only.
- ODM tracking: `OpenDroneMap/ODM` PR #2074 "Openmvs 240" (opened 2026-08-31,
  open) targets the 2.4.0 line, which predates the Metal merge.
- No OpenCL/HIP/Vulkan/SYCL path exists upstream. Requests: #206 (2017, asked for
  kernel sources to write `SceneRefineOpenCL.cpp`), #1076 (2023-10-25, same),
  #1085 (2023-11-28) — none received a maintainer reply. Upstream also carries a
  vcpkg `ports/siftgpu` (CUDA SiftGPU) behind `OpenMVS_USE_SIFTGPU`, unrelated
  to ODM.

---

## 10. Portability matrix (PatchMatch path)

| CUDA feature (where) | Metal | HIP/ROCm | Vulkan compute / WGSL |
|---|---|---|---|
| 2-D float texture with bilinear filter, wrap addressing (`PatchMatchCUDA.cpp:137-145`; `tex2D<float>` in `.cu`) | `texture2d<float>` + `sampler(filter::linear, address::repeat)`; upstream chose `clamp_to_edge` | `hipTextureObject_t` / `tex2D` — supported on AMD (gfx9+) | `VkImage R32_SFLOAT` + sampler `REPEAT`, `textureSampleLevel` (WGSL); note WGSL `r32float` is **not filterable** without the `float32-filterable` feature — may need `rg16float` or manual bilinear |
| `cudaMallocArray` + `cudaMemcpy2DToArray` uploads | `MTLTexture replaceRegion:` | `hipMallocArray` / `hipMemcpy2DToArray` | staging buffer + `vkCmdCopyBufferToImage` |
| Per-pixel `curandStateXORWOW` (`.cu:564`, 48 B/px) | no curand; upstream uses PCG hash from `(x,y,iter)` in a `uint` buffer | `hiprand` (rocRAND) has XORWOW-compatible state | implement PCG/xorshift in shader; 4 B/px |
| Deterministic seed `curand_init(1234, y, x)` | reproduce with counter RNG seeded by pixel index | same API → bit-identical if hiprand XORWOW matches (not guaranteed) | same as Metal |
| Kernel launch `<<<grid,block>>>` 32×16, checkerboard remap of `p.y` | `dispatchThreads:threadsPerThreadgroup:` (non-uniform grids allowed on Apple GPUs) | identical `hipLaunchKernelGGL` | `vkCmdDispatch` with `local_size 32×16`; needs explicit bounds check |
| `cudaDeviceSynchronize` between Black/Red passes | encode all passes in one command buffer; barriers implicit between compute encoders (or `memoryBarrier`) | `hipDeviceSynchronize` | pipeline barriers `SHADER_WRITE→SHADER_READ` between dispatches |
| Eigen in device code (`Matrix3::inverse`, `hnormalized`) with `--expt-relaxed-constexpr` | rewrite as `float3x3` (MSL has built-in `transpose`; no `inverse` in MSL — hand-code) | hipcc compiles Eigen device code (HIP has `__host__ __device__` parity); Eigen ≥3.4 has HIP support | hand-written GLSL/WGSL matrices |
| Large per-thread local arrays (`costArray[8][32]`) | private memory; Apple GPUs spill to device memory — reduce `MAX_VIEWS` or tile | same as CUDA (scratch memory); AMD wave64 halves occupancy | same; consider `MAX_VIEWS 16` |
| Bubble sort / branchy per-pixel loops (`.cu:107-120, 425-445`) | fine | fine | fine |
| `exp`, `sqrt`, `sin/cos` single precision | `metal::precise::` or fast-math; results not bit-identical | same | `exp`, `sqrt` — precision differs |
| Structs passed by value as kernel params (`Params`, `Camera*`) | `constant Params&` buffer (upstream did this) | identical | uniform buffer / push constants (Params ≤ 128 B fits push constants) |
| Driver API device pick by GFLOPS (`UtilCUDA.cpp:110-160`) | `MTLCreateSystemDefaultDevice` (single GPU on Apple Silicon) | `hipGetDeviceProperties` | `vkEnumeratePhysicalDevices` |
| CC ≥ 3 gate | n/a (Metal 3 / macOS 13+) | gfx arch list at build (`--offload-arch`) | SPIR-V 1.3+ |
| Compile-time `nNumViews 8 / iters 4` defaults under `_USE_CUDA` | should be made a `_USE_GPU` condition | — | — |

Non-PatchMatch (not needed for ODM): `SceneRefineCUDA.cpp` relies on driver-API
PTX JIT, texture *references* and surface references — no Metal/Vulkan
equivalent; would need a rewrite. Upstream already rewrote it as `.cu`.

---

## 11. Open questions

1. Effective fatbin contents when both `CUDA_ARCHITECTURES "50;72;75"` (target
   property) and `-gencode` flags from `MAX_CUDA_COMPATIBILITY` are present —
   verify with `cuobjdump --list-elf` on a built `DensifyPointCloud`.
2. Whether the `sem(1)` semaphore truly serialises the two worker threads'
   `pmCUDA->EstimateDepthMap` calls, or whether both threads can enter
   `RunCUDA` concurrently on the default stream (would be safe but slow).
3. `sizeof(curandStateXORWOW_t)` on CUDA 12 (assumed 48 B).
4. Whether ODM will adopt upstream's `--gpu-device` string API (PR #2074 is
   still on 2.4.0, which has neither Metal nor the rename).

## Sources

- Clone: `https://github.com/OpenDroneMap/openMVS` tag `355` → `8e4c5f31a16e0e329132c8f26e20a8c5d7b47f53`
- `libs/MVS/PatchMatchCUDA.{cu,cpp,inl,h}`, `libs/MVS/SceneDensify.cpp`, `libs/MVS/DepthMap.{cpp,h}`, `libs/Common/UtilCUDA.{cpp,h}`, `apps/DensifyPointCloud/DensifyPointCloud.cpp`, `CMakeLists.txt`, `libs/MVS/CMakeLists.txt` (line refs above)
- ODM: `/Users/avneh/Code/ADM/ODM/stages/openmvs.py`, `/Users/avneh/Code/ADM/ODM/opendm/gpu.py`, `/Users/avneh/Code/ADM/ODM/SuperBuild/cmake/External-OpenMVS.cmake`, `/Users/avneh/Code/ADM/ODM/SuperBuild/CMakeLists.txt:62-67`, `/Users/avneh/Code/ADM/ODM/pixi.toml`
- GitHub API: `repos/cdcseacave/openMVS/compare/develop...OpenDroneMap:openMVS:355`; `repos/cdcseacave/openMVS/pulls/1279` (+ `/files`, issue comments); `repos/cdcseacave/openMVS/releases`; `repos/cdcseacave/openMVS/git/trees/develop?recursive=1`; upstream `develop` contents of `CMakeLists.txt`, `libs/MVS/CMakeLists.txt`, `libs/MVS/PatchMatchMetal.*`, `libs/MVS/SceneDensify.cpp`, `apps/DensifyPointCloud/DensifyPointCloud.cpp`, `libs/Common/UtilCUDA.cpp`
- https://github.com/cdcseacave/openMVS/pull/1279
- https://github.com/cdcseacave/openMVS/issues/206, /issues/1076, /issues/1085
- https://github.com/OpenDroneMap/ODM/pull/2074
- ACMH: Xu & Tao, "Multi-Scale Geometric Consistency Guided Multi-View Stereo" / "Asymmetric Checkerboard Propagation and Multi-Hypothesis Joint View Selection", CVPR 2018 (cited in `.cu:387-388`)
