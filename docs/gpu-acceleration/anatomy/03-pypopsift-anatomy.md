# pypopsift / PopSift anatomy

Scope: what ODM 3.6.2 builds for GPU SIFT. Read from clones of
`https://github.com/OpenDroneMap/pypopsift` at `fe2d1ccc63877ba315e65f34d2adeadd838b3ac3`
(2025-09-04, "Merge pull request #15 from NathanMOlson/ODM") and
`https://github.com/alicevision/popsift` (default branch, HEAD `36d704d39`
2026-08-07). pypopsift pins PopSift commit `a19d394d2` (2025-08-29, "Merge pull
request #169 from NathanMOlson/cuda13"); the GitHub compare
`a19d394d...master` reports 0 commits ahead, so the pinned commit is the
`master` tip and the default-branch clone is newer. LOC and line numbers below
come from the clone; kernel structure is unlikely to differ between the two,
but treat exact line numbers for PopSift as approximate for `a19d394d`.

---

## 1. pypopsift: what it exposes

Tiny project (~350 LOC of code):

| File | LOC |
|---|---|
| `src/popsift.cpp` | 137 |
| `src/popsift.h` | 71 |
| `src/main.cpp` | 38 |
| `CMakeLists.txt` | 26 |
| `setup.py` | 79 |
| `example.py`, `tests/test.py`, `test.sh` | 57 / 3 / 4 |

Python API (`src/main.cpp:19-31`):

```python
pypopsift.popsift(image, peak_threshold=0.1, edge_threshold=10,
                  target_num_features=4000, use_root=True, downsampling=-1)
    -> [points: float32 (N,4), desc: float32 (N,128)]    # py::list of two numpy arrays
pypopsift.fits_texture(width, height, downsampling=-1) -> bool
pypopsift.__version__
```

`image` is `py::array_t<unsigned char, c_style|forcecast>` (`popsift.h:19`),
interpreted as `height = shape(0)`, `width = shape(1)` (`popsift.cpp:78-79`) —
i.e. a single-channel uint8 image (OpenSfM passes the grayscale resized image,
`features.py:422`). A multi-channel array would be silently mis-sized.

Implementation (`src/popsift.cpp`):

- A process-global `PopSiftContext* ctx` guarded by `std::mutex g_mutex`
  (lines 9-10, 83-88): the GIL is released (76) but all callers serialise on
  the mutex, so OpenSfM's multi-process feature extraction is what gives
  parallelism, not threads; each worker process creates its own CUDA context.
- `PopSiftContext()` (12-28): `cudaGetDevice`, on failure `cudaDeviceReset`
  and retry, else throws `runtime_error("Cannot use CUDA device: ...")`;
  `popsift::cuda::device_prop_t deviceInfo; deviceInfo.set(0, false)`; new
  `popsift::Config`.
- `setup()` (37-62): on any parameter change → `config->setThreshold(peak)`,
  `setEdgeLimit(edge)`, `setNormMode(use_root ? RootSift : Classic)`,
  `setFilterSorting(LargestScaleFirst)`, `setMode(popsift::Config::OpenCV)`,
  `setDownsampling(downsampling)`, then re-creates
  `PopSift(*config, ExtractingMode, PopSift::ByteImages)` (commented-out
  `setOctaves(4)` / `setLevels(3)`, lines 51-52 — defaults apply).
- `popsift()` (68-128): loop — enqueue the image, `job->get()`, count features;
  if `numFeatures >= target_num_features || peak_threshold < 0.0001` build
  outputs, else `peak_threshold = peak_threshold*2/3` and retry (119-122) —
  the same rule OpenSfM applies to CPU SIFT. Output packing (97-111):

  ```cpp
  for (oriIdx = 0; oriIdx < pFeat.num_ori; oriIdx++) {       // multiple orientations...
      desc[128*i+k] = pDesc->features[k];                     // ...overwrite the same slot i
      points[4*i+0] = min(round(xpos), width-1);              // rounded + clamped -> integer pixel coords
      points[4*i+1] = min(round(ypos), height-1);
      points[4*i+2] = sigma;  points[4*i+3] = orientation[oriIdx];
  }
  ```

  Two consequences worth knowing: (a) sub-pixel keypoint positions are
  discarded (rounded to integers), unlike OpenCV SIFT; (b) for keypoints with
  several orientations only the **last** orientation/descriptor survives
  (`N` = keypoint count, not descriptor count). Neither is documented in ODM.
- `fitsTexture(w, h, downsampling)` (130-136): `ctx->setup(0.06, 10, true,
  downsampling)` then `PopSift::testTextureFit(w,h) == AllocTest::Ok`.

Build (`CMakeLists.txt`): CMake ≥ 3.14, C++11, `FetchContent` of PopSift at
`a19d394d` with `PopSift_BUILD_EXAMPLES OFF`, `find_package(pybind11)`
(system pybind11 since PR #15; the old submodule was removed),
`pybind11_add_module(pypopsift src/main.cpp src/popsift.cpp)`,
`target_link_libraries(pypopsift PUBLIC popsift)`, optional `OUTPUT_DIR`.
No CUDA arch list of its own — PopSift's CMake decides (below). `setup.py`
is a thin CMake driver (`-j4`).

ODM side (`/Users/avneh/Code/ADM/ODM/SuperBuild/cmake/External-PyPopsift.cmake`):
built only `if(SB_ENABLE_CUDA)`, `DEPENDS opensfm`,
`-DOUTPUT_DIR=${SB_INSTALL_DIR}/bin/opensfm/opensfm` so the module lands next
to OpenSfM's own extension modules. On non-CUDA builds it prints "Skipping
pypopsift". Runtime gate: `opendm/gpu.py:12-35`.

pypopsift commit history of note: `838b7aecc` 2021-12-17 "Add fitsTexture
method"; `7d6b6039d` 2022-01-12 "Fail gracefully on CUDA failure";
`96910582f` 2022-07-24 "Fix segmentation fault"; PR #15 (2025-08/09,
NathanMOlson) switched from the `uav4geo/popsift develop` fork to upstream
alicevision at `a19d394d` (which fixed CUDA 13 compilation) and to system
pybind11. Open issue #7 (2024-08-09): segfault calling `popsift()`.

---

## 2. PopSift: build and CUDA assumptions

`popsift/CMakeLists.txt`: `cmake_minimum_required(VERSION 3.24)` (line 2),
`project(PopSift VERSION 0.10.1 LANGUAGES CXX CUDA)` (19),
`CMAKE_CUDA_ARCHITECTURES` defaults to `"all-major"` — or `"53;62;72;87"`
when `/etc/nv_tegra_release` exists (7-17) — `CMAKE_CUDA_SEPARABLE_COMPILATION ON`
(107), shared libs by default (36), options `PopSift_USE_GRID_FILTER`,
`PopSift_ERRCHK_AFTER_KERNEL`, `PopSift_USE_TEST_CMD`. `all-major` with a CUDA
12 toolkit means sm_50…sm_90 fatbins.

Source size: `src/` = 12,051 LOC total; `src/popsift/` ≈ 11,200 (`.cu` + `.h`),
`src/application/` 825 (CLI demos). Largest files: `s_extrema.cu` 636,
`common/plane_2d.h` 540, `s_pyramid_build.cu` 535, `popsift.cu` 497,
`sift_octave.cu` 452, `s_orientation.cu` 435, `sift_pyramid.cu` 433,
`sift_conf.h` 409, `gauss_filter.cu` 391, `s_filtergrid.cu` 338,
`features.cu` 338, `common/device_prop.cu` 321, `sift_conf.cu` 321.

CUDA API surface (grep counts over `src/popsift`):

| Feature | Files / uses | Where it matters |
|---|---|---|
| `cudaMalloc3DArray` with `cudaArrayLayered | cudaArraySurfaceLoadStore` | 1 / 6 | `sift_octave.cu:220-227, 299-306` — every octave's Gaussian stack (`_data`), intermediate (`_intm`) and DoG (`_dog_3d`) are **layered 2-D arrays** (`depth = levels+3`) |
| `cudaCreateSurfaceObject` | 1 / 3 | `sift_octave.cu:247, 326` — kernels **write** pyramid layers via `surf2DLayeredwrite` (5 files / 17 uses) |
| `cudaCreateTextureObject` | 2 / 7 | `sift_octave.cu:260, 271, 339, 344` (point and linear samplers per array), `s_image.cu:150` (input image: pitched uint8, `normalizedCoords=1`, `cudaAddressModeClamp`, `cudaReadModeNormalizedFloat` — hardware uint8→float, `cudaFilterModeLinear`, lines 128-146) |
| `tex2D` / `tex2DLayered` reads | 3 / 14, 1 / 1 | blur, DoG, extrema, gradients |
| `__shfl*` | 1 / 16 | warp reductions in extrema/orientation |
| `__ballot` + `__popc` | 1 / 2, 2 / 3 | `s_extrema.cu:21-40` warp-level stream compaction of extrema (write index = `atomicAdd(counter, popc(mask))` + `popc(mask & lanemask_lt)`) — **assumes 32-lane warps** |
| `__shared__`, `__syncthreads` | 5 / 10, 11 / 28 | orientation histograms, descriptor tiles |
| `atomicAdd` / `atomicMin` | 3 / 5, 1 / 1 | extrema counters, histogram bins (`s_orientation.cu:152`) |
| `__constant__` | 4 / 6 | `d_consts` (sigma0, levels, threshold, edge limit, max extrema), Gauss filter tables (`gauss_filter.cu`, `GAUSS_ALIGN 32`, `GAUSS_LEVELS 12`) |
| `thrust::` | 1 / 102 | `s_filtergrid.cu` — sort/partition to keep at most `_filter_max_extrema` per grid cell (`filter_grid_size 2`); PopSift can be built without it (`PopSift_USE_GRID_FILTER`) |
| `cudaStream*` / `cudaEvent*` | 13 / 94, 3 / 30 | per-octave streams; two host threads pipeline upload and extract (`popsift.cu:43-47`: `uploadImages` and `extractDownloadLoop` via `SyncQueue`) |
| `cudaMallocHost` (pinned) | 3 / 5 | host feature buffers |
| `cudaMemcpy2DAsync`, `cudaMallocPitch` | 1 / 2, 1 / 1 | input image upload |
| Not used | cub, cooperative groups, curand, half precision, dynamic parallelism | |

Device requirement: whatever `all-major` emits; ODM additionally gates on
CC ≥ 3.5 (`gpu.py:17`). Layered surfaces need CC ≥ 2.0; `__shfl` ≥ 3.0.

---

## 3. Algorithm structure

Pipeline (`sift_pyramid.cu:220-232`): `step1 = build_pyramid(conf, img)`;
`step2 = find_extrema → orientation → descriptors`; then
`get_descriptors` downloads (`275-304`, `cudaStreamSynchronize(_download_stream)`).

1. **Input & upscaling.** `Config` defaults (`sift_conf.cu:19-37`):
   `_upscale_factor 1.0` (= `downsampling −1` since `setDownsampling(v)` sets
   `_upscale_factor = −v`, `sift_conf.cu:250`), `octaves −1` (auto),
   `levels 3`, `sigma 1.6`, `_edge_limit 10`, `_threshold 0.04`,
   `_max_extrema 100000`, `_filter_grid_size 2`, `_assume_initial_blur true`
   / `_initial_blur 0.5`, `_desc_mode Loop`, `_sift_mode PopSift`. pypopsift
   overrides mode to `OpenCV` and passes ODM's `downsampling` default −1, so the
   first octave is **2× the input size** (`popsift.cu:109-117`:
   `scaleFactor = 1/2^(−upscale)`) — a 2736×1824 feature image becomes a
   5472×3648 octave-0 with `levels+3 = 6` float layers ≈ 480 MB for the
   Gaussian stack alone (plus `_intm` and DoG), which is the real reason large
   images can fail on small GPUs even when `fits_texture` says OK.
2. **Gaussian pyramid** (`s_pyramid_build.cu` + `_aa/_ai/_ra` variants;
   kernels `horiz`, `vert`, `horiz_all`, `vert_all_abs0`, `vert_abs0`,
   `horiz_level`, `get_by_2_pick_every_second`, `make_dog`, `octave_fixed`):
   separable blurs sampled from textures, writing layers through surface
   stores; `GaussMode` variants (`VLFeat_Compute`, `VLFeat_Relative`,
   `VLFeat_Relative_All`, `OpenCV_Compute`, `Fixed9/15`) control how the
   incremental sigmas are computed (`sift_conf.h:38-48`,
   `gauss_filter.cu`). Next octave = pick every second pixel of the level
   with `sigma·2`.
3. **DoG** (`make_dog`): difference of adjacent layers into `_dog_3d`.
4. **Extrema** (`s_extrema.cu`, kernel `find_extrema_in_dog`, template on
   `sift_mode`): 3×3×3 neighbourhood test through the DoG texture (299-380),
   3×3 Hessian and `solve(A,b)` (`s_solve.h`, line 424), up to
   `MAX_ITERATIONS` sub-pixel/sub-scale refinement (`refine`, 134-258 per
   mode), edge response `tr²/det < edge_limit` (472), contrast threshold
   (484); survivors compacted with ballot/popc/atomicAdd (21-40). Grid filter
   (`s_filtergrid.cu`, thrust) caps extrema per tile when
   `setFilterMaxExtrema` is used (pypopsift does not).
5. **Orientation** (`s_orientation.cu`, kernels `ori_par`, `ori_prefix_sum`):
   36-bin histogram (`ORI_NBINS 36`), Gaussian window `ORI_WINFACTOR 1.5·σ`
   (97), `atomicAdd` into shared `hist[]` (152), two smoothing passes
   (161-165), peaks above 0.8·max become up to `ORIENTATION_MAX_COUNT` (3 or 4,
   `sift_constants.h:58-60`) orientations; a prefix sum assigns descriptor
   slots (`max_orientations = max_extrema + max_extrema/4`,
   `sift_constants.cu:31`).
6. **Descriptors** (`s_desc_{loop,iloop,grid,igrid,notile,vlfeat}.cu`, one
   `ext_desc_*` kernel each; `DescMode`, `sift_conf.h:74-89`): 4×4 spatial ×
   8 orientation bins = 128 (`DESC_BINS 8`, `DESC_MAGNIFY 3.0`), gradients via
   `s_gradiant.h` texture reads, then normalisation `s_desc_norm_l2.h`
   (classic, clip 0.2 & renorm) or `s_desc_norm_rs.h` (RootSIFT: L1 then
   sqrt) selected by `NormMode` (`sift_conf.h:95-98`) — the same mapping as
   OpenSfM's `root_feature`.
7. **Download** (`features.cu`, `prep_features` kernel `sift_pyramid.cu:244`):
   `Feature{xpos, ypos, sigma, num_ori, orientation[..], desc[..]}` (positions
   already divided by the upscale factor) → `FeaturesHost`. `features.cu` also
   contains an on-GPU brute-force matcher (`compute_distance`,
   `show_distance`) used only by the demo apps.

---

## 4. The texture-size limit and why it exists

`PopSift::testTextureFit` (`popsift.cu:168-196`):

1. `checkLimit_2DtexLinear(width, height)` — the **input** image is a pitched
   linear texture; compared against `cudaDeviceProp::maxTexture2DLinear`
   (`device_prop.cu:120-146`, which actually reads `maxTexture2DLayered[0..1]`
   — a quirk, but the values are the same order of magnitude).
2. `private_apply_scale_factor(width, height)` (2× by default), `depth =
   levels + 3 = 6`, then `checkLimit_2DsurfLayered(w, h, depth)` against
   `maxSurface2DLayered[0..2]` (`device_prop.cu:260-`). Failure returns
   `ImageExceedsLayeredSurfaceLimit`, with an error string that suggests a
   larger downsampling (198-237).

On every CC ≥ 3.0 device `maxSurface2DLayered` is 32768×32768×2048 and
`maxTexture2DLinear` ≥ 65000 wide (per the CUDA Programming Guide compute
capability table; not verified on the owner's hardware), so with ODM's
`feature_process_size ≤ 4480` (→ 8960 px after upscale) the check essentially
always passes on modern GPUs. What *does* fail is the subsequent
`cudaMalloc3DArray` when VRAM is exhausted (this surfaced in
ODM #1365 for the pre-popsift backend and as the "GPU memory warning" in
community thread 14070); pypopsift converts that into an exception that ODM
catches at `osfm.py:424-433` and falls back to CPU SIFT. ODM's `1.02` margin
(`gpu.py:27`) is arbitrary head-room, not derived from PopSift.

---

## 5. Feasibility of a Metal port (mapping, no implementation)

Metal on Apple Silicon has every primitive PopSift relies on; the port is
mechanical but not small (≈11 kLOC, of which the kernel code is perhaps 4-5 k):

| PopSift construct | Metal equivalent | Notes |
|---|---|---|
| Layered `cudaArray` + surface writes / texture reads (`sift_octave.cu`) | `MTLTextureType2DArray`, `MTLPixelFormatR32Float`, `MTLTextureUsageShaderRead | ShaderWrite`; MSL `texture2d_array<float, access::read_write>` `.write()` / `.sample()` with a `sampler(filter::linear)` | Apple GPUs allow read_write on R32Float textures; array length ≥ 6 trivially; max texture size 16384² on Apple Silicon (M1+) — a 5472×3648 upscaled octave fits, but 16 MP+ feature images at `ultra` quality would exceed 16384 after 2× upscale → must cap `feature_process_size` or force `downsampling 0` |
| Input uint8 texture with hardware normalisation (`s_image.cu:132`) | `MTLPixelFormatR8Unorm` sampled as float | identical semantics |
| `__constant__ d_consts`, Gauss tables | `constant` address-space buffers or function constants | |
| `__shared__` + `__syncthreads` | `threadgroup` memory + `threadgroup_barrier(mem_flags::mem_threadgroup)` | |
| `__shfl_*`, `__ballot`, `__popc` (32-lane) | `simd_shuffle*`, `simd_ballot` (returns `simd_vote`), `popcount`; Apple SIMD width is 32 | `simd_ballot` requires Metal 2.2+/Apple family 6 (A13/M1+) — fine for M4 Pro |
| `atomicAdd` on global counters and shared histograms | `atomic_uint` in `device` / `threadgroup` with `atomic_fetch_add_explicit` | float atomics for histogram bins: Metal has no `atomic<float>` add on Apple GPUs (only integer + Apple-family-specific `atomic_float` for `min/max`? — **uncertain**); use fixed-point integer bins or per-thread partial histograms |
| `thrust::sort/partition` in `s_filtergrid.cu` | none built-in; write bitonic/radix sort or run the grid filter on CPU (N ≤ 100k) | pypopsift never enables the grid filter, so this can be left out initially |
| Streams / events / two host threads | `MTLCommandQueue`, `MTLSharedEvent`; same producer/consumer threads | unified memory removes the D2H copy: use `MTLResourceStorageModeShared` |
| `cudaMallocHost` pinned buffers | shared-mode `MTLBuffer` | |
| `all-major` fatbin | single `.metallib` (or runtime `newLibraryWithSource`) | |
| `device_prop_t` texture-limit probes (`fits_texture`) | `MTLDevice.maxTextureSize` (via feature-set tables) / `supportsFamily:` | keep the `fits_texture` name for drop-in use |

Numerical parity: Metal `fast::` math vs CUDA's `__fdividef`/`rintf` choices
(`s_orientation.cu:141`) will not be bit-identical; descriptor differences
should be sub-1 % if the same Gaussian tables are reused.

Alternatives that already exist:

- **VulkanSift** (`maelaubert/VulkanSift`, MIT, C, Vulkan compute; README
  states it was tested on NVIDIA RTX/GTX, Qualcomm Adreno 615 and **Apple M2**
  via MoltenVK) — a possible non-CUDA GPU SIFT for both macOS and AMD without
  writing kernels, at the cost of a new pybind wrapper and parity testing.
- **OpenCL**: ODM's original 2021 GPU SIFT was `silx.image.sift`
  (pyopencl), removed in Dec 2021 when pypopsift landed
  (`OpenSfM` commit `c90b0344a`). No OpenCL, HIP or Metal fork of PopSift
  itself was found (searches for "popsift OpenCL/HIP/Metal/Vulkan port"
  returned only VulkanSift and unrelated projects).
- **HIP/ROCm**: `hipify-perl` would translate most of PopSift; the risk points
  are (a) layered surfaces (`hipArrayLayered` + `hipCreateSurfaceObject` exist
  but AMD surface-write support has been patchy — verify on the target ROCm),
  (b) wave64 on GCN/CDNA vs the hard-coded 32-lane ballot/popc logic in
  `s_extrema.cu:21-40` (RDNA runs wave32 by default and is the safer target),
  (c) rocThrust for the grid filter, (d) no ROCm support for most integrated
  Radeon APUs — check the owner's exact GPU against the ROCm support matrix
  before planning a HIP route.
- **CPU**: OpenCV SIFT / DSPSIFT already work on Apple Silicon; the README's
  "about 2x faster than the CPU on a typical consumer laptop" (ODM
  `README.md:97`) bounds the payoff of a Metal SIFT port.

---

## 6. Open questions

1. `sizeof`/exact texture limits on the M4 Pro (`MTLDevice` reports 16384 for
   2-D textures on Apple family GPUs — confirm whether 2-D *array* textures
   share the limit).
2. Whether Metal float atomics are acceptable for the orientation histogram or
   an integer-bin rewrite is needed.
3. Exact PopSift line numbers at the pinned `a19d394d` (clone was the newer
   default branch; the compare API shows master == pinned, so drift is only
   in unmerged develop commits).
4. Whether the "last orientation wins" packing in `pypopsift/src/popsift.cpp:100-111`
   is intentional; a port could instead emit one row per orientation (OpenCV
   behaviour), which changes `N`.

## Sources

- `https://github.com/OpenDroneMap/pypopsift` @ `fe2d1ccc63877ba315e65f34d2adeadd838b3ac3`: `src/main.cpp`, `src/popsift.cpp`, `src/popsift.h`, `CMakeLists.txt`, `setup.py`, `README.md`; PR #15 files (`repos/OpenDroneMap/pypopsift/pulls/15/files`); commit list; issue #7
- `https://github.com/alicevision/popsift` (clone HEAD `36d704d39`; pinned `a19d394d2`): `CMakeLists.txt`, `src/popsift/{popsift.cu,popsift.h,sift_conf.{h,cu},sift_constants.{h,cu},sift_pyramid.cu,sift_octave.cu,s_image.cu,s_pyramid_build*.cu,s_extrema.cu,s_orientation.cu,s_desc_*.cu,s_filtergrid.cu,features.cu,common/device_prop.{h,cu}}`; `repos/alicevision/popsift/compare/a19d394d...master`
- ODM: `/Users/avneh/Code/ADM/ODM/SuperBuild/cmake/External-PyPopsift.cmake`; `/Users/avneh/Code/ADM/ODM/opendm/gpu.py`; `/Users/avneh/Code/ADM/ODM/opendm/osfm.py:299-317, 418-439`; `/Users/avneh/Code/ADM/ODM/README.md:95-115`
- OpenSfM fork commit `c90b0344a` (2021-12-16 "GPU code cleanup, pypopsift in, silx out"); ODM issue #1365 (https://github.com/OpenDroneMap/ODM/issues/1365)
- Griwodz, Calvet, Halvorsen, "Popsift: A Faithful SIFT Implementation for Real-time Applications", MMSys 2018, doi:10.1145/3204949.3208136
- https://github.com/maelaubert/VulkanSift
- https://community.opendronemap.org/t/why-webodm-not-use-my-gpu/14070
