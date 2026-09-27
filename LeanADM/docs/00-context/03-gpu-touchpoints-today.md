# Where ODM 3.6.2 touches a GPU today (all CUDA-only)

Exact locations in `/Users/avneh/Code/ADM/ODM` at commit `1a71341`.

## 1. Detection: `opendm/gpu.py`

* `has_gpu(args)` (line 73): returns False if env `ODM_NO_GPU` is set (line 9) or
  `--no-gpu` (line 77). Otherwise on Windows it checks for
  `%SYSTEMROOT%\system32\nvcuda.dll`; on Linux/macOS it checks `shutil.which('nvidia-smi')`.
  **On this Mac it always returns False.**
* `get_cuda_compute_version()` (line 38): ctypes-loads `libcuda.so`/`nvcuda.dll`, calls
  `cuInit`, `cuDeviceGetCount`, `cuDeviceComputeCapability`. Requires CC >= 3.5.
* `has_popsift_and_can_handle_texsize(w, h)` (line 12): imports `opensfm.pypopsift` and
  asks `pypopsift.fits_texture(int(w*1.02), int(h*1.02))`.

Any Metal/HIP work needs a new detection function here (or a sibling module) that does not
assume NVIDIA. The CLI flag `--no-gpu` (`opendm/config.py` line 388) is generic and reusable.

## 2. Feature extraction: `opendm/osfm.py` lines 298-317

```python
if feature_type == "SIFT":
    if has_gpu(args) and max_dims is not None:
        ... compute (w, h) after feature_process_size scaling ...
        if has_popsift_and_can_handle_texsize(w, h):
            feature_type = "SIFT_GPU"
```

Only when the user asks for `--feature-type sift` (default is `dspsift`, see
`config.py` line 233) *and* CUDA is present does OpenSfM use `SIFT_GPU` (PopSift).
Everything else (dspsift, hahog, akaze, orb, matching, bundle adjustment) is CPU.
`pypopsift` is not even built without `ODM_ENABLE_CUDA` (`SuperBuild/cmake/External-PyPopsift.cmake`).

## 3. Dense reconstruction: `stages/openmvs.py` lines 82-86, 111-114

```python
use_gpu = has_gpu(args)
gpu_config.append("--cuda-device -1" if use_gpu else "--cuda-device -2")
...
if e.errorCode == 1 and use_gpu:   # GPU run failed -> retry on CPU
    gpu_config = ["--cuda-device -2"]
```

`DensifyPointCloud` from OpenMVS is the single most expensive stage in a normal run and
the only place a GPU makes a large difference. The binary is compiled with
`-DOpenMVS_USE_CUDA=${SB_ENABLE_CUDA}` (`External-OpenMVS.cmake`), i.e. OFF on macOS; the
`--cuda-device` flag is then inert.

## 4. AI models: `opendm/skyremoval/skyfilter.py` line 15, `opendm/bgfilter.py` line 16

```python
provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
```

Used by `--sky-removal` and `--bg-removal`. The conda-forge `onnxruntime 1.22.2 ... _cpu`
build is installed on osx-arm64, so only `CPUExecutionProvider` exists. A CoreML EP would be
the natural Apple path (see `gpu-acceleration/metal/03-onnxruntime-coreml-mps.md`).

## 5. Build plumbing

* `pixi.toml`: `[feature.gpu.activation] env = { ODM_ENABLE_CUDA = "ON" }`; gpu feature
  platforms limited to `linux-64-cuda-12-0` and `win-64-cuda-12-0`.
* `SuperBuild/CMakeLists.txt`: `SB_ENABLE_CUDA` from `$ENV{ODM_ENABLE_CUDA}`.
* `External-OpenMVS.cmake`: libcuda stub search on UNIX; downloads `nvcuda_dummy.dll` on
  Windows so the binary launches without a driver; `OpenMVS_MAX_CUDA_COMPATIBILITY=ON`.
* `gpu.Dockerfile`: `nvidia/cuda:12.9.1-runtime-ubuntu24.04` runtime image.

## Everything else is CPU

OpenSfM matching/BA (Ceres), PoissonRecon (OpenMP), dem2mesh, texrecon (mvs-texturing,
OpenMP), PDAL, GDAL, odm_orthophoto, Obj2Tiles, Entwine, OpenPointClass, report generation.
Any "make it fast on a laptop" plan therefore has two halves: (a) GPU for densify + features
+ ONNX, and (b) CPU tuning for the rest (see `gpu-acceleration/cpu-baseline/`).
