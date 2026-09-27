# Build recipes: ODM 3.6.2 SuperBuild with an AMD Radeon iGPU (Linux and Windows 11)

Documented, **not executed**. Every path below assumes the HIP ports described in
`01-openmvs-hip-port.md` and `02-popsift-and-features-on-amd.md` exist (OpenMVS
`OpenMVS_USE_HIP`, PopSift `USE_HIP`). Without them the only AMD-usable piece is the ONNX
runtime (doc 03), which needs no SuperBuild changes.

The existing CUDA plumbing this mirrors:

| Piece | Where | What it does today |
|---|---|---|
| `[feature.gpu.activation] env = { ODM_ENABLE_CUDA = "ON" }` | `pixi.toml` | switches the SuperBuild |
| `[feature.gpu-build.target.linux-64.dependencies] cuda-toolkit = "12.*"`; win-64 `cuda-nvcc_win-64`, `cuda-libraries-dev` | `pixi.toml` | compiler + libs from conda-forge |
| `[feature.gpu.dependencies] onnxruntime = { build = "*cuda*" }` | `pixi.toml` | CUDA EP |
| `environments.gpu = ["build","gpu-build","gpu"]`, `gpu-prod = ["gpu"]` | `pixi.toml` | solve group `gpu` |
| `set(SB_ENABLE_CUDA OFF); if("$ENV{ODM_ENABLE_CUDA}" STREQUAL "ON") ...` | `SuperBuild/CMakeLists.txt` L62–67 | env → CMake var |
| `-DOpenMVS_USE_CUDA=${SB_ENABLE_CUDA}`, `-DOpenMVS_MAX_CUDA_COMPATIBILITY=ON`, `GPU_CMAKE_ARGS` (`libcuda.so` stub dir on Linux), `nvcuda_dummy.dll` download on Windows | `SuperBuild/cmake/External-OpenMVS.cmake` | |
| `if(SB_ENABLE_CUDA)` around the pypopsift ExternalProject | `SuperBuild/cmake/External-PyPopsift.cmake` | |
| `has_gpu()`: `nvidia-smi` on PATH (Linux) / `%SYSTEMROOT%\system32\nvcuda.dll` (Windows); `ODM_NO_GPU`, `--no-gpu` | `opendm/gpu.py` | runtime switch |
| `--cuda-device -1` / `-2`, exit-1 CPU fallback, OOM → `--fusion-mode 1` tiling | `stages/openmvs.py` | |
| `CONDA_OVERRIDE_CUDA=12.0` at image build, `nvidia/cuda:12.9.1-runtime` base | `gpu.Dockerfile` | |

---

## A. Linux (Ubuntu 24.04) with ROCm

### A.1 Kernel, driver, ROCm

1. Ubuntu 24.04.4 with the 6.8 GA or 6.17 HWE kernel is the validated pair for ROCm 7.x. For
   Strix Halo (`gfx1151`) use ≥ 6.16.9 so ROCm sees the whole GTT pool; RDNA3 Phoenix is fine
   on 6.8. [ROCm Linux requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html),
   [ROCm#5444](https://github.com/ROCm/ROCm/issues/5444)
2. Groups and headers:
   ```
   sudo usermod -a -G render,video $LOGNAME
   sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
   ```
3. AMD repo + packages (quick-start, version current at time of research):
   ```
   wget https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb
   sudo apt install ./amdgpu-install_7.2.4.70204-1_all.deb && sudo apt update
   sudo apt install rocm            # full stack, or: rocm-hip-sdk  (compiler + HIP libs only)
   # amdgpu-dkms is optional on an APU with a recent in-tree amdgpu; install it only if
   # rocminfo cannot see the GPU with the stock kernel.  [uncertain: AMD's guide installs it]
   sudo reboot
   ```
   [ROCm quick start](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
4. Verify: `rocminfo | grep -E 'Name:|gfx'` should show `gfx1103` / `gfx1150`; `clinfo` if
   `rocm-opencl` is installed. `/opt/rocm/bin` on `PATH`, `/opt/rocm/lib` in `ldconfig`.
5. Memory: BIOS "UMA Frame Buffer Size" → smallest/Auto (0.5–1 GB); raise GTT via
   `ttm.pages_limit` (pages = MiB×256) or legacy `amdgpu.gttsize=<MiB>` in
   `GRUB_CMDLINE_LINUX`; check `rocm-smi --showmeminfo vram gtt`.
   [Strix Halo guide](https://rocmdocs.amd.com/en/develop/how-to/system-optimization/strixhalo.html),
   [k8s.it recipe](https://www.k8s.it/posts/running-a-local-llm-on-amd-radeon-780m-gfx1103-rocm-and-the-gpu-that-wasnt-supposed-to-work/)
6. gfx1103 only: no override should be needed for code compiled with
   `--offload-arch=gfx1103`; keep `HSA_OVERRIDE_GFX_VERSION=11.0.2` and `HSA_ENABLE_SDMA=0` in
   your back pocket for libraries (MIGraphX/rocBLAS) that ship no gfx1103 kernels.
   [inferena#61](https://github.com/kvark/inferena/issues/61)

### A.2 Toolchain: conda-forge HIP packages vs system ROCm

conda-forge has a HIP stack: the `hip` feedstock outputs `hip-devel`, `hip-runtime-amd`,
`hipcc`, `rocm-opencl`, `rocm-opencl-runtime` (linux-64 only), and separate feedstocks for
`rocm-device-libs`, `rocm-cmake`, `rocm-comgr`, `rocm-smi`. Versions and which gfx targets
the device libs cover were not verified; I found no `hiprand`/`rocrand` feedstock, and the
HIP compiler is `hipcc` driving conda's clang (OpenMM documents building with
`conda install -c conda-forge hip-devel hipcc rocm-cmake rocm-device-libs`).
[hip-feedstock](https://github.com/conda-forge/hip-feedstock),
[rocm-device-libs-feedstock](https://github.com/conda-forge/rocm-device-libs-feedstock),
[OpenMM guide](https://docs.openmm.org/latest/userguide/library/02_compiling.html)

Recommendation: **build against system ROCm in `/opt/rocm`** (known-good hipRAND, matching
runtime, same libs Docker/WSL use) and treat the conda packages as a follow-up experiment.
Pixi has no `rocm` system-requirement key and conda has no `__rocm` virtual package, so the
env cannot be gated on the driver the way `cuda = "12.0"` gates `__cuda`; the `amd` feature is
just opt-in. [pixi system requirements](https://pixi.prefix.dev/latest/workspace/system_requirements/)

An alternative "ROCm in a venv" is AMD's TheRock pip SDK
(`rocm[libraries,devel]` from `https://rocm.nightlies.amd.com/v2/gfx110X-all/` or the
`gfx115X-all` family) — I could confirm `win_amd64` wheels for `rocm-sdk-core` there but did
not confirm Linux wheels for that family; verify. [TheRock](https://github.com/ROCm/TheRock)

### A.3 `pixi.toml` sketch: an `amd` feature mirroring `gpu`

```toml
[workspace]
platforms = ["linux-64", "win-64", "linux-aarch64",
    { platform = "osx-arm64", macos = "14.0" },
    { platform = "linux-64", cuda = "12.0" },
    { platform = "win-64", cuda = "12.0" }]
# no rocm platform variant exists; the amd feature reuses plain linux-64 / win-64

[feature.amd]
platforms = ["linux-64", "win-64"]

# Read by SuperBuild/CMakeLists.txt (new SB_ENABLE_HIP)
[feature.amd.activation]
env = { ODM_ENABLE_HIP = "ON", ROCM_PATH = "/opt/rocm", HIP_PATH = "/opt/rocm",
        ODM_HIP_ARCHS = "gfx1103;gfx1150;gfx1151;gfx1100;gfx1101;gfx1102" }

[feature.amd.target.win-64.activation]
env = { ODM_ENABLE_HIP = "ON", ODM_HIP_ARCHS = "gfx1150;gfx1151;gfx1100;gfx1101;gfx1102" }
# HIP_PATH is set by the HIP SDK installer on Windows

# Build-time deps if the conda-forge HIP stack is used instead of /opt/rocm (linux-64 only,
# unverified adequacy — see A.2):
# [feature.amd-build.target.linux-64.dependencies]
# hip-devel = "*"
# hipcc = "*"
# rocm-device-libs = "*"
# rocm-cmake = "*"

# Runtime ONNX providers (doc 03); conda-forge has no rocm/directml onnxruntime builds
[feature.amd.target.win-64.pypi-dependencies]
onnxruntime-directml = "==1.24.4"
[feature.amd.target.linux-64.pypi-dependencies]
onnxruntime-migraphx = { version = "==1.23.2", index = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/" }
# NOTE: the conda-forge `onnxruntime==1.22.2` in [dependencies] must be excluded from this
# solve group or the two runtimes collide (both install the `onnxruntime` Python package).

[environments]
amd      = { features = ["build", "amd"], solve-group = "amd" }
amd-prod = { features = ["amd"],          solve-group = "amd" }
```

Open question for whoever implements this: pixi's `[dependencies]` block is global, so the
`onnxruntime` conda pin needs to move into `[feature.cpu.dependencies]`/`[feature.gpu...]` so
the `amd` solve group can drop it. The pypi `index`/`find-links` syntax for per-dependency
indexes should be checked against the pixi version in use.

### A.4 SuperBuild CMake deltas (described, not applied)

`SuperBuild/CMakeLists.txt` (next to L62–67):
```cmake
set(SB_ENABLE_HIP OFF)
if("$ENV{ODM_ENABLE_HIP}" STREQUAL "ON")
    set(SB_ENABLE_HIP ON)
endif()
message(STATUS "HIP support: ${SB_ENABLE_HIP}")
```

`SuperBuild/cmake/External-OpenMVS.cmake`:
* Guard the `GPU_CMAKE_ARGS` libcuda-stub search with `if(UNIX AND SB_ENABLE_CUDA)`, and the
  `nvcuda_dummy.dll` download with `if(WIN32 AND NOT SB_ENABLE_HIP)` (the HIP build links
  `amdhip64` from the driver, no dummy needed).
* Add to `CMAKE_ARGS`:
  ```
  -DOpenMVS_USE_HIP=${SB_ENABLE_HIP}
  -DCMAKE_HIP_ARCHITECTURES=$ENV{ODM_HIP_ARCHS}
  -DCMAKE_HIP_COMPILER=$ENV{ROCM_PATH}/lib/llvm/bin/clang++      # Linux
  -DCMAKE_PREFIX_PATH=$ENV{ROCM_PATH}\;$ENV{CONDA_PREFIX}
  ```
  and drop `-DOpenMVS_MAX_CUDA_COMPATIBILITY=ON` when HIP is on (it calls `nvcc`).
* Windows: see B.2 — the OpenMVS ExternalProject needs `-DCMAKE_C_COMPILER=clang
  -DCMAKE_CXX_COMPILER=clang++` from `%HIP_PATH%\bin` instead of the `cl` the win-64
  `build` task passes, because "CMake HIP Language: Unsupported" on the HIP SDK.

`SuperBuild/cmake/External-PyPopsift.cmake`: `if(SB_ENABLE_CUDA OR SB_ENABLE_HIP)`, pass
`-DUSE_HIP=${SB_ENABLE_HIP}` and point `GIT_TAG` at a pypopsift revision whose FetchContent
pulls a PopSift with PR #186 (`AMD-Ecosystem/popsift:moat-port` until merged).

### A.5 Runtime detection: an AMD analogue of `has_gpu()`

Describe-only sketch for `opendm/gpu.py`:

```python
def has_amd_gpu():
    if sys.platform == 'win32':
        # HIP SDK ships hipInfo.exe; the runtime DLL name is versioned (amdhip64_N.dll)
        return shutil.which('hipInfo') is not None or _hip_device_count('amdhip64_7.dll') > 0
    return os.path.exists('/dev/kfd') and (shutil.which('rocminfo') is not None
                                          or _hip_device_count('libamdhip64.so') > 0)

def _hip_device_count(lib):
    # ctypes: hipInit(0); hipGetDeviceCount(&n)  (same shape as get_cuda_compute_version)
```

`has_gpu()` returns `True` if either vendor probe passes (still honouring `ODM_NO_GPU` and
`--no-gpu`), and `has_popsift_and_can_handle_texsize()` skips the `libcuda` compute-capability
check on AMD. `stages/openmvs.py` keeps `--cuda-device -1` if the HIP OpenMVS keeps the option
name; the exit-code-1 fallback to `-2` already covers "HIP init failed". The Windows DLL name
`amdhip64_7.dll` is from HIP SDK 7.x packaging conventions and should be confirmed.

### A.6 Docker

Mirror `gpu.Dockerfile` with a ROCm base and device passthrough:

```
FROM rocm/dev-ubuntu-24.04:7.2 AS runtime   # or install rocm-hip-runtime on ubuntu:24.04
...
docker run --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
  --group-add video --group-add render -v $PWD/datasets:/datasets odm:amd --project-path /datasets project
```
`/dev/kfd` is the compute interface; `/dev/dri` (or specific `renderD12x` nodes) selects GPUs.
No equivalent of `CONDA_OVERRIDE_CUDA` is needed because nothing in the solve is ROCm-gated.
[ROCm Docker docs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)
GTT allocations are not charged to the container's memory cgroup on RDNA3.5 APUs, so
`--memory` limits will not see GPU-mapped pages. [ROCm#6370](https://github.com/ROCm/legacy-rocm-build/issues/6370)

## B. Windows 11

### B.1 What is officially there

* HIP SDK for Windows 7.2.0: `gfx1150` (Ryzen AI 300) and `gfx1151` supported; `gfx1103`
  absent (community reports the runtime still runs kernels on 780M; rocBLAS is what is
  missing and OpenMVS/PopSift do not use it). Windows 11 22H2+.
  [HIP SDK requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html),
  [ROCmLibs for gfx1103](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU)
* Included: HIP runtime, `hipcc`/clang++, HIP math + primitive libs (hipRAND is a math lib),
  HIPIFY, Visual Studio plugin. Not included: MIOpen, MIGraphX, PyTorch; `rocminfo` →
  `hipInfo`; CMake HIP language unsupported.
  [Component support](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html)
* Adrenalin driver supplies `amdhip64` at runtime, so an end user needs the driver only, no SDK.

### B.2 Toolchain interaction with ODM's MSVC build

ODM's win-64 build is `cl` + Ninja (`pixi.toml` `[target.win-64.tasks] build`, with
`vs2022_win-64` and `scripts/activate-msvc.bat` clearing conda's clang link flags and forcing
`CMAKE_GENERATOR=Ninja`). The HIP SDK compiles device code with its own clang and, per
llama.cpp's recipe, the practical approach is to build the *whole* HIP-enabled project with
`clang`/`clang++` from `%HIP_PATH%\bin` under Ninja from a VS x64 Native Tools prompt:

```
set PATH=%HIP_PATH%\bin;%PATH%
cmake -S openmvs -B build -G Ninja -DOpenMVS_USE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1150;gfx1151 ^
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
```
[llama.cpp Windows HIP](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

Consequences for the SuperBuild: only the OpenMVS (and pypopsift) ExternalProjects switch to
clang; they must still consume conda-forge's MSVC-built OpenCV/Boost/CGAL, which clang on
Windows (MSVC ABI) can do. `activate-msvc.bat`'s `INCLUDE` tweak for CUDA CCCL headers is
irrelevant here. Risk is moderate; budget a week (doc 01 §6).

### B.3 ONNX on Windows

No ROCm ONNX Runtime on Windows. Use `onnxruntime-directml` (any DX12 GPU including Vega 8)
or the WebGPU EP (D3D12). Details and the `skyfilter.py`/`bgfilter.py` provider change in doc 03.

### B.4 Vulkan-based alternatives on Windows

None exist for OpenMVS. For features, VulkanSift would run on the Adrenalin Vulkan driver
(doc 02 §2.3). If the Windows HIP toolchain proves too intrusive, "DirectML for ONNX + CPU for
everything else" is the honest Windows story.

## C. WSL2 and containers on Windows

* ROCm on WSL2 now officially reaches **gfx11.0.3 (Radeon 780M)** and gfx11.5.2 iGPUs via
  `librocdxg`/ROCDXG 1.2 (May 2026), in addition to Strix/Strix Halo; needs Windows 11,
  Adrenalin ≥ 26.2.2, Ubuntu 22.04/24.04 in WSL2, and `amdgpu-install --usecase=wsl,rocm --no-dkms`.
  Limitations: `rocm-smi` unsupported, **MIGraphX unsupported**, no multi-GPU, "performance
  may be limited compared to native Linux". So the ONNX MIGraphX path does not work in WSL;
  HIP-compiled OpenMVS/PopSift should.
  [WSL how-to](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html),
  [Phoronix librocdxg 1.2](https://www.phoronix.com/news/librocdxg-1.2-More-Ryzen-ROCm)
* Docker Desktop on WSL2: the GPU shows up as `/dev/dxg`, not `/dev/kfd`; the ROCm-in-WSL
  path relies on librocdxg inside the distro. Whether that works inside a container is
  **unverified** — treat "Docker + AMD iGPU on Windows" as unsupported until tested.
* Native Linux Docker: A.6.

## D. Verification checklist (for whoever runs this)

1. `rocminfo`/`hipInfo` lists the iGPU with the expected gfx name.
2. `DensifyPointCloud --help` shows the GPU option; run the ODM test dataset with
   `--cuda-device -1` and `-2` and compare point counts/timings (doc 01 §7 method).
3. `python -c "from opensfm import pypopsift; print(pypopsift.fits_texture(4000,3000))"` in the
   env when PopSift-HIP is built.
4. `python -c "import onnxruntime as o; print(o.get_available_providers())"` shows
   `MIGraphXExecutionProvider` (Linux) or `DmlExecutionProvider` (Windows).
5. Watch `rocm-smi --showmeminfo gtt` / Task Manager shared GPU memory during densification and
   confirm ODM's own `--max-concurrency` × ~1 GB/thread still fits in what is left.

## Sources

- ODM sources: `pixi.toml`, `SuperBuild/CMakeLists.txt`, `SuperBuild/cmake/External-OpenMVS.cmake`, `SuperBuild/cmake/External-PyPopsift.cmake`, `opendm/gpu.py`, `stages/openmvs.py`, `scripts/activate-msvc.bat`, `gpu.Dockerfile`
- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html
- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html
- https://rocmdocs.amd.com/en/develop/how-to/system-optimization/strixhalo.html
- https://github.com/ROCm/ROCm/issues/5444 , https://github.com/ROCm/legacy-rocm-build/issues/6370
- https://github.com/kvark/inferena/issues/61 , https://www.k8s.it/posts/running-a-local-llm-on-amd-radeon-780m-gfx1103-rocm-and-the-gpu-that-wasnt-supposed-to-work/
- https://github.com/conda-forge/hip-feedstock , https://github.com/conda-forge/rocm-device-libs-feedstock , https://docs.openmm.org/latest/userguide/library/02_compiling.html
- https://pixi.prefix.dev/latest/workspace/system_requirements/
- https://github.com/ROCm/TheRock , https://rocm.nightlies.amd.com/v2/gfx110X-all/
- https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ (onnxruntime_migraphx-1.23.2 cp310/cp312) , https://pypi.org/project/onnxruntime-directml/
- https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html , https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html
- https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html , https://www.phoronix.com/news/librocdxg-1.2-More-Ryzen-ROCm
- https://github.com/alicevision/popsift/pull/186 , https://github.com/OpenDroneMap/pypopsift
