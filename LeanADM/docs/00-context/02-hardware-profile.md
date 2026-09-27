# Target hardware profile

Captured 2026-09-13 with `system_profiler SPDisplaysDataType`, `sysctl`, `sw_vers`.

| Item | Value |
|---|---|
| Machine | MacBook Pro (Apple silicon), `arm64` |
| SoC | Apple **M4 Pro** |
| CPU | 12 cores: **8 Performance + 4 Efficiency** (verified: `hw.perflevel0.logicalcpu`=8 Performance, `hw.perflevel1.logicalcpu`=4 Efficiency) |
| GPU | Apple M4 Pro, **16 cores**, built-in, vendor 0x106b |
| Metal | **Metal 4** (`Metal Support: Metal 4`) |
| Unified memory | 24 GB (`hw.memsize` = 25769803776) |
| Display | Liquid Retina XDR 3024x1964 |
| OS | macOS **26.6** (build 25G5065a) |
| Toolchain in repo | conda-forge clang 19.1.7, libcxx 22.1.8, llvm-openmp 22.1.8, cmake 3.31.8 (see `04-toolchain-osx-arm64.md`) |
| NVIDIA / CUDA | none (so `opendm/gpu.py::has_gpu()` returns False) |
| Apple dev tools | Command Line Tools only (`/Library/Developer/CommandLineTools`); **`xcrun metal` is NOT available** (needs Xcode + Metal Toolchain component) |
| OpenCL.framework | present at `/System/Library/Frameworks/OpenCL.framework` (deprecated API, still ships) |
| pixi | **not installed** (`which pixi` fails) |

## Implications

* **Unified memory**: CPU and GPU share the 24 GB. A Metal compute path gets zero-copy access
  to buffers allocated with shared storage mode, which removes the host<->device copy cost
  that dominates small kernels on discrete GPUs. But GPU work also competes with the CPU
  stages (and with OpenMVS's multi-GB depth-map working set) for the same 24 GB.
* **24 GB is tight for ODM at high settings.** OpenMVS densification at `--pc-quality high`
  on 20 MP images can exceed this; `stages/openmvs.py` already catches OOM exit codes
  (137/143) and retries with `--fusion-mode 1` sub-scene tiling.
* **No OpenCL/CUDA vendor stack**: OpenCL is deprecated on macOS but the framework still
  exists; whether OpenCV's OpenCL T-API works on Apple silicon under macOS 26 must be
  verified (see `gpu-acceleration/metal/04-*.md`).
* **Laptop thermals**: sustained all-core CPU load will throttle; measure with
  `sudo powermetrics --samplers cpu_power,gpu_power`.

## Second target (assumed, not present here)

The "integrated Radeon" laptop is not this machine. Docs in `gpu-acceleration/radeon/`
assume a Ryzen APU (Radeon 780M `gfx1103`, 890M `gfx1150`, or Vega `gfx90c`) on Linux or
Windows 11. If the actual machine differs, re-check the ROCm support matrix first.
