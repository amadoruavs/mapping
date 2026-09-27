# osx-arm64 toolchain pinned by `ODM/pixi.lock`

Extracted 2026-09-13 with grep over `pixi.lock` (conda-forge, osx-arm64 subdir).

| Package | Version / build | Notes |
|---|---|---|
| clang / clangxx | 19.1.7 (`clang_osx-arm64-19.1.7`) | conda-forge clang, not Apple clang. LLVM 19 knows `-mcpu=apple-m4`? (verify; apple-m4 landed in LLVM 19/20 — see cpu-baseline docs) |
| compiler-rt | 19.1.7 | |
| libcxx | 22.1.8 | newer than the compiler |
| llvm-openmp | 22.1.8 | OpenMP runtime available for `-fopenmp` |
| cmake | 3.31.8 | Ninja generator forced via `CMAKE_GENERATOR=Ninja` in the build feature |
| python | 3.12.9 | |
| ceres-solver | 2.2.0 `cpugplhc6e4162_210` | CPU build, links conda `libblas` |
| libblas / liblapack | 3.11.0 `*_openblas` | **OpenBLAS**, not Apple Accelerate. Swappable via `libblas=*=*accelerate` (research in cpu-baseline/02) |
| libopenblas | 0.3.33 `openmp_*` | |
| suitesparse | 7.10.1 | used by Ceres sparse solvers |
| libopencv | 4.12.0 `qt6_py312*_604` | check OpenCL/T-API support in this build |
| onnxruntime | 1.22.2 `*_cpu` | no CoreML EP unless a different variant exists |
| tbb | 2022.3.0 | |
| gdal | 3.11.1 | |

## Not in the environment

* No Apple `metal` shader compiler on this host (verified 2026-09-13:
  `xcrun -sdk macosx metal --version` -> "unable to find utility metal"; only Command Line
  Tools are installed at `/Library/Developer/CommandLineTools`). Compiling `.metal` sources
  offline requires Xcode plus its Metal Toolchain component (Xcode 16/26 ships it as a
  separate download: `xcodebuild -downloadComponent MetalToolchain`). Alternative that
  needs no toolchain: compile MSL **at runtime** from source with
  `MTLDevice.newLibrary(source:options:)`, which is what a first Metal prototype should do.
  This is a **host requirement**, not a conda package.
* `pixi` itself is not installed yet (`brew install pixi` or the official installer).
* No CUDA (by design on macOS).
* No `hip`/`rocm` packages (irrelevant on macOS).

## Why this matters for the Metal plan

The SuperBuild compiles everything with conda clang 19. Objective-C++ that links
`Metal.framework`/`Foundation.framework` compiles fine with conda clang as long as the macOS
SDK is available (`SDKROOT`, normally set by the conda compiler activation scripts). The
shader compilation step (`xcrun metal`) is separate and needs the Apple toolchain. The
build recipe doc (`gpu-acceleration/metal/05-build-recipe-macos-metal.md`) covers both.
