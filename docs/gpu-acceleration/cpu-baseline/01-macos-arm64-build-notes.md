# Building ODM 3.6.2 natively on osx-arm64 (Apple M4 Pro): what the SuperBuild does today

State of this checkout on 2026-09-13: `pixi` is not installed (`which pixi` fails), there is no
`ODM/.pixi/` environment and no `ODM/SuperBuild/{build,src,download,install}`. Only Apple
Command Line Tools are present (`/Library/Developer/CommandLineTools`, Apple clang 17.0.0,
SDK 26.0; `xcodebuild`/`xctrace` refuse to run without full Xcode). Free disk on the data
volume was ~11 GB of 460 GB (`df -h`), which is **not enough** for a build (section 6).

## 1. The exact commands

From `README.md:149-171` and `pixi.toml [tasks]`:

```bash
brew install pixi            # or: curl -fsSL https://pixi.sh/install.sh | sh  (SSL failures were reported; Homebrew worked in PR #2037 testing)
cd /Users/avneh/Code/ADM/ODM
pixi install --locked        # solves nothing (uses pixi.lock), downloads ~conda packages into .pixi/envs/default
pixi run build               # = cmake -S SuperBuild -B SuperBuild/build && cmake --build SuperBuild/build
pixi run test                # = python scripts/test.py  (unittest discovery in tests/, pattern test_*.py)
pixi run smoke               # = python scripts/smoke.py (pdal+untwine on PATH, DensifyPointCloud/ReconstructMesh --help, run.py --help, imports opensfm/gdal/cv2/pdal)
pixi run odm -- --project-path ~/datasets aukerman [flags]   # = python run.py ...
pixi run clean               # rm -rf SuperBuild/build SuperBuild/src SuperBuild/download SuperBuild/install
```

Environments (`pixi.toml [environments]`): `default` = features `build` + `cpu` (compilers,
cmake 3.31, ninja, make, pybind11, tbb-devel + runtime); `prod` = `cpu` only (runtime, used by
the Docker image); `gpu`/`gpu-prod` exist only for linux-64 and win-64 with CUDA 12. On macOS
only `default` and `prod` resolve.

Activation env on macOS (`pixi.toml [target.osx.activation.env]`):

```
DYLD_LIBRARY_PATH          = $PIXI_PROJECT_ROOT/SuperBuild/install/lib:$DYLD_LIBRARY_PATH
DYLD_FALLBACK_LIBRARY_PATH = $CONDA_PREFIX/lib:$HOME/lib:/usr/local/lib:/lib:/usr/lib
CMAKE_GENERATOR            = Ninja          (feature.build)
```

The comment in `pixi.toml` says the fallback path lets PyPI wheels find system libs while
conda deps resolve from the env.

## 2. What is prebuilt (conda-forge) vs compiled here

`SuperBuild/CMakeLists.txt` requires a conda env (`CONDA_PREFIX`, else FATAL_ERROR), adds it
to `CMAKE_PREFIX_PATH`, and `odm_require_conda_package()`s GDAL, OpenCV, GFlags, Ceres, PDAL
(`Conda-Packages.cmake`). Everything else is an `ExternalProject`:

| Target | Source | Build | osx-arm64 specifics |
|---|---|---|---|
| `opensfm` | OpenDroneMap/OpenSfM `c5328439` (+ submodules) | CMake, `--parallel nproc`, into `SuperBuild/install/bin/opensfm` (built in place; `INSTALL_COMMAND ""`) | patch `opensfm-aarch64-abs.patch`: changes `#ifdef __aarch64__` to `#if defined(__aarch64__) && !defined(__APPLE__)` in `triangulation.h` so Apple libc++ uses `abs(det)` not `std::abs<T>` |
| `openmvs` | OpenDroneMap/openMVS tag `355` + `vcg` `285` + `eigen34` | CMake | `-DOpenMVS_USE_SSE=OFF` when `CMAKE_SYSTEM_PROCESSOR` is `aarch64`/`arm64`; `-DOpenMVS_USE_CUDA=OFF`; on APPLE runs `patch-cgal-iterator.cmake` which edits `$CONDA_PREFIX/include/CGAL/boost/graph/iterator.h` **in the conda env** (removes three `explicit operator bool()` blocks that clang 19 rejects; CGAL #8313) |
| `mvstexturing` (+`mve` 356) | OpenDroneMap/mvs-texturing `c5a4d0c` | CMake | `mvstexturing.patch` removes `-march=native` and skips `-mfpmath=sse -funroll-loops` on arm64; bumps bundled Eigen to 3.4.0 |
| `poissonrecon` | OpenDroneMap/PoissonRecon `290` | **hand-written Makefile**: `bash -lc "make -j${nproc} -f Makefile.macos poissonrecon COMPILER=${CMAKE_CXX_COMPILER} CONDA_PREFIX=$CONDA_PREFIX"`; binary copied from `Bin/Linux/PoissonRecon` | patches: `poissonrecon-macos-makefile.patch` (`-fopenmp=libomp`, `-lomp -L$(CONDA_PREFIX)/lib`, `-I$(CONDA_PREFIX)/include`, `-Wno-missing-template-arg-list-after-template-kw`), `poissonrecon-png-mac.patch` and `poissonrecon-zlib-byte.patch` (bundled PNG/zlib headers mis-detect `TARGET_OS_MAC`), `poissonrecon-sparsematrix-rows.patch` (`A.rows` -> `A.rows()`) |
| `dem2mesh` | OpenDroneMap/dem2mesh `334` | CMake | `dem2mesh-vertex-ring-null.patch` (NULL edge in jmeshlib); its own CMake sets `-Xclang -fopenmp` + `libomp` on APPLE/Clang |
| `dem2points`, `odm_orthophoto` (355), `fastrasterfilter`, `lastools` (250), `renderdem`, `fpcfilter` (331), `entwine`, `openpointclass` | CMake | | `openpointclass`: `-DPORTABLE_BUILD=ON` because it hardcodes `-march=native` otherwise; `draco-no-start-group.patch`: no `--start-group` on APPLE ld |
| `obj2tiles` | prebuilt zip `Obj2Tiles-Osx64.zip` v1.4.0 | download only | `OBJ2TILES_ARCH` = `Osx64` on APPLE (a .NET single-file binary; whether it is arm64 or x64-under-Rosetta is UNVERIFIED) |
| `pypopsift` | skipped | | `message(STATUS "Skipping pypopsift: ODM_ENABLE_CUDA not set ...")` |
| `nvcuda_dummy.dll` | Windows only | | not downloaded on macOS |

RPATH: "On Linux the conda-forge toolchain bakes an RPATH ... macOS clang does not, so
libraries linked with an @rpath install name (e.g. conda boost) fail to load at runtime"; on
APPLE the SuperBuild adds `-DCMAKE_INSTALL_RPATH=$CONDA_PREFIX/lib
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON` to every CMake sub-project (`CMakeLists.txt:65-76`).
This was added after `dyld: Library not loaded: @rpath/libboost_iostreams.dylib` from
`DensifyPointCloud` during PR #2037 testing (https://github.com/OpenDroneMap/ODM/pull/2037).

Parallelism of the build: `ProcessorCount(nproc)` unless `CMAKE_BUILD_PARALLEL_LEVEL` is set;
ninja nests inside ninja (SuperBuild -> each ExternalProject), and CI caps it at 3 because
"each level runs cores+2 jobs and ~30 compilers exhaust the 16 GB runner"
(`.github/workflows/native-builds.yaml`). On 24 GB with 12 cores, start with
`CMAKE_BUILD_PARALLEL_LEVEL=4` (OpenMVS/CGAL and OpenSfM translation units are large).

## 3. Toolchain and math libraries pinned by `pixi.lock` (osx-arm64)

| Package | Version / build | Relevance |
|---|---|---|
| `clang_osx-arm64`, `clangxx_osx-arm64` | 19.1.7 (`clang-19.1.7-default_hf9bcbb7_9`, `compiler-rt 19.1.7`) | conda-forge LLVM clang, not Apple clang. LLVM 19 is the first release whose `AArch64Processors.td` defines `apple-m4` (https://raw.githubusercontent.com/llvm/llvm-project/release/19.x/llvm/lib/Target/AArch64/AArch64Processors.td). |
| `libcxx` 22.1.8, `libcxx-devel` 19.1.7 | | headers from 19, runtime 22 |
| `llvm-openmp` 22.1.8 | | provides `libomp.dylib`; OpenMP available for `-fopenmp` in every sub-project |
| `cmake` 3.31.8, `ninja` 1.13.2, `make` | | |
| `python` 3.12.9, `numpy` 2.3.2, `scipy` 1.16.1, `py-opencv` 4.12.0 (`qt6_py312h1f2130c_604`) | | numpy/scipy/opencv depend on generic `libblas`/`libcblas`/`liblapack` |
| `libblas`/`libcblas`/`liblapack`/`liblapacke` 3.11.0 `8_*_openblas` | | **OpenBLAS**; `libopenblas 0.3.33 openmp_*` (depends on `llvm-openmp >= 19.1.7`) |
| `ceres-solver` 2.2.0 `cpugplhc6e4162_210` | | "cpu" + "gpl" variant: depends on `liblapack`, `suitesparse 7.10.1` (libcholmod 5.3.1, libspqr, libamd, libcolamd, libklu, libumfpack ...), `eigen 3.4.0`, `glog 0.7.1`. The feedstock builds with `-DBLA_VENDOR=Generic -DACCELERATESPARSE=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_UNITY_BUILD=ON` and `-DSUITESPARSE=ON` for the gpl variant (https://raw.githubusercontent.com/conda-forge/ceres-solver-feedstock/main/recipe/build.sh). |
| `suitesparse` 7.10.1, `libcholmod` | | `libcholmod` depends on `libblas`, `liblapack`, `llvm-openmp` -> CHOLMOD is BLAS-swappable |
| `libopencv` 4.12.0 | | depends on `libcblas`, `liblapack`, `liblapacke`; OpenCL/T-API status unverified |
| `gdal` 3.11.1, `pdal` 2.8.4, `python-pdal` 3.4.5, `untwine`, `cgal` 5.6.1 / `cgal-cpp` 5.6.1, `libboost` 1.86.0, `eigen` 3.4.0, `tbb` 2022.3.0 | | |
| `onnxruntime` 1.22.2 (CPU build) | | sky/bg removal on CPU EP only |

## 4. Compiler flags actually in effect

* conda-forge clang activation exports `CFLAGS="-ftree-vectorize -fPIC -fstack-protector-strong
  -O2 -pipe"` and `CXXFLAGS="... -O2 -pipe -stdlib=libc++ -fvisibility-inlines-hidden
  -fmessage-length=0"`, plus `-isystem $CONDA_PREFIX/include`, `-mmacosx-version-min=14.0`
  (from `macos = "14.0"` in `pixi.toml`) and `LDFLAGS="-Wl,-headerpad_max_install_names
  -Wl,-dead_strip_dylibs -Wl,-rpath,$PREFIX/lib -L$PREFIX/lib"`. **There is no `-march` or
  `-mcpu` on arm64** (x86_64 gets `-march=core2 -mtune=haswell -mssse3`)
  (https://raw.githubusercontent.com/conda-forge/clang-compiler-activation-feedstock/main/recipe/build.sh).
* Nothing in the ODM SuperBuild adds `-O3`, `-march` or `-mcpu`: `grep -rn -e -O3 -e march
  -e mcpu SuperBuild opendm stages pixi.toml` only hits the OpenPointClass comment and the
  patches that *remove* `-march=native`. CMake's own `CMAKE_CXX_FLAGS_RELEASE` (`-O3 -DNDEBUG`)
  is appended after the env `-O2` for the CMake sub-projects (`CMAKE_BUILD_TYPE` defaults to
  Release, `CMakeLists.txt:9-12`), so C++ code compiles at `-O3` there; PoissonRecon's
  Makefile uses `-O3 -DRELEASE -ffast-math -g0`.
* OpenSfM passes `-DCMAKE_CXX_FLAGS="$CXXFLAGS -DGLOG_USE_GLOG_EXPORT -DGLOG_USE_GFLAGS"`
  (`External-OpenSfM.cmake:29-33`) and its own CMake adds `-Wall -Wextra` and
  `-DVL_DISABLE_AVX` (VLFeat); no arm-specific code.
* Consequence: with no `-mcpu`, LLVM's default CPU for the `arm64-apple-macos` triple is
  `apple-m1` (verified locally with Apple clang 17: `clang -### -c t.c` prints
  `-target-cpu apple-m1`; conda clang 19 should behave the same, UNVERIFIED until installed).
  That already enables NEON, FP16, DotProd, etc. (armv8.5-a class features) but not the M4
  scheduling model or SME. See `02-apple-silicon-cpu-tuning.md`.

## 5. Known and suspected macOS problems

1. **PoissonRecon + `libomp.dylib` (suspected still broken on master).** During PR #2037
   testing on Apple silicon: `dyld: Library not loaded: @rpath/libomp.dylib` from
   `PoissonRecon` at `odm_meshing`; every retry with fewer threads aborted (exit 134)
   (https://github.com/OpenDroneMap/ODM/pull/2037). Root cause as read from the code:
   `Makefile.macos` links `-lomp -L$(CONDA_PREFIX)/lib` without `-Wl,-rpath` and does not
   use the env `LDFLAGS` (which carry conda's `-Wl,-rpath,$PREFIX/lib`), so the binary
   references `@rpath/libomp.dylib` with no `LC_RPATH`. ODM launches every tool through
   `subprocess.Popen(cmd, shell=True)` = `/bin/sh -c` (`opendm/system.py:103`); `/bin/sh` is
   SIP-protected and **strips all `DYLD_*` variables** (verified locally on macOS 26.6:
   `DYLD_LIBRARY_PATH=/tmp/x /bin/sh -c 'echo $DYLD_LIBRARY_PATH'` prints nothing; SIP
   enabled). `system.run` works around this only for `DYLD_LIBRARY_PATH` by prefixing the
   command string with `export DYLD_LIBRARY_PATH="..." &&` (`system.py:90-92`); the
   `DYLD_FALLBACK_LIBRARY_PATH` set by pixi activation is lost. Since `DYLD_LIBRARY_PATH`
   only contains `SuperBuild/install/lib` (no `libomp.dylib` there), PoissonRecon cannot
   resolve libomp. Verification after a build: `otool -L SuperBuild/install/bin/PoissonRecon`
   and `otool -l ... | grep -A2 LC_RPATH`. Fix candidates (not applied): add
   `-Wl,-rpath,$(CONDA_PREFIX)/lib` to `LFLAGS` in `poissonrecon-macos-makefile.patch`, or
   also re-export `DYLD_FALLBACK_LIBRARY_PATH` in `system.run`, or copy `libomp.dylib` into
   `SuperBuild/install/lib`. The CMake-built binaries (OpenMVS, texrecon, dem2mesh, ...) are
   fine because of the `CMAKE_INSTALL_RPATH` above. Also check `Obj2Tiles` and the OpenSfM
   pybind `.so` (loaded by Python, not via `/bin/sh`, so `DYLD_*` survive there).
2. **CGAL header patched inside the conda env** (`patch-cgal-iterator.cmake`). `pixi install
   --locked` re-materializes packages from the cache; if pixi re-links `cgal-cpp` the patch is
   lost until the next `pixi run build` reruns the PATCH step (it does, as `PATCH_COMMAND` reruns
   when its literal command line changes or on a fresh checkout; on an unchanged build tree it
   may not). PR #2074 (open, 2026-08-31) moves to CGAL 6 which removes the need.
3. **Patch re-application on incremental builds**: `apply-patch.cmake` skips when
   `git apply --reverse --check` succeeds; the community pixi thread reported a
   patch-reapplication bug on incremental builds that this script addresses
   (https://community.opendronemap.org/t/moving-the-odm-build-system-to-pixi/26795).
4. **`OpenMVS_USE_SSE=OFF`**: harmless on arm64 (no SSE), but it means the OpenMVS fork's
   x86-only intrinsics paths are disabled; NEON auto-vectorization is what you get.
5. **`free -t -m` absent on macOS**: OpenSfM's `memory_available()` returns None, so its
   memory-aware worker throttling is off (see `00-pipeline-stage-map.md` section 3). Not a
   build issue, but it makes `--max-concurrency` the only memory guard.
6. **Mac runtime was never fully validated upstream**: PR #2037's checklist "[ ] Mac runtime"
   and "[ ] Arm runtime" were unchecked at merge; CI only runs `pixi run test` (unit tests)
   and `pixi run smoke` on `macos-latest`, never a dataset. The last community reports of
   native macOS runs predate the pixi move (2022-2024) and are unresolved
   (https://community.opendronemap.org/t/odm-on-macos/20064,
   https://community.opendronemap.org/t/native-install-macos/14117). Older issues:
   #1763 (2024, Homebrew `tbb@2020` gone, maintainers recommend Docker), #1314/#1240 (2021).
7. `Obj2Tiles-Osx64.zip` architecture (x64 vs arm64) is unverified; if x64 it needs Rosetta 2
   (`softwareupdate --install-rosetta`), which is only reached with `--3d-tiles`.
8. No GPU env on macOS: `pixi install -e gpu` will fail to solve (platform not listed). Any
   Metal work must add its own feature/flags (see `docs/gpu-acceleration/metal/`).
9. `xctrace`/Instruments need full Xcode; only CLT is installed. `sample` and `powermetrics`
   work without Xcode (`04-benchmark-and-profiling-plan.md`).

## 6. Expected build time and disk usage

* CI (`macos-latest`, GitHub-hosted Apple silicon runner, `CMAKE_BUILD_PARALLEL_LEVEL=3`):
  "Build SuperBuild" step took **34 min** (run 34507541585, 2026-09-10), **41 min** (run
  34494496280) and **49 min** (run 34345241946, 2026-09-09); ubuntu-latest 17-19 min,
  ubuntu-24.04-arm 14-15 min, windows-2022 21-24 min. `pixi install --locked` took ~1-2 min
  on each (measured from the job step timestamps via `gh run view --json jobs`).
* Upstream author's Linux dev box: cold build "from 40 minutes to 7m" with pixi
  (https://github.com/OpenDroneMap/ODM/pull/2037). Expect **15-30 min** on the M4 Pro with
  `CMAKE_BUILD_PARALLEL_LEVEL=4-6` (ESTIMATE: CI runners are 3-4 core M1/M2-class VMs at
  parallel level 3; this laptop has 8 P-cores). The first `pixi install` also downloads the
  conda packages (a few GB; ESTIMATE 3-5 GB extracted for `default`).
* Disk: `.pixi/envs/default` (ESTIMATE 4-6 GB incl. Qt6 OpenCV, GDAL, PDAL, CUDA-free ceres),
  `.pixi/envs/prod` if installed (~3 GB), `SuperBuild/src` + `SuperBuild/build` (OpenSfM with
  submodules, OpenMVS, VCG, Eigen, entwine, draco, LAStools; ESTIMATE 3-6 GB of sources and
  objects), `SuperBuild/install` (~0.5-1 GB). Budget **15-20 GB** for the toolchain and build,
  plus datasets and outputs (a 100-image 18 MP run produces several GB of undistorted TIFs,
  depth maps and point clouds; `--optimize-disk-space` trims that). The data volume currently
  has ~11 GB free, so free space first.
* Build memory: OpenMVS and OpenSfM C++ translation units use 1-2 GB each at `-O3`;
  `CMAKE_BUILD_PARALLEL_LEVEL=4` keeps peak under ~10 GB (ESTIMATE, based on the CI comment
  that ~30 compilers exhaust 16 GB).

## 7. Suggested first-run checklist (documented, not executed)

```bash
brew install pixi
cd /Users/avneh/Code/ADM/ODM
pixi install --locked
CMAKE_BUILD_PARALLEL_LEVEL=4 pixi run build 2>&1 | tee ../build.log
otool -L SuperBuild/install/bin/PoissonRecon | grep -i omp          # expect @rpath/libomp.dylib
otool -l SuperBuild/install/bin/PoissonRecon | grep -A2 LC_RPATH    # expect nothing -> issue 5.1
pixi run smoke
pixi run test
pixi run odm -- --project-path ~/datasets aukerman --end-with odm_meshing   # exercises PoissonRecon early
```

If `odm_meshing` aborts with the libomp error, the smallest local workaround without editing
ODM is `cp .pixi/envs/default/lib/libomp.dylib SuperBuild/install/lib/` (DYLD_LIBRARY_PATH
covers that dir) - a change to the repo is out of scope for this research.

## Sources

* `pixi.toml`, `pixi.lock`, `README.md`, `SuperBuild/CMakeLists.txt`, `SuperBuild/cmake/*.cmake`,
  `SuperBuild/cmake/*.patch`, `scripts/smoke.py`, `scripts/test.py`,
  `.github/workflows/native-builds.yaml`, `opendm/system.py` (commit 1a71341)
* PoissonRecon `Makefile.macos` (tag 290): https://github.com/OpenDroneMap/PoissonRecon/blob/290/Makefile.macos
* dem2mesh CMake Apple/OpenMP block: https://github.com/OpenDroneMap/dem2mesh/blob/334/CMakeLists.txt
* OpenSfM src CMake: https://github.com/OpenDroneMap/OpenSfM/blob/c5328439465e6ace011f39077d1077d7b1cdd65d/opensfm/src/CMakeLists.txt
* conda-forge clang activation flags: https://raw.githubusercontent.com/conda-forge/clang-compiler-activation-feedstock/main/recipe/build.sh
* conda-forge ceres recipe/build: https://raw.githubusercontent.com/conda-forge/ceres-solver-feedstock/main/recipe/recipe.yaml ,
  https://raw.githubusercontent.com/conda-forge/ceres-solver-feedstock/main/recipe/build.sh
* LLVM 19 AArch64 processors (apple-m4): https://raw.githubusercontent.com/llvm/llvm-project/release/19.x/llvm/lib/Target/AArch64/AArch64Processors.td
* ODM PR #2037 "Moving build system to pixi" (macOS dyld reports, build times): https://github.com/OpenDroneMap/ODM/pull/2037
* ODM PR #2074 "Openmvs 240" (CGAL 6): https://github.com/OpenDroneMap/ODM/pull/2074
* ODM PR #2053 (OpenSfM upgrade, FPCFilter untested on Mac): https://github.com/OpenDroneMap/ODM/pull/2053
* ODM issue #1763 (Homebrew native install broken, Docker recommended): https://github.com/OpenDroneMap/ODM/issues/1763
* ODM issues #1314, #1240 (native macOS / M1 history): https://github.com/OpenDroneMap/ODM/issues/1314 , https://github.com/OpenDroneMap/ODM/issues/1240
* Community: https://community.opendronemap.org/t/moving-the-odm-build-system-to-pixi/26795 ,
  https://community.opendronemap.org/t/the-future-of-the-build-system/26730 ,
  https://community.opendronemap.org/t/odm-on-macos/20064 ,
  https://community.opendronemap.org/t/native-install-macos/14117
* CI timings: `gh run view 34507541585 / 34494496280 / 34345241946 --repo OpenDroneMap/ODM --json jobs`
* SIP and DYLD_* stripping: verified locally (`csrutil status` enabled; `/bin/sh` drops
  `DYLD_LIBRARY_PATH`); background: https://developer.apple.com/library/archive/documentation/Security/Conceptual/System_Integrity_Protection_Guide/RuntimeProtections/RuntimeProtections.html
