# Build recipe: ODM 3.6.2 on the M4 Pro with pixi, and how an `osx-metal` feature would be added

Research date: 2026-09-13. **Documented, not executed** — no build has been run and nothing
under `ODM/` was modified. Facts marked **[verified locally]** were checked on the target
machine (macOS 26.6, Command Line Tools at `/Library/Developer/CommandLineTools`, no Xcode.app,
`pixi` not installed).

---

## 1. Prerequisites on the machine

1. **Command Line Tools** are present (`xcode-select -p` → `/Library/Developer/CommandLineTools`,
   Apple clang 17.0.0, SDK `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`)
   **[verified locally]**. They provide the macOS SDK, `clang`, `ld`, `git`, `make`. They do
   **not** provide `xcrun metal`/`metallib` (`xcrun --find metal` fails) and `xcodebuild`
   refuses to run ("requires Xcode") **[verified locally]**.
2. **Xcode.app + Metal Toolchain** are *optional*: needed only if you want to precompile
   `.metal` → `.metallib` at build time, or use the Metal debugger/profiler. Install Xcode 26
   from the App Store/developer site, then `xcodebuild -downloadComponent MetalToolchain`
   (~700 MB; per-user visibility quirks: [FB20389216](https://openradar.appspot.com/FB20389216),
   [polpiella.dev](https://www.polpiella.dev/metal-toolchain-ci-cd)). Point `xcode-select` at
   Xcode if you do (`sudo xcode-select -s /Applications/Xcode.app`) — this also changes which
   SDK conda clang sees.
3. **pixi**: `curl -fsSL https://pixi.sh/install.sh | sh` (installs to `~/.pixi/bin`, adds it
   to the shell rc) or `brew install pixi`; verify with `pixi --version`
   ([pixi installation](https://pixi.prefix.dev/latest/installation/)). **[verified locally]**
   `which pixi` finds nothing today.
4. Disk/RAM: the SuperBuild clones ~10 repos and builds OpenSfM, OpenMVS, etc.; budget ~10 GB
   under `SuperBuild/` plus the `.pixi/envs/default` env (~4 GB). 24 GB RAM is enough for
   `cmake --build` with `-j12` **[uncertain: OpenMVS `SceneDensify.cpp` and OpenSfM Ceres
   templates are memory-hungry TUs; if OOM, set `CMAKE_BUILD_PARALLEL_LEVEL=6`]**.
5. Do **not** install Rosetta-dependent x86_64 tooling into the path. pixi will fall back to
   `osx-64` packages if an `osx-arm64` package is missing and run them under Rosetta 2
   ([pixi #1750](https://github.com/prefix-dev/pixi/issues/1750)); ODM's `pixi.toml` pins the
   platform as `{ platform = "osx-arm64", macos = "14.0" }` so the whole env is native. A mixed
   env would break Metal: an x86_64 process under Rosetta can create a Metal device but is
   "opted into several Metal consistency features" and cannot be linked against arm64 conda
   libraries ([WWDC20 10631](https://developer.apple.com/videos/play/wwdc2020/10631/),
   [Rosetta 2 security guide](https://support.apple.com/guide/security/rosetta-2-on-a-mac-with-apple-silicon-secebb113be1/web)).
   Check with `file SuperBuild/install/bin/DensifyPointCloud` → must say `arm64`.

---

## 2. What the repo already does for osx-arm64 (read from the sources)

`pixi.toml`:

* `platforms` includes `{ platform = "osx-arm64", macos = "14.0" }`; `feature.cpu` lists
  `osx-arm64-macos-14-0`; `feature.gpu` (CUDA) is linux-64/win-64 only.
* `[feature.build.dependencies]`: `c-compiler`, `cxx-compiler` (→ conda-forge
  `clangxx_osx-arm64 19.1.7` in the lock), `cmake 3.31.*`, `ninja`, `make` (PoissonRecon),
  `git`, `pybind11`, `pkg-config`, `tbb-devel 2022.3.*`; `CMAKE_GENERATOR=Ninja`.
* `[target.osx.activation.env]`: `DYLD_LIBRARY_PATH=$PIXI_PROJECT_ROOT/SuperBuild/install/lib:…`
  and `DYLD_FALLBACK_LIBRARY_PATH=$CONDA_PREFIX/lib:$HOME/lib:/usr/local/lib:/lib:/usr/lib`
  — the fallback lets PyPI wheels resolve system libs while conda deps resolve in the env.
  Note: `DYLD_*` variables are stripped from *protected* (Apple-signed/hardened) processes by
  SIP; they survive for the unsigned/ad-hoc-signed binaries ODM builds and for
  `$CONDA_PREFIX/bin/python` **[uncertain: for the `pixi run` subprocess chain specifically]**.
* Tasks: `build = "cmake -S SuperBuild -B SuperBuild/build && cmake --build SuperBuild/build"`,
  `clean`, `test`, `smoke`, `odm`.

`SuperBuild/CMakeLists.txt` and `cmake/*`:

* Refuses to configure without `CONDA_PREFIX`; sets `CONDA_CMAKE_ARGS`
  (`CMAKE_PREFIX_PATH`, `Python3_ROOT_DIR`).
* `ODM_ENABLE_CUDA=ON` → `SB_ENABLE_CUDA` — the pattern to copy for Metal.
* **RPATH**: on APPLE adds `-DCMAKE_INSTALL_RPATH=$CONDA_PREFIX/lib` and
  `CMAKE_INSTALL_RPATH_USE_LINK_PATH=ON` because "macOS clang does not [bake an RPATH]" and
  conda dylibs have `@rpath` install names.
* **PoissonRecon**: `make -f Makefile.macos … COMPILER=${CMAKE_CXX_COMPILER} CONDA_PREFIX=…`
  with patches `poissonrecon-macos-makefile.patch` (`-fopenmp=libomp -lomp -L$CONDA_PREFIX/lib`,
  `-Wno-missing-template-arg-list-after-template-kw` for clang 19),
  `poissonrecon-png-mac.patch`, `-zlib-byte`, `-sparsematrix-rows`.
* **OpenMVS**: `patch-cgal-iterator.cmake` edits `$CONDA_PREFIX/include/CGAL/boost/graph/iterator.h`
  in place (CGAL 5.6.1 safe-bool operators rejected by clang 19; CGAL #8313);
  `-DOpenMVS_USE_SSE=OFF` on arm64; `-DOpenMVS_USE_CUDA=${SB_ENABLE_CUDA}` (upstream forces
  OFF on APPLE anyway).
* **OpenSfM**: `opensfm-aarch64-abs.patch`; `-DGLOG_USE_GLOG_EXPORT -DGLOG_USE_GFLAGS`
  in `CMAKE_CXX_FLAGS` (glog 0.6+); Ceres from `$CONDA_PREFIX`.
* **mve/mvs-texturing/dem2mesh/odm_orthophoto/…**: plain CMake ExternalProjects with
  `CONDA_CMAKE_ARGS`; `FPCFilter` and `lastools` intentionally without.
* **draco**: `draco-no-start-group.patch` (`--start-group` is a GNU ld flag).

`opendm/gpu.py` / `stages/openmvs.py` / filters: CUDA-only detection (nvidia-smi, nvcuda.dll,
`libcuda.so` ctypes, `CUDAExecutionProvider`).

---

## 3. Step-by-step: baseline CPU build (as the repo stands)

```
cd /Users/avneh/Code/ADM/ODM
pixi install                       # solves the 'default' env (build + cpu features) for osx-arm64
pixi run build                     # cmake -S SuperBuild -B SuperBuild/build && cmake --build …
pixi run smoke                     # python scripts/smoke.py  (imports + CLI)
pixi run odm -- --project-path /path/to/projects <dataset>
```

What to expect / check (document only):

1. `pixi install` also installs the PyPI deps (rasterio 1.4.3, fiona, codem, rawpy, …) as wheels;
   these are the consumers of `DYLD_FALLBACK_LIBRARY_PATH`.
2. The first configure clones every ExternalProject (needs network); `SuperBuild/download`,
   `src`, `build`, `install` are created. Use `CMAKE_BUILD_PARALLEL_LEVEL` to cap parallelism.
3. Verify OpenMP was found for PoissonRecon (`Makefile.macos` link line shows `-lomp`),
   mvs-texturing and OpenMVS (`Found OpenMP`), see `04-*.md` §8. If OpenMVS reports
   `OpenMP_ROOT` at `/opt/homebrew/opt/libomp`, that is upstream's hard-coded default; the
   conda `llvm-openmp` should still be found through `CMAKE_PREFIX_PATH`
   **[uncertain until a build log exists]**.
4. `otool -L SuperBuild/install/bin/DensifyPointCloud` should show `@rpath/libboost_*.dylib`
   resolving via the baked `$CONDA_PREFIX/lib` RPATH.
5. Codesigning: on Apple silicon every executable must carry at least an ad-hoc signature; the
   Apple/LLVM linker (`ld64`/`ld-prime`) ad-hoc-signs automatically, and conda's clang uses
   the same system linker, so no `codesign` step is needed for local use
   ([Eclectic Light](https://eclecticlight.co/2020/08/22/apple-silicon-macs-will-require-signed-code/),
   [smileykeith](https://www.smileykeith.com/2021/10/05/codesign-m1/)). Two pitfalls: (a) if a
   post-link tool rewrites the binary (`install_name_tool`, `patchelf`-style RPATH edits) the
   signature is invalidated and the process is killed with SIGKILL — re-sign with
   `codesign -s - -f <binary>`; (b) notarization/Gatekeeper only concerns *downloaded*
   apps, so locally built binaries are unaffected. Hardened runtime is not required and would
   actually block `DYLD_*` injection.

---

## 4. Designing an `osx-metal` pixi feature/environment (document only)

Goal: a build of ODM where OpenMVS (pin rebased onto upstream `develop` ≥ `4694ded` with the
Metal backend, stacked on ODM PR #2074's `openmvs-240` branch — `01-*.md` §5 (ii-a)) compiles
its Metal path, onnxruntime has the CoreML EP, BLAS is Accelerate, and Python knows the
machine has a Metal GPU. Two host constraints drive the design **[verified locally]**: no
Xcode/Metal Toolchain (`xcrun -sdk macosx metal` absent) → shaders must be compiled at
runtime (upstream already does this) unless Xcode is installed; and `pixi` must be installed
first (§1). Sketch of the `pixi.toml` additions:

```toml
[feature.metal]
platforms = ["osx-arm64-macos-14-0"]          # reuse the existing custom platform name

[feature.metal.activation]
env = { ODM_ENABLE_METAL = "ON" }              # read by SuperBuild/CMakeLists.txt

[feature.metal.dependencies]
onnxruntime = ">=1.26"                          # conda-forge osx-arm64 builds with CoreML EP (PR #182, 2026-05)
libblas = { version = "*", build = "*accelerate*" }   # Apple Accelerate BLAS/LAPACK (verify build string)
llvm-openmp = "*"                               # explicit OpenMP runtime for all OpenMP consumers
pyobjc-framework-metal = "*"                    # optional: Metal probe from Python (conda-forge package exists)

[environments]
metal      = { features = ["build", "cpu", "metal"], solve-group = "metal" }
metal-prod = { features = ["cpu", "metal"],          solve-group = "metal" }
```

Notes on the syntax: features may restrict `platforms` to names declared in
`[workspace].platforms` (including the synthesised `osx-arm64-macos-14-0`); `[feature.X.activation]
env` and `[environments]` composition are documented in
[pixi multi-platform configuration](https://pixi.prefix.dev/latest/workspace/multi_platform_configuration/).
Keep `metal` in its own `solve-group` because the onnxruntime/BLAS overrides differ from
`default`. The onnxruntime pin bump must be checked against the Linux `gpu` feature's
`==1.22.2 *cuda*` pin — a per-target override (`[feature.metal.target.osx-arm64.dependencies]`)
avoids touching the CUDA environments.

**CMake plumbing** (mirror of the CUDA path):

```cmake
# SuperBuild/CMakeLists.txt
set(SB_ENABLE_METAL OFF)
if(APPLE AND "$ENV{ODM_ENABLE_METAL}" STREQUAL "ON")
    set(SB_ENABLE_METAL ON)
endif()
message(STATUS "Metal support: ${SB_ENABLE_METAL}")
```

```cmake
# SuperBuild/cmake/External-OpenMVS.cmake (additions)
if(APPLE)
  list(APPEND OPENMVS_APPLE_ARGS -DOpenMP_ROOT=$ENV{CONDA_PREFIX})   # override upstream's /opt/homebrew default
  if(SB_ENABLE_METAL)
    list(APPEND OPENMVS_APPLE_ARGS -DCMAKE_OBJCXX_COMPILER=${CMAKE_CXX_COMPILER}
                                   -DCMAKE_OSX_SYSROOT=$ENV{SDKROOT})   # SDK for Metal.framework headers
  else()
    # upstream enables Metal automatically when Metal.framework is found; to force it off
    # a small patch adding an OpenMVS_USE_METAL option would be needed (document only)
  endif()
endif()
```

Upstream openMVS' CMake auto-detects Metal (`FIND_LIBRARY(METAL_FRAMEWORK Metal)` →
`ENABLE_LANGUAGE(OBJCXX)` → `_USE_METAL`), so with the fork rebased the only real requirements
are that (a) CMake's OBJCXX language works with the conda clang and (b) the SDK is visible.
conda-forge's clang activation exports `CONDA_BUILD_SYSROOT`/`SDKROOT` when an SDK is found and
passes `-isysroot`; with only CLT installed the SDK is
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk` **[verified locally that the path
exists; not verified that pixi's activation picks it up]**. If `FIND_LIBRARY(Metal)` fails,
set `CMAKE_OSX_SYSROOT` explicitly as above or `export SDKROOT=$(xcrun --show-sdk-path)`
before `pixi run build`.

**Shader compilation strategy.** Two supported modes should be documented:

1. *Runtime compile (default, no Xcode)*: upstream generates `PatchMatchMetal_msl.h` from the
   `.metal` file at configure time and calls `newLibraryWithSource:` at start-up. Cost ~0.1 s
   per process; DensifyPointCloud is one long process, so this is negligible. Works with CLT
   only **[verified locally that runtime MSL compile works on this machine]**.
2. *Ahead-of-time `.metallib` (optional)*: if `xcrun --find metal` succeeds, add a CMake
   custom command: `xcrun -sdk macosx metal -O2 -std=metal4.0 -c PatchMatchMetal.metal -o
   PatchMatchMetal.air && xcrun -sdk macosx metallib PatchMatchMetal.air -o PatchMatchMetal.metallib`,
   install it next to the binary (`SuperBuild/install/bin/`), and load with
   `newLibraryWithURL:` (falls back to source if missing). Precompiled loads in ~20 ms vs
   ~130 ms source ([Apple forum 696543](https://developer.apple.com/forums/thread/696543)); the
   `-std=` flag names track the MSL version ([forum 743992](https://developer.apple.com/forums/thread/743992)).
   Guard with `find_program(METAL_COMPILER …)` and never make it required.

**Host code language.** Upstream uses Objective-C++ (`.mm`, `-fobjc-arc`). If ODM-owned Metal
code is added elsewhere (e.g. `odm_orthophoto`), the alternatives are the same `.mm` approach
or metal-cpp (header-only, C++17, `NS_PRIVATE_IMPLEMENTATION` in one TU, link
`-framework Metal -framework Foundation -framework QuartzCore`) which keeps CMake in pure CXX
([apple/metal-cpp](https://github.com/apple/metal-cpp)). Both compile with the CLT clang
**[verified locally for `.mm`]**; the conda clang should behave identically given the SDK
sysroot **[uncertain: not tried]**.

---

## 5. Runtime wiring (document only)

* `stages/openmvs.py`: rename `--cuda-device` → `--gpu-device` (`-1` GPU, `-2` CPU; upstream
  option name, compiled only when `_USE_CUDA || _USE_METAL` — `01-*.md` §2.2) once the pin
  moves past `4694ded`; keep the exit-code-1 → CPU retry; consider lowering `--max-threads`
  when the GPU path is active since GPU memory is the same 24 GB (`00-*.md` §4).
* `opendm/gpu.py`: add `has_metal()` (§6) and use `has_gpu(args) or has_metal(args)` in the
  OpenMVS stage; leave PopSift's CUDA checks alone.
* `SuperBuild/CMakeLists.txt` / `External-OpenMVS.cmake`: `ODM_ENABLE_METAL` → `SB_ENABLE_METAL`
  (§4); upstream has no `OpenMVS_USE_METAL` option — Metal is auto-enabled on APPLE when the
  frameworks are found, so the ODM switch mainly controls the Python side and any optional
  `.metallib` prebuild.
* `opendm/skyremoval/skyfilter.py`, `opendm/bgfilter.py`: provider list with
  `CoreMLExecutionProvider` + `ModelFormat=MLProgram` (`03-*.md` §4).
* `opendm/config.py`: `--no-gpu` already exists and should disable Metal/CoreML too.
* `DYLD_LIBRARY_PATH` is irrelevant to Metal (system framework, in the dyld shared cache).

---

## 6. Python-side detection: a `has_metal()` next to `has_gpu()`

Options, cheapest first (all document-only):

1. **`system_profiler`** — `system_profiler SPDisplaysDataType` prints
   `Chipset Model: Apple M4 Pro`, `Total Number of Cores: 16`, `Metal Support: Metal 4`
   **[verified locally]**. Parse `Metal Support:`; costs ~1 s; also works to log the GPU name.
   `-json` output is available for robust parsing.
2. **`sysctl`/`platform`** — `platform.machine() == "arm64"` and `sys.platform == "darwin"`
   is a sufficient proxy for "Apple-silicon Mac ⇒ Metal 3+ GPU", since every Apple-silicon Mac
   supports Metal (Apple Support 102894). Zero cost; cannot detect Rosetta (under Rosetta
   `platform.machine()` reports `x86_64`), which is desirable — treat that as "no Metal".
3. **ctypes** — `ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal").MTLCreateSystemDefaultDevice()`
   returns a non-NULL `id`; querying properties requires `objc_msgSend` gymnastics. Fine as a
   boolean probe **[uncertain: not exercised; Metal.framework has no on-disk binary in the
   shared cache but `dlopen` by framework path works, as it did for OpenCL locally]**.
4. **PyObjC** — `from Metal import MTLCreateSystemDefaultDevice; d = MTLCreateSystemDefaultDevice();
   d.name(), d.supportsFamily_(1009)  # MTLGPUFamilyApple9`, `d.recommendedMaxWorkingSetSize()`
   ([pyobjc-framework-Metal](https://pypi.org/project/pyobjc-framework-Metal/),
   [API notes](https://pyobjc.readthedocs.io/en/latest/apinotes/Metal.html),
   [conda-forge package](https://anaconda.org/conda-forge/pyobjc-framework-metal)). Most
   informative (can report the memory budget and family), adds a dependency; known PyObjC
   over-retain leak on `new*` methods ([pyobjc #690](https://github.com/ronaldoussoren/pyobjc/issues/690))
   is irrelevant for a one-shot probe.
5. Let the C++ binary decide: upstream's `AllocateMetalPool` already probes `IsValid()` and
   silently falls back to CPU; ODM only needs the Python check for logging and for choosing
   `--gpu-device`.

Suggested shape (description, not code): `opendm/gpu.py` gains `has_metal()` =
`sys.platform == "darwin" and platform.machine() == "arm64" and not gpu_disabled_by_user_env()
and not args.no_gpu`, optionally refined by `system_profiler` to log the chip and the Metal
version; `has_gpu()` keeps its CUDA meaning and callers ask for `has_gpu() or has_metal()`
where the downstream tool supports both.

---

## 7. Rosetta and architecture pitfalls (summary)

* Never mix `osx-64` and `osx-arm64` packages; pixi's platform pin prevents it, but a
  `x86_64` Homebrew/system Python on `PATH` inside custom scripts would not be Metal-capable
  in the same way (Rosetta Metal is supported but is a translated x86_64 process and cannot
  load arm64 dylibs). `arch` → must print `arm64` inside `pixi shell`.
* Rosetta does not translate AVX/AVX2/AVX-512 — irrelevant for native builds, but explains
  crashes if an x86_64 wheel with AVX kernels is pulled in accidentally.
* `-march=native` in mvs-texturing is fine locally; for distributing binaries to other Macs
  prefer `-mcpu=apple-m1` or no flag **[uncertain: exact conda clang flag support]**.
* Apple's CLT `swiftc` currently fails to import `Metal` due to an SDK/compiler version skew
  on this machine **[verified locally]** — another reason to keep ODM's Metal code in
  Objective-C++/C++, which is unaffected.

---

## 8. Verification checklist for when the recipe is executed

1. `pixi info` shows `osx-arm64` and the `metal` environment.
2. Configure log of OpenMVS: `Enabling Metal compute backend on macOS`, `Found OpenMP`.
3. `SuperBuild/install/bin/DensifyPointCloud --help | grep gpu-device` shows the option.
4. `file` on binaries → `Mach-O 64-bit executable arm64`; `codesign -dv` → `Signature=adhoc`.
5. `pixi run python -c "import onnxruntime as o; print(o.get_available_providers())"` includes
   `CoreMLExecutionProvider`.
6. `pixi run python -c "import numpy; numpy.show_config()"` reports Accelerate.
7. `pixi run odm -- --project-path … --pc-quality medium` on a small aerial dataset with the
   log showing `Using provider CoreMLExecutionProvider` and OpenMVS using the Metal pool
   (upstream logs per-map timings with `-v 2`); Activity Monitor's GPU history shows load.
8. Compare `odm_georeferenced_model.laz` point count and DEM stats with the CPU run.

---

## Sources

- ODM (read locally): `/Users/avneh/Code/ADM/ODM/pixi.toml`, `/Users/avneh/Code/ADM/ODM/pixi.lock`, `/Users/avneh/Code/ADM/ODM/SuperBuild/CMakeLists.txt`, `/Users/avneh/Code/ADM/ODM/SuperBuild/cmake/External-OpenMVS.cmake`, `External-OpenSfM.cmake`, `External-PyPopsift.cmake`, `External-MvsTexturing.cmake`, `External-FPCFilter.cmake`, `ExternalProject-Setup.cmake`, `patch-cgal-iterator.cmake`, `poissonrecon-macos-makefile.patch`, `/Users/avneh/Code/ADM/ODM/opendm/gpu.py`, `/Users/avneh/Code/ADM/ODM/stages/openmvs.py`
- pixi installation: https://pixi.prefix.dev/latest/installation/ ; multi-platform configuration: https://pixi.prefix.dev/latest/workspace/multi_platform_configuration/ ; pixi #1750 (osx-64 fallback under Rosetta): https://github.com/prefix-dev/pixi/issues/1750 ; #1404: https://github.com/prefix-dev/pixi/issues/1404
- conda-forge macOS ARM builds: https://conda-forge.org/blog/2020/10/29/macos-arm64/ ; clang activation / SDKROOT: https://github.com/conda-forge/clang-compiler-activation-feedstock/issues/94 ; knowledge base: https://conda-forge.org/docs/maintainer/knowledge_base/ ; conda-build compiler tools: https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html
- conda-forge Accelerate BLAS: https://conda-forge.org/news/2025/07/31/new-accelerate-macos/ ; onnxruntime CoreML on conda-forge (PR #182): https://github.com/conda-forge/onnxruntime-feedstock/pull/182 ; pyobjc-framework-metal on conda-forge: https://anaconda.org/conda-forge/pyobjc-framework-metal
- cdcseacave/openMVS develop CMakeLists (Metal auto-detect, OpenMP_ROOT): https://raw.githubusercontent.com/cdcseacave/openMVS/develop/CMakeLists.txt ; libs/MVS/CMakeLists (MSL embedding): https://raw.githubusercontent.com/cdcseacave/openMVS/develop/libs/MVS/CMakeLists.txt ; PR #1279: https://github.com/cdcseacave/openMVS/pull/1279
- Apple metal-cpp: https://github.com/apple/metal-cpp ; https://developer.apple.com/metal/cpp/
- Xcode 26 Metal Toolchain: https://www.polpiella.dev/metal-toolchain-ci-cd ; https://github.com/actions/runner-images/issues/13080 ; https://openradar.appspot.com/FB20389216 ; https://christiantietze.de/posts/2026/01/manually-mount-metal-toolchain/ ; Apple forum 787294: https://developer.apple.com/forums/thread/787294 ; llama.cpp #6946: https://github.com/ggml-org/llama.cpp/issues/6946
- Metal compile timings and flags: https://developer.apple.com/forums/thread/696543 ; https://developer.apple.com/forums/thread/743992 ; Metal Programming Guide (Functions and Libraries): https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Prog-Func/Prog-Func.html
- Code signing on Apple silicon: https://eclecticlight.co/2020/08/22/apple-silicon-macs-will-require-signed-code/ ; https://www.smileykeith.com/2021/10/05/codesign-m1/ ; SuperCollider #5603: https://github.com/supercollider/supercollider/issues/5603
- Rosetta 2: https://support.apple.com/guide/security/rosetta-2-on-a-mac-with-apple-silicon-secebb113be1/web ; WWDC20 10631 "Bring your Metal app to Apple silicon Macs": https://developer.apple.com/videos/play/wwdc2020/10631/ ; Apple, Rosetta translation environment: https://developer.apple.com/documentation/apple-silicon/about-the-rosetta-translation-environment
- Apple Support, Metal on Apple devices: https://support.apple.com/en-us/102894 ; system_profiler usage: https://gist.github.com/dvu4/8b786513fbe6fb55eb6504d585155dcf
- PyObjC Metal: https://pypi.org/project/pyobjc-framework-Metal/ ; https://pyobjc.readthedocs.io/en/latest/apinotes/Metal.html ; https://github.com/ronaldoussoren/pyobjc/issues/690 ; minimal Metal-from-Python gist: https://gist.github.com/alvinwan/f7bb0cdd26c018f40052f9944fc5c679
