# CPU-side tuning for ODM on an Apple M4 Pro (8P+4E, 24 GB)

All hardware facts below were read on the target machine on 2026-09-13 unless a URL is given.
Everything marked ESTIMATE or UNVERIFIED must be confirmed by the benchmark plan in
`04-benchmark-and-profiling-plan.md` before being relied on.

## 1. The machine, precisely

| Item | Value | Source |
|---|---|---|
| SoC | Apple M4 Pro, 12-core: **8 Performance + 4 Efficiency** | `sysctl hw.perflevel0.physicalcpu`=8 (`name: Performance`), `hw.perflevel1.physicalcpu`=4 (`name: Efficiency`); Apple tech specs list the 12-core M4 Pro as 8P+4E (https://support.apple.com/en-us/121553) |
| Clocks | P-cores up to ~4.5 GHz, E-cores ~2.6 GHz | Notebookcheck (https://www.notebookcheck.com/Apple-M4-Pro-12-cores-Prozessor-Benchmarks-und-Specs.922298.0.html); not exposed by sysctl |
| Caches | P: L1d 128 KB, L1i 192 KB, **L2 16 MB per 4-core cluster** (two clusters); E: L1d 64 KB, L2 4 MB shared by 4 cores; cache line 128 B; page size 16 KB | `sysctl hw.perflevel0.l1dcachesize hw.perflevel0.l2cachesize hw.perflevel0.cpusperl2 ...` |
| Memory | 24 GB LPDDR5X unified, **273 GB/s** | `hw.memsize`=25769803776; bandwidth from Apple (https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/) |
| ISA features | NEON, FP16, BF16, DotProd, I8MM, LSE/LSE2 atomics, **SME + SME2** (`FEAT_SME_F64F64`, `FEAT_SME_I16I64`); **no SVE** | `sysctl hw.optional.arm.FEAT_*` |
| GPU | 16-core, shares the 24 GB | `docs/00-context/02-hardware-profile.md` |
| OS / tools | macOS 26.6; Apple clang 17 (CLT only); conda clang 19.1.7 once pixi env exists | `sw_vers`, `clang --version`, `pixi.lock` |

Practical reading: 8 fast cores with a big 16 MB L2 per cluster and very high memory
bandwidth. The pipeline's hot loops (SIFT scale space, FLANN kd-tree search, patch-match
depth estimation, Poisson octree solves, MRF view selection) are mostly compute- or
latency-bound, not bandwidth-bound; 273 GB/s is ample. What hurts is (a) work that is
single-threaded (OpenSfM `reconstruct`, PDAL translate, GDAL writes), (b) oversubscription
across P+E cores, and (c) memory pressure with 24 GB.

## 2. Compiler: what `-mcpu=apple-m4` would and would not buy

* conda-forge's clang activation sets `-O2 -ftree-vectorize` and **no `-mcpu`/`-march` on
  arm64** (https://raw.githubusercontent.com/conda-forge/clang-compiler-activation-feedstock/main/recipe/build.sh);
  CMake Release adds `-O3 -DNDEBUG`. LLVM's default CPU for `arm64-apple-macos` is
  `apple-m1` (verified with Apple clang 17: `clang -### -c t.c` -> `-target-cpu apple-m1`).
  So generated code already uses NEON, FP16 arithmetic, DotProd and the M1 scheduling model.
* `apple-m4` exists in **LLVM 19** (`AArch64Processors.td` on `release/19.x`:
  `ProcessorModel<"apple-m4", CycloneModel, ProcessorFeatures.AppleM4, [TuneAppleM4]>`;
  absent from `release/18.x`;
  https://raw.githubusercontent.com/llvm/llvm-project/release/19.x/llvm/lib/Target/AArch64/AArch64Processors.td).
  The conda toolchain pinned by `pixi.lock` is clang **19.1.7**, so `-mcpu=apple-m4` is
  accepted. Its feature set is an **armv8.7-a base** with SHA3, FullFP16, FP16FML, AES, BF16,
  SME, SME2 and *no* SVE (LLVM commit eba1ef5a, Aug 2024,
  https://github.com/llvm/llvm-project/commit/eba1ef5a1b7a84ed1954797dcd6d6f073b1f1a56).
  `TuneAppleM4` adds fusion/zero-cycle-move tuning; `CycloneModel` is the same generic
  scheduling model used for M1, so the codegen difference between `apple-m1` and `apple-m4`
  is small (BF16/I8MM/SME availability and armv8.7 atomics/LSE2). ESTIMATE: 0-5 % on the
  C++ hot loops; do not expect a step change. `-mcpu=native` resolves through
  `getHostCPUName()`; whether clang 19.1.7 maps this host to `apple-m4` is UNVERIFIED (check
  `clang -mcpu=native -### -c t.c 2>&1 | grep target-cpu`).
* Where to inject it if tried: `CXXFLAGS`/`CFLAGS` in the pixi activation (`pixi.toml`
  `[feature.build.activation] env`) so every CMake sub-project and PoissonRecon's Makefile see
  it; OpenSfM forwards `$ENV{CXXFLAGS}` explicitly (`External-OpenSfM.cmake:33`). Two
  sub-projects override architecture flags themselves: mvs-texturing (patch removes
  `-march=native`) and OpenPointClass (`PORTABLE_BUILD=ON`). Note conda packages (OpenCV,
  Ceres, OpenBLAS, GDAL, PDAL) are prebuilt and cannot be retuned without rebuilding; that is
  what prefix.dev's "CPU optimized packages" note in PR #2037 refers to
  (https://github.com/OpenDroneMap/ODM/pull/2037).
* `-ffast-math`: PoissonRecon already uses it; OpenMVS uses `-frounding-math` unless
  `ENABLE_FAST_MATH` (its `build/Utils.cmake`). Changing this alters results; leave alone
  for baselines.
* SME: LLVM will not auto-vectorize to SME; only Accelerate (AMX/SME-backed BLAS) uses the
  matrix unit today. This is the strongest argument for the Accelerate BLAS experiment below.

## 3. NEON auto-vectorization in the libraries

* **Eigen** (Ceres, OpenSfM, OpenMVS, odm_orthophoto): NEON packet math is on whenever
  `__ARM_NEON` is defined (always on arm64-apple); SVE/SME paths are opt-in and irrelevant
  here (https://gitlab.com/libeigen/eigen/-/raw/master/Eigen/src/Core/util/ConfigureVectorization.h).
  OpenMVS and mvs-texturing bundle Eigen 3.4.0 headers (`SuperBuild` `eigen34`), Ceres/OpenSfM
  use conda `eigen 3.4.0`.
* **OpenCV 4.12 (conda)**: AArch64 baseline is NEON with runtime dispatch for
  `NEON_FP16;NEON_BF16;NEON_DOTPROD` (`OpenCVCompilerOptimizations.cmake`; SVE explicitly
  excluded on Apple: "Current Apple silicone M4 does not support SVE",
  https://raw.githubusercontent.com/opencv/opencv/4.x/cmake/OpenCVCompilerOptimizations.cmake).
  Whether the conda-forge build keeps those dispatch defaults is UNVERIFIED (`cv2.getBuildInformation()`
  after install; look for "Dispatched code generation").
* **Ceres 2.2**: C++ thread pool only (OpenMP/TBB removed in 2.2.0,
  https://raw.githubusercontent.com/ceres-solver/ceres-solver/2.2.0/docs/source/version_history.rst).
  OpenSfM configures `DENSE_SCHUR`, `num_threads = processes`, 10 iterations
  (`opensfm/src/sfm/src/ba_helpers.cc:262-264`), and leaves
  `dense_linear_algebra_library_type` at its default `EIGEN`, so BA dense algebra is Eigen,
  not BLAS. `SPARSE_SCHUR`/CHOLMOD would be BLAS-bound; that path is only reachable by
  changing OpenSfM's solver string (out of scope; note for the Metal track that Ceres also has
  CUDA-only dense solvers).
* **VLFeat SIFT** in OpenSfM compiles with `-DVL_DISABLE_AVX`; on arm64 it is scalar plus
  whatever clang auto-vectorizes. DSP-SIFT (ODM default `--feature-type dspsift`) is the
  OpenCV SIFT path in OpenSfM's `features.py`, i.e. OpenCV's NEON kernels. UNVERIFIED which
  exact code path dominates; profile `opensfm detect_features` (plan in 04).

## 4. BLAS/LAPACK: OpenBLAS today, Accelerate as an experiment

* Today: `libblas/libcblas/liblapack 3.11.0 *_openblas` with `libopenblas 0.3.33 openmp_*`
  (`pixi.lock`). Consumers in the ODM env: numpy, scipy, OpenCV (`liblapack`, `libcblas`),
  Ceres (`liblapack`), SuiteSparse/CHOLMOD (`libblas`, `liblapack`). All link the generic
  `libblas` shim, so they are swappable without rebuilding, which is the documented
  conda-forge mechanism: `libblas=*=*_openblas | *_accelerate | *_newaccelerate | *_mkl ...`
  (https://conda-forge.org/docs/maintainer/knowledge_base/#switching-blas-implementation).
  `newaccelerate` (macOS 13.3+) uses Apple's new ILP64-capable BLAS/LAPACK with LAPACK fixes
  (https://conda-forge.org/news/2025/07/31/new-accelerate-macos/). In pixi this is a
  dependency pin, e.g. `libblas = { version = "*", build = "*newaccelerate" }` (syntax
  UNVERIFIED; derived from the build-string convention), which changes `pixi.lock`.
* Ceres from conda-forge is built with `-DBLA_VENDOR=Generic -DACCELERATESPARSE=OFF` and
  `-DSUITESPARSE=ON` (gpl variant) (https://raw.githubusercontent.com/conda-forge/ceres-solver-feedstock/main/recipe/build.sh),
  so: LAPACK calls go through the swappable `liblapack`; Accelerate's *sparse* solvers are
  compiled out; and, as noted, OpenSfM's BA does not use LAPACK anyway. **Expected win for
  ODM from Accelerate: small** - numpy/scipy code in ODM (alignment, GSD, DEM helpers) is not
  BLAS-heavy, and the C++ hot paths are Eigen or hand-written. Worth a 30-minute A/B on the
  `opensfm` stage only; ESTIMATE <5 % end-to-end.
* Thread interaction: OpenBLAS `openmp_*` build spawns its own OpenMP team; combined with
  OpenSfM's `processes` threads and Ceres' pool it can oversubscribe. Set
  `OPENBLAS_NUM_THREADS=1` (or `OMP_NUM_THREADS` for the Python process) when profiling
  shows OpenBLAS threads in the OpenSfM stage. Ceres' own docs warn that OpenBLAS threading
  "must be disabled to avoid conflicts with Ceres threading"
  (http://ceres-solver.org/installation.html).

## 5. Threads on P/E cores: what macOS lets you do

* There is **no thread pinning** on macOS. libomp compiles affinity out on Darwin
  (`KMP_AFFINITY_SUPPORTED` undefined; https://github.com/llvm/llvm-project/pull/156708);
  `OMP_PROC_BIND` / `OMP_PLACES` / `KMP_AFFINITY` are documented for Windows/Linux only
  (https://openmp.llvm.org/design/Runtimes.html). The scheduler places threads by QoS: QoS
  Background (and lower) runs only on E-cores; higher QoS gets a P-core "if one is available"
  otherwise an E-core cluster at high frequency
  (https://eclecticlight.co/2024/12/17/tune-for-performance-core-types/,
  https://eclecticlight.co/2022/01/24/how-you-cant-promote-threads-on-an-m1/).
* `OMP_NUM_THREADS` default = all 12 logical CPUs. ODM sets it per tool from
  `--max-concurrency` (`odm_orthophoto`, `renderdem`, `fastrasterfilter`, `pcclassify`) and
  passes explicit thread counts elsewhere (`DensifyPointCloud --max-threads`, `PoissonRecon
  --threads N-1`, `texrecon --num_threads`, `FPCFilter --concurrency`, `dem2mesh
  -maxConcurrency`, OpenSfM `processes`). So `--max-concurrency` is the single knob.
  Recommendations to test (all ESTIMATE until measured):
  * `--max-concurrency 12` (default): uses E-cores too; best for throughput-style stages
    (feature extraction, matching, depth maps) if memory allows; E-cores add maybe 15-25 %
    over 8 P-cores given ~2.6 vs ~4.5 GHz and narrower cores.
  * `--max-concurrency 8`: safer for memory (OpenMVS at `--pc-quality high`, PoissonRecon
    depth 12) and avoids the P/E straggler effect in barrier-synchronized OpenMP loops
    (E-core threads finish last, P-cores idle at the barrier). Static-scheduled `omp for`
    loops in OpenMVS/PoissonRecon are the likely victims; dynamic schedules are not.
  * `KMP_BLOCKTIME`/`OMP_WAIT_POLICY=PASSIVE`: reduces spin when the pipeline has serial
    Python between parallel regions; default libomp block time is 200 ms.
* Demotion, not promotion: `taskpolicy -b <cmd>` forces a whole run onto E-cores (useful to
  keep the laptop usable during an overnight run; expect ~3x slower);
  `taskpolicy -c utility` clamps QoS (https://ss64.com/mac/taskpolicy.html). Python's
  `os.setpriority`/`nice` does not promote either.
* Apple's QoS API (`pthread_set_qos_class_self_np`, `QOS_CLASS_USER_INITIATED`) is the only
  way to *ask* for P-cores, and it is per-thread inside the binary
  (https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheTaskLevel.html).
  Default QoS for a process launched from a terminal is already high enough to use P-cores,
  so no change is needed unless a tool is observed on E-cores (`powermetrics
  --show-process-amp` shows per-process P/E split).

## 6. Memory: 24 GB, page-outs, and how ODM sizes things

* Rules of thumb from the docs: 40 images -> 4 GB, 250 -> 16 GB, 500 -> 32 GB
  (https://docs.opendronemap.org/installation/); `--max-concurrency` help: "Peak memory
  requirement is ~1GB per thread and 2 megapixel image resolution"
  (https://docs.opendronemap.org/arguments/). Community: 16 GB machines crash on ~500 images
  at defaults (Shitan 493 crashed at 47 min on a 16 GB Ryzen laptop, odm-benchmarks CSV,
  https://raw.githubusercontent.com/OpenDroneMap/odm-benchmarks/master/data/benchmarks.csv).
  So 24 GB comfortably handles 100-250 images at defaults; 300-500 needs `--pc-quality low`,
  `--max-concurrency 8` or `--split`.
* ODM's own sizing (`opendm/concurrency.py`): `get_max_memory()` = half of the *currently
  free* percentage (min 5 %) for GDAL cache in DEM/report steps; `get_max_memory_mb()` = half
  of available MB (min 100) for `odm_orthophoto`'s `GDAL_CACHEMAX`. Both read `vmem`'s
  `virtual_memory()` at stage start, so a browser-heavy desktop shrinks ODM's caches; close
  memory hogs before a benchmark. OpenMVS OOM (exit 137/143) is caught and rerun with
  `--fusion-mode 1` tiling (`stages/openmvs.py:100-105`); PoissonRecon failure retries with
  half the threads (`opendm/mesh.py:172-181`); `dem2mesh` likewise halves concurrency.
* OpenSfM's memory throttles are inactive on macOS (`memory_available()` needs `free`;
  see 00 section 3). With 12 detection workers on 18 MP images at `--feature-quality ultra`
  (4480 px scale space ~ hundreds of MB each) the Python process can exceed 10 GB; the
  laptop will compress then swap instead of failing.
* Observe: `vm_stat 1` (watch "Pageouts" and "Pages occupied by compressor"; page size 16 KB),
  `memory_pressure` (prints system-wide free %), Activity Monitor's Memory Pressure graph
  (yellow/red = compress/swap;
  https://support.apple.com/guide/activity-monitor/check-if-your-mac-needs-more-ram-actmntr34865/mac).
  macOS compresses before it swaps, and the compressor burns P-core time
  (https://blog.greggant.com/posts/2024/07/03/macos-memory-management.html), so a run that
  "works" under pressure can still be 2x slower - record `vm_stat` deltas per stage.

## 7. Thermals on a laptop

* `sudo powermetrics --samplers cpu_power,gpu_power,thermal -i 1000 -n <N>` reports CPU/GPU
  power, per-cluster frequencies and "thermal pressure" (Nominal / Moderate / Heavy /
  Trapping / Sleeping); `--show-process-amp` gives per-process P vs E time; `--show-plimits`
  shows CPU limiting (local `powermetrics -h`, https://ss64.com/mac/powermetrics.html,
  https://fleetdm.com/tables/macos_thermal_pressure). Requires root.
* Reviews of the 14" MacBook Pro with the base 12-core M4 Pro report ~11 % multi-core loss
  under sustained Cinebench even in High Power mode
  (https://www.notebookcheck.net/Apple-MacBook-Pro-14-2024-M4-Pro-base-SKU-laptop-review-Who-needs-the-M4-Pro.932642.0.html).
  ODM runs are 15 min to hours of all-core load, so expect the steady state, not the burst
  clock. Log `powermetrics` alongside `benchmark.txt` and compare the first and last stage
  that use all cores (feature extraction vs texturing) for frequency drift.
* Mitigations: plug in and set System Settings > Battery > Energy Mode = High Power (16"
  only exposes it fully; 14" M4 Pro has it too per the review above), keep the lid open, use
  `--max-concurrency 8` overnight to lower package power for a small throughput cost.

## 8. Python side

* Multiprocessing start method on macOS is **spawn** since Python 3.8 ("fork ... should be
  considered unsafe" on macOS; https://docs.python.org/3.12/library/multiprocessing.html).
  ODM's own parallelism is threads (`opendm/concurrency.py::parallel_map`) and OpenSfM's
  `parallel_map` is joblib with the `threading` backend (`opensfm/context.py`), so spawn cost
  does not hit the hot stages; it matters only for any future Pool-based code.
* numpy 2.3.2 / scipy 1.16.1 from conda-forge use the env's `libblas` (OpenBLAS now,
  Accelerate if switched; SciPy >= 1.14 supports Accelerate on macOS 13.3+,
  https://github.com/scipy/scipy/pull/20510). ODM's numpy work is light.
* GIL: OpenSfM's detection/matching threads release the GIL inside OpenCV/pybind calls, so
  threads scale; pure-Python parts (track creation, incremental reconstruction bookkeeping)
  do not, which is why `reconstruct` looks single-core. Python 3.12 has no free-threading
  build in this env.
* `cv2.setNumThreads(0)` is set by OpenSfM around `parallel_map` to stop OpenCV's internal
  pool from multiplying threads; leave it.

## 9. Priority list for the CPU baseline (cheapest first)

1. Measure defaults with `benchmark.txt` + `powermetrics` + `vm_stat` (04). Nothing below is
   worth doing blind.
2. `--max-concurrency 8` vs 12 on the same dataset; keep whichever is faster per stage
   (it can differ: 12 for opensfm, 8 for openmvs). Free win, no code change.
3. `OPENBLAS_NUM_THREADS=1` (or `=4`) exported before `pixi run odm` if OpenBLAS threads
   show up in `sample` of the opensfm process.
4. `libblas=*=*_newaccelerate` A/B (pixi pin; lockfile change). Low expected gain, low cost.
5. `CXXFLAGS="$CXXFLAGS -mcpu=apple-m4"` rebuild of the SuperBuild (not the conda packages).
   ESTIMATE 0-5 %; mainly a sanity check that nothing regresses.
6. Anything that needs source changes (Ceres `SPARSE_SCHUR`, OpenMP schedule tweaks, QoS
   calls) belongs with the Metal/engineering track, not the baseline.

## Sources

* Local: `sysctl hw.*`, `sw_vers`, `clang -###`, `powermetrics -h`, `taskpolicy`, `vm_stat`,
  `memory_pressure` (2026-09-13); code paths cited from `opendm/concurrency.py`,
  `opendm/osfm.py`, `stages/openmvs.py`, `opendm/mesh.py`, `SuperBuild/cmake/*.cmake`.
* Apple M4 Pro specs: https://support.apple.com/en-us/121553 ,
  https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/ ,
  https://en.wikipedia.org/wiki/Apple_M4 ,
  https://www.notebookcheck.com/Apple-M4-Pro-12-cores-Prozessor-Benchmarks-und-Specs.922298.0.html
* `sysctl` key meanings: https://eclecticlight.co/sysctl-information/
* LLVM apple-m4: https://raw.githubusercontent.com/llvm/llvm-project/release/19.x/llvm/lib/Target/AArch64/AArch64Processors.td ,
  https://github.com/llvm/llvm-project/commit/eba1ef5a1b7a84ed1954797dcd6d6f073b1f1a56 ,
  https://github.com/llvm/llvm-project/commit/fcca10c69aaab539962d10fcc59a5f074b73b0de (`-march=native` on Apple silicon)
* conda-forge flags: https://raw.githubusercontent.com/conda-forge/clang-compiler-activation-feedstock/main/recipe/build.sh
* conda-forge BLAS switching: https://conda-forge.org/docs/maintainer/knowledge_base/#switching-blas-implementation ,
  https://conda-forge.org/news/2025/07/31/new-accelerate-macos/
* Ceres: https://raw.githubusercontent.com/conda-forge/ceres-solver-feedstock/main/recipe/build.sh ,
  https://raw.githubusercontent.com/ceres-solver/ceres-solver/2.2.0/docs/source/version_history.rst ,
  https://raw.githubusercontent.com/ceres-solver/ceres-solver/2.2.0/CMakeLists.txt ,
  http://ceres-solver.org/installation.html , https://ceres-solver.readthedocs.io/latest/nnls_solving.html
* Eigen NEON: https://gitlab.com/libeigen/eigen/-/raw/master/Eigen/src/Core/util/ConfigureVectorization.h
* OpenCV AArch64 dispatch: https://raw.githubusercontent.com/opencv/opencv/4.x/cmake/OpenCVCompilerOptimizations.cmake ,
  https://github.com/opencv/opencv/wiki/CPU-optimizations-build-options
* OpenMP on macOS: https://openmp.llvm.org/design/Runtimes.html ,
  https://github.com/llvm/llvm-project/pull/156708 ,
  https://eclecticlight.co/2022/01/24/how-you-cant-promote-threads-on-an-m1/ ,
  https://eclecticlight.co/2024/12/17/tune-for-performance-core-types/ ,
  https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/PrioritizeWorkAtTheTaskLevel.html ,
  https://ss64.com/mac/taskpolicy.html
* Thermals: https://ss64.com/mac/powermetrics.html , https://fleetdm.com/tables/macos_thermal_pressure ,
  https://www.notebookcheck.net/Apple-MacBook-Pro-14-2024-M4-Pro-base-SKU-laptop-review-Who-needs-the-M4-Pro.932642.0.html
* Memory: https://support.apple.com/guide/activity-monitor/check-if-your-mac-needs-more-ram-actmntr34865/mac ,
  https://blog.greggant.com/posts/2024/07/03/macos-memory-management.html ,
  https://docs.opendronemap.org/installation/ , https://docs.opendronemap.org/arguments/ ,
  https://raw.githubusercontent.com/OpenDroneMap/odm-benchmarks/master/data/benchmarks.csv
* Python: https://docs.python.org/3.12/library/multiprocessing.html , https://github.com/scipy/scipy/pull/20510
* OpenSfM threading: https://github.com/OpenDroneMap/OpenSfM/blob/c5328439465e6ace011f39077d1077d7b1cdd65d/opensfm/context.py ,
  https://github.com/OpenDroneMap/OpenSfM/blob/c5328439465e6ace011f39077d1077d7b1cdd65d/opensfm/src/sfm/src/ba_helpers.cc
