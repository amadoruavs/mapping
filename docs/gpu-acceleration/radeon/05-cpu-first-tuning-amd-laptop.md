# CPU-first tuning for ODM on a Ryzen laptop (Zen 4 / Zen 5 APU)

The iGPU work in docs 01–03 is speculative and, at best, buys 2–4× on one stage. The items
below are sure wins on an 8-core Ryzen 7 7840HS/8845HS-class laptop (and apply unchanged to
Strix Point). Flags reference ODM 3.6.2's `opendm/config.py`; behaviour references the stage
code. Nothing here requires a rebuild unless stated.

---

## 1. Know the machine

* Ryzen 7 7840HS: 8 cores / 16 threads, Zen 4, AVX-512 (executed on 256-bit datapaths),
  Radeon 780M (12 CU), DDR5-5600 / LPDDR5x-7500, configurable TDP 35–54 W (AMD spec page;
  the page refused connections during research — verify on
  [amd.com](https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-7-7840hs.html)).
  Notebookcheck lists the 780M at up to 3 GHz / 768 shaders.
  [notebookcheck](https://www.notebookcheck.net/Radeon-780M-vs-Vega-7_11564_10098.247598.0.html)
* Everything shares one power/thermal budget and one memory bus. A sustained 16-thread
  ODM run will sit at the laptop's PL1, so **plug in, pick the "performance" power profile,
  and put it on a hard surface**; on Linux check `amd-pstate` is in `active`/`performance`
  (`cat /sys/devices/system/cpu/amd_pstate/status`, `cpupower frequency-info`).
* Memory is the wall. ODM's own help text: "Peak memory requirement is ~1GB per thread and
  2 megapixel image resolution" (`--max-concurrency`). A 16 GB laptop with a 2–4 GB BIOS UMA
  carve-out for the iGPU has ~11 GB usable. **Set the BIOS "UMA Frame Buffer Size" to the
  minimum/Auto** for CPU-first use; on Linux the iGPU can still borrow via GTT when needed.
  [Strix Halo guide (same GTT model)](https://rocmdocs.amd.com/en/develop/how-to/system-optimization/strixhalo.html)
* Dual-channel/dual-rank memory matters more than CPU model: a single-channel DDR5-5600
  Phoenix box measured ~31 GB/s and lost to expectations across the board.
  [k8s.it](https://www.k8s.it/posts/running-a-local-llm-on-amd-radeon-780m-gfx1103-rocm-and-the-gpu-that-wasnt-supposed-to-work/)

## 2. The flags that move the needle (ODM 3.6.2)

| Flag | Default | What it does | Laptop advice |
|---|---|---|---|
| `--max-concurrency N` | `num_cores` (16 on a 7840HS incl. SMT) | processes/threads for OpenSfM, OpenMVS `--max-threads`, DEM, ortho | Memory-bound stages OOM before they saturate 16 threads. Start at `8` (physical cores) on 16 GB; `12–16` on 32 GB+. |
| `--pc-quality` | `medium` | OpenMVS `--resolution-level`; "Each step up in quality increases processing time roughly by a factor of 4x" | `medium` is the laptop sweet spot; `low` for previews. `ultra` on a laptop: 1 h vs 13 min for Aukerman (benchmarks below). |
| `--pc-skip-geometric` | off | passes `--geometric-iters 0` to `DensifyPointCloud` (skips the second, geometric-consistency pass) | Roughly halves densification time; slightly noisier cloud. Good default for previews. |
| `--feature-quality` | `high` | image size used for feature extraction (`feature_process_size`) | `medium` cuts extraction and matching time substantially with little SfM loss on typical drone sets. |
| `--feature-type` | `dspsift` | CPU SIFT variants; only `sift` can ever use a GPU | keep `dspsift`; `hahog` is the BoW-compatible fallback |
| `--matcher-neighbors N` | `0` (triangulation) | when > 0, match only the N GPS-nearest images | `--matcher-neighbors 8–12` on GPS-tagged sets bounds matching cost; leave 0 when overlap is irregular |
| `--matcher-type` | `flann` | `bow` faster, `bruteforce` very slow | `flann` |
| `--fast-orthophoto` | off | skip dense reconstruction and mesh; orthophoto from sparse | the single biggest saving when only an orthomosaic is needed (minutes instead of tens of minutes) |
| `--skip-3dmodel` | off | no textured 3D model | saves texturing time and RAM; DSM/ortho unaffected |
| `--skip-report` | off | no PDF report | small win |
| `--use-hybrid-bundle-adjustment` | off | incremental BA less often | helps on 500+ image sets |
| `--optimize-disk-space` | off | deletes intermediate files as stages finish | essential on 512 GB laptop SSDs; `datasets/*/opensfm/undistorted` and `openmvs` are the hogs |
| `--orthophoto-resolution`, `--dem-resolution` | 5 cm/px | output GSD | do not ask for finer than the GSD; it only costs time and disk |
| `--mesh-octree-depth` | 11 | PoissonRecon depth | 10 on laptops when a mesh is needed at all |
| `--min-num-features` | 10000 | per image | leave; lowering starves matching |
| `--auto-boundary` | off | crop to camera footprint | trims DEM/ortho work at the edges |
| `--no-gpu` / `ODM_NO_GPU=1` | — | disables GPU code paths | irrelevant on AMD today (no GPU is detected), useful once a HIP build exists and misbehaves |

Sources: `opendm/config.py` help strings; `stages/openmvs.py` (`--geometric-iters 0`,
`--max-threads`, `--resolution-level`); [ODM arguments reference](https://docs.opendronemap.org/arguments/).

### Out-of-memory behaviour you get for free

`stages/openmvs.py` catches exit codes 137/143/3221226505 from `DensifyPointCloud`, logs
"OpenMVS ran out of memory, we're going to turn on tiling", appends `--fusion-mode 1` and
re-runs with sub-scenes (there is no user flag; `pc_tile` is internal). If densification is
being killed repeatedly, lower `--max-concurrency` or `--pc-quality` rather than relying on
the retry, which repeats the failed work. `opendm/concurrency.get_max_memory()` also caps
GDAL/PDAL steps at 50% of RAM.

## 3. Thread placement and BLAS

* OpenSfM is run with `processes: <max_concurrency>` (`opendm/osfm.py`), each worker importing
  numpy/OpenCV/Ceres. If OpenBLAS/OpenMP inside each worker also spins 16 threads you get
  16×16 oversubscription during matching and BA. Export `OPENBLAS_NUM_THREADS=1` and
  `OMP_NUM_THREADS=1` for the OpenSfM stages (ODM already passes `OMP_NUM_THREADS` explicitly
  only to `pcclassify` and the orthophoto step). Mark as "measure it": some configurations
  are already single-threaded per worker.
* BLAS choice in the pixi env: conda-forge's `libblas` selector supports `openblas`, `blis`,
  `mkl`, `netlib` (`libblas=*=*blis`). AMD's AOCL-BLAS is on conda-forge as `aocl-blas 5.2`
  (linux-64/win-64) but does not appear to be wired into the selector; BLIS is its parent,
  so `blis` is the nearest Zen-tuned option. Expect low single-digit % on BA; not a priority.
  [blas-feedstock](https://github.com/conda-forge/blas-feedstock/blob/master/README.md),
  [aocl-blas](https://anaconda.org/conda-forge/aocl-blas)
* Pin ODM to physical cores when the laptop is also in use: `taskset -c 0-7` (Linux) or
  Process Lasso/affinity (Windows); SMT siblings add little to SIMD-heavy stages.

## 4. Rebuilding the SuperBuild for Zen 4 (`-march=znver4`)

The source-built components (OpenSfM's C++ extensions, OpenMVS, mvs-texturing, PoissonRecon,
FPCFilter, OpenPointClass, RenderDEM, Obj2Tiles) are compiled by the conda-forge compilers
that `[feature.build]` installs. conda's compiler activation exports conservative
`CFLAGS`/`CXXFLAGS` (generic x86-64 tuning), so nothing is Zen-specific by default.

* GCC's `-march=znver4` enables AVX-512F/DQ/BW/VL/VNNI/BF16/VBMI etc. on "AMD Family 19h
  core based CPUs"; `-march=znver5` adds AVX-VNNI, AVX512VP2INTERSECT, PREFETCHI for Family
  1Ah (Strix Point/Krackan are Zen 5). `-march=native` picks the compiling machine's ISA.
  [GCC x86 options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html)
* How to inject without editing CMake files: a pixi feature that appends to the flags
  conda already sets, e.g.
  ```toml
  [feature.znver4.activation.env]
  CFLAGS   = "$CFLAGS -march=znver4 -mtune=znver4"
  CXXFLAGS = "$CXXFLAGS -march=znver4 -mtune=znver4"
  ```
  (Linux/macOS only; `scripts/activate-msvc.bat` deliberately clears `CXXFLAGS` on Windows
  because conda's clang flags break `link.exe` — on MSVC use `/arch:AVX512` via
  `CMAKE_CXX_FLAGS` in the ExternalProject args instead.) Then `pixi run -e <env> clean` and
  `build`. Note `External-OpenMVS.cmake` already toggles `OpenMVS_USE_SSE` off only on ARM;
  OpenMVS's own SSE paths stay on.
* Expected gain: modest (5–15% on OpenMVS CPU densification and PoissonRecon, where the
  compiler can vectorise), zero on Python-bound stages, and the binaries stop running on
  older CPUs. The conda-forge prebuilt libraries (OpenCV, Ceres, GDAL, PDAL, numpy/OpenBLAS)
  are unaffected — conda-forge ships baseline x86-64 builds with runtime dispatch where the
  upstream library has it (OpenBLAS, OpenCV's `cv::hal`), so most of the hot loops already
  pick AVX2/AVX-512 kernels at runtime. Whether `-march` helps enough to justify a
  non-portable binary is a measurement, not a given.

## 5. Power, thermals and the iGPU sharing the budget

* Laptop firmware caps sustained package power (PL1) at 35–54 W for HS parts; a 16-thread
  run drops from ~5 GHz boost to whatever the cooler sustains (commonly 3.3–3.8 GHz all-core).
  A 4–5 °C cooler chassis (stand, fan) is worth more than any flag here.
* If a HIP path is ever enabled, remember the GPU draws from the same PL1: the CPU stages
  running concurrently (OpenMVS keeps CPU workers busy while the GPU densifies) slow down.
  Linux exposes the GPU DPM knob (`power_dpm_force_performance_level`); the 780M idled at
  800 MHz until forced `high` (37 → 54 tok/s prefill in a compute-bound test). [k8s.it]
* Windows: "Best performance" power mode; disable Smart App Control only if you go the
  ROCm-on-Windows route (AMD lists it as a requirement).
  [Ryzen limitations](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.1.1/docs/limitations/limitationsryz.html)

## 6. What to expect: numbers from the community benchmarks

From `OpenDroneMap/odm-benchmarks` `data/benchmarks.csv` (241 rows; defaults unless noted):

| Dataset | Machine | ODM | Time | Note |
|---|---|---|---|---|
| Brighton Beach (18 img) | Ryzen 7 8845HS, 16 thr, 8 GB, Win 11, Docker | 3.5.4 | 4 min | Zen 4 laptop, no GPU |
| Toledo (~100 img) | Ryzen 7 8845HS, 16 GB, native | 3.5.4 | 27 min | Zen 4 laptop, no GPU |
| Aukerman (77 img) | i7-11800H, 32 GB, native | 2.7.2 | 13 min | CPU only |
| Aukerman | same + RTX 3050 Ti | 2.7.2 | 11 min | GPU buys ~15% |
| Aukerman, `--pc-quality ultra` | same + RTX 3050 Ti | 2.7.2 | 1 h 0 min | quality knob dominates |
| Aukerman | Ryzen 5 5500U, 6 c, 40 GB, Docker | 2.7.2 | 9 min | Zen 2 laptop |
| Brighton Beach | i5-8250U, 16 GB, no GPU | 2.7.0 | 16 min | |
| Brighton Beach | i5-8250U + MX150 | 2.7.0 | 12 min | ~25% |
| Sheffield Cross | Ryzen 9 5900X, 64 GB + RTX 3060 | 3.5.5 | 32 min | desktop reference |
| Zoo (~500 img) | Ryzen 5 1500X, 32 GB | 2.1.0 | 4 h 14 min | old ODM, 4 cores |

Sources: [benchmarks.csv](https://github.com/OpenDroneMap/odm-benchmarks/blob/master/data/benchmarks.csv),
[odm-benchmarks README](https://github.com/OpenDroneMap/odm-benchmarks),
[ODM data benchmarks page](https://opendronemap.org/odm-data-benchmarks/).

Take-aways: (1) a 2024 Zen 4 laptop does a 100-image set in ~30 min at defaults; (2) small
laptop GPUs historically bought 15–25% end-to-end, which is the ceiling an iGPU port should be
measured against; (3) `--pc-quality` moves time by 4× per step, far more than any hardware
change. The MDPI multi-platform study concurs that ~20 cores and 64 GB is the sweet spot for
~1000 images and 8000-image sets need up to 256 GB — beyond any laptop.
[MDPI 2023](https://www.mdpi.com/2673-7086/3/3/23)

## 7. Suggested laptop profiles

```
# preview / field check (minutes)
--fast-orthophoto --feature-quality medium --matcher-neighbors 8 --max-concurrency 8 --skip-report --optimize-disk-space

# deliverable ortho + DSM, 16 GB laptop
--pc-quality medium --pc-skip-geometric --feature-quality high --max-concurrency 8 --skip-3dmodel --optimize-disk-space

# best quality overnight, 32 GB laptop, plugged in
--pc-quality high --feature-quality ultra --max-concurrency 12 --use-hybrid-bundle-adjustment --optimize-disk-space
```

Note: the ODM docs' installation page still says "a graphics card (GPU) currently has no
impact on performance" in its hardware recommendations — outdated for NVIDIA, accurate for
AMD today. [ODM docs installation](https://docs.opendronemap.org/installation/)

## 8. Measure before you tune

ODM writes `log.json` into the project directory (`opendm/log.py`) with per-stage start/end
times and memory; the console log carries the same stage boundaries. (ODM 3.6.2 has no
`--time` flag; older docs mention one.)
Use the same small public dataset (Brighton Beach, 18 images; Aukerman, 77 images — both in
the benchmarks index) as the yardstick so results are comparable to the community table:

1. Baseline: defaults, plugged in, laptop idle, `--optimize-disk-space` off, record RAM peak
   (`/usr/bin/time -v` on Linux, Task Manager on Windows) and the per-stage times.
2. Change **one** knob per run: `--max-concurrency 8`, then `--pc-skip-geometric`, then
   `--feature-quality medium`. Densification (`openmvs`) and matching (`opensfm`) are the two
   stages that should move.
3. Only then consider a `-march=znver4` rebuild or an iGPU path; if densification is already
   under a third of the total, the iGPU port's 2–4× (doc 01 §7) translates to well under 30%
   end-to-end — the same order as the historical laptop-dGPU gain in §6.
4. Re-check thermals: if all-core clocks fall below ~3 GHz mid-run (`turbostat`, HWiNFO), fix
   cooling or lower `--max-concurrency` before touching software.

## Sources

- ODM sources: `opendm/config.py`, `stages/openmvs.py`, `opendm/osfm.py`, `opendm/concurrency.py`, `opendm/gpu.py`, `SuperBuild/cmake/External-OpenMVS.cmake`, `scripts/activate-msvc.bat`, `pixi.toml`
- https://docs.opendronemap.org/arguments/ , https://docs.opendronemap.org/installation/
- https://github.com/OpenDroneMap/odm-benchmarks , https://github.com/OpenDroneMap/odm-benchmarks/blob/master/data/benchmarks.csv , https://opendronemap.org/odm-data-benchmarks/
- https://www.mdpi.com/2673-7086/3/3/23
- https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
- https://github.com/conda-forge/blas-feedstock/blob/master/README.md , https://anaconda.org/conda-forge/aocl-blas
- https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-7-7840hs.html , https://www.notebookcheck.net/Radeon-780M-vs-Vega-7_11564_10098.247598.0.html
- https://rocmdocs.amd.com/en/develop/how-to/system-optimization/strixhalo.html
- https://www.k8s.it/posts/running-a-local-llm-on-amd-radeon-780m-gfx1103-rocm-and-the-gpu-that-wasnt-supposed-to-work/
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.1.1/docs/limitations/limitationsryz.html
