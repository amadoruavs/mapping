# GPU acceleration roadmap for ODM on laptops (Apple M4 Pro first, AMD APU second)

Written 2026-09-13. Synthesises `anatomy/`, `cpu-baseline/`, `metal/`, `radeon/`.
Research only; no code changed. Effort figures are estimates from the docs, not measurements.

## Headline findings

1. **ODM on this Mac is 100 % CPU today.** Every GPU touchpoint is CUDA-only
   (`../00-context/03-gpu-touchpoints-today.md`). `--no-gpu` is a no-op on macOS.
2. **Upstream openMVS already has a Metal PatchMatch backend.** cdcseacave/openMVS PR #1279
   (merged 2026-06-26, ~1 kLOC `PatchMatchMetal.{h,mm,metal}`, MSL compiled at runtime with
   `newLibraryWithSource`, so no Xcode needed, CMake auto-enables it on APPLE, CLI flag
   `--gpu-device` replaces `--cuda-device`). ODM's pinned fork (tag 355, based on a 2023-08
   upstream) and the pending ODM PR #2074 (openMVS 2.4.0) both predate it.
   → `metal/01-openmvs-on-metal.md`, `anatomy/01-openmvs-cuda-anatomy.md`.
3. **PopSift has an open HIP port from AMD** (alicevision/popsift PR #186, June 2026, tested on
   a Strix Halo APU on Windows). No OpenMVS HIP fork exists; the Metal backend is the template
   for one. → `radeon/01`, `radeon/02`.
4. **GPU SIFT only matters with `--feature-type sift`.** ODM's default is `dspsift` (CPU), so
   PopSift/Metal-SIFT is a secondary win. → `anatomy/02`, `anatomy/03`.
5. **Densification is the prize.** OpenMVS `DensifyPointCloud` dominates wall time in a normal
   3D run; upstream measured 5–6× vs CPU for the Metal backend on Apple silicon (4-image
   validation only, real aerial datasets unverified). Everything else (BA, Poisson, texturing,
   DEM, ortho) is CPU and stays CPU. → `cpu-baseline/00-pipeline-stage-map.md`.
6. **Host facts verified on this machine**: 8P+4E cores, 16-core GPU, 24 GB, Metal 4,
   Command Line Tools only (no `xcrun metal`), OpenCL 1.2 still enumerates the GPU,
   runtime MSL compile works, `pixi` not installed, ~11 GB free disk (build needs 15–20 GB).

## Phased plan (Mac)

### Phase 0 — baseline (no GPU work) · ~1 day
* Install pixi, `pixi install`, `pixi run build` (CI takes 34–49 min on macOS), `pixi run smoke`.
* Free disk first (build + env need 15–20 GB).
* Check the known macOS trap: PoissonRecon may fail to load `libomp.dylib` because
  `Makefile.macos` links `-lomp` without an rpath and `/bin/sh` strips `DYLD_*` under SIP
  (`cpu-baseline/01`, open question 1). Verify with `otool -l`.
* Run the benchmark plan on Toledo (87 × 12 MP) and Aukerman (77 × 18 MP) with the three
  presets in `cpu-baseline/03`; record per-stage times from `benchmark.txt`/`log.json`
  (`cpu-baseline/04`). This is the number every later step is judged against.

### Phase 1 — free CPU wins · ~1 day
* `libblas=*=*accelerate` in pixi (low yield: OpenSfM BA is `DENSE_SCHUR`/Eigen, but free).
* Make sure OpenMP is actually found/linked for OpenMVS, PoissonRecon, texrecon.
* `-mcpu=apple-m4` via `CXXFLAGS` in pixi activation (single-digit % at most).
* `--max-concurrency 8` on 24 GB at `--pc-quality high`; OpenSfM's memory throttle is dead
  on macOS (`memory_available()` shells out to Linux `free`). → `cpu-baseline/02`, `03`.

### Phase 2 — Metal densification · ~1–2 weeks
* Rebase (or cherry-pick PR #1279 onto) ODM's openMVS fork; resolve ODM-specific commits
  (scalable texturing, `INSTALL_USE_SUBDIR`). → `metal/01` §5.
* ODM plumbing: `External-OpenMVS.cmake` option, `stages/openmvs.py` `--cuda-device` →
  `--gpu-device`, a `has_metal()` probe in `opendm/gpu.py`, a pixi `osx-metal` feature
  setting `ODM_ENABLE_METAL`. → `metal/05`.
* Validate accuracy and memory on real aerial datasets vs the Phase 0 baseline; watch the
  24 GB unified budget (OpenMVS ~76 B/px + 4 B/px/view per depth map, `anatomy/01`).

### Phase 3 — ONNX on CoreML · ~2 days
* Only affects `--sky-removal` / `--bg-removal`. conda-forge onnxruntime gained CoreML at
  1.26; the pin is 1.22.2 (CPU). Provider list change in `skyfilter.py`/`bgfilter.py`,
  `ModelFormat=MLProgram` to avoid the fp16 pitfall. Expect 2–4× on that stage. → `metal/03`.

### Phase 4 — features (optional) · weeks
* Metal SIFT: no drop-in exists; colmap-metal is the reference; or learned features via
  ONNX/CoreML behind a new OpenSfM `feature_type`. Only worth it for `--feature-type sift`
  users or if Phase 0 shows feature extraction is a bottleneck. → `metal/02`, `anatomy/02`.

### Phase 5 — `odm_orthophoto` as a Metal render pass (optional)
* Single-threaded CPU rasteriser today; a natural Metal render target job. → `metal/04`.

## AMD integrated Radeon (Linux first, Windows second)

* **Support reality**: gfx1150/gfx1151 (890M / Strix) are official in ROCm 7.2 Linux and the
  Windows HIP SDK 7.2; gfx1103 (780M) is unofficial (compiles natively, community override
  `HSA_OVERRIDE_GFX_VERSION=11.0.2`, no rocBLAS/MIOpen kernels, one reported hard lock);
  gfx90c is a legacy override case. → `radeon/00`.
* **Path**: HIP port of OpenMVS PatchMatch (hipify; only textures + curand + default stream
  are used; the Metal backend shows the structure), 4–7 person-weeks, honest expectation
  2–4× on densify, 1.2–1.6× end-to-end. → `radeon/01`. Track PopSift PR #186 → `radeon/02`.
* **ONNX**: ROCm EP was removed in ORT 1.23; MIGraphX wheels exist (crash on gfx1103);
  DirectML is in maintenance mode; WebGPU EP is the wildcard. → `radeon/03`.
* **Build**: system ROCm at `/opt/rocm` rather than conda-forge `hip-*`; an `amd` pixi
  feature mirroring `gpu`; Docker needs `--device=/dev/kfd --device=/dev/dri`. → `radeon/04`.
* **Do the CPU tuning first** (`-march=znver4`, thread counts, iGPU steals RAM). → `radeon/05`.

## Decision table

| Workload | Mac M4 Pro | AMD APU Linux | AMD APU Windows |
|---|---|---|---|
| Densify (OpenMVS) | Metal backend exists upstream — rebase | HIP port needed (weeks) or ZLUDA (fragile) | HIP SDK + same port |
| SIFT (only `--feature-type sift`) | no port; ONNX/CoreML features later | PopSift HIP PR #186 | PopSift HIP PR #186 |
| Sky/bg ONNX | CoreML EP (ORT ≥ 1.26) | MIGraphX (gfx1150+) | DirectML (maintenance) |
| BA, Poisson, texturing, DEM, ortho | CPU (Accelerate, OpenMP) | CPU (AOCL/OpenBLAS) | CPU |

## Consolidated open questions

See `../research-log.md` §Open questions (merged list from all six research tracks).
