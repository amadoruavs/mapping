# Python parallelism and accelerator validation — 2026-09-16

This is a tested Python orchestration update, not a full GPU/NPU rewrite of ODM.
No new XDNA2 kernel is included. A Mac cannot compile/run-test the laptop's HIP
runtime or XDNA driver. Target verification is required before describing those
backends as validated. Linux alone never qualifies an accelerator for selection.

## Installed behavior

- ODM uses a bounded per-image OpenSfM feature scheduler. It retains upstream
  Ultra DSPSIFT, precision and feature settings. No unbounded decoded-image queue.
- ODM CPU FSR tiles run concurrently within the requested thread budget. GDAL
  reads/writes remain serialized; exact halo handling and geographic metadata
  are preserved. HIP FSR startup validation now requires zero pixel difference,
  rather than the previous allowance of one code value.
- MapRepair prepares photographs/features concurrently, then applies alignment,
  witness and repair decisions in the existing order. Its grading strips run
  concurrently with unchanged math and halos.
- Exact integer alpha compositing is implemented in Python for NumPy, PyTorch
  Metal and PyTorch ROCm. Auto selection checks pixel parity and benchmarks the
  transfer-inclusive operation before choosing a GPU. Small inputs stay on CPU.
  Missing runtime or failed auto-device initialization falls back to CPU.
  Explicit `metal`/`hip` requests fail rather than silently report a GPU success.
- Reconstruction, sparse solvers, meshing and texturing retain native upstream
  implementations. OpenMP passive waiting is requested to reduce idle spinning.
- Every sixth unique photo, Ultra, the 24k feature target, mesh settings and
  2.3 cm/pixel remain unchanged. FSR remains display upscaling, not measured detail.

## What passed on the Mac

- Four real photographs at 3840px feature processing / 24k target: serial and
  new parallel DSPSIFT points, descriptors and colors are bit-identical.
  Serial 63.67 s; new parallel 22.25 s. **Existing upstream parallel also took
  22.28 s**, so this establishes parity and bounded scheduling, not a material
  speedup over upstream parallel extraction. Timings were under concurrent load.
- MapRepair original-vs-new on six real photographs: repaired pixels, ownership
  and geometry decisions identical; 4.84 s original versus 2.97 s new for this
  scan/repair test. A synthetic known-displacement case also matched exactly.
- Grading JPEG bytes match the original implementation with CPU and Metal
  compositing, including sharpening and non-white backgrounds.
- Metal: 100 randomized batches, 25,048,960 input pixels, zero pixel differences
  for compositing and nearest-neighbor alpha enlargement. Live device allocation
  returned to zero. Driver/cached allocation growth is measured separately.
- A focused 512x1024 composite benchmark: CPU 13.07 ms versus Metal 3.34 ms.
  This is one kernel, not a map-generation speedup.
- HIP FSR device equations compiled as host C++: 36 cases exactly match CPU.
  **This is a source-math check, not HIP compilation or AMD GPU execution.**
- CPU FSR tile seams, alpha and georeference tests pass. Native Qt integration
  and the MapRepair suite pass. Scheduler stress ran 1,000 jobs with a 24 MiB
  estimated reservation ceiling and successfully propagated injected failure.
- Fresh six-photo Ultra ODM pipeline completed including GeoTIFF and FSR in
  185.65 s with another 94-photo job running. This is a smoke test, not a clean
  benchmark. All six feature point/descriptor arrays matched the historical run.
  **The historical orthophoto grid differed**, so full reconstruction pixel
  parity is NOT established. Historical concurrency was 8 versus this run's 4;
  downstream stochastic/numerical reconstruction must be controlled separately.

## Completed 36-photo Ultra benchmark

All 36 already-deduplicated originals were used directly, with no deduplication
or stride, eight CPU workers, Ultra DSPSIFT/dense reconstruction and 2.3 cm/pixel.
End-to-end wall time including FSR: **522.75 s (8 min 43 s)**. Sampled process-tree
RSS peaked at **4.39 GiB**. The older 94-photo job was suspended during this run
and automatically resumed afterward.

Sampled stage times: OpenSfM 169.84 s; dense MVS 288.87 s; point filtering 8.25 s;
meshing 15.46 s; texturing 16.63 s; georeferencing 8.23 s; orthophoto 2.06 s;
reporting 7.21 s. Boundaries have roughly one-second sampling precision.
All 36 cameras registered in one reconstruction, with 83,722 sparse points and
0.982 px reported reprojection error (not surveyed positional accuracy).
Base GeoTIFF: 2070x1499; FSR: 4140x2998. FSR bounds and nearest-2x alpha are exact.
Full reconstruction pixel parity is still not established.

Reports: `output/odm_macos_ultra36_parallel_20260916/{benchmark,verification}.json`
in the development workspace. Photos and results are excluded from the archives.
The new target is under 300 s for roughly 250 already-deduplicated 4K images.
**That target is not met or verified.** See `TARGET_250.md` for the performance
and quality acceptance criteria. There is no remote access to the Fedora laptop;
its included target verification commands must be run there.

## 640px experiment

Coarse matching is an isolated experiment, not production configuration.
On one pair, coarse SIFT produced a 2.17px held-out median residual; original
resolution produced 0.85px. These use different held-out match sets, not ground
truth. Full-resolution verification always runs; baseline pairs are not discarded.
No accuracy-preserving end-to-end speedup from this strategy has been established.

## Memory limits and scope

The Python scheduler reserves estimated scratch and retained results before
submitting each task; it bounds queued futures and joins workers on errors or
cancellation. Defaults use the smaller of 20 GiB and 60% of currently available
memory, respecting visible cgroup limits. Meshing also has its prior RAM-based
worker limit. Pixel operations use bounded row batches.

These estimates do not intercept every native allocation, account for every
other application's future allocation, or impose one global budget across both
apps. The stress tests do not prove that a complete native reconstruction can
never exhaust RAM. Avoid concurrent heavy jobs on the 32 GB shared-memory laptop;
allocation failures must be reported rather than sacrificing resolution.

## Controls

- `ADM_PARALLEL=0`: upstream ODM feature scheduling; serial MapRepair preparation,
  grading and ODM FSR tile dispatch. Native libraries may still use their configured
  threads; this is not a global one-core mode.
- `ADM_PARALLEL_WORKERS=4`: bound Python task concurrency.
- `ADM_MEMORY_MB=8192`: additional Python working-memory ceiling; it cannot raise
  the limit above the available-memory fraction.
- `ADM_PIXEL_BACKEND=cpu|auto|metal|hip`: compositing backend. Auto is default.
- `ODM_FSR_BACKEND=cpu|auto|hip`: separate ODM FSR backend.

PyTorch is optional and not added to the base app requirements. A compatible
ROCm PyTorch build is needed for `ADM_PIXEL_BACKEND=hip`; the HIP FSR library uses
HIP directly and does not require PyTorch. Do not install a random PyTorch wheel
and assume it includes ROCm. Existing MapRepair native FSR remains CPU.

## On-laptop validation

After installing/updating ODM:

```bash
cd ~/Applications/ODM-Fedora
./verify-acceleration.sh             # CPU tests, diagnostics, HIP source-math check
./verify-acceleration.sh hip         # compile gfx1150 HIP FSR and require real-device parity
# Optional, only with a compatible ROCm PyTorch already installed in this env:
./verify-acceleration.sh hip-python
```

After installing/updating MapRepair:

```bash
cd ~/Applications/MapRepair
./verify-acceleration.sh             # CPU and application tests
./verify-acceleration.sh hip         # requires ROCm PyTorch in MapRepair's runtime
```

Reports go under `fedora/validation/target` or `validation/target`. Device-required
commands return an error if the requested hardware/runtime is absent. They do
not relabel CPU execution as an accelerator pass.

For real-image DSPSIFT parity on Fedora:

```bash
.pixi/envs/default/bin/python fedora/validate_parallel_features.py \
  --project /path/to/completed/project/flight/opensfm \
  --output /path/to/new-empty-validation-folder
```

## Upstream documentation

AMD HIP, ROCm, amd/xdna-driver and Xilinx/mlir-aie documentation was downloaded
at pinned Git commits. `AMD_DOCS_MANIFEST.json` records exact URLs and SHA256s.
The development snapshots are in `acceleration/docs/upstream/` in the ADM workspace.
HIP streams require explicit dependency ordering; obtaining a CPU NumPy result
from the Python GPU operation synchronizes before the data is used. XDNA2 needs
its own AIE2P compiler/runtime path; it is not a HIP target and a Metal test does
not validate XDNA execution. Kernel/driver builds on Fedora remain target work.
