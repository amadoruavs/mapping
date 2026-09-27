# Target: 250 already-deduplicated 4K photographs in under five minutes

This is a performance target, not achieved performance. The completed 36-photo
Ultra run took 522.75 seconds on an M4 Pro with eight workers. It is not a Ryzen
measurement and cannot establish a 250-photo runtime. The target counts all 250
selected originals: no deduplication, stride, reduced feature/depth resolution,
reduced geometric iterations or sparse-surface substitution during the benchmark.

## Pipeline and next optimization priorities

| Stage | 36-photo Mac time | Accuracy-preserving direction to investigate |
|---|---:|---|
| Image metadata, decoding, features | Included in OpenSfM | Bounded per-image parallelism, reuse decoded data; unchanged DSPSIFT arrays |
| Matching, tracks, reconstruction, bundle adjustment, undistortion | OpenSfM total 169.84 s | Profile separately; retain candidate pairs and solver criteria; accelerate independent work before changing solvers |
| Dense depth estimation, geometric consistency, fusion | 288.87 s | Main GPU-port candidate; retain full depth resolution, sampling, iterations and visibility checks |
| Point filtering | 8.25 s | Parallel spatial queries with stable decisions |
| Surface meshing | 15.46 s | Bounded tile work; retain mesh resolution and topology criteria |
| Texture selection and seam processing | 16.63 s | Profile shared-lock contention; preserve patch selection, color and seam equations |
| Georeferencing | 8.23 s | Preserve coordinates/CRS and original transforms |
| Orthophoto rasterization | 2.06 s | Tile parallelism with identical sampling and halos |
| Reports and FSR/export | Remaining time | Overlap independent reporting/export only within shared memory budget |

Times are sampled stage boundaries, approximately one-second precision. These are
not proposed 250-image budgets. OpenSfM plus dense MVS account for about 88% of
this measured run; optimizing FSR or UI code alone cannot meet the target.

Python orchestrates native work but does not replace the main native kernels.
A compatible HIP dense-reconstruction implementation does not currently exist
in this bundle. A Metal prototype and host math test cannot certify its Ryzen
implementation. XDNA2 has a separate AIE2P toolchain, memory layout and precision
constraints; no NPU kernel is included. Do not choose a device merely because
Linux is detected. Detect the actual runtime, validate and measure the device.

## Why fast orthophoto is not the acceptance baseline

ODM fast orthophoto retains sparse SfM but skips dense MVS. In this source tree,
stages/odm_meshing.py also multiplies the DSM grid spacing by eight and selects
Poisson instead of gridded meshing. This is an algorithm/geometry change, not
simply the same reconstruction scheduled faster. Both methods can texture from
original photos and export sharp-looking pixels; sparse height errors can still
misplace those pixels. No percentage quality loss has been measured here.

Official description: https://docs.opendronemap.org/arguments/fast-orthophoto/
FSR improves display sampling/sharpness; it does not recover missing surveyed
geometry or prove additional resolved ground detail.

## Acceptance before promoting an accelerator

1. Cold complete run, all 250 originals, same dataset/quality/output resolution,
   start-to-finished GeoTIFF and FSR under 300 seconds on the actual 32 GB Ryzen
   laptop. Record hardware, versions, thermals/power mode, stage times and memory.
   Repeated cache-resume time is a separate result.
2. Require bit-identical output for unchanged deterministic pixel kernels.
   Compare CPU repetitions first for native reconstruction variability. Compare
   camera registration, retained observations, reprojection and independent
   checkpoints when available; reprojection alone is not positional accuracy.
3. Compare maps at identical geographic grid and sampling. Inspect grass/detail,
   roof/line edges, tall objects, seams, holes and alpha. Do not count FSR output
   dimensions as new reconstructed detail. No unapproved numerical tolerance or
   quality-setting reduction qualifies as parity.
4. Stress full native stages with representative 250-image data and record
   process-tree RSS, accelerator allocations and system pressure. Current Python
   allocation estimates are not a global hard native-memory bound.
5. Pass target CPU/HIP verification on the laptop. There is no remote access;
   the user will run verify-acceleration.sh there. Fail explicit GPU requests if
   unavailable; retain tested CPU fallback for normal auto selection.

## Direct on-laptop timing, bypassing selection

After ./install-fedora.sh, with no other heavy job running:

```bash
python3 fedora/benchmark_selected.py \
  --input /path/to/already-selected-250-photos \
  --output /path/to/new-empty-benchmark-folder \
  --threads 8 --expect-images 250
```

This measures the currently shipped CPU reconstruction, followed by CPU FSR.
It does not test a nonexistent dense HIP or NPU backend. benchmark.json records
elapsed time, sampled process-tree RSS, options and stage times. The output path
must not already exist. Run the separate HIP verification before using HIP FSR.
