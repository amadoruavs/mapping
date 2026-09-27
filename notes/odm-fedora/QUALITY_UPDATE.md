# Ryzen AI 9 365 / 32 GB quality update

Default resolution is **2.3 cm per pixel**, not pixels per cm. Exact duplicates
are removed, then unique photos 1, 7, 13, … are retained, unchanged.

Reconstruction now uses DSPSIFT ultra, a 24,000 minimum feature target, ultra
dense reconstruction, 500,000 mesh vertices and octree depth 12. Camera
calibration, terrain reconstruction and texture seam leveling stay enabled.
The original GeoTIFF remains available; a separate lossless 2× FSR GeoTIFF is
created automatically and downloadable in the web UI. Its pixel spacing is
half the actual original spacing, with identical geographic bounds and mask.
ODM retains its GSD limit: requesting smaller pixels cannot force new detail.
FSR1 EASU reconstructs edges; restrained RCAS (.25) adds crispness without
changing the underlying measured reconstruction. It cannot recover missing
coverage or guarantee fine detail absent from the photos.

## Speed and memory

- Native dependencies are rebuilt with `-O3 -march=native -mtune=native
  -fno-math-errno` inside each CMake dependency project, including OpenSfM,
  OpenMVS and texturing. Build on the laptop where the binaries will run.
- FSR uses explicit 128-bit RGB SIMD for RCAS, hoists pass selection out of
  pixel loops, and uses parallel CPU rows. Fast-math is not enabled.
- Default concurrency reserves 8 GiB, budgets roughly 3 GiB per worker and
  caps at 8 workers. This is a heuristic, not a guarantee against OOM with
  arbitrarily large photos. Ultra reconstruction is substantially more
  expensive than medium; preserving quality takes priority over runtime.
- FSR reads overlapping 512-pixel tiles, bounds GDAL cache to 256 MiB, and
  writes compressed tiled GeoTIFF incrementally. No full upscaled raster
  needs to fit in RAM. Output is atomic; original GeoTIFF is never overwritten.
- Existing engine downloads are reused. Changed build settings trigger a
  rebuild. Existing reconstruction projects reject changed engine fingerprints;
  choose a new output folder for the new quality profile.

## ROCm / HIP

AMD's [Ryzen Linux compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)
lists Ryzen AI 9 365 / gfx1150 under ROCm 7.2.1, with Ubuntu support. This is
not a certification of this Fedora installation. Fedora's HIP package is an
optional target-machine setup:

```bash
sudo dnf install rocm-hip-devel
./fedora/finish/build-hip.sh
.pixi/envs/default/bin/python fedora/finish/validate.py --hip --output hip-validation.json
```

HIP support here accelerates **FSR only**. OpenSfM/OpenMVS reconstruction remains
CPU; installing ROCm does not turn their CUDA backend into HIP. The optional
HIP kernel targets gfx1150, uses FP32 EASU/RCAS, and must pass a CPU parity check
before auto-selection. A missing compiler/runtime or failed startup parity
check falls back to CPU with a log message. `ODM_FSR_BACKEND=cpu` forces CPU;
`ODM_FSR_BACKEND=hip` requires HIP and fails explicitly if unavailable.
Runtime errors during a selected HIP export fail the job; restarting can use CPU.

The HIP code could not be compiled/run on this Apple Silicon development host.
Treat it as experimental until the target-machine validation passes. No Ryzen
speedup or full-reconstruction comparison with Docker has been measured.
Native execution alone is not proof of faster reconstruction.

[AMD FSR documentation](https://gpuopen.com/fidelityfx-superresolution/),
[HIP compiler flags](https://rocm.docs.amd.com/projects/llvm-project/en/develop/reference/rocmcc.html),
and [GCC optimization options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
were checked for this update.

## Quick local validation

On Apple Silicon: C++ application builds; Qt pipeline/localhost integration
tests pass (Linux process cleanup test is skipped). Real local `odm.png`
877×678 -> 1754×1356 FSR works. Alpha, tile seams and geographic bounds are
exact in synthetic fixtures. SIMD output is byte-identical to the pre-change
CPU reference on that real map. Seven interleaved measurements: reference
median 0.04599 s, SIMD 0.04555 s, approximately 1% improvement on this host;
this small difference is not evidence of a material Ryzen speedup. A six-photo native Mac Ultra smoke test now passes. Full reconstruction pixel
parity and installation/GPU execution on Fedora remain unverified; see
PYTHON_PARALLEL_UPDATE.md.
