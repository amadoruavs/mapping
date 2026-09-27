# Ultra map update: mixed terrain and difficult photographs

The 306 images are Google Maps screenshots used as a visual terrain reference.
All 306 local files are 640x640. They are not a real multiview 4K drone flight,
and their embedded camera/GPS metadata cannot establish 3D reconstruction accuracy.
The profile must handle real images of grass, pavement/runway, trees, tents,
cars and roofs, including exposure changes, motion blur and imperfect alignment.

## What changes

- Independent per-photo feature extraction now consumes completed tasks as soon
  as they finish. Slow/complex vegetation images no longer block dispatch behind
  completed easier images. The same bounded memory reservations and detector run.
- A byte-bounded cache reuses decoded feature arrays across matching and
  reconstruction. Concurrent requests share one load. Pair selection, FLANN,
  RANSAC, descriptors and solver criteria are unchanged. Cache defaults to the
  smaller of 4 GiB and one quarter of the existing Python memory budget.
- Initial undistortion workers share one camera-map computation. The native
  mapping equations, interpolation and exported mappings are unchanged.
- PDF/statistics-chart generation is omitted by default. Set ODM_FULL_REPORT=1
  before launch when you need the complete diagnostic report. This avoids work
  that does not change map pixels; it also omits those report statistics.
  A small quality_summary.json still lists registration counts and missing photos.

Full dense depth, geometric consistency iterations, outlier rejection, camera
optimization, mesh detail, original photo texturing and seam leveling remain.
The map-only 2.5D path already omitted the separate full 3D model. In the default profile, no flat-plane
assumption, coarse sparse-surface replacement, reduced feature/depth resolution,
640px reconstruction shortcut, quantized inference or neural inpainting is added.
FSR remains a separate display upscale; the original GeoTIFF remains available.
Default sampling is still every sixth unique photo; the direct benchmark CLI
bypasses both sampling and deduplication for already-selected input.

## Controls and target checks

```bash
# Normal app: Ultra, optimized work reuse, omit PDF/diagnostic charts
./run-fedora.sh
# Retain the complete quality report
ODM_FULL_REPORT=1 ./run-fedora.sh
# Reference runtime for comparisons: old dispatch, no new caches, full report
ODM_TERRAIN_RUNTIME=0 ./run-fedora.sh
# Optional smaller decoded-feature cache; zero disables retained values
ODM_FEATURE_CACHE_MB=1024 ./run-fedora.sh

./verify-acceleration.sh
./verify-acceleration.sh hip
```

The HIP check is for FSR, not a dense-reconstruction GPU backend. Experimental XDNA2 compositing and hybrid sources are now included; see AMD_KERNELS.md. CPU fallback remains the default when real-device checks fail.

Real-photo cache, descriptor and undistortion parity:

```bash
.pixi/envs/default/bin/python fedora/validate_terrain_runtime.py \
  --project /path/to/completed/flight/opensfm \
  --output /path/to/new-empty-parity-folder
```

Noise, exposure, motion blur and JPEG degradation at original image dimensions:

```bash
.pixi/envs/default/bin/python fedora/validate_high_error.py \
  --project /path/to/completed/flight/opensfm \
  --output /path/to/new-empty-high-error-folder
```

Tests compare unchanged feature arrays and run a controlled non-planar geometric
outlier fixture. They do not prove surveyed accuracy, model real GPS/rolling
shutter errors, or promise recovery from photographs without useful overlap.

For the roughly 250-image performance target, use the same already-deduplicated
originals and separate fresh output directories. Do not run the trials together:

```bash
ODM_TERRAIN_RUNTIME=0 python3 fedora/benchmark_selected.py \
  --input /path/to/250-originals --output /path/to/new-reference \
  --threads 8 --expect-images 250
python3 fedora/benchmark_selected.py \
  --input /path/to/250-originals --output /path/to/new-optimized \
  --threads 8 --expect-images 250
```

Compare benchmark.json and actual maps. Do not compare a cached resume against a
fresh run. The under-five-minute target has not been demonstrated on the Ryzen.
Runtime estimates/cache limits are not a global native-memory cap.

## Measured local component checks

306 screenshot features: all decoded arrays and sampled descriptor matches exact.
Three repeated raw-feature passes: 1.463 s -> 0.590 s. This is a deliberately
cache-heavy component workload, not a 2.5x reconstruction gain. The original
camera-map creation ran eight times on concurrent first requests; the new path
ran once, with identical arrays. Repeated-loading tests on the 36 real 4K photos
also matched exactly: 2.165 s -> 1.020 s. These are single trials, not medians.

Controlled geometric correspondence tests passed at 15/35/50% injected outliers
and 1/2/3 px coordinate noise: precision and recall each above 99% in the three
fixtures. This validates existing filtering, not a new robustness algorithm.

The new 36-photo Ultra run took 495.44 s (previous 522.75 s), with all 36
registered in one component, peak sampled process-tree RSS 4.41 GiB. Geometry
is stochastic; this single comparison is not a hardware speed guarantee.
Four real 4K photos degraded with noise, exposure reduction, motion blur and
JPEG compression produced identical feature arrays under both schedulers.
Their timings were 12.035 s and 12.030 s; no material speed gain in this tiny test. Full end-to-end pixel parity is not claimed;
upstream reconstruction is sensitive to stochastic/numerical execution order.
