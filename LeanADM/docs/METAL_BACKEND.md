# Metal and MPS pipeline

`--backend metal` runs the accelerated pipeline on macOS. The CPU backend remains
the numerical reference. Hardware validation uses an Apple M4 Pro; cooperative
matching currently requires a 32-lane Metal SIMD group.

```sh
.venv/bin/python -m adm.pipeline /Users/avneh/Downloads/101SIYI_IMG \
  --output output/101SIYI_IMG_metal \
  --config output/101SIYI_IMG/run_config.json \
  --exclusions output/101SIYI_IMG/exclusions.json --gps-centers --backend metal

.venv/bin/python scripts/validate_metal.py
.venv/bin/python scripts/benchmark_metal.py
```

The first call builds an Objective-C dynamic library with `xcrun clang` and compiles
MSL through the Metal runtime. Subsequent calls reuse the device, command queue,
compute pipelines, MPS matrix operators, Gaussian filters, and bounded texture cache.
No standalone Metal compiler, PyObjC, PyTorch, or new pip dependency is required.
A process sandbox may hide the GPU; the backend reports this explicitly.

Add `--recompute-warps` when disk space or intermediate I/O is a constraint. Metal
reruns the fused warp during source selection and preserves the float16 confidence
rounding policy, producing the same result as its disk-cache mode. It creates no
`warps/` directory. Raw RGB tiles, coverage, source IDs, final maps, and reports are
still exported. Existing output directories are not cleaned automatically.

## Stage implementation

| Work | Implementation |
| --- | --- |
| Selection, EXIF, JPEG decode | Existing CPU implementation |
| SIFT extraction | OpenCV CPU |
| Descriptor matching | Tiled `MPSMatrixMultiplication`; MSL descriptor norms and SIMD top-two L2 reduction |
| Corner verification | MSL projective rectification and cooperative 128-thread NCC reduction; existing CPU acceptance and geographic checks |
| RANSAC, global alignment | Existing float64 CPU geometry and sparse SciPy solver |
| 21×21 RGB index data | MSL channel conversion, tile rearrangement, and zero padding; CPU geographic coordinates and SQLite |
| Mosaic warping | MSL inverse homography sampling fused with source edge-confidence sampling |
| Mosaic mean and sampling | Resident MSL accumulation, coverage, mean, correlated Gaussian-field CDF, Gumbel score, selected pixels, and source IDs |
| Seam correction | RGBA residual/coverage packing, two separable `MPSImageConvolution` passes, MSL normalization |
| Flat lighting | One fused MSL sRGB decode, lighting, linear export, and sRGB encode pass |
| Terrain lighting | MSL normals and early-exit shadow rays; MPS visibility blur; resident environment evaluation, selection, and export |
| PNG, HDR, NPY exports | CPU codecs and existing file formats |

MPS handles dense products and separable convolution. The sparse float64 alignment
solve remains on the CPU; a dense float32 replacement would change conditioning
and memory complexity. Flat-ground environments share the same lighting equation,
so their redundant evaluation and random selection collapse to one pass.

## Data movement and memory

`adm/metal.py` binds the C ABI with `ctypes`. Metal owns shared buffers; NumPy views
them directly after command completion and retains the underlying allocation.
Inputs copy once into shared storage. GPU intermediate results do not pass through
subprocesses or temporary pixel files. Existing disk-backed warped observations
remain deliberate pipeline artifacts for the compositor's second pass.

MSL and MPS encodes share a command buffer until a required CPU read. Tracked
resource hazards order texture reuse and buffer dependencies. A backend is confined
to its creating thread; another thread can create its own backend.

- Descriptor products occupy at most `512 × target_feature_count` float32 values;
  descriptors and norms stay resident across pairs.
- MPS filters reuse three private RGBA32Float textures, each at most
  `(2048 + 2r) × (256 + 2r)`. Reflection halos match OpenCV's `BORDER_REFLECT_101`,
  including one-pixel image dimensions and tile boundaries.
- Terrain candidates cover 128 output rows at a time. Shadow rays sample the full
  input terrain, so distant occluders remain visible across stripes.
- The compositor retains full-map reductions: float4 mean, seam input, blurred
  output, BGRA selection, coverage, and source IDs total roughly 56 bytes per output
  pixel, plus a warped ROI and scratch. Memory is bounded in image count, **not** map
  area. Larger mosaics still need an output-tile scheduler.
- Existing float16 confidence files retain the CPU policy. Compute uses float32;
  global geometry uses float64. Device allocation limits produce explicit errors.
- Live device memory is checked before every allocation against 60% of physical
  memory (`ADM_METAL_MEMORY_LIMIT_MB` overrides). Exceeding it raises `MemoryError`
  in this process rather than exhausting unified memory, which panics macOS.
- Every exported C function drains its own autorelease pool. Python calls in on a
  thread without one, and an autoreleased retain with no pool leaks the buffer.
  `gpu_report.json` records `device_allocated_bytes` at the end of a run.

## Numerical behavior

OpenCV 4 uses a 5-bit interpolation table; OpenCV 5 uses full bilinear precision.
Shaders follow the installed version. Runtime math uses safe mode; sun vectors are
computed once per environment.

Metal computes exact descriptor neighbors; CPU FLANN is approximate. Accepted
matches and fitted transforms can therefore differ across backends. The 0.70 ratio
test, RANSAC criteria, 2% delta-RMSE threshold, uniqueness requirement, and geographic
checks are unchanged. Match fingerprints include backend, shader/bridge content,
and OpenCV version.

Source selection preserves seeded coarse Gaussian samples and uses a normal CDF
approximation with absolute error below 7.5e-8. Small arithmetic differences can
change a categorical winner near a tie. Terrain sampling advances PCG64 to preserve
the CPU's environment-major random stream across stripes. Repeated backend runs
are deterministic in the validation suite.

Float operations can differ by a few ulps; an sRGB value near a half-integer may
quantize one code value differently. Tests use explicit tolerances for radiance,
interpolation, selected IDs, coverage, and encoding. They do not require byte equality
between independently fitted CPU and GPU mosaics.

## Validation and measurement

`scripts/validate_metal.py` runs the original lighting equation check, CPU tests,
and hardware tests for accelerated stages, including geographic processing and
resume. It writes `output/metal_validation.json`. `--quick` runs only the original
check. Regular unittest discovery skips hardware tests unless `ADM_TEST_METAL=1`.

The benchmark warms each operation and records repeated wall times, including
uploads and synchronization, excluding image codecs. Descriptor comparison uses
exact OpenCV BF matching so both sides perform the same task, rather than FLANN.

Initial M4 Pro measurements at 1536×1152, three repetitions:

| Stage | CPU median | Metal median | Ratio |
| --- | ---: | ---: | ---: |
| Flat lighting + encoding, 5 environments | 681.95 ms | 2.72 ms | 250.6× |
| Gaussian σ=7, four channels | 11.36 ms | 9.43 ms | 1.2× |
| Warp + confidence | 12.63 ms | 2.66 ms | 4.7× |
| Exact 7000×7000 descriptor matching | 57.35 ms | 6.44 ms | 8.9× |
| Terrain, 512×384, 5 environments × 48 shadow steps | 228.53 ms | 13.58 ms | 16.8× |

Flat-lighting acceleration includes removing identical environment passes and large
intermediate arrays; this does not mean GPU arithmetic alone is 250× faster.
These are stage timings, not full-pipeline speedups. Raw samples are recorded in
`output/metal_benchmark.json`.

Per-run `gpu_report.json` records the device, submissions, dispatches, MPS operations,
and GPU time. GPU time excludes queue wait, Python work, and I/O; `report.json`
contains full-pipeline wall time and identifies remaining CPU work.

References: Apple's [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication),
[MPSImageConvolution](https://developer.apple.com/documentation/metalperformanceshaders/mpsimageconvolution),
and [shared storage](https://developer.apple.com/documentation/metal/mtlstoragemode/shared);
OpenCV's [migration guide](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration).
