# ADM implementation memory

## Current revision: lean pipeline (2026-09-14 evening)

This section supersedes the historical inspection below. See `docs/LEAN_PIPELINE.md` for implementation details, measurements, and artifact links.

- CLI now defaults to `png`: one neutral map, geometry/transforms and small reports. Match caching is optional via `--cache-matches` or `--resume`; diagnostic profile retains it. Corner dumps, support PNGs, lighting, GPS leave-one-out/heading diagnostics, and post-registration footprint-index exports are omitted from the normal PNG path. Full-pose corner checks used for actual matching are retained.
- Vector exports actually bypass compositing and save `project.json`. Config() still defaults to diagnostic for programmatic compatibility; CLI overrides it. Normal coherent output uses 8 rather than 12 primary full-resolution bytes/pixel by omitting support arrays. Diagnostic rendering still exposes coverage/source IDs.
- Matching removes per-descriptor Python DMatch construction, unused square roots and GPS-mode native-coordinate projections. Match settings/candidates are unchanged. `matching_report.json` separates features/upload, descriptor matching, verification/bookkeeping, and cache/diagnostic export time.
- Solver remains one C ABI operation with GPU numerical work, but checks convergence between four-iteration command batches. Actual encoded iterations are reported separately from the maximum budget. At most three frozen iterations remain within the final batch. Cache identity now includes bridge and registration source.
- Fresh full-process timings: SIYI 9.55985 s (previous 30.00013); Archive 9.11155 s (previous 33.63460); 306-image 128 MiB profile 83.20550 s (previous 119.48827). All neutral BGRA pixels and image-to-map transforms are exactly identical to saved baselines. New directories are `metal_lean`, with sibling `metal_lean_timing.json` and `evaluation_lean/comparison.json`.
- New output sizes: SIYI 64.31 MiB, Archive 47.37 MiB, 306 images 52.73 MiB. 306-image internal times: preparation 15.604 s; SIFT/upload 6.996 s; descriptor matching 43.173 s; verification/bookkeeping 7.175 s; alignment/geography 7.398 s; composition/exports 2.395 s. Alignment encoded 12 iterations (9 active) rather than 60.
- Crispness comparison now area-filters minification before geographic warping and uses the exact shared eroded coverage mask. ADM/ODM mean-gradient ratios: SIYI .7841 at 5.00 cm/pixel, Archive 1.0663 at 21.64 cm/pixel. Laplacian-variance ratios .4478 and 1.3401. These are edge/texture proxies, not overall quality or resolved-detail ratios. Earlier bilinear-only, separate-footprint sharpness figures overstate ADM and are not interchangeable. Crop figures are in each dataset's `evaluation_lean/crispness_crops.png`.
- Validation: 38 pipeline/GPU tests passed after implementation; two additional comparison tests verify alias filtering and shared-mask/parity behavior. All 40 pass. Preserved source before trimming is `/private/tmp/adm-before-trim-u0369xhm` (temporary, not guaranteed durable). No saved reference datasets or old outputs were deleted.

## Historical inspection before the lean changes

Inspected 2026-09-14 from local source and saved artifacts. The ADM root has no Git repository. The legacy source checkpoint is `checkpoints/full_metal_86s/`; the separate `ODM/` repository is at `1a7134170e6100cade3236486a279d6241f11f99`.

Source snapshot SHA-256: `aacfc4f06387ed28a46cb26be0ac612f5cab0cdc56ed4794c8dea04917009529`. Computed over sorted `adm/*.py`, `shaders/*.m`, and `shaders/*.metal`, concatenating each relative path, NUL, file bytes, NUL.

## Identity and current behavior

- `metal_optimized` is an output directory/profile name, not a module. `--backend metal --gpu-optimized` sets `gpu_alignment`, `coherent_composite`, `compact_outputs`, and `metal_recompute_warps` true.
- ADM is a planar photomosaic with one homography (eight projective parameters) per photograph. It does not reconstruct cameras in 3D, triangulate scene points, estimate depth, or build a mesh. Its optional input heightmap affects lighting, not registration or orthorectification.
- The optimized profile retains visual feature matching. Do not confuse it with the old `metal_fast` GPS-only preview experiment.
- CPU work: selection, image decoding, metadata, OpenCV SIFT/RANSAC, graph setup, geographic fitting, codecs, and reports. Metal/MPS handles exact descriptor matching, iterative homography alignment, coherent source selection, sampling, and supported lighting/cropping.
- The latest source changes include lasso cropping, output budgets/export profiles, buffer lifetime safeguards, and configurable coarse-compositor memory. Older README/GPU documents describe earlier behavior.

## Selection, matching, and geography

Sources: `adm/selection.py`, `metadata.py`, `geometry.py`, `vocabulary.py`, `registration.py`, and `pipeline.py`.

1. Sort supported input image filenames. Hash inputs and exclude configured initial frames, explicit exclusions, unreadable images, exact duplicates, detected blue annotation labels, and near duplicates. Defaults: skip first 7; thumbnail MAE threshold 1.5. Saved comparison configurations set skip-first to zero.
2. Read EXIF/XMP and optional explicit sidecar overrides. Altitude AGL comes from configuration/sidecar, never automatically from EXIF sea-level altitude. Default AGL 30 m is a placeholder; HFOV 81 degrees, nadir pitch, and square-pixel VFOV derivation are assumptions. Unsupported EXIF orientation is rejected.
3. Saved optimized runs use `--gps-centers`: a KD-tree selects potentially overlapping camera-center circles from AGL/FOV plus GPS tolerance (default 5 m). Recorded yaw does not place images. Without this flag, full camera pose predicts footprints and permits geographic corner verification. Explicit local mode uses sequence neighbors (default 6).
4. Resize each image to registration width (default 1400; Archive/killerofmaps use 640). CPU OpenCV SIFT requests 7000 features with contrast threshold .025. Keep descriptors resident on GPU. MPS computes batched descriptor dot products (512 query rows); MSL norms and 32-lane top-two reductions recover exact squared L2 distances with stable index ties.
5. Apply the .70 nearest/second-nearest ratio test (squared threshold .49 on GPU). Require at least 35 ratio matches. CPU RANSAC fits a homography with 2.5 px tolerance; require at least 30 inliers, 25% inlier fraction, and 3.5% source-image convex-hull coverage. Full-pose mode additionally checks geographic disagreement. Refit accepted inliers and retain at most 160 evenly indexed correspondence pairs per edge; edge `n` retains the original inlier count.
6. Four 50x50 mean-centered corner vocabularies are a separate path: predicted-pose rectification, NCC search radius 24, std >=3, NCC >=.8, delta RMSE <=.02, uniqueness margin >.02, geographic support when available. Four accepted corners can supply an edge. In GPS-center mode these checks are diagnostic-only, after SIFT; non-diagnostic profiles skip them. The latest 306-image runs have zero evaluated corners and all 11,254 edges labeled SIFT fallback.
7. After visual alignment, convert registration-space homographies to native image coordinates. GPS-center fitting uses a CPU float64 robust four-parameter similarity to local WGS84 east/north coordinates, with a vertical-axis flip, soft-L1 scale .75 m, and leave-one-out diagnostics. Requires >=3 images, pitch within 5 degrees of nadir, >=3 m center spread, and median residual <= configured GPS tolerance. GPS establishes scale/north; these residuals are not independent survey accuracy.

## GPU alignment

Sources: `adm/gpu_alignment.py`, `shaders/alignment.metal`, `shaders/ADMMetal.m::adm_align`.

- Anchor image is index N//2. Initialize all homographies by greedily extending the connected graph using the largest-inlier-count crossing edge. Reject disconnected graphs instead of dropping frames.
- Normalize coordinates by registration width. Upload eight float32 homography parameters per image, correspondences, ownership, edge lookup, and incidence arrays. Anchor parameters remain fixed through the normal equations.
- Default 60 iterations, allowed 1..200. One C ABI call encodes the whole budget into one command submission: analytical Jacobians/coordinate-wise soft-L1 weights; per-edge Hessian/gradient blocks; normal-matrix assembly; MPS Cholesky factorization/solve; trial cost reductions; device damping/acceptance/commit.
- Objective is the sum of `sqrt(1+r_px^2)-1` per residual coordinate. Initial damping .001; accepted cost-decreasing steps multiply damping by .35 (floor 1e-7); rejected steps multiply by 8 (ceiling 1e12). Diagonal regularization is damping times max(diagonal,1e-8), plus 1e-9.
- Objective convergence: accepted improvement <1e-6*max(old objective,1). Step convergence: squared relative step <=9e-14, successful factorization, damping <.01, and either a prior accepted step or an already exact solution. Trial projective denominators must exceed .05. After convergence parameters freeze, but all encoded stages still execute; this is not early termination of the command buffer.
- Download final parameters/state; convert to float64 for diagnostics/geography. Nonfinite results or nonconvergence block compositing.
- Jacobian uses 32 floats per correspondence; normal/factor matrices are each (8N)^2 float32 entries. Planned scratch = observations*156 + edges*1088 + (8N)^2*8 + (8N)*12 bytes, limited to 512 MiB, excluding inputs and MPS internals. This is not constant-memory in image/edge count.

## Coherent compositor

Sources: `adm/composite.py`, `adm/coherent_composite.py`, `shaders/coherent.metal`.

- Reject homographies whose image footprint crosses the projective horizon. Fit all footprints into a canvas with longest side `max_output_dimension` (default 6500), a small margin, and north-up orientation when geographic. Output pixel spacing follows this canvas sizing, not a requested physical GSD.
- Coarse scale is min(1, 384/max(W,H), sqrt(budget/(N*8*W*H))). Round coarse dimensions, with minimum 2 per axis. `coherent_budget_mb` defaults to 32 MiB and targets observations+confidence+unary storage; rounding and other buffers mean it is not an exact total-allocation cap.
- First source pass: decode one image at a time, inverse-warp samples onto the coarse grid. Trim normalized edge distance below .02 (about 1% of image dimensions). Confidence is squared normalized edge distance. Store uchar4 observations and half confidence; form confidence-weighted mean colors.
- Unary source cost: `3*(1-sqrt(confidence)) + photometric*min(RGB_squared_distance/(3*22^2),4) + .0001*source_index`. Photometric weight is .15 only if weighted mean per-channel variance <225; otherwise zero to avoid voting displaced foreground objects away. The 22 scale is hardcoded here, not read from `selection_sigma_rgb`.
- Initialize each coarse label from minimum unary cost, then do 32 red/black relaxation sweeps. Four-neighbor Potts penalties favor matching labels, with weight `2.5/(1+neighbor_mean_RGB_squared_distance/(3*30^2))`. Preserve existing labels on ties. This is local label relaxation, not a globally optimal graph cut.
- Second source pass: compute per-source low-frequency exposure bias against the mean, excluding variance >=900. Clamp offsets to +/-32 in 8-bit channels, confidence-weight them, blur numerator/weight with MPS Gaussian sigma 6 coarse pixels, then interpolate/normalize/clamp correction at rendering time.
- Render each source footprint ROI directly from the original decoded photograph. Score = normalized edge distance +2 when that source matches the coarse label. A valid labeled source therefore wins; where unavailable, choose the best geometric confidence. Count coverage independently. Each output pixel has one source's detail plus smooth exposure adjustment; no stochastic switching or high-frequency averaging between photographs.
- Full-resolution primary buffers total 12 bytes/pixel: best float32 (4), BGRA uint8 (4), coverage uint16 (2), source ID uint16 (2). There are additional coarse buffers, decoded source/upload, filtering scratch, CPU masks, and codec allocations. At 6500x4394 the primary buffers alone are 342,732,000 bytes. No output-tile scheduler exists.
- Source IDs are zero-based; 65535 means uncovered. Footprint vectors reference original source images; index lookup returns image support with tile ID -1, not the legacy raw tile API.
- A flat homography cannot correct general 3D parallax. Coherent labels can still cut elevated objects. Seed controls other pipeline operations but this compositor does not sample random source labels.

## Runtime and optional stages

- `adm/metal.py` builds an Objective-C dylib using xcrun clang, keyed by bridge-source hash. The bridge compiles concatenated pipeline/alignment/coherent/lasso MSL at runtime in safe math mode. Shared Metal buffers expose NumPy views whose owner retains the allocation; CPU reads require synchronization. Backend is thread-local and restricted to its creating thread.
- MPS product cache clears at 64 shapes; filtering scratch cache at 4 shapes. Gaussian convolution uses reflected halos and three private RGBA32F textures per cached shape, at most (2048+2r)x(256+2r). Every exported bridge function has an autorelease pool.
- Python Buffer allocation checks live device allocation against 60% physical RAM, overridable with `ADM_METAL_MEMORY_LIMIT_MB`. Direct C solver scratch and private MPS textures are not individually guarded by that Python check; the solver has its separate planned-scratch limit. Device max-buffer size and end-of-run allocated bytes are not peak memory measurements.
- Optional lasso (`adm/lasso.py`, `shaders/lasso.metal`) builds an opaque/nonblack coarse mask, opens the footprint, casts 512 inward boundary rays, reduces to 64 polygon anchors, and refines eight times. Default inset 1%; reject missing central support or removal of >half covered pixels. Retained colors and canvas/world transform stay unchanged; removed coverage/IDs clear. It does not repair internal registration.
- Diagnostic compact flat lighting uses a fused sRGB decode/warm treatment/encode kernel and omits HDR/float lighting exports. Supplied height fields enable the other terrain-lighting path. Optional shadows use contrast-derived relief and three short screen-space rays; they are artistic, separate from neutral mapping, and do not reconstruct geometry.

## Export/cache discrepancies: preserve these distinctions

- CLI defaults to `--export-profile vectors`, while `Config()` defaults to `diagnostic`. Programmatic `run()` calls without an export_profile argument can therefore behave differently from CLI runs.
- Despite CLI help, **current `compose()` always computes the raster and writes `map_neutral.png`, `coverage.png`, `source_ids.png`, transforms, and geometry under all profiles**. `vectors` is not yet a vector-only execution path; `png` is not yet a single-raster export path.
- `adm/project.py::write_recipe()` exists but has no pipeline call site. Current runs do not write `project.json`, despite the lighting summary referring to a saved render recipe.
- Profiles vectors/png skip the entire lighting stage, including requested `--stylized-shadows`. Historical SIYI/Archive optimized runs predate this default and include golden/shadow PNGs. Their old benchmark command now has different behavior unless diagnostic is explicit.
- Explicit CLI profiles activate output quotas: 32 MiB default for vectors, 256 MiB otherwise; `--max-output-mb` overrides. Managed exports use guarded atomic writes, conservative raw-size PNG preflight, and a 512 MiB free-disk reserve. Current raster exports can defeat the nominal vector budget. Existing files count toward the cap.
- Compact mode omits raw `tiles/`, SQLite tiles, `warps/`, and full linear lighting exports, **but still saves `registration.npz` even without --resume**, plus metadata/transforms/support images. Diagnostic mode also saves corner vocabulary caches. No automatic cleanup of previous outputs is performed.
- Resume only reuses accepted match edges when the fingerprint agrees; alignment and rendering rerun. Fingerprint includes selection/input hashes, metadata, configuration, GPS mode, OpenCV, and backend fingerprint. Backend fingerprint hashes MSL files plus `metal.py` and OpenCV; contrary to older wording it does not directly hash `ADMMetal.m` or all registration/pipeline code. Dylib build hashing is separate.

## Saved measurements and storage (inspected, not newly benchmarked)

| Run | Images | Full process wall | Alignment + geography | Match edges / observations |
|---|---:|---:|---:|---:|
| SIYI optimized | 29 | 30.000 s | .459 s | 187 / 26,555 |
| Archive optimized | 63 | 33.635 s | 2.800 s | 1,551 / 241,429 |
| killerofmaps optimized, 32 MiB coarse | 306 | 121.529 s | 35.337 s | 11,254 / 1,747,493 |
| killerofmaps optimized_128mb | 306 | 119.488 s | 35.222 s | 11,254 / 1,747,493 |

Latest completed variant is `output/killerofmaps/metal_optimized_128mb` (Sept 14 00:42 local). AGL 60 m, registration width 640, skip-first 0, canvas 6500x4394, coarse grid 285x193 versus 142x96 at 32 MiB. Both large runs use vectors and explicitly raised output cap to 2048 MiB; both save neutral/support rasters but no golden/shadow output. Baseline 306-image residual median/p95 .22791/.68777 registration pixels; planned solver scratch 332,824,268 bytes. Its matching takes 67.889 s and reports 9 executed / 60 encoded iterations.

Measured logical file sizes below are MiB, exclude input photographs, and do not deduplicate hard links across separate output directories:

| Run | Match cache | Vocabulary cache | Maps/previews | Support rasters | Metadata/transforms | Total |
|---|---:|---:|---:|---:|---:|---:|
| SIYI optimized | .39 | 2.48 | 193.27 | .62 | .31 | 197.07 |
| Archive optimized | 3.28 | 4.37 | 142.31 | .92 | 1.78 | 152.68 |
| killerofmaps optimized | 23.92 | 0 | 52.35 | 1.62 | 2.29 | 80.18 |
| killerofmaps optimized_128mb | 23.92 | 0 | 52.35 | 1.64 | 2.29 | 80.20 |

The 306-image `registration.npz` is 25,077,282 bytes. Coarse budget 128 MiB is runtime working storage, not a 128 MiB saved cache file. Earlier legacy SIYI output was 1,422.04 decimal MB before cleanup. A saved cleanup manifest records 4,130,376,735 bytes removed across regenerable intermediates/trial outputs; those are historical, not currently present cache bytes.

## Exact ODM comparison

- Saved baseline is ODM 3.6.2 in ARM64 Docker, 12 visible CPUs and 7.75 GiB guest RAM. Commands are `python3 /code/run.py --project-path /datasets <dataset> --orthophoto-png --optimize-disk-space`. Reconstruction quality remains default: DSPSIFT/high, 10,000 minimum feature target, FLANN, incremental SfM, camera-parameter optimization, medium point-cloud quality, default 5 cm requested orthophoto resolution. Densification logs explicitly use `--cuda-device -2` (CPU). ADM runs natively with six OpenCV workers and Metal/MPS.
- ODM stages (`ODM/stages/odm_app.py`): dataset -> split/merge -> OpenSfM features/matching/tracks -> 3D reconstruction/bundle adjustment and undistortion -> OpenMVS dense depth/fusion -> point filtering -> 3D and 2.5D meshing -> texturing -> georeferencing -> optional DEM products -> orthophoto -> reports.
- Saved options have skip_3dmodel=false and use_3dmesh=false: ODM builds/textures both 3D and 2.5D meshes; the orthophoto uses the 2.5D mesh. DSM/DTM exports are false; do not claim those raster products were enabled merely because internal surface reconstruction occurs.
- ODM estimates 3D camera/scene geometry and camera parameters; ADM aligns 2D homographies then fits a geographic similarity. ADM has no lens-distortion calibration/undistortion stage. ODM texture selection/seam leveling operates on reconstructed surfaces; ADM selects coarse connected photo regions in the flat map and applies bounded exposure bias.
- Saved feature-processing widths: ODM 1920 SIYI /640 Archive; ADM 1400/640. ODM output rasters: 877x678 at ~5 cm/pixel (SIYI), 1621x1384 at ~21.64 cm/pixel (Archive, GSD-limited). ADM: 6500x5458 at .7947 cm/pixel; 6500x5689 at 5.6664 cm/pixel. Finer raster sampling does not establish more resolved detail or accuracy.
- ODM wall times: 279.824 s SIYI and 352.776 s Archive; ratios to saved optimized ADM are 9.33x and 10.49x. They compare different workloads and environments. From consecutive ODM log stage starts, dense MVS + meshing + texturing take about 207.52 s SIYI and 303.04 s Archive; ADM has no corresponding stages. ADM's own legacy->optimized SIYI alignment improves 49.921 s to .459 s; that comparison isolates its alignment revision better than ODM totals.
- Actual retained ODM outputs with optimize-disk-space enabled, excluding input image folders: 144.94 MiB SIYI, 145.15 MiB Archive. Corresponding ADM optimized directories are 197.07 and 152.68 MiB. Do not claim ADM always saves less disk: these ADM outputs are larger and include three large PNGs at different sampling resolutions. Persistent match caches, diagnostic output, runtime memory, and total retained disk are different measures.
- Archive held-out nominal-center RMSE: ADM 2.987 m, ODM 2.982 m. These are supplied synthetic/crop centers, not surveyed checkpoints. SIYI GPS participates in fitting. Homography correspondence residuals and ODM 3D reprojection residuals have different definitions and are not directly rankable as accuracy.
- Neutral-map comparison resamples onto a common geographic grid/physical resolution. SIYI optimized SSIM against ODM .07410, Archive .60021. Agreement with ODM is not ground truth. The optimized compositor improves the saved goal-detail example versus legacy ADM while SIYI geographic SSIM falls from .10432; no blanket quality win.
- No corresponding saved ODM timing was located for the 306-image killerofmaps run. Do not extrapolate a measured ODM speedup for it. ODM's `--fast-orthophoto` exists but was not the measured baseline; it skips dense MVS and forces skip-3dmodel while retaining SfM and surface mapping.

## Evidence and validation

Quality attribution reviewed against saved tests/comparisons:

- No stage-by-stage percentage attribution exists. GPU numerical parity tests, synthetic failure cases, and multi-change real-run comparisons answer different questions. Do not present runtime shares or source-boundary counts as percentages of map quality.
- Latest 306-image 128 MiB run: preparation 14.462 s (12.16%), matching 67.104 s (56.43%), alignment/geographic fit 35.222 s (29.62%), composition/index/export 2.136 s (1.80%), lighting skipped. Shares use the 118.924 s internal timer; external process time is 119.488 s. Existing timers do not isolate SIFT, descriptor matching, RANSAC, cache writing, GPS fitting, or individual compositor stages.
- GPS-only versus optimized Archive comparisons: geographic ODM-agreement SSIM .28493 -> .60021; MAE 46.959 ->13.215 on a common physical grid. This compares multiple changed stages, not an isolated matching ablation; it is also agreement with ODM rather than independent accuracy.
- SIYI source-transition fraction drops .17888065 -> .00232177 (17.888% -> .232%, a 98.70% reduction). This quantifies label coherence, not overall image-quality improvement. Geographic SSIM against ODM simultaneously drops .10432 -> .07410 versus legacy visually aligned ADM.
- Synthetic displaced-object test asserts output interior equals one original photograph, preserving one object without duplicated/erased copies. Exposure test uses source values 80/120 and requires neighboring output jumps <8 levels in the tested interior. These validate specific cases, not a real-dataset average benefit.
- Corner delta threshold .02 -> .10 experiment: 226 evaluated patches, 0 ->119 passing delta alone, 0 ->2 passing all gates, zero pairs with four corners. All 190 edges/homographies/correspondence sets remain identical; no map-quality improvement demonstrated (`output/101SIYI_IMG_corner10/RESULTS.md`). This experiment predates the 187-edge optimized SIYI comparison.
- Lasso synthetic tests preserve surviving pixels exactly, remove protruding/black borders, retain >80% of covered pixels in the test fixture, preserve an interior dark region and canvas, and reject unsupported central geometry. It is off in the latest measured run.
- 32 ->128 MiB coarse budget changes 306-image grid 142x96 ->285x193; saved alignment residuals are identical. No independent compositor-quality comparison for this budget change was located. Larger budget does not have a measured quality percentage.
- Existing legacy Metal test confirms recomputed warps versus disk-cache mode are pixel-identical; optimized resume test confirms unchanged neutral output and unchanged match-cache timestamp. Cache persistence is for reuse, not additional map information.

- Implementation sources above; saved results: `output/comparison/{siyi,archive}/metal_optimized/`, sibling timing/comparison JSONs, `output/killerofmaps/metal_optimized*/`, and `output/comparison/{siyi,archive}/odm/{options.json,log.json,opensfm/config.yaml,timing.json}`.
- Historical `output/metal_validation.json` reports 38 tests, zero failures/errors, lighting max absolute error 1.19209e-7. Timestamp Sept 14 00:34 precedes the final configurable-budget edit at 00:40. This review read the tests/reports; it did not rerun the GPU suite or full benchmarks.
- Review ran a tiny temporary-file probe: parsed actual CLI default as vectors and invoked the common compositor export path to confirm neutral/coverage/source-ID PNGs are written and no project.json appears. Temporary probe data were removed automatically. No implementation changes or cache deletion were performed.
