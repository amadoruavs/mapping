# GPU alignment and coherent compositing

The timings below describe the preserved earlier diagnostic runs. The current default is the lean PNG profile; see [fresh measurements, export behavior and crispness](LEAN_PIPELINE.md).

On the 29 SIYI images, the measured full run drops from 86.18 seconds to 30.00 seconds, including decoding, matching, alignment, three full-size PNG exports and process startup. Global alignment drops from 49.92 seconds to 0.46 seconds. Both runs use the same 187 edges and 26,555 observations. Median / p95 alignment error changes from 1.12133 / 3.40180 pixels to 1.12061 / 3.39743 pixels at registration width 1400.

These are measured improvements to the planar pipeline. They do not establish independent survey accuracy or an entirely GPU implementation. The new neutral image has cleaner goal-frame detail, while geographic SSIM against default ODM decreases from 0.10432 to 0.07410. The comparison records that tradeoff; no blanket improvement claim is made.

The final 63-image Archive run takes 33.63 seconds, including all exports, versus 352.78 seconds for saved default ODM. Its alignment stage takes 2.80 seconds. The supplied-center horizontal RMSE is 2.987 m for ADM and 2.982 m for ODM; this does not establish an accuracy win. See the [full measured comparison](../output/comparison/GPU_REVISION_REPORT.md) for images, CSVs, failures and limitations.

## Run

```sh
.venv/bin/python -m adm.pipeline output/comparison/siyi/odm/images \
  --output output/my_gpu_run \
  --config output/comparison/siyi/adm_config.json \
  --gps-centers --backend metal --gpu-optimized
```

Use a new output directory for a fresh comparison. Matching is retained. `--resume` is explicit and requires a matching input/configuration/backend fingerprint. The original profile remains available without `--gpu-optimized`; its source checkpoint is `checkpoints/full_metal_86s/`.

## The GPU operation

`adm.gpu_alignment.solve` packs correspondence vectors, graph incidence and eight homography parameters per image. `adm_align` encodes four iterations per batch, then reads the device convergence flag before submitting another batch. Each iteration runs analytical Metal Jacobians and robust weights, per-edge Hessian blocks, normal-matrix assembly, MPS Cholesky factorization and solve, MPS objective reductions, and device-side damping, acceptance and parameter updates. Parameters freeze on convergence, with at most three extra iterations in the final batch. Only the convergence flag is checked on the host between batches; solution parameters are read back at the end.

The objective retains the previous coordinate-wise soft-L1 loss with a one-pixel scale and the same anchored image. The host's sparse finite-difference nonlinear solve is replaced by damped analytical Gauss-Newton. Float32 is used on the GPU; coordinate normalization and robust damping protect numerical conditioning. Failed convergence is reported and blocks compositing. The exact-initial-solution case is covered by validation.

This is one host operation built from ordered GPU stages. It is not one literal MSL kernel: global normal-equation reductions and MPS factorizations require ordered dispatches. CPU work still includes SIFT extraction, RANSAC, graph packing and initial spanning-tree composition, double-precision GPS fitting, metadata and image codecs. The reports name those stages explicitly.

The Jacobian stores 32 floats per correspondence; only the small 8N-by-8N normal matrix is dense. Planned solver scratch is 4.78 MB for SIYI and 41.39 MB for Archive, with a 512 MiB application limit. These figures exclude input vectors and MPS internal allocations. `gpu_report.json`'s `max_buffer_bytes` is the device's maximum allowed allocation size, **not measured memory usage**. Peak child RSS comes from the external timing report and does not measure total system or GPU-private memory.

## Ghosting and exposure

The previous compositor used spatially varying stochastic source selection and a mean-derived correction. With parallax, that can repeatedly switch between displaced observations of the same structure. The new compositor optimizes coherent source labels on a coarse grid with red/black Potts relaxation, then renders detail from one photograph per output pixel. High disagreement disables the consensus term so that a displaced foreground object is not voted away as background.

Exposure differences are corrected with bounded, smooth per-source bias fields using MPS Gaussian filtering. Strong cross-view disagreements are excluded. This adjusts low-frequency color without transferring another view's high-frequency structure. The displaced-structure test checks that the object remains present exactly once; the exposure-step test checks that a 40-level seam is reduced below eight levels between neighboring pixels.

Source labels, coverage and a same-world-coordinate goal crop are saved for inspection. Lower source-boundary density is a diagnostic, not proof of accuracy. Coherent regions can still cross elevated objects, and some seams remain. A planar homography cannot recover depth, fully correct 3D parallax, or replace ODM's reconstruction.

## Shadows and storage

`map_neutral.png` is the mapping output; `map_golden.png` retains the previous SDR lighting equation. Optional `map_shadows.png` uses three short screen-space rays over a contrast-derived relief proxy and MPS softening. It is an artistic effect, not reconstructed geometry or physically correct lighting. It does not modify the neutral PNG and is excluded from accuracy metrics.

The compact profile keeps original pixels in the source photographs and writes `footprint_index.json` with image paths, transforms and footprint vectors. It does not materialize raw tile copies, warped images, linear HDR or full float lighting arrays. The vector index supplies image support, not the old `TileIndex.pixels()` raw-tile API; its lookup returns tile ID -1 explicitly. Existing successful PNGs and geometry remain available after cache cleanup.

SIYI outputs occupy 206.64 MB versus 1,422.04 MB before cleanup for the old profile. The new compositor's primary full-resolution buffers use 12 bytes per output pixel plus bounded coarse observations, MPS scratch and one decoded source at a time. The filesystem cleanup manifest is `output/comparison/gpu_revision_cleanup.json`. It records 4.13 GB of removed regenerable arrays and superseded trial PNGs, retaining reference PNGs and measurements. A disk-full Archive attempt is preserved separately and excluded from completed timings.

## Validation and comparison artifacts

```sh
.venv/bin/python scripts/validate_metal.py
.venv/bin/python scripts/validate_gpu_alignment.py
.venv/bin/python scripts/benchmark_gpu_revision.py siyi
.venv/bin/python scripts/compare_lean.py siyi
```

The benchmark writes `metal_lean` and refuses to overwrite a nonempty result. Use `archive` or `killerofmaps` for the other datasets. The suite now has 40 passing tests, including analytical Jacobians, early convergence, disconnected-graph rejection, ghosting/erasure, exposure correction, neutral/shadow separation, PNG/diagnostic parity, cache reuse, vector-only exports and comparison filtering.

For each dataset, `comparison_optimized.json` contains timings, peak process RSS, registration residuals, geographic residuals, common-resolution PSNR/SSIM/MAE, coverage overlap, sharpness diagnostics, optional visual registration and position errors. `evaluation_optimized/metrics.csv` and `position_errors.csv` provide tabular data. `side_by_side_optimized.png` compares the neutral maps. Evaluation runs outside the pipeline timer using saved ODM rasters.

SIYI has no independent survey truth: its GPS is a fitting prior. Archive's nominal synthetic centers are held out from reconstruction, but are not surveyed checkpoints. The saved ODM runs use ARM64 Docker, while ADM runs natively. These are single fresh-run timings with filesystem caches and background desktop activity uncontrolled. Default ODM performs full 3D reconstruction at different output dimensions; the image metrics first resample both maps to the same physical resolution.
