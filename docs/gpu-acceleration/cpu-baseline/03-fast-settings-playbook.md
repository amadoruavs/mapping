# Fast-settings playbook: ODM CLI presets for the M4 Pro laptop

Flags and defaults are from `opendm/config.py` at commit 1a71341; the numeric effect of each
flag is traced in `00-pipeline-stage-map.md`. Speed multipliers are ESTIMATES (from the
docs' 4x rule, the code's scaling tables and community benchmarks) until the plan in
`04-benchmark-and-profiling-plan.md` has been run on this machine. Invocation form:

```bash
cd /Users/avneh/Code/ADM/ODM
pixi run odm -- --project-path ~/datasets <name> [flags]
```

Only `default` (CPU) is available on macOS; `--no-gpu` is a no-op here because `has_gpu()`
already returns False without `nvidia-smi`.

## 1. Decision table

| Goal | Preset flags | Stages run | Relative time (defaults = 1.0) | What you give up |
|---|---|---|---|---|
| **A. Fast orthophoto only** (map in minutes) | `--fast-orthophoto --skip-report --orthophoto-resolution 5 --feature-quality medium --matcher-neighbors 12 --max-concurrency 12 --optimize-disk-space` | dataset, opensfm (features/match/reconstruct/undistort/export_ply), filterpoints, meshing (2.5D from sparse, DSM x8 coarser, Poisson), mvs_texturing (2.5D only), georef, dem (off), orthophoto | **0.25-0.4** (community laptop: 5 min vs 9 min default on Aukerman; on larger sets the saving grows because openmvs is skipped) | No dense point cloud, no 3D mesh, DSM/DTM off unless `--dsm`; orthophoto built from the sparse cloud so relief displacement on buildings/trees is worse; feature medium halves extraction resolution (fewer matches on low-texture scenes) |
| **B. Balanced 3D** (usable mesh + ortho, laptop-friendly) | `--pc-quality medium --feature-quality high --mesh-octree-depth 11 --mesh-size 200000 --matcher-neighbors 0 --max-concurrency 8 --dsm --skip-report` (i.e. defaults plus `--dsm`, capped concurrency) | all except report | **~1.0** (maybe 0.9-1.1: `--max-concurrency 8` can be faster than 12 on memory-bound openmvs, slower on features) | Nothing vs defaults; DSM adds a few percent; report skipped |
| **C. Max quality overnight** | `--pc-quality high --feature-quality ultra --min-num-features 16000 --mesh-octree-depth 12 --mesh-size 400000 --use-hybrid-bundle-adjustment --dsm --dtm --orthophoto-resolution 2 --dem-resolution 2 --texturing-single-material --max-concurrency 8 --optimize-disk-space` | all | **4-12x** (pc high = ~4x on openmvs, octree 12 = up to 8x on Poisson, ultra features ~2-4x on extraction/matching; the community "3D" preset with octree 12/mesh 300k/pc high took 6h vs 36 min for a "high res" preset on the same 8-core desktop) | Time and memory; risk of OOM on 24 GB above ~200 images at 18-20 MP (OpenMVS then falls back to tiled fusion automatically) |

Corollary rows:

| Situation | Add | Why |
|---|---|---|
| Only need DSM/DTM + ortho, want dense accuracy | `--skip-3dmodel --dsm [--dtm]` (keep openmvs) | Skips PoissonRecon + 3D texturing (two of the heavier post-MVS steps); 2.5D mesh still built |
| 3D only, no map | `--skip-orthophoto --use-3dmesh` | `--use-3dmesh` drops the 2.5D mesh + its texturing run; orthophoto skipped |
| >300 images on 24 GB | `--split 200 --split-overlap 100` (local split-merge) or `--pc-quality low --max-concurrency 8` | Bounds per-submodel memory; local split runs submodels sequentially so total time is similar but peak RAM drops |
| Dataset without GPS (indoor, handheld) | `--matcher-type bow` is chosen automatically; consider `--matcher-order 10` for video/sequential captures | Avoids O(n^2) pair matching |
| Re-tune later stages without redoing SfM/MVS | `--rerun-from odm_meshing` (or `odm_dem`, `odm_orthophoto`) | Stage results on disk are reused; `--optimize-disk-space` breaks this by deleting intermediates |
| Laptop must stay usable | `--max-concurrency 6` or `taskpolicy -b pixi run odm -- ...` | Leaves cores / pushes work to E-cores (see 02) |

## 2. Flag-by-flag notes (what each one changes in the code)

* `--fast-orthophoto` (default False): drops the `openmvs` stage from the graph
  (`stages/odm_app.py:69-73`), forces `--skip-3dmodel`, sources the point cloud from
  `opensfm/reconstruction.ply` (`export_ply --no-cameras --point-num-views`), makes the 2.5D
  DSM 8x coarser and uses `dem2points` + PoissonRecon for the 2.5D mesh
  (`stages/odm_meshing.py:44-60`), and uses decimation step 4 instead of 40 in the cropper.
  Docs: "generates an orthophoto directly from the sparse reconstruction"
  (https://docs.opendronemap.org/arguments/).
* `--skip-3dmodel`: no PoissonRecon 3D mesh and no 3D `texrecon` run; the 2.5D path stays.
  Ignored if `--use-3dmesh` is also set (`config.py:948-950`).
* `--use-3dmesh`: orthophoto is rendered from the 3D textured mesh; the 2.5D mesh and its
  texturing are skipped (`stages/mvstex.py:34-53`, `stages/odm_orthophoto.py:52-55`). Docs:
  "a bit faster and provides similar results in planar areas".
* `--pc-quality {ultra,high,medium,low,lowest}` (medium): depth-map size factor
  0.5/0.25/0.125/0.0675/0.03375 of the longest image side (`opendm/utils.py`); each step ~4x
  time and memory; `low`/`lowest` also lower OpenMVS `--min-resolution` to 320.
* `--pc-skip-geometric`: `--geometric-iters 0` for `DensifyPointCloud`; large saving on
  the openmvs stage (ESTIMATE 30-50 % of that stage) at the cost of noisier depth maps;
  docs say geometric estimates "may not be usable in larger datasets".
* `--pc-filter 0`: skips the OpenMVS `--filter-point-cloud` pass (and FPCFilter's statistical
  filter becomes a no-op); saves minutes on big clouds, keeps outliers.
* `--feature-quality {ultra,high,medium,low,lowest}` (high): scale 1/0.5/0.25/0.125/0.0675,
  clamp 320..4480 px (`opendm/osfm.py:202-232`). Extraction and matching cost scale with the
  pixel count (~4x per step) but the descriptor count is capped by `--min-num-features`.
* `--min-num-features N` (10000): OpenSfM `feature_min_frames`; OpenSfM lowers the peak
  threshold until N features exist, so higher N = more extraction time *and* more matching
  time (FLANN cost ~ N log N per pair). 8000 is a common "fast" value; 16000-20000 for
  low-texture scenes.
* `--matcher-type {flann,bow,bruteforce}` (flann): FLANN KD-tree (`flann_algorithm: KDTREE`);
  `bow` forces HAHOG features (not SIFT) and is faster but "can sometimes miss valid matches";
  `bruteforce` is "very slow but robust" (`config.py:253`).
* `--matcher-neighbors N` (0): N>0 -> GPS nearest-N pairs, no graph rounds; 0 -> Delaunay
  graph with 50 rounds. With good GPS and regular flight lines, `--matcher-neighbors 8-12`
  cuts pair count noticeably on large sets; on small sets (<100) the default is fine.
* `--matcher-order N` (0): filename-order neighbors, non-georeferenced sets only.
* `--use-hybrid-bundle-adjustment`: local BA per image plus global every 100 images
  (`bundle_interval: 100`, `local_bundle_radius: 1`); docs: "Speeds up reconstruction for
  very large datasets". Small sets see no gain; >500 images do.
* `--sfm-algorithm planar`: "can be much faster" for nadir-only fixed-altitude planar scenes
  (`config.py:332`); not in the presets because it changes robustness; try on flat farmland.
* `--mesh-octree-depth D` (11, clamped 1..14): PoissonRecon `--depth`; ~8x memory/time per
  +1 (octree). `--mesh-size V` (200000): `ReconstructMesh --target-face-num 2V`; larger
  meshes also make texturing and orthophoto slower. Docs recommend 8-12 and mesh-size
  300000-600000 for urban detail (https://docs.opendronemap.org/tutorials/).
* `--orthophoto-resolution R` / `--dem-resolution R` (5 cm/px): both floored at 90 % of the
  estimated GSD (`gsd.cap_resolution`), so values below the flight's GSD do nothing;
  `--orthophoto-resolution` also feeds `undistorted_image_max_size` (power-of-two downscale of
  the undistorted images that OpenMVS and texrecon read), which is why a coarse ortho
  resolution can speed up MVS on high-GSD flights. `--ignore-gsd` disables the cap and is
  "memory and processor hungry"; docs: "do not use it".
* `--dem-decimation N` (1): decimates points before DEM rasterization; 10-100 for large
  clouds when only coarse DEMs are needed. `--dem-gapfill-steps` (3): each step is another
  IDW raster; 0 disables.
* `--optimize-disk-space`: deletes `features/`, `matches/`, `reports/`, `tracks.csv`,
  depth maps, `scene_dense*`, undistorted images, unfiltered cloud, meshes after texturing
  (`stages/run_opensfm.py`, `openmvs.py`, `mvstex.py`, `odm_filterpoints.py`,
  `odm_georeferencing.py`). Saves multiple GB on the ~11 GB-free volume; disables
  `--rerun-from` of most stages.
* `--max-concurrency N` (12 here): every thread count in the pipeline (see 00 section 7).
  PoissonRecon gets N-1. On 24 GB, 8 is the safe value for `--pc-quality high`+.
* `--no-gpu` / `ODM_NO_GPU=1`: no effect on macOS today (no CUDA); keep for parity with the
  Metal track's future switch.
* `--skip-report`: skips `export_stats` in opensfm (`run_opensfm.py:60-67`) and the whole
  `odm_report` stage (PDF, overlap raster via PDAL, previews); a few percent, plus avoids the
  `reports/` disk usage.
* `--rerun-from <stage>`, `--rerun <stage>`, `--end-with <stage>`: stage names are
  `dataset split merge opensfm openmvs odm_filterpoints odm_meshing mvs_texturing
  odm_georeferencing odm_dem odm_orthophoto odm_report odm_postprocess` (`config.py:13-15`).
  `rerun_stages` maps every flag to the earliest stage it affects (`config.py:17-110`), e.g.
  `mesh_octree_depth -> odm_meshing`, `pc_quality -> opensfm` (because undistort size
  depends on it), `orthophoto_resolution -> odm_orthophoto`.
* `--texturing-single-material`: post-packs textures into one atlas (Python `obj_pack`);
  costs time, saves files; only for the primary/3D run. `--texturing-skip-global-seam-leveling`
  skips a texrecon pass (small saving, visible seams). `--gltf` adds a GLB export.
* `--split N` / `--split-overlap M` / `--sm-cluster URL` / `--sm-no-align`: submodel
  split-merge; locally submodels run one after another with `max_concurrency - 1` when a
  cluster is used (`stages/splitmerge.py:50-53`); docs: "will require less memory" and
  "3D textured meshes are currently not being merged" (https://docs.opendronemap.org/large/).
  Without a ClusterODM node there is no parallel speedup, only a memory bound.
* Multispectral/thermal flags (`--radiometric-calibration`, `--primary-band`,
  `--skip-band-alignment`) multiply the undistort/texturing/orthophoto work by the band
  count; not covered by the presets.

## 3. Expected numbers on this laptop (to be replaced by measurements)

Anchors from the community and the CPU rows in odm-benchmarks
(https://raw.githubusercontent.com/OpenDroneMap/odm-benchmarks/master/data/benchmarks.csv):

| Dataset | Images / MP | Reference machine | Default | Fast ortho | High |
|---|---|---|---|---|---|
| Aukerman | 77 / 18 | Ryzen 9 5900X 12c/24t, 64 GB, ODM 3.5.5 native Win | 15m52s | - | - |
| Aukerman | 77 / 18 | i7-10700K 8c/16t, 96 GB, ODM 2.7.1 | ~9 min (laptop Acer Aspire 5: 9 min) | 5 min (laptop) | 30-36 min (feature ultra, pc high, 2 cm) |
| Toledo | 87 / 12 | Ryzen 9 5900X | 14m05s | - | - |
| Toledo | 87 / 12 | Ryzen 7 8845HS laptop 8c/16t, 16 GB | 27 min | - | - |
| Bellus | 122 / (Canon) | i7-10700K | - | - | 1h03m at pc high |
| Shitan | 493 | Ryzen 9 5900X / 64 GB | 1h12m | - | - (16 GB laptop crashed) |

ESTIMATE for the M4 Pro (8 P-cores at ~4.5 GHz, comparable per-core to the 5900X, fewer
threads): Aukerman default 12-20 min, fast-ortho 4-7 min, preset C 1.5-3 h; Toledo default
10-18 min. Thermal steady state and the 24 GB ceiling are the unknowns.

## 4. Copy-paste command lines

Dataset layout expected by ODM: `~/datasets/<name>/images/*.JPG`; outputs land next to
`images/` (`odm_orthophoto/`, `odm_dem/`, `odm_texturing/`, `odm_georeferencing/`,
`benchmark.txt`, `log.json`).

```bash
# A. fast orthophoto (minutes)
pixi run odm -- --project-path ~/datasets aukerman \
  --fast-orthophoto --skip-report --feature-quality medium --matcher-neighbors 12 \
  --orthophoto-resolution 5 --max-concurrency 12 --optimize-disk-space

# B. balanced 3D (defaults + DSM, memory-safe concurrency)
pixi run odm -- --project-path ~/datasets aukerman \
  --dsm --max-concurrency 8 --skip-report

# C. max quality overnight (plug in, High Power mode, close other apps)
pixi run odm -- --project-path ~/datasets aukerman \
  --pc-quality high --feature-quality ultra --min-num-features 16000 \
  --mesh-octree-depth 12 --mesh-size 400000 --use-hybrid-bundle-adjustment \
  --dsm --dtm --orthophoto-resolution 2 --dem-resolution 2 \
  --texturing-single-material --max-concurrency 8 --optimize-disk-space

# Re-tune only the DEM/orthophoto after any of the above (keeps SfM/MVS/mesh)
pixi run odm -- --project-path ~/datasets aukerman --rerun-from odm_dem --dsm --dem-resolution 3
```

`settings.yaml` at the repo root is loaded as argparse defaults (`config.py:170-179`,
hyphens converted to underscores); the shipped file sets only `project_path: '/'`, so every
flag above can instead be written there (e.g. `max_concurrency: 8`) to avoid retyping.

## 5. Flag interactions enforced by `config.py` (read before combining flags)

| If you set | ODM also does | Where |
|---|---|---|
| `--fast-orthophoto` | sets `--skip-3dmodel` | `config.py:936-938` |
| `--pc-rectify` | turns on `--pc-classify` (SMRF + OpenPointClass, extra minutes) | `config.py:940-942` |
| `--dtm` | turns on `--pc-classify` | `config.py:944-946` |
| `--skip-3dmodel` together with `--use-3dmesh` | ignores `--skip-3dmodel` (3D mesh is required for the ortho) | `config.py:948-950` |
| `--orthophoto-cutline` with `--crop 0` | sets `--crop 0.01` | `config.py:952-954` |
| `--sm-cluster URL` | contacts the node at parse time and exits if offline | `config.py:956-961` |
| `--sfm-algorithm triangulation` without omega/phi/kappa in EXIF | falls back to `incremental` | `stages/dataset.py:321-326` |
| `--rolling-shutter` on a non-georeferenced set | disabled | `stages/dataset.py:329-331` |
| `--matcher-order N` on a georeferenced set | ignored | `opendm/osfm.py:268-272` |
| `--matcher-type` not given and no GPS | switches to `bow` (HAHOG features) | `opendm/osfm.py:286-288` |
| `--boundary` on a non-georeferenced set | ignored | `stages/dataset.py:313-318` |
| <= 3 photos | forces `matching_gps_neighbors: 3` | `opendm/osfm.py:245-247` |
| deprecated `--time --verbose --debug --resize-to --depthmap-resolution --pc-geometric ...` | warned and ignored (no error) | `config.py:919-926` |

Cost ordering (most to least expensive knob at defaults, ESTIMATE from the scaling tables):
`--mesh-octree-depth` (8x per step, 3D mesh only) > `--pc-quality` (4x per step) >
`--feature-quality` (4x pixels per step, capped by `--min-num-features`) > `--min-num-features`
(linear-ish in extraction, N log N in matching) > `--mesh-size` (linear in texturing/ortho) >
`--orthophoto-resolution`/`--dem-resolution` (quadratic in output pixels but floored by GSD) >
`--dem-gapfill-steps`, `--pc-filter`, `--crop`, report/tiles/COG flags (single-digit %).

## 6. Sanity checks before trusting a "fast" preset

* Compare `odm_report/stats.json` -> `reconstruction_statistics` (reprojection error, number
  of reconstructed shots) between default and fast runs; if `--feature-quality medium` or
  `--matcher-neighbors` loses cameras, the speedup is not real.
* Compare orthophoto extent (`odm_orthophoto/odm_orthophoto.tif` bounds) and DSM histogram
  (`odm_dem/dsm.tif`) - `--fast-orthophoto` shrinks coverage at the edges.
* Keep the `benchmark.txt`/`log.json` of each preset next to its outputs so the per-stage
  split is preserved (plan in 04).

## Sources

* `opendm/config.py`, `stages/odm_app.py`, `stages/run_opensfm.py`, `stages/openmvs.py`,
  `stages/odm_meshing.py`, `stages/mvstex.py`, `stages/odm_orthophoto.py`,
  `stages/splitmerge.py`, `opendm/osfm.py`, `opendm/utils.py`, `opendm/gsd.py`,
  `opendm/mesh.py` (commit 1a71341)
* ODM docs, arguments (per-flag speed guidance): https://docs.opendronemap.org/arguments/
* ODM docs, tutorials (mesh-size / octree recommendations, HPC example): https://docs.opendronemap.org/tutorials/
* ODM docs, large datasets / split-merge: https://docs.opendronemap.org/large/
* ODM docs, hardware: https://docs.opendronemap.org/installation/
* odm-benchmarks (total times per dataset/machine): https://github.com/OpenDroneMap/odm-benchmarks ,
  https://raw.githubusercontent.com/OpenDroneMap/odm-benchmarks/master/data/benchmarks.csv
* Community "ODM Benchmarks Update" (fast/default/high on a laptop; 3D preset 6 h): https://community.opendronemap.org/t/odm-benchmarks-update/10367
* Community "Processing Time Differences" (`--pc-skip-geometric` advice): https://community.opendronemap.org/t/processing-time-differences/15255
* Community "OpenSfM Processing Time" (SfM dominates large sets): https://community.opendronemap.org/t/opensfm-processing-time/11252
* Community "Processing time on a local machine" (RAM ceilings): https://community.opendronemap.org/t/processing-time-on-a-local-machine/17006
