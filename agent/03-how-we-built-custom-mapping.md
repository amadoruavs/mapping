# How the custom mapper was made

This is the story a future agent needs. Dates are local 2026-09-13 through 2026-09-17 unless noted. Measurements were taken on an Apple M4 Pro / 24 GB / macOS 26. Recheck [MEMORY.md](../MEMORY.md) and the code before quoting them as current.

## Why not just use OpenDroneMap

ODM 3.6.2 (commit `1a7134170e6100cade3236486a279d6241f11f99` in the original clone) is a full photogrammetry stack:

```
dataset → OpenSfM features/matches/tracks → incremental SfM + BA
  → OpenMVS dense depth → filter → 2.5D/3D mesh → texture → georeference → ortho
```

Saved ODM wall times on ARM64 Docker, 12 CPUs, 7.75 GiB guest RAM:

| Set | Photos | ODM wall |
|---|---:|---:|
| SIYI | 29 | 279.8 s |
| Archive | 63 | 352.8 s |

The club wanted something that could map a few hundred nadir 4K frames in about two minutes on a laptop. That is a different product: **planar mosaicking**, not SfM.

ODM was kept as a quality reference, then later ported natively (Pixi SuperBuild) for Fedora and macOS. Those trees are not in this git repo. Notes are in `notes/odm-fedora/`.

## Stage 0 — research docs (2026-09-13)

`docs/` was written **before** the custom GPU mapper existed as a replacement. It maps ODM's CUDA touchpoints, Metal/Radeon options, and a golden-hour look. Start at `docs/README.md`. The research-log marks verified vs estimated claims. No ODM code was changed in that pass.

## Stage 1 — ADM on Metal

A from-scratch pipeline in `adm/` + `shaders/`:

1. Select and hash frames.
2. Read EXIF/XMP. AGL is configured, never taken from EXIF altitude.
3. Propose overlaps from GPS camera-center circles (`--gps-centers`) or a sequence window.
4. Resize to a registration width (640 on the 306-photo set, 1280 later in ISM).
5. OpenCV SIFT (7000 features, contrast 0.025) on CPU; descriptors stay on GPU.
6. MPS exact L2 + 0.70 ratio test; CPU RANSAC homography (2.5 px, ≥30 inliers, 25% inlier fraction, 3.5% hull coverage).
7. Global 8-DOF alignment on Metal, middle image anchored.
8. Robust 4-parameter similarity to GPS centers (soft-L1 scale 0.75 m, median residual ≤ 5 m).
9. Coherent source selection + bounded exposure.
10. Optional lighting / lasso / presentation.

`--backend metal --gpu-optimized` turns on GPU alignment and the coherent compositor. `metal_optimized` is that profile name, not a Python module.

Saved optimized-profile walls (before the lean trim):

| Set | Photos | Wall | Alignment+geo | Edges |
|---|---:|---:|---:|---:|
| SIYI | 29 | 30.0 s | 0.46 s | 187 |
| Archive | 63 | 33.6 s | 2.80 s | 1,551 |
| killerofmaps 128 MiB | 306 | 119.5 s | 35.2 s | 11,254 |

Matching, not alignment, dominated the 306-photo run (~67 s).

## Stage 2 — lean pipeline (2026-09-14 evening)

Document: `docs/LEAN_PIPELINE.md`. Frozen copy: `LeanADM/`.

What was removed from the *normal* path: match-cache default, corner dumps, support PNGs, lighting, GPS leave-one-out extras, full coverage/source-ID buffers.

What was changed: GPU solver stops between 4-iteration batches instead of always encoding 60; matcher results stay as arrays (no per-match Python `DMatch`); primary buffers 12 → 8 bytes/pixel.

Fresh walls, **byte-identical** maps vs the optimized baselines:

| Set | Previous | Lean |
|---|---:|---:|
| SIYI 29 | 30.000 s | **9.560 s** |
| Archive 63 | 33.635 s | **9.112 s** |
| killerofmaps 306 | 119.488 s | **83.206 s** |

306-image internal: prep 15.6, SIFT/upload 7.0, descriptor match 43.2, verify 7.2, align+geo 7.4 (9 active / 12 encoded iterations), compose 2.4.

This is the last "validated lean backup." User said not to rerun it. `LeanADM/benchmark/` (the actual PNG) was not uploaded.

## Stage 3 — FSR experiments (do not treat as the good default)

Actual AMD FidelityFX Frame Interpolation 1.1.3 and FSR1 EASU/RCAS were translated to Metal 4. Unmodified GLSL → glslang → spirv-opt → SPIRV-Cross → MSL.

Using generated views as extra compositor inputs on the 306-photo set produced 915 virtual views and a 143.6 s run. Later, generated-frame road fragments appeared (duplicated/warped pavement). The software gate is `adm/fsr_geometry.py`: reject pairs with rotation > 15°, huge motion, horizon midpoints, or parent inconsistency. Recompose of the 94-photo case dropped the large artifacts; 20 of 93 pairs were skipped.

ISM now defaults `interpolation_levels=0` and `fsr=False`. Real photos only. Do not turn interpolation back on because it "uses AMD FSR."

Final FSR **upscale** (2× spatial) is a presentation tool. It is not extra GSD.

## Stage 4 — ISM package

`ISM/` is the handheld product: one `ISM.metal` (92 entry points), `runtime/`, `run.py` with SIYI defaults, optional visual HTML runner.

User-preserved defaults that still matter:

- Input was `/Users/avneh/Downloads/mapv2` (94 photos) or `/Users/avneh/Downloads/maps` (564).
- AGL 51.816 m.
- Registration width 1280.
- Alignment cap 600 with early stop; workspace 1024 MiB.
- `allow_gps_mismatch=False`.
- `alignment_symmetric_transfer=True`.
- `geographic_solver='visual'` (the OpenSfM residual port is opt-in).
- Lasso crop after grade.

## Stage 5 — the collapse, the rollback, the transfer residual

This is the most important geometry lesson in the repo.

A 564-photo / 600-iteration run ran 1094 s and reported a tiny median residual (~0.0028 px). Inspection of `transforms.npy` showed **523/564 footprints under 1% of the registration area**. The solver had shrunk the photos. More iterations cannot fix a collapsing objective.

The user then demanded a rollback to `ISM.metal` SHA-256 `c610ad5fc0a456f6978ead738a9aeb073695faa16b1f2436ec0be7283e0cc2c2` (from `ISM/runtime.zip`). GPS/yaw priors added after that shader were removed. Backup: `backups/ism_before_rollback_20260914_232001/`.

After rollback, 94 photos aligned visually but failed the GPS gate: **median 30.09 m vs 5 m**. Mosaic-space area had collapsed to ~0.008 of nominal.

The fix that actually passed the gate: `alignment_symmetric_transfer=True`. Measure i→j and j→i transfer in **original resized photo pixels**, half-weighted. Same 94 photos: median/p95 transfer 1.10 / 3.19 px; GPS median/p95 **2.38 / 6.83 m**. All 94 retained.

If you see 30.09 m again, you are in the collapsed-scale regime. Do not raise the GPS tolerance to "make it work" unless the user explicitly wants a bypass.

## Stage 6 — OpenSfM geographic port (opt-in)

Pinned OpenSfM revision `c5328439465e6ace011f39077d1077d7b1cdd65d`. Ported `DataPriorError`, `UpVectorError` + Cauchy, and WGS84/ECEF/topocentric equations onto Metal as a 6-DOF physical camera on a flat ground plane. Document: `docs/OPENSFM_PORT.md`.

Direct metadata init failed: every `mapv2` frame says `NO GPS connected`, and recorded yaw can be ~180° off. The working path is: visual solve → geographic similarity → decompose to physical poses → joint refine.

On 94 photos it slightly **improved** the image-center/GPS proxy (2.38 → 2.14 m) and slightly **worsened** visual transfer (1.10 → 1.23 px). It is not the default. It is not full ODM.

## Stage 7 — presentation, then MapRepair

Cinematic grade, SIYI chroma cleanup, lasso-as-final-step, and later MapRepair were built so a map (ISM or ODM) could be checked against source photos and finished on CPU. See [04-maprepair.md](04-maprepair.md).

## Stage 8 — native ODM and AMD sketches

A Fedora-first native ODM install (Pixi 0.77.1, no Docker) plus a macOS build, every-sixth-image selection, FSR GeoTIFF export, and HIP/XDNA2 kernels were added for a Ryzen laptop target (~250 photos under 5 minutes). That target was **not met**. Host math was tested; HIP/XDNA2 devices were not available on the development Mac. Do not advertise those kernels as measured speedups.

## Design choices that were deliberate

- **Reject disconnected graphs** rather than silently drop frames. Recovery may add verified visual bridges; it never adds GPS-only edges.
- **One source per output pixel.** Averaging photos at seams looked worse than coherent labels plus low-frequency exposure.
- **No invented terrain.** Flat-plane lighting is a no-op (or a warm wash). Shadows in the photos stay. Optional OBJ is the only real occluder.
- **Empty output folders only.** Never overwrite a finished run.
- **Reports over vibes.** `registration_report.json`, `geographic_alignment.json`, `matching_report.json` exist so you can see a collapse.

## What we did not build

- Lens calibration / undistortion.
- Rolling-shutter correction.
- GCPs.
- Dense depth, DSM/DTM, or textured mesh (except via the separate ODM port).
- A Windows or Radeon backend for ISM.
- A guarantee that 564 4K frames finish end-to-end. That full flight was never verified after the workspace-guard and recovery work.
