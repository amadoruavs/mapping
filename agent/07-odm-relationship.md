# How this repo relates to OpenDroneMap

## ODM is not in this git repository

The original machine had:

- `ODM/` — clone at `1a7134170e6100cade3236486a279d6241f11f99` (~672 MB).
- `ODM-Fedora/` — native SuperBuild + Pixi (~8.2 GB including installs).
- `ODM-Fedora.tar.gz` — source share of the Fedora port (~1 MB). SHA-256 is in `notes/odm-fedora/`. The tarball itself was left behind as optional; recreate from notes if needed.

What *is* here: `notes/odm/`, `notes/odm-fedora/`, and the research docs under `docs/`.

## What was compared

Saved default ODM 3.6.2 runs (ARM64 Docker, `--orthophoto-png --optimize-disk-space`):

- DSPSIFT / high / 10,000 feature target / FLANN / incremental SfM.
- Medium point cloud. `--cuda-device -2` (CPU).
- `skip_3dmodel=false`, `use_3dmesh=false` → both 3D and 2.5D meshes; the ortho uses the 2.5D mesh.
- Requested GSD 5 cm. Archive came out GSD-limited at ~21.64 cm/px.

ADM/ISM does none of those stages. Speed ratios (ODM 280–350 s vs ADM 9–30 s on the small sets) compare **different workloads**. Never write "we are 10× faster than ODM" without saying planar vs SfM+MVS.

## Geographic residuals vs ODM accuracy

Archive held-out nominal-center RMSE was ~2.99 m for both ADM and ODM. Those were supplied/synthetic centers, not surveyed checkpoints. SIYI GPS participates in ADM's fit, so residuals there are not independent.

SSIM of ADM vs ODM on a common grid: SIYI 0.074, Archive 0.600. Agreement with ODM is not ground truth. The optimized compositor made SIYI SSIM *worse* than legacy ADM while making a goal-detail crop look better. There is no blanket quality win.

## Native ODM port (Fedora / macOS)

User asked for standard WebODM/ODM quality, native Python orchestration, no Docker, Fedora x86_64, then a localhost site with whole-folder input.

Selection rule they insisted on: SHA-256 dedup, then **every sixth unique photo starting at the first** (1, 7, 13, …). That reduces overlap. Full-flight WebODM quality parity was **not** established.

Later a 36-photo Ultra run (no stride) on the Mac completed in 523 s, then 495 s in a follow-up. Target of <300 s for ~250 deduplicated 4K photos without quality loss was **unmet**. `--fast-orthophoto` is not quality-equivalent (skips dense, 8× coarser DSM step).

Default GSD instruction at one point: 2.3 cm/pixel.

`QUALITY_UPDATE.md` / `PERFORMANCE_UPDATE.md` / `AMD_KERNELS.md` in `notes/odm-fedora/` describe FSR GeoTIFF export, CPU caps, and HIP/XDNA2 work. HIP auto-select requires byte parity **and** ≥10% transfer-inclusive speedup. No HIP device was present on the development Mac.

## When to use which

| Goal | Tool |
|---|---|
| Survey-ish 3D, mesh, DSM, "real" ortho | ODM (upstream or the Fedora port) |
| Fast nadir mosaic on Apple silicon | ISM |
| Reproducible 306-photo lean number | LeanADM |
| Check / finish an existing ortho | MapRepair |
| Laptop AMD experiment | notes + `acceleration/`; do not claim device speed |

## Licensing reminder

ODM / OpenSfM are AGPL. We ported OpenSfM *equations* into ISM (`docs/OPENSFM_PORT.md`). If you copy more OpenSfM code into this repo, you inherit AGPL obligations. Do not vendor ODM into a casual "MIT" dump.
