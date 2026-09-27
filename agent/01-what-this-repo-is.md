# What this repository is

A source archive, not a full data archive. Large maps, venvs, ODM SuperBuilds, and browser-test screenshots were left on the original machine. See [15-what-was-left-behind.md](15-what-was-left-behind.md).

## Layout

```
mapping/
  AGENTS.md                 Future-agent entry (read first)
  README.md                 Human entry
  MEMORY.md                 Dated implementation log (2026-09-13 … 2026-09-17)
  NOTICE.md                 Licenses
  requirements.txt          Python pins for the Metal mapper
  MapRepair/                Portable map checker + finisher + web UI
  ISM/                      Integrated Metal mapper (latest custom package)
  LeanADM/                  Frozen non-FSR mapper + 83.2 s / 306-photo claim
  adm/                      Shared Python package used by the root mapper
  shaders/                  Metal/ObjC for the root mapper (ISM.metal is separate)
  docs/                     Technical write-ups (lean pipeline, FSR, OpenSfM port, GPU)
  agent/                    These notes
  acceleration/             Shared CPU/HIP/XDNA2 runtime sketches (host-tested)
  experiments/              GPS chunk-stack experiment (no visual graph)
  checkpoints/full_metal_86s/  Historical source checkpoint
  backups/                  ISM rollback snapshot from 2026-09-14
  tests/                    Root mapper tests
  scripts/                  Validation / benchmark helpers
  third_party/              Small FSR1 pin + presentation notices
  notes/odm/                ODM clone README only
  notes/odm-fedora/         Native Fedora/macOS ODM port notes
  MapRepair-linux.tar.gz    Source-only MapRepair share archive
```

## The three code lineages

Treat them as siblings, not imports of each other at runtime unless a file says so.

| Lineage | Role | Imports the others? |
|---|---|---|
| `adm/` + `shaders/` | Original experimental mapper at repo root | No |
| `LeanADM/` | Clean copy of the lean non-FSR mapper, 2026-09-14 | Self-contained `adm/` + `shaders/` |
| `ISM/` | Later integrated package: one `ISM.metal`, `runtime/adm`, SIYI defaults | Self-contained. Does **not** import parent `adm/` |
| `MapRepair/` | Separate finisher | Does **not** import ADM/ISM at runtime |

If you "fix ISM by editing `adm/` at the repo root", ISM will not see it. Edit `ISM/runtime/adm/` and `ISM/ISM.metal`.

## What each program consumes

| Program | Input | Output |
|---|---|---|
| ISM / LeanADM / `adm.pipeline` | A folder of nadir JPEGs | `map_neutral.png`, optional `map_graded.png`, geometry JSON, reports |
| MapRepair | An existing map raster + the source photo folder | `final_2x.jpg` / PNG, alpha TIFF, world file, check report |
| `experiments/gps_chunk_stack` | Photos + GPS + assumed AGL/FOV | Diagnostic tile stack, not a serious map |

## Historical names you will see in MEMORY.md

- **ADM** — Aerial Delta Mapping, the project name.
- **metal_optimized** — a CLI profile (`--backend metal --gpu-optimized`), not a module.
- **metal_lean** — the 2026-09-14 trimmed pipeline.
- **killerofmaps** — the 306-photo dataset used for the 83–120 s timings.
- **mapv2** — 94 SIYI photos used for ISM geographic work.
- **101SIYI_IMG** — 29–40 photo SIYI set used in earlier comparisons.
- **South 1st Place** — a user-supplied GeoTIFF that MapRepair repaired.
- **ODM-Fedora** — native ODM 3.6.2 SuperBuild for Fedora x86_64, later also built on macOS.

## Files that look important but are not the live mapper

- `checkpoints/full_metal_86s/` — old snapshot.
- `backups/ism_before_rollback_20260914_232001/` — rejected GPS/yaw-prior experiment.
- `ISM/runtime.zip` — byte-for-byte rollback source used on 2026-09-14. The live tree is `ISM/runtime/` + `ISM.metal`.
- `OPTIMIZATION.md` — deferred performance plan, not a description of current code.
