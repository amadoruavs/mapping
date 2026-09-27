# What was not uploaded

The original `/Users/avneh/Code/ADM` tree was about **20+ GB**. This git repo is ~18 MB of source plus notes. After the local folder is deleted, the following are gone unless someone else still has them.

## Large trees (do not expect them)

| Path | Approx size | What it was |
|---|---:|---|
| `ODM-Fedora/` | 8.2 GB | Native ODM SuperBuild, Pixi env, install prefix |
| `output/` | 4.8 GB | All maps, comparisons, killerofmaps, SIYI/Archive ODM rasters |
| `MapRepair/` caches | ~3 GB | `.runtime`, `web_runs`, `web_runs_mac`, `build` |
| `odmtester/` | 1.5 GB | ODM test harness / data |
| `.venv/` | 1.1 GB | Development Python env |
| `ODM/` | 672 MB | Full ODM 3.6.2 clone |
| `acceleration/validation/` | 612 MB | Pixel-parity rasters |
| `third_party/fidelityfx_sdk/` | 395 MB | Full AMD SDK |
| `MapRepair/validation/web_ui/` | 350 MB | Browser screenshots / downloaded JPEGs |
| `final_test/` | 109 MB | Last ISM output attempts |
| `ISM/benchmarks/` rasters | 72 MB | Grader / geo comparison images |
| `LeanADM/benchmark/` | 53 MB | The actual 306-photo `map_neutral.png` |
| `third_party/opensfm_reference/` | 53 MB | OpenSfM checkout used for the port |
| `build/` | 45 MB | Root Metal dylibs |

## Still on someone's Downloads if they did not delete those

These were **never in git**. They are the real flights:

- `/Users/avneh/Downloads/mapv2` — 94 SIYI photos
- `/Users/avneh/Downloads/maps` — 564 4K photos
- `/Users/avneh/Downloads/101SIYI_IMG` — SIYI set (29–40 depending on the run)
- `/Users/avneh/Downloads/odm.png` — 877×678 ODM PNG used with MapRepair
- `/Users/avneh/Downloads/South-1st-Place-9-15-2026-orthophoto (1).tif` — GeoTIFF MapRepair input
- `/Users/avneh/Downloads/predict` — early ADM demo frames

## What you *do* have if you need to rebuild

- All MapRepair source, web UI, OpenRT pin, tests, and `MapRepair-linux.tar.gz`.
- All ISM source, `ISM.metal`, runtime, tests, `runtime.zip`.
- All LeanADM source and docs (not the 306-photo PNG).
- Root `adm/` + `shaders/` + `tests/` + `scripts/`.
- Research docs and MEMORY.md.
- A small MapRepair sample under `MapRepair/results/siyi_odm_openrt_golden_hour/`.
- ODM *notes* plus `notes/odm-fedora/ODM-Fedora.tar.gz` (source share of the Fedora port, ~1 MB). The SuperBuild/Pixi install is still gone.

## If you need ODM again

Clone upstream OpenDroneMap 3.6.2 (or current) and read `notes/odm-fedora/README_FEDORA.md` + `README_MACOS.md`. The SuperBuild is a long install. Do not assume the Pixi lock from 2026-09-16 still resolves.

## If you need the 306-photo map

It is not here. Re-run LeanADM on the original JPEGs if those JPEGs still exist. The timing will not match 83.2 s unless hardware and thermal conditions do.

## If you need browser-test evidence

JSON/log fragments from MapRepair validation were copied when they were small. The 350 MB screenshot set was not.
