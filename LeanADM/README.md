# LeanADM — validated lean Metal backup

Clean copy of the non-FSR ADM pipeline, made on 2026-09-14. Experimental FSR,
frame interpolation and cinematic grading code are excluded. The working ADM
project remains in the parent folder.

## Saved benchmark

- 306 source photos, all retained; registration width 640, AGL assumption 60 m.
- 6500 × 4394 output; 128 MiB coarse compositor budget.
- Full-process wall time: **83.2055 seconds** in the saved run.
- Map pixels and geometry match the earlier optimized 128 MiB run exactly.
- Map: `benchmark/306/map/map_neutral.png`.
- Full timing and command: `benchmark/306/timing.json`.
- Matching, registration, geometry and configuration reports accompany the map.

The saved timing describes the original validated run. This backup has not been
rerun, as requested.

## Run independently

Requires macOS, a Metal/MPS-capable GPU, Apple Command Line Tools and Python 3.12.
Create an environment inside this folder and install the pinned requirements:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m adm.pipeline /path/to/photos \
  --output output/map --config config_306.json \
  --gps-centers --backend metal --gpu-optimized --export-profile png
```

The source photos and Python environment are not duplicated in the backup.
`config_306.json` is the saved benchmark configuration; use the correct flight
assumptions for other datasets. Normal runs create no large matching cache unless
`--cache-matches` or `--resume` is requested.

## Verification

Historical project notes are in `MEMORY.md`, and the original project README is
preserved at `docs/PROJECT_README.md`.
