# LeanADM

Clean copy of the **non-FSR** ADM pipeline, frozen 2026-09-14. This is the last mapper whose 306-photo result was explicitly accepted as the lean baseline.

README: [LeanADM/README.md](../LeanADM/README.md).
Notes: [LeanADM/MEMORY.md](../LeanADM/MEMORY.md), [LeanADM/docs/LEAN_PIPELINE.md](../LeanADM/docs/LEAN_PIPELINE.md).

## Why it exists

FSR frame generation, cinematic experiments, and later ISM presentation work live in the parent/ISM trees. LeanADM exists so those experiments cannot contaminate the measured 83.2 s / 306-photo run.

The user said there was no need to rerun this backup. It was **not** separately re-validated after the copy.

## Saved claim (do not inflate)

- 306 source photos, all retained.
- Registration width 640, AGL assumption 60 m.
- Canvas 6500 × 4394, coarse compositor 128 MiB.
- Full-process wall **83.2055 s**.
- Neutral BGRA pixels and image-to-map transforms identical to the previous optimized 128 MiB run.
- Command: `adm.pipeline` with `--gps-centers --backend metal --gpu-optimized --export-profile png` and `config_306.json`.

The actual `benchmark/306/map/map_neutral.png` was **not uploaded** (tens of megabytes). Timing JSON and the map itself lived only on the original disk.

## How to run

```sh
cd LeanADM
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m adm.pipeline /path/to/photos \
  --output output/map --config config_306.json \
  --gps-centers --backend metal --gpu-optimized --export-profile png
```

Use the correct AGL/FOV for the new flight. `config_306.json` is the killerofmaps config, not a SIYI 170 ft config.

## How it differs from ISM

| | LeanADM | ISM |
|---|---|---|
| Shader layout | split `shaders/*.metal` | one `ISM.metal` |
| Default registration width | 640 in the 306 config | 1280 |
| FSR interpolation | absent | present but **off** |
| OpenSfM planar solver | absent | opt-in |
| Symmetric transfer residual | not the later default | **on** |
| Presentation / lasso | lean PNG, no SIYI grade | SIYI + pro_camera + lasso |
| 94-photo GPS collapse fix | not in this tree | yes |

If you need the validated speed number, quote LeanADM. If you need the current handheld mapper, use ISM.

## Failed LeanADM run you should know about

On 2026-09-15 LeanADM was pointed at `mapv2` (94 photos) with GPS centers, registration 640, coarse 128 MiB. It failed after 9.46 s: **disconnected match graph** despite 512 verified pairs of 2994 candidates. No map. Reports were under `output/leanadm_mapv2_20260915/` on the original machine. ISM's match-graph recovery was not backported into this backup.
