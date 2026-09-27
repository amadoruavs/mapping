# How to run

## Shared Python for ISM / LeanADM / root `adm`

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Pinned in `requirements.txt`:

```
numpy==2.5.3
opencv-python-headless==5.0.0.93
pillow==12.3.0
scipy==1.18.1
```

Needs: Apple silicon, macOS 26+, Xcode Command Line Tools, a Metal GPU visible to the process.

## ISM (make a map)

```sh
.venv/bin/python ISM/run.py \
  --input /path/to/photo-folder \
  --output /path/to/brand-new-empty-folder

.venv/bin/python ISM/run.py --compile-only
.venv/bin/python ISM/run.py --self-test
.venv/bin/python ISM/run.py --geographic-solver opensfm_planar \
  --input /path/to/photos --output /path/to/empty
```

Or edit `INPUT_REPO` / `OUTPUT_DESTINATION` / `SETTINGS` at the top of `ISM/run.py`.

Output cap: 768 MiB (`MAX_OUTPUT_MB`). Logs: `run.log`, `timing.json` in the output folder.

## LeanADM (frozen lean mapper)

```sh
cd LeanADM
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m adm.pipeline /path/to/photos \
  --output output/map --config config_306.json \
  --gps-centers --backend metal --gpu-optimized --export-profile png
```

Change AGL in the config for a different flight. `config_306.json` assumes 60 m and width 640.

## Root `adm` (experimental tree)

```sh
.venv/bin/python -m adm.pipeline /path/to/photos \
  --allow-local \
  --output output/project
```

`--allow-local` is required when photos have no GPS. `--gps-centers` when they do but yaw is junk. `--metadata flight.json` for overrides (`metadata.example.json`). `--exclusions` for a denylist.

CLI default export profile historically drifted between `vectors` and `png`. Read `adm/config.py` and `docs/LEAN_PIPELINE.md` before promising what files appear. MEMORY.md records cases where `compose()` still wrote rasters under every profile — **recheck the code**.

## MapRepair (finish a map)

```sh
cd MapRepair
bash run.sh
# http://127.0.0.1:8765
```

Flags:

```
bash run.sh --no-browser
bash run.sh --port 0
bash run.sh --setup-only
MAPREPAIR_INSTALL_SYSTEM=0 bash run.sh
MAPREPAIR_PYTHON=/path/to/python3 bash run.sh
```

CLI:

```sh
python -m pip install -r MapRepair/requirements.txt
python MapRepair/run.py --compile-only
python MapRepair/run.py --map MAP.tif --photos PHOTOS --output EMPTY
python MapRepair/run.py --finish legacy ...
python -m unittest discover -s MapRepair/tests -v
```

MapRepair extras vs the Metal mapper: `pyproj`, `rasterio`. C++17 compiler required. OpenRT: CMake + C++20 + OpenCV 4 *dev* libraries (Homebrew `opencv@4` on Mac; keg-only, the launcher searches for it).

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python ISM/run.py --self-test
.venv/bin/python -m unittest discover -s MapRepair/tests -v
.venv/bin/python -m unittest discover -s LeanADM/tests -v
```

GPU tests need a real Metal device. Sandboxes can hide it.

## GPS chunk-stack experiment

```sh
.venv/bin/python experiments/gps_chunk_stack/run.py
```

No visual registration. Diagnostic only.

## Environment variables you may see

| Variable | Effect |
|---|---|
| `ADM_METAL_MEMORY_LIMIT_MB` | Override the 60%-of-RAM Metal allocation check |
| `MAPREPAIR_PYTHON` | Interpreter for `run.sh` |
| `MAPREPAIR_INSTALL_SYSTEM` | `0` skips apt/dnf/pacman |
| `OpenCV_DIR` | CMake prefix for OpenRT / native |
| `CXX` | Compiler for `native.cpp` |
| `ADM_ENABLE_XDNA` | Opt-in NPU path; needs Python 3.12 + XRT; untested on device here |
| `ODM_OFFLINE` / `ODM_FULL_REPORT` / `ODM_HYBRID` | Native ODM port only (not in this repo's binaries) |

## Output hygiene

- New empty folder every run.
- Do not commit `output/`, `web_runs/`, `.venv/`, `build/`.
- Input photos stay where they are. Nothing should rewrite JPEGs.
