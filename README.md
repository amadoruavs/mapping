# mapping

Archive of AmadorUAVs aerial mapping work. Two programs live here:

- **[MapRepair](MapRepair/)** — drop an existing orthophoto (TIFF/PNG/JPEG) plus the source photos. It checks geometry against those photos, blends conservative patches, and finishes the map (sharpen, grade, optional OpenRT lighting, 2× FSR1). Portable Python + C++. Has a localhost web UI (`bash MapRepair/run.sh`).
- **The custom mapper** — a planar Metal/MPS photomosaic written for Apple silicon. Latest integrated package is **[ISM/](ISM/)**. The validated non-FSR backup is **[LeanADM/](LeanADM/)**. Shared Python lives in **[adm/](adm/)** and shaders in **[shaders/](shaders/)**.

This is **not** OpenDroneMap. ODM 3.6.2 was used as a comparison and later ported natively (`notes/odm-fedora/`). Those SuperBuild trees were too large to keep in this repository.

## Start here

| Audience | Document |
|---|---|
| Future agents | [AGENTS.md](AGENTS.md) then [agent/00-start-here.md](agent/00-start-here.md) |
| Humans who want to finish a map | [MapRepair/README.md](MapRepair/README.md) |
| Humans who want to *make* a map on a Mac | [ISM/GUIDE.md](ISM/GUIDE.md) |
| How we got here | [agent/03-how-we-built-custom-mapping.md](agent/03-how-we-built-custom-mapping.md) |
| Dated measurements and caveats | [MEMORY.md](MEMORY.md) |

## Quick run

MapRepair (Linux or macOS):

```sh
cd MapRepair
bash run.sh
# opens http://127.0.0.1:8765
```

ISM (Apple silicon, macOS 26+, Metal 4):

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python ISM/run.py --input /path/to/photos --output /path/to/empty-folder
```

Edit the path constants at the top of `ISM/run.py` and `MapRepair/run.py` before using the hardcoded defaults. Those defaults still point at a 2026 development machine.

## What this software is

A **planar photomosaic**. Each photo gets one homography. GPS, when present, sets scale and north after visual alignment. It does not do SfM, dense matching, meshing, or survey-certified orthorectification.

MapRepair is a **finisher**, not a reconstructor.

## Hardware that was actually used

Development: Apple M4 Pro, 24 GB unified memory, macOS 26, Metal 4. MapRepair was also installed in an emulated Ubuntu 24.04 x86-64 container. AMD/Fedora GPU kernels were written and host-tested; they were **not** executed on Radeon or XDNA2 hardware in this archive.

## License

Mixed. Custom code is for AmadorUAVs. Bundled third-party code keeps its original licenses (AMD FidelityFX MIT, OpenRT BSD-3, OpenCV/Pillow notices, OpenSfM AGPL for the residual port). See [NOTICE.md](NOTICE.md) and [agent/14-licenses-provenance.md](agent/14-licenses-provenance.md).
