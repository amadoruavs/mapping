# Notices

This repository mixes original AmadorUAVs code with pinned third-party sources.

## AmadorUAVs original work

Python launchers, Metal host code, MapRepair geometry checking, web UI, adaptive finishing, and the custom planar mapper in `adm/`, `ISM/`, `LeanADM/`, and `shaders/` were written for AmadorUAVs in 2026.

## Third-party code that ships here

| Component | Location | License |
|---|---|---|
| AMD FidelityFX FSR1 (EASU + RCAS) | `third_party/fidelityfx_fsr1/`, `MapRepair/FSR_LICENSE.txt`, `ISM/runtime/LICENSES.txt` | MIT (AMD) |
| OpenRT ray tracer (pinned) | `MapRepair/third_party/OpenRT/` | BSD-3-Clause |
| OpenCV / Pillow presentation ports | notices in `third_party/PRESENTATION_NOTICES.txt` and `ISM/runtime/LICENSES.txt` | Intel / Pillow licenses |
| OpenSfM geographic residuals (ported equations) | documented in `docs/OPENSFM_PORT.md` and `ISM/runtime/OPENSFM_PORT.md` | AGPL-3.0 (upstream OpenSfM). The port is a residual/equation adaptation, not a full OpenSfM binary. |

The full FidelityFX SDK (~395 MB) and an OpenSfM reference tree (~53 MB) were **not** uploaded. Pins and hashes are recorded in `MapRepair/source_provenance.json` and `ISM/runtime/bundle_manifest.json`.

OpenDroneMap itself is AGPL and is **not** vendored here. See `notes/odm/` and `notes/odm-fedora/` for the port notes only.
