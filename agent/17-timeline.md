# Timeline (September 2026)

Compressed from MEMORY.md. Use this to see what superseded what.

| When | What happened |
|---|---|
| 09-13 | Research docs in `docs/` for ODM GPU/golden-hour. No code change. ADM already exists as a Metal planar mapper. |
| 09-14 | GPU-optimized profile documented (30/34/120 s). Lean trim: 9.6 / 9.1 / **83.2 s**, byte-identical maps. LeanADM frozen. |
| 09-14 night | User rolls ISM back to `runtime.zip` / `ISM.metal` `c610ad5f…`. GPS/yaw priors removed. |
| 09-14–15 | 94-photo GPS fail 30.09 m. Symmetric transfer residual added. GPS 2.38 m. |
| 09-15 | LeanADM fails disconnected on mapv2. FSR road artifacts gated. User requests GPS mismatch *bypass* (ISM `allow_gps_mismatch` exists; current default False). |
| 09-15 | OpenSfM planar residual port. 94-photo old vs new ~21 s, no quality winner. Default stays `visual`. |
| 09-15 | Presentation: SIYI profile, supplied `pro_camera`, lasso-as-final, Metal grade. |
| 09-15 | MapRepair created as a separate folder. 94-photo scan+finish ~53 s then faster as blending/TIFF/JPEG landed. |
| 09-15 | OpenRT vendored. Golden-hour look on `odm.png` + 40 SIYI, 14.2 s. |
| 09-15 | MapRepair web UI + `run.sh`. 30 tests. Ubuntu container setup works. |
| 09-15 | Fedora native ODM app (every sixth photo, localhost 8765). Fedora machine never in front of us. |
| 09-16 | Mac native ODM + Pixi. MapRepair grading sliders. Shared `acceleration/` runtime. HIP/XDNA2 kernels host-only. 36-photo Ultra 523 s then 495 s. <300 s / 250 photos unmet. |
| 09-17 | MapRepair adaptive cinema default. 45 tests. Linux tarball rebuilt. |
| 09-26 | ISM `run.py` on `mapv2` again hits **median 30.09 m** GPS gate (see terminal archive). Not fixed in this dump. |

## What "current" meant on upload day

- MapRepair default finish: **adaptive**, OpenRT golden-hour, FSR 2×, web UI.
- ISM default: **visual** alignment, symmetric transfer **on**, interpolation **off**, SIYI + pro_camera, lasso on, GPS mismatch **not** bypassed.
- LeanADM: frozen lean backup.
- ODM native port: documented, SuperBuild not uploaded.

If those disagree with the files you are looking at, the files win.
