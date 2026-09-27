# Start here

You are looking at the AmadorUAVs mapping archive. The local checkout that produced this work was deleted after upload. If a fact is not in this repository, it is gone unless it was already on GitHub or in someone's Downloads folder.

## What the club actually wanted

Fast, decent-looking maps from SIYI A8 Mini nadir photos, on a laptop, without Docker if possible. OpenDroneMap (ODM) 3.6.2 was the quality reference. A custom Metal planar mapper was built because ODM's full SfM + OpenMVS + mesh + ortho path is a different, slower workload.

Two products came out of that:

1. **ISM** — make a map from photos (Apple silicon only).
2. **MapRepair** — take a map you already have (ODM or ISM) and check / finish it on CPU (Mac or Linux).

## Read in this order

1. This file.
2. [01-what-this-repo-is.md](01-what-this-repo-is.md) — folder map.
3. [03-how-we-built-custom-mapping.md](03-how-we-built-custom-mapping.md) — the actual design history.
4. [08-hard-lessons.md](08-hard-lessons.md) and [09-do-not-claim.md](09-do-not-claim.md) — so you do not lie in a write-up.
5. [12-known-failures.md](12-known-failures.md) — the bugs that will bite you on the next flight.
6. The package you were asked to change: [04-maprepair.md](04-maprepair.md) or [05-ism.md](05-ism.md).
7. [MEMORY.md](../MEMORY.md) for dated measurements. Recheck source before quoting a number.

## One-sentence identity test

Ask: "Does this code estimate 3D cameras and a surface, or does it warp photos onto a flat plane?"

- ODM: 3D cameras + surface + ortho.
- ADM / ISM / LeanADM: flat plane, one homography per photo, GPS similarity after the fact.
- MapRepair: no reconstruction. It warps source photos onto an existing map to look for disagreements.

If you catch yourself writing "orthorectified" or "photogrammetric reconstruction" about ISM, stop.

## Default flight assumptions that are still in the launchers

These are **not** read from EXIF altitude:

| Field | Typical value in this repo | Meaning |
|---|---|---|
| AGL | 51.816 m (170 ft) in ISM; 30 m placeholder in generic `adm` | Height above the mapped ground |
| HFOV | 81° | SIYI A8 Mini published 1× zoom |
| Pitch | −90° (nadir) | Assumed unless XMP says otherwise |
| GPS | present in EXIF **and** tagged `NO GPS connected` on the SIYI flights | Unverified |

See [11-flight-settings.md](11-flight-settings.md).

## If you were asked to "just run it"

- Finishing an existing GeoTIFF/PNG: `MapRepair/run.sh`.
- Making a new map on a Mac: `ISM/run.py` with a **new empty** output folder.
- Reproducing the 306-photo lean result: `LeanADM/` + `config_306.json`. That backup was not rerun after it was frozen.

Output directories must be empty. Existing results are treated as sacred.

## If the GPS check fails

ISM will raise:

```
GPS camera positions contradict the visual mosaic: median X.XX m exceeds 5.00 m
```

This happened on the 94-photo `mapv2` set when the visual solve collapsed in scale (median residual ~30 m). The measured fix was bidirectional source-pixel transfer error (`alignment_symmetric_transfer=True`), which brought the same set to ~2.4 m median. Turning on `allow_gps_mismatch` only skips the final gate. It does not repair geometry. Details: [12-known-failures.md](12-known-failures.md).
