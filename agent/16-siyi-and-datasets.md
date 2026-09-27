# SIYI camera and the datasets we actually used

## Camera

SIYI A8 Mini on a gimbal, nadir mapping. See [11-flight-settings.md](11-flight-settings.md).

Typical raster: 3840×2160 JPEG. Metadata is messy: GPS numbers exist, `NO GPS connected` is set, yaw is untrustworthy, altitude is not AGL.

## Datasets named in MEMORY.md

None of the photo folders are in this git repo.

| Name | ~Count | Used for |
|---|---:|---|
| `101SIYI_IMG` | 29 in ADM comparisons; 40 in a MapRepair OpenRT run (IMG_0007–0046) | ODM vs ADM, golden-hour finish |
| `mapv2` | 94 | ISM geographic work, GPS collapse, MapRepair 94-photo scans, the 2026-09-26 30.09 m failure |
| `maps` | 564 4K | Disconnected graph, recovery, 600-iter collapse. E2E never finished |
| `killerofmaps` | 306 | Lean 83.2 s and earlier 119–143 s Metal runs |
| Archive | 63 | ODM comparison (21.64 cm/px ODM output) |
| `predict` | small, no EXIF | Early local-mode demo |
| South 1st Place TIFF | 1 map + 94 photos | MapRepair GeoTIFF path, 4679×3941, EPSG:32615 |
| `odm.png` | 1 map 877×678 RGBA, no CRS | MapRepair OpenRT demo |

## Selection rules that were applied to some runs

Generic ADM: skip first 7 sorted names, drop exact SHA-256 dups, drop near-duplicate thumbnails (MAE 1.5), drop blue detector-label heuristics.

ISM: `skip_first=0`.

Fedora ODM app: SHA-256 dups, then every sixth unique photo starting at index 1.

A later 36-photo Ultra ODM test used **all** already-deduplicated photos, no stride. Do not mix those timings with the production every-sixth app.

## Filenames

Inputs were sorted numerically. `IMG_0006`, `IMG_0007`, … SIYI names. Sequence-neighbor fallback assumes that order is flight order.

## What "retained" means

ISM/LeanADM either keep a photo in the connected graph or fail the run. They should not quietly omit frames. MapRepair can leave a photo unused if it never wins a tile.

## Ground sample distance

ADM/ISM canvas longest side is 6500 px by default. GSD is whatever that canvas implies after the GPS similarity, **not** a requested cm/px. ODM was asked for 5 cm (and later 2.3 cm on the native port). Finer ADM sampling is not more resolved detail.
