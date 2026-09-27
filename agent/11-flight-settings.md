# Flight settings

Wrong AGL/FOV/yaw is the most common way to get a pretty, georeferenced, **wrong-scale** mosaic.

## Camera that was actually used

SIYI A8 Mini, published:

- Sony 1/1.7-inch class sensor, 8 MP, 3840×2160 typically.
- 21 mm equivalent, F2.8.
- 81° horizontal FOV at 1× zoom.
- FAQ also lists 65° VFOV and 93° diagonal. Those three numbers do **not** describe a consistent 16:9 pinhole. We used 81° HFOV and derived VFOV from pixel aspect (square pixels). At 3840×2160 that is about 51.32°.
- Rolling shutter is specified. **We do not model it.**

Sources are linked from the original `README.md` (SIYI spec/FAQ/manual v1.6).

## Fields in code

| Field | Where | Rule |
|---|---|---|
| `altitude_agl_m` | ISM SETTINGS, `adm/config.py` `ALTITUDE_AGL_M` | Height above the **mapped ground**. Never EXIF GPSAltitude. |
| `HORIZONTAL_FOV_DEG` | `adm/config.py` | Default 81. |
| `VERTICAL_FOV_DEG` | `adm/config.py` | `None` → derive from raster size. |
| Pitch / roll | metadata, default −90 / 0 | Non-nadir needs real gimbal values. |
| Yaw | EXIF `GPSImgDirection` or XMP | Clockwise from true north. Aircraft course + relative gimbal yaw is not a substitute. |
| `skip_first` | config | Generic ADM default 7; ISM 0; comparison configs 0. |
| `registration_width` | config | 640 on 306-photo lean; 1280 on later ISM. |
| `gps_tolerance_m` | config | Default 5. Final median gate. |

## SIYI metadata trap

On the `mapv2` 94-photo set:

- Every frame had GPS coordinates in EXIF/XMP.
- Every frame also had **`NO GPS connected`**.
- There were no horizontal-accuracy numbers.
- Recorded yaw could disagree with image orientation by ~180°.

Successful geographic fits treated GPS as a noisy camera-center prior and **ignored recorded yaw** for placement. Neighbor search used AGL + FOV circles + the 5 m tolerance, not yawed footprints.

`adm/metadata.py` is the adapter. It audits JPEG APPs and records provenance. `--metadata flight.json` overrides per frame. `gps_reference: "camera"` means the GPS is the camera; rays hit a horizontal plane. `"image_center"` means the GPS is the ground point under the principal point — only use that if you know it.

## AGL values that appear in this archive

| Value | Used for |
|---|---|
| 30.0 m | Generic `adm` placeholder. **Not measured.** |
| 51.816 m | ISM / 170 ft SIYI flights (`mapv2`, MapRepair neighbor search) |
| 60.0 m | LeanADM 306-photo `config_306.json` and older killerofmaps runs |

If someone copies ISM SETTINGS onto a 60 m flight, scale is ~14% off before you look at anything else.

## Neighbor search

With `--gps-centers` / ISM GPS mode:

- KD-tree of camera centers.
- Overlap if circles of radius ~ AGL * tan(HFOV/2) plus `gps_tolerance_m` intersect.

Wrong AGL → wrong pairs → disconnected graph or too many false pairs.

Without GPS: filename-order window (default 6 neighbors). Assumes the card order is the flight order.

## Geographic acceptance

GPS-center similarity requires:

- ≥ 3 images
- pitch within 5° of nadir
- ≥ 3 m center spread
- median residual ≤ `gps_tolerance_m` (5 m)

Soft-L1 scale on the residual is 0.75 m. Leave-one-out diagnostics exist in diagnostic profiles.

ISM writes `geographic_alignment.json` on failure so you can see the 30 m collapse without rerunning.

## World files

MapRepair JPEG gets `.jgw` + `.prj`. PNG grade downloads need `.pgw`. CRS is whatever the **input map** had. ISM local ENU is a small-site tangent plane, not a state plane until the similarity is applied.

Do not interpret a world-file `x` pixel size as degrees.
