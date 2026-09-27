# GPS chunk stacking — Python prototype

This experiment places 50×50-pixel pieces of source photographs using GPS and
camera angles. It runs independently of LeanADM and ISM, on CPU with Python,
NumPy, Pillow and OpenCV. The existing LeanADM and ISM code is unchanged.

## Run

From `/Users/avneh/Code/ADM`:

```sh
.venv/bin/python experiments/gps_chunk_stack/run.py \
  --input /Users/avneh/Downloads/mapv2 \
  --output output/gps_chunk_stack_next \
  --count 10
```

Or edit the constants at the top of `run.py`: input/output paths, photo count,
chunk size, flight height, horizontal field of view and maximum map dimension.
The output directory must be new or empty. `--count 0` selects the whole folder;
only the first 10 photos have been benchmarked. `--no-save-chunks` omits the raw
tile pixel file while still projecting and compositing every tile.

Dependencies are already installed in the project's `.venv`. No compiler,
Metal device, ADM package, network service or pretrained model is required.

## Exact procedure

1. Select the first 10 JPEGs in filename order: `IMG_0006.jpg` through
   `IMG_0060.jpg`, from the current 94-photo `mapv2` folder. Read native
   3840×2160 pixels, GPS and gimbal yaw/pitch/roll directly from EXIF/XMP.
2. Divide each photo into a 77×44 grid: **3,388 tiles per photo, 33,880 total**.
   There are 1,200 partial edge tiles across the ten photos. Pad these only for
   storage and save their true dimensions so padding never becomes map content.
3. Convert camera lat/long to a shared local east/north coordinate frame, using
   WGS84 ellipsoid radii at the first camera position.
4. Use the configured **51.816 m above-ground height**, **81° horizontal FOV**,
   square pixels and central principal point. Rotate camera rays using the
   recorded gimbal angles, then intersect each ray with the flat ground plane:
   `ground = camera + (-camera_height / ray_up) * ray`.
   Apply this to tile centers and corners, preserving rotation and perspective.
5. Save every tile center as `(latitude, longitude, ground_height=0)` and
   `(east, north, up=0)`, with its photo index, source rectangle and footprint.
   The interactive viewer separates photo layers vertically for inspection;
   **that vertical spacing is photo index, not measured elevation**.
6. Project each tile to a north-up output grid and sample its pixels with
   bilinear interpolation. Shared tile edges use adjacent source pixels for
   filtering. A half-open ownership rule ensures one observation per photo per
   map pixel, avoiding artificial cracks and duplicate contributions at cuts.
7. Save two composites from the overlapping observations:
   - `map_selected.png`: choose the photo whose projected pixel is closest to
     its optical center in normalized image coordinates. Preserve one source's
     detail at each output pixel.
   - `map_mean.png`: blend observations using those same centrality weights.
     This is useful for seeing disagreement and doubled edges.
8. Save transparency, coverage counts, selected photo IDs, previews, metadata,
   coordinates, the interactive stack, and measured timings. No grading or
   sharpening is applied.

## Measured run — 2026-09-15

Results: `/Users/avneh/Code/ADM/output/gps_chunk_stack_10_verified`.

| Measurement | Result |
| --- | ---: |
| Photos | 10 of 94 |
| Tiles | 33,880 |
| Output | 2494×4096 |
| Output sampling | 0.03764 m/pixel, an assumed scale |
| Whole subprocess, including startup/exit | **6.1599 s** |
| Metadata, projection, coordinate export | 0.1155 s |
| Decode, tile storage, compositing, tile preview | 3.4323 s |
| Raster, contact-sheet and viewer export | 2.1841 s |
| Final diagnostic metrics | 0.2757 s |
| Peak child process RSS, external timer | 984,907,776 bytes / 0.917 GiB |
| Covered output pixels | 9,381,058 |
| Pixels with at least two observations | 7,765,982 |
| Maximum observations per pixel | 6 |

The external `timing.json` measures the entire subprocess; `report.json` also
contains an internal timer. Use the external values for end-to-end time and peak
memory. An initial run in `output/gps_chunk_stack_10` took 6.5776 s; after bounding
diagnostic scratch allocations, the repeat above took 6.1599 s. Both maps,
coverage, source IDs and the complete RGB tile stack were byte-identical.
These are ten-photo timings, not a 94- or 306-photo benchmark. Source tile storage
is 254,100,000 raw RGB bytes plus a NumPy header, included in both runs.

**Visual result:** the experiment produces a mosaic, but road/sidewalk pieces
jump at source boundaries. Averaging creates multiple outlines and duplicated
cars. The overlap mean-versus-selected difference is 11.9245/255 per channel;
this is a compositing disagreement measure, not a position or accuracy score.

All ten files contain GPS coordinates and a contradictory `GpsStatus` value
of `NO GPS connected`. Both are retained in the report. Gimbal heading and FOV
are used as provided/configured, without independent calibration. Reported XMP
relative altitude is logged; the projection uses the configured AGL consistently.

## Why tiles do not fully fix registration

The LeanADM attempt stopped because its verified visual-match graph was
disconnected: not every photograph could be placed relative to the others.
This experiment can place every photograph because GPS supplies an independent
position estimate. It does not require that visual graph.

Splitting a photograph into tiles gives smaller independently addressable
pieces, but it creates no new measurements. An error in camera GPS, heading,
height or FOV moves its tiles together. Terrain/building parallax also remains.
This result therefore supports a possible auxiliary path for initial placement
and overlap lookup. A further refinement would need observed local image
agreement to correct tile positions before producing a seamless final map.

## Saved outputs

- `map_selected.png`, `map_mean.png`: full-resolution BGRA maps.
- `comparison.png`: source selection beside overlap average.
- `preview.png`: labeled overview of the selected-source map.
- `input_contact_sheet.png`, `source_chunk_grid.png`: the actual inputs and cuts.
- `stack.html`: self-contained interactive point-stack viewer; drag to rotate,
  scroll to zoom, or flatten to the ground plane.
- `chunk_centers.csv`: one row per tile, including filename, valid dimensions,
  lat/long and east/north ground coordinates.
- `coordinates.npz`: numeric arrays, photo indices, footprint corners, camera
  positions and image-to-ground/image-to-map transforms. Float64 for coordinates.
- `chunks_rgb.npy`: memory-mappable `uint8[N,50,50,3]` RGB tiles, padded at edges.
- `coverage.png`: uint16 observation count; zero is uncovered.
- `source_ids.png`: uint16 selected photo index; 65535 is uncovered.
- `map_geometry.json`: local origin, assumed ground plane and raster transform.
  The PNG is not itself a tagged GeoTIFF.
- `metadata.json`, `report.json`, `timing.json`, `run.log`: provenance and results.

Read the tile stack without loading it all into RAM:

```python
import numpy as np
from pathlib import Path

out = Path('/Users/avneh/Code/ADM/output/gps_chunk_stack_10_verified')
tiles = np.load(out / 'chunks_rgb.npy', mmap_mode='r')
coordinates = np.load(out / 'coordinates.npz')
i = 0
x, y, width, height = coordinates['source_rect_xywh'][i]
rgb = tiles[i, :height, :width]
lat, lon, ground_height = coordinates['center_lat_lon_ground_z'][i]
```

## Verification and a future GPU implementation

Run the five focused CPU tests:

```sh
cd experiments/gps_chunk_stack
../../.venv/bin/python test_geometry.py
```

Tests verify camera axes and yaw orientation, geographic round trips, rejection
of rays crossing the horizon, exact partial-tile coverage, and bit-identical
whole-image versus tiled projection on a rotated/projective synthetic fixture.

For a future Metal auxiliary pass, the reusable inputs are the source rectangles,
photo IDs and image-to-map matrices. Projection, tile footprint indexing,
inverse sampling and source selection could become kernels. JPEG decoding,
metadata reading and file output still need a host. No Metal port was made in
this experiment, and faster compositing alone would not correct pose errors.
