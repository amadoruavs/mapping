# Aerial Delta Mapping

A complete, runnable experimental mapping pipeline with a CPU reference and an in-process Metal/MPS backend for matching, tile packing, compositing, and lighting. It includes the proposed 21×21 spatial color index, four 50×50 corner vocabularies, mean-centered matching, geographic verification, global planar registration, probabilistic source selection, and golden-hour lighting exports.

OpenDroneMap is cloned into `ODM/` at commit `1a7134170e6100cade3236486a279d6241f11f99`. ADM is a separate planar implementation. Saved default ODM 3.6.2 comparisons for SIYI and Archive are under `output/comparison/`; ODM performs a different, full 3D reconstruction workload.

The GPU alignment revision is available with `--backend metal --gpu-optimized`. It keeps visual matching, runs homography alignment on Metal/MPS with convergence checks, and selects coherent source regions. The default CLI output is one neutral PNG plus geometry and reports; match caches and diagnostic images are optional. Use `--export-profile vectors` for source references and transforms without raster rendering, or `--export-profile diagnostic --stylized-shadows` for inspection and lighting images. See the [lean pipeline, fresh timings and crispness comparison](docs/LEAN_PIPELINE.md). The original successful code is preserved in `checkpoints/full_metal_86s/`; historical results are documented in [GPU alignment](docs/GPU_ALIGNMENT.md).

## Run the supplied frames

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m adm.pipeline /Users/avneh/Downloads/predict \
  --allow-local --exclusions predict_exclusions.json
```

Outputs default to `output/project/`. `map_neutral.png` is the default map; transparent pixels mean no image coverage. The diagnostic export profile additionally writes `map_golden.png` and `preview.jpg`. No map contains generated scenery or filled-in terrain.

The provided JPEGs have no EXIF or XMP metadata. Local mode is explicitly requested and never assigns invented latitude/longitude. The geographic branch is tested end to end using synthetic overlapping photographs with actual embedded GPS EXIF.

## Flight settings and the EXIF adapter

Edit the constants at the top of `adm/config.py`:

- `ALTITUDE_AGL_M = 30.0` is a **placeholder**, not a measured flight altitude. Set the actual altitude above the mapped ground.
- `HORIZONTAL_FOV_DEG = 81.0` is the SIYI A8 Mini's published horizontal FOV at 1× zoom.
- `VERTICAL_FOV_DEG = None` derives vertical FOV from the actual raster dimensions, assuming square pixels. At 3840×2160 this gives approximately 51.32°. It is an explicit calibration assumption.
- Pitch defaults to −90° and roll to 0°. Non-nadir frames need correct gimbal orientation. Camera yaw is clockwise from true north.

SIYI specifies a Sony 1/1.7-inch optical-format sensor, 8 MP effective resolution, 21 mm equivalent focal length, F2.8, 81° horizontal FOV, and 93° diagonal FOV. The FAQ lists 65° vertical FOV and a rolling shutter. Those three FOV values do not describe a consistent 16:9 pinhole crop. No physical sensor dimensions or lens distortion coefficients are invented. Actual raster dimensions come from the file header; digital zoom and crop must agree with calibration.

Sources: [SIYI specifications](https://www.siyi.biz/en/product/tri-axis-single-camera-gimbal/a8-mini/spec/), [SIYI FAQ](https://www.siyi.biz/en/product/tri-axis-single-camera-gimbal/a8-mini/faq/), [SIYI manual v1.6](https://siyi.biz/siyi_file/A8%20mini/A8%20mini%20User%20Manual%20v1.6.pdf).

`adm/metadata.py` is the adapter to change when the real metadata arrives. It reads EXIF GPS coordinates, true-north `GPSImgDirection`, timestamps, orientation, and selected XMP absolute camera yaw/pitch/roll fields. It audits JPEG APP segments and records field provenance. EXIF altitude is recorded separately because it is typically not altitude above ground. Aircraft course and relative gimbal yaw are not substitutes for absolute camera yaw.

Canonical overrides can be supplied as `--metadata flight.json`. See `metadata.example.json` for the schema; its coordinates are examples only. Provide an entry for each frame whose metadata is incomplete. `gps_reference: "camera"` means GPS is the camera location and the model intersects each camera ray with the ground plane. Use `"image_center"` only if the supplied GPS describes the ground point at the image centre. Coordinates do not rotate with yaw; pixel offsets around the camera/centre do.

```sh
.venv/bin/python -m adm.pipeline /path/to/originals \
  --altitude 42 --metadata flight.json --output output/geographic
```

The default refuses geographic processing when GPS or yaw is missing. A JSON `--config settings.json` can override any `Config` field, including `skip_first` for a different flight. Do not apply the sample exclusion manifest to unrelated flights.

For near-nadir frames whose GPS is usable but recorded gimbal yaw is relative or inconsistent with image rotation, `--gps-centers` aligns the visual mosaic to GPS camera positions. It estimates map scale and north orientation from camera motion, reports GPS residuals and leave-one-out checks, and writes `heading_validation.json`. In this mode the 21×21 geographic index is built after visual alignment and is explicitly marked as inferred geometry, not an independent geographic fact checker. The default full-pose mode remains available.

The unannotated SIYI originals can be reproduced with:

```sh
.venv/bin/python -m adm.pipeline /Users/avneh/Downloads/101SIYI_IMG \
  --output output/101SIYI_IMG --config output/101SIYI_IMG/run_config.json \
  --exclusions output/101SIYI_IMG/exclusions.json --gps-centers
```

## Implemented stages

1. **Selection and audit.** Exclude the first seven sorted frames, manifest exclusions, conservative blue detector-label detections, byte duplicates, and near-identical thumbnails. Every decision is recorded in `selection.json`; originals are untouched. Label heuristics may need a reviewed exclusion manifest for other detector colors.
2. **Projection and neighbors.** A calibrated pinhole ray intersects a horizontal ground plane using altitude, camera yaw, pitch, and roll. WGS84 local east/north conversion supplies geographic coordinates. A spatial tree proposes intersecting footprints. Without metadata, a bounded sequence-neighbor window supplies visual candidates. This fallback assumes filename order approximates flight order.
3. **21×21 spatial multimap.** Every non-overlapping tile is retained in RGB `.npy` arrays, with partial edge tiles padded and their valid dimensions recorded. `tiles.sqlite` indexes quantized coordinate cells; multiple images in the same cell remain separate observations. In geographic mode this index is built before matching from the camera priors. In local mode it uses the resulting map coordinates and is explicitly a consistency index, not independent ground truth.
4. **50×50 vocabularies.** Four full-resolution corner patches per image are saved, together with signed per-channel deviations from each patch's mean. The display representation maps zero to middle gray; matching retains unclipped float deltas.
5. **Matching and verification.** The predicted transform rectifies the adjacent view before normalized correlation searches for each kernel. Acceptance requires normalized RGB-delta RMSE ≤0.02, sufficient correlation, a distinct peak, texture, and available independent geographic support. Four accepted corners can estimate a homography. Because corners often fall outside overlaps, SIFT/RANSAC is an explicit fallback. `--strict-corners` disables that fallback and fails on a disconnected graph; it is an experiment mode, not a promise that four corners suffice.
6. **Global alignment.** Robust sparse optimization jointly fits image homographies using pairwise matches. With metadata, a robust similarity aligns the visual solution with camera footprints, and inconsistent geometry causes an error. Registration errors are reported in pixels at the working image resolution. This is planar mosaicking, not 3D photogrammetric reconstruction or a survey-certified orthophoto.
7. **Source selection.** Disk-backed warped observations replace a huge in-memory image stack. The color mean defines selection weights `edge_confidence × exp(−mean_RGB_squared_distance / (2σ²))`. Seeded Gumbel sampling chooses actual source pixels; smoothly correlated random fields retain the correct per-pixel probabilities while reducing speckle and visible block boundaries. Low-frequency seam correction is applied after selection. Coverage and selected source IDs are exported for inspection.
8. **Lighting.** Decode sRGB to linear light, simulate several restrained sun environments around 30° elevation, and select using distance from the candidate mean. With an aligned height map, normals and approximate height-field cast shadows affect lighting. Without one, the stage uses a flat surface and a subtle warm SDR treatment. It does not infer heights from image brightness. Both the CPU shader and actual Metal compute kernel implement the same lighting equation.

## Geographic lookup

```python
from adm.tiles import TileIndex
from adm.geometry import LocalEarth

index = TileIndex('output/geographic/tiles.sqlite')
earth = LocalEarth(*index.info['origin_latlon'])
x, y = earth.xy(latitude, longitude)
observations = index.lookup(float(x), float(y), radius=1.0)
for image_id, tile_id, *_ in observations:
    rgb_21x21 = index.pixels(image_id, tile_id)
index.close()
```

`map_geometry.json` gives the pixel-to-local-world matrix and the geographic origin when available. The local tangent approximation is intended for a small drone site; continental mapping requires a proper projected CRS. Floating-point coordinates are not used as exact hash keys. GPS/calibration-derived positions are priors with uncertainty, not measured ground truth.

## Lighting and Metal

To provide terrain, use `--heightmap terrain.npy`, a finite 2D elevation array in metres with exactly the output mosaic shape. In local mode also provide `--heightmap-spacing metres_per_pixel`, and ensure map axes correspond to east/right and north/up for the azimuth to be meaningful. Geographic outputs are north-up. Existing photographic shadows remain baked into the source; this relighting cannot fully replace the original illumination.

`--backend metal` selects MPS matrix multiplication for exact descriptor matching, custom MSL for corner verification and mosaic processing, MPS separable image filtering, and fused or tiled lighting. A persistent Objective-C library shares buffers with NumPy and caches pipelines. No standalone `metal` compiler is required. Add `--gpu-optimized` to run global alignment on the GPU and replace raw tile storage with footprint vectors. SIFT extraction, RANSAC, metadata, geographic fitting and image codecs still use the CPU. The legacy profile retains CPU alignment and SQLite tiles. The CPU backend works on other platforms; Windows/Linux Radeon requires a separate backend.

See [Metal implementation and measurements](docs/METAL_BACKEND.md) for memory limits, numerical differences, validation, and benchmarks. Use a distinct output directory when comparing backends: exact GPU matching can choose different correspondences from CPU FLANN, and match caches are fingerprinted separately.

```sh
.venv/bin/python scripts/validate_metal.py
.venv/bin/python scripts/benchmark_metal.py
.venv/bin/python -m unittest discover -s tests -v
```

GPU validation covers filtering, matching, warping, tile packing, source selection, flat and terrain lighting, and a synthetic geographic run with resume. A GPU must be visible to the process; sandboxed environments can hide it. The reference `.hdr` and float `.npy` exports preserve linear values above 1, but they do **not** recover real HDR radiance lost from JPEGs. PNG is an 8-bit SDR display export.

## Outputs and next work

`report.json`, `matching_report.json`, `registration_report.json`, `metadata.json`, and `resolved_config.json` record decisions, assumptions and timings. Diagnostic exports additionally include corner reports, `source_ids.png` (0-based image indices; 65535 means uncovered), and `coverage.png`. Use `--cache-matches` to retain correspondence data for a later `--resume`; normal optimized PNG/vector runs omit this cache. The vector profile saves `project.json` for deferred rendering. Input hashes and implementation/configuration fingerprints protect cache reuse.

See `OPTIMIZATION.md` for the deferred performance plan. Reading all pixels and writing the output imposes a linear lower bound: the full job cannot scale sublinearly or constantly with the number of newly added pixels. No 300–450-frame benchmark or integrated-Radeon speed claim is made.
