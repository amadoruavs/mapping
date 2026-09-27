# MapRepair

Small, standalone **Python + C++17** map checker, local repair tool and finishing
pipeline. Edit the paths at the top of `run.py`. ISM is not imported at runtime;
the current ISM sharpening and supplied cinematic presets are copied here.

## Run

Requires Python 3.10+, the packages in `requirements.txt`, and a C++17 compiler.
Use Xcode Command Line Tools on macOS, GCC/Clang on Linux, or a Visual Studio
C++ developer terminal on Windows. It uses CPU threads and requires no GPU API.
This version is tested on macOS/Apple silicon; Linux/Windows builds have not
been executed here. Devices need a compatible Python/OpenCV installation and
enough storage for the input/output map.

```sh
python -m pip install -r requirements.txt
python run.py --compile-only
python run.py
```

Using the existing ADM environment, from the ADM directory:

```sh
.venv/bin/python MapRepair/run.py
# Diagnostics only:
.venv/bin/python MapRepair/run.py --check-only --output /tmp/map_check
# Optional command-line overrides:
.venv/bin/python MapRepair/run.py --map /path/map.png --photos /path/photos --output /path/new_result
# Tests:
.venv/bin/python -m unittest discover -s MapRepair/tests -v
```

The launcher compiles `native.cpp` automatically on first use. `CXX` can select
another compiler. `CMakeLists.txt` also builds the native shared library for C/C++
integration. Python, dependencies and a native compiler remain required; this
is source-portable software, not a precompiled binary for every device.

### Editable settings

At the top of `run.py`:

- `PHOTOS_FOLDER`, `MAP_IMAGE`, `OUTPUT_FOLDER` — source photos, existing map and
  a new/empty output folder. Defaults use the 94-photo `mapv2` batch and the
  ungraded `final_test/map_neutral.png`.
- `SHARPEN_PROFILE`, `COLOR_PRESET`, `REGIONAL_PRESET` — ISM recipes. Current
  defaults are `siyi_low_light`, `pro_camera`, and no regional grade.
- `RAYTRACE`, `MESH_OBJ`, `RAY_STRENGTH`, sun elevation/azimuth — lighting.
- `HIGHLIGHT_BLOOM=.02` — subtle highlight glow; zero disables it.
- `FSR_SHARPNESS=.2` — final FSR1 RCAS strength. Final enlargement is always
  **2× width and 2× height** (four times the pixel count).
- `CPU_THREADS=8`, `SAVE_STAGES=True`, `MAX_OUTPUT_PIXELS=200_000_000`.

Geometry thresholds live in `map_geo_checker.Settings`. It defaults to SIFT
for drone-image robustness and also supports ORB. The original ORB-style
`find_alignment` entry point is retained for small standalone checks.

## Exact pipeline

1. Read the map and source photos. Preserve the input map's alpha footprint.
2. Detect map features once at a bounded working size. Cache photo features.
   GPS selects search regions when explicit map georeferencing is available;
   a weak regional fit retries the whole map.
3. Fit photo-to-map homographies using ratio-tested, one-to-one matches and
   RANSAC. Hold out 20% of matches from fitting. Require held-out agreement,
   inlier support across the image, and a noncollapsed, plausible transform.
4. Track local source-to-map differences with forward/backward optical flow
   and normalized patch correlation. Flag coherent displacement in small
   tiles; low-texture, ambiguous and unsupported areas stay unresolved.
5. Independently match overlapping source photos. Compare their tie points in
   **native map pixels**, rather than mixing map and source-photo pixel units.
   A repair tile needs an agreeing neighbor whose actual tie-point hull covers
   that tile and whose direct transform also agrees at its center.
6. Patch only those verified tiles. Select one source per tile using fit
   confidence, spatial support, sharpness and local tracking scatter. Warp the
   source with its fitted homography into the affected bounding box. Feather
   inward at the patch boundary and at the real source-image boundary.
   Geometry repair applies no exposure or color matching.
7. ISM finishing: bounded chroma cleanup → configured Lanczos scale **1** →
   positive detail → blended Lab CLAHE → fine sharpen → contrast. Then apply
   the supplied `pro_camera` grade, including its own clarity/sharpness.
8. C++ ray tracing, then optional gentle highlight bloom.
9. C++ **FSR1 EASU → noise-aware RCAS**, at 2× dimensions, as the last image
   processing stage. Preserve transparency and update the pixel-to-world
   transform for the new pixel centers. Export PNGs, metadata and timings.

The accumulator in the pasted script is replaced by local source selection:
overlapping inconsistent photos are not averaged throughout their footprints.
The distance-transform mask has an explicit zero border, so feathering really
falls to zero at image edges. Inputs and unrelated map pixels are preserved
during the geometry-repair stage. The later finishing stages intentionally
change the whole map's presentation.

## GPS and georeferencing

- Reads standard EXIF GPS and the SIYI XMP fields used by this batch.
- Understands ISM's `map_geometry.json`/`presentation_geometry.json` with its
  explicit local origin, as well as world files with `.prj` or `MAP_CRS`.
- Manual `MAP_BOUNDS` means actual WGS84 **outer** map bounds; transforms use
  pixel centers. Rotated world-file axes are retained.
- A world file without a declared CRS triggers an explicit configuration
  error. Projected metres are never silently interpreted as longitude/latitude.
- Photo GPS extents do not establish the map's real bounds. If georeferencing
  is absent, use visual diagnostics without an absolute-position verdict.
- A camera tag reporting `NO GPS connected` is recorded as untrusted; it may
  narrow an initial search but does not certify absolute position.
- Image-center versus camera-GPS distance assumes near-nadir capture. It is
  reported as a proxy. A GPS-only discrepancy never causes an automatic patch.

This checks consistency, not survey accuracy. A globally shifted map can remain
internally consistent. Perspective/parallax, dynamic objects, weak overlap and
badly corrupted regions may remain unresolved. A single planar homography
cannot replace a terrain/mesh orthorectification model. `PATCHED` means some
supported tiles were replaced, not that an entire photo footprint is certified.

## Ray tracing and quality

`native.cpp` builds a triangle BVH and traces orthographic camera rays, sun
visibility and cosine-weighted sky rays, with up to two diffuse bounces. This is
actual triangle intersection code, independent of Metal hardware ray tracing.

The default scene is the previously selected **flat map plane**. Its illumination
is the neutral reference, so it preserves the photograph exactly. It cannot
produce building/tree shadows from a flat RGB map. The optional OBJ must already
be registered: X is input-map column, Y is input-map row, Z is height in the same
pixel-unit scale. For metre heights, divide by metres per map pixel. An OBJ is
additional geometry supplied by the operator, not geometry inferred from RGB.
Relighting photographed pixels is an artistic approximation to reflectance.

The small highlight bloom supplies the optional visible glow. It does not
estimate terrain, invent scene objects or blur the underlying detail layer.
FSR reconstructs sharper output edges; more output pixels do not add measured
ground detail. The source equations are a CPU adaptation of ISM's FP32
[FSR1 EASU/RCAS implementation](https://github.com/GPUOpen-Effects/FidelityFX-FSR).
AMD's MIT notice is included in `FSR_LICENSE.txt`.

## Files and resource use

- `run.py` — paths, settings, pipeline and wall/stage timers.
- `map_geo_checker.py`, `georeference.py` — geometric validation and repair.
- `presentation.py`, `cinematic_grade.py` — copied/adapted ISM CPU recipes.
- `native.cpp`, `native.py` — threaded FSR, triangle BVH, C ABI and local build.
- `tests/`, `validation/` — checks and measurements; no production map data.

Outputs include `geometry_report.json`, `repair_tiles.png`, optional intermediate
PNGs, `raytrace_report.json`, `final_2x.png`, `preview.png`, `final_geometry.json`,
world/projection sidecars when available, and `timing.json`.
`repair_tiles.png` is a coarse uint16 source-index grid (zero means unpatched),
not a full-resolution image; its transform/tile size is in the geometry report.

Photo features are cached at reduced resolution. Full source colors are decoded
again only for selected repairs. Warps are bounded to affected boxes. Chroma
scratch works in strips. Final FSR intermediates use disk-backed arrays; allow
about eight bytes per final pixel for its two temporary BGRA files, plus PNG
exports. Those temporary files are removed after a successful export. Large
maps still require RAM for image codecs, the map and OpenCV/Pillow filters.

## Validation

Eight portable tests cover known-error repair, unchanged unrelated pixels,
held geometry, feather boundaries, CRS/pixel-center transforms, FSR thread/chunk
repeatability, alpha, an unlit flat plane, an occluder shadow and the full pipeline.
The synthetic test's damaged-region mean RGB error fell **28.37 → 1.85**
(0–255 scale); this measures a known planar fixture, not real-flight accuracy.
The full 960×720 fixture produced a 1920×1440 final PNG.

The copied ISM sharpening is byte-identical on the compared fixtures. C++ FSR
versus the existing ISM Metal path differs by at most one code after EASU and
two after RCAS; mean error is below .001. One-thread/multithread/chunked C++
outputs are byte-identical. See `validation/ism_parity.json` and `tests.log`.

### Full 94-photo check and finishing run

The four-photo real-map diagnostic completed in 2.44 s. The full 94-photo run
completed in **52.62 s** on the Apple M4 Pro with eight CPU threads, starting
from the existing 6500×5978 neutral map. This excludes the original mapping run;
the C++ library was already compiled. Timings include this tool's decoding,
checking, repairs, finishing and PNG exports.

| Stage | Seconds |
|---|---:|
| Map/photo checks and source-pair verification | 24.35 |
| Local repair | 4.65 |
| ISM sharpening | 6.19 |
| ISM color grade | 3.69 |
| Flat-plane ray tracing | 0.004 |
| Highlight glow | 0.57 |
| Final 2× EASU + RCAS | 2.84 |
| Final PNG/preview/metadata export | 4.99 |
| Total, including load and intermediate exports | **52.62** |

**42 tiles from 29 photos** were patched (1,072,000 affected pixels). Final
photo statuses: 29 patched, 22 consistent, 32 flagged, 11 unresolved. Of 145
tested source pairs, 90 agreed, 25 disagreed and 30 could not be confirmed.
These counts describe the checks, not survey certification or a claim that
every visible seam has been fixed. Source/exposure seams remain visible.

The final image is **13000×11956**. Its alpha is the nearest-neighbor 2× source
footprint; no lasso is applied by this separate tool. Original inputs and the
main ISM package are unchanged. The flat-plane lighting is intentionally
neutral; the visible finishing comes from the requested sharpening, grade,
subtle glow and FSR.

- [Full-resolution result](../output/maprepair94_20260915/final_2x.png)
- [Preview](../output/maprepair94_20260915/preview.png)
- [Geometry report](../output/maprepair94_20260915/geometry_report.json)
- [Timing report](../output/maprepair94_20260915/timing.json)

Large run outputs live outside this small code folder. The input map hash,
source settings and stage times are retained in the reports.
