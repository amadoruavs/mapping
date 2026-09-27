> September 16 AMD kernel/hybrid test bundle: see [AMD_KERNELS.md](AMD_KERNELS.md) and [UPDATE_LAPTOP.md](UPDATE_LAPTOP.md).

## Offline adaptive cinema (2026-09-17)

New runs default to `FINISH_MODE='adaptive'`: scene-dependent tone and neutral
balance, noise-gated fine/medium texture enhancement, restrained local contrast,
and cinematic color with gamut compression. Existing local luminance extrema
limit sharpening halos around roof edges and runway markings. There is no
added grain, vignette or bloom in this mode. Coverage, map dimensions and
georeferencing remain unchanged by finishing. Analysis is recorded in
`adaptive_finish.json`; CPU strip processing is bounded and parallel. No model,
cloud service or network request is used during enhancement or grading.

The actual bundled [OpenRT](https://github.com/Project-10/OpenRT) sun/sky tracer
still runs. `ADAPTIVE_RAY_STRENGTH=.35` applies restrained golden illumination;
`ADAPTIVE_FSR_SHARPNESS=.10` avoids stacking heavy sharpening after the detail
pass. With no `MESH_OBJ`, the traced geometry is a flat plane: this is artistic
lighting, not recovered tree/building geometry or a new sun-position estimate.
The registered mesh option and neutral lighting remain available. Use
`--finish legacy` at the CLI (or `FINISH_MODE='legacy'`) for the previous recipe.

After processing, **Auto cinema · offline** analyzes the completed map and sets
editable exposure, white balance, S-curve, detail and noise-threshold controls.
**Inspect full-resolution detail**, or clicking the full image, shows a lossless
640×480 crop using full-resolution filter radii. Its pixels match the same area
of a full export. The comparison button also switches the detail inspector.
**Lossless PNG** is the default graded download: exact graded BGRA pixels and
transparency, with no JPEG recompression. JPEG quality 97 / 4:4:4 is optional.
PNG does not embed georeferencing: retain/rename the world file and CRS sidecar
(`.jgw` becomes `.pgw` for PNG) from the output folder when importing into GIS.
Older JPEG-only jobs cannot recover compression already present in their source.

Validation: 45 tests passed, including exact serial/parallel strip output,
hidden-pixel invariance, alpha, a monotone grayscale ramp, no overshoot on a
synthetic step edge, noise suppression gates, native crop/full export parity,
and byte-exact decoded PNG export. Real-image comparisons cover the 306-image
screenshot map and both field maps. Enhancement alone measured 0.25 s at
1838×1114, 2.51 s at 6500×4394, and 1.99 s at 6500×5978 on the development Mac;
these are single trials, excluding loading, ray tracing, FSR and export.
Browser Auto cinema + preview measured 0.49 s; no external requests, JS errors
or mobile overflow. Artifacts: `validation/adaptive_cinema/` (not in bundles).

Sharpening improves existing texture contrast; it cannot recover individual
grass blades that were never resolved, motion blur lost frequencies, clipped
sensor detail, or screenshot compression. FSR increases sampling, not measured
ground detail. These checks establish specific invariants, not universal image
quality or recovery of ground truth. The original inputs are always retained.

## Full-image grading preview (2026-09-16)

After processing, **Color grading & detail** shows the complete map fitted within
1920 × 1080, preserving its aspect ratio. Releasing a slider, editing a number,
or releasing a curve point automatically updates the preview. The two most
recent reduced images are cached in memory; stale requests cannot overwrite
newer settings. **Download full-resolution grade** applies a snapshot of the
current settings to the original lossless master. **Download original finish**
keeps access to the unadjusted finished JPEG.

52 numeric controls cover exposure, temperature/tint, contrast, highlights,
shadows, whites/blacks, gamma, saturation, vibrance, hue, fade, eight-color HSL,
RGB balance in shadows/midtones/highlights, clarity, texture, sharpening and halo protection.
Master and individual RGB curves support up to 16 editable points, smooth
shape-preserving interpolation, S-curve/matte presets, exact input/output
values and keyboard editing. Click to add, drag to reshape, arrow keys to nudge,
Delete to remove an interior point. Settings persist per map in this browser.
**Show original** compares against the same full-image preview; **Reset all**
restores neutral adjustments.

Preview detail radii scale with its reduction factor. Inspect the full-size
export to judge fine sharpening. New jobs retain `grade_master.bgra` (4 bytes
per output pixel); older jobs use their original JPEG. The source is never
modified, and repeated grading never regrades an earlier export. Full downloads
default to lossless PNG, with JPEG quality 97, 4:4:4 available. The existing alpha and geographic sidecars continue
to describe the same geometry; copy sidecars to the graded filename for GIS use.

Validated: 38 tests, including curves, selective HSL, alpha, full-frame preview
corners, cache invalidation and identical strip/whole-image grading. Browser
check on a real 3676×2228 master produced a 1782×1080 preview and 3676×2228 export;
slider-release update measured ~192 ms on the development Mac. Mobile has no
horizontal overflow; no browser JS errors. Results: `validation/grading_1080p/`.

## One-command local web UI

Copy the **whole MapRepair folder** to the other computer. From that folder:

```sh
bash run.sh
```

The launcher installs missing Linux build dependencies (apt-get on Ubuntu/Debian,
dnf on Fedora, or pacman on Arch), creates a private Python environment, installs
`requirements.txt`, compiles the C++ processors and opens **http://127.0.0.1:8765**.
First setup needs internet and may ask for a sudo password to install system
packages; run the script as your normal user. Arch retains its package-manager
confirmation for the required full upgrade. Later launches reuse the environment
and compiled code. Python 3.10+ and OpenCV 4 development libraries are required;
older/unsupported Linux distributions may need those installed manually.

1. Drop your TIFF (PNG/JPEG also accepted).
2. Drop the source **photos folder**, or click **Choose folder**. An optional
   path field reads photos directly without copying them.
3. Click **Run MapRepair**. Follow upload progress, current stage, elapsed time
   and the expandable log.
4. Click **Download full-resolution grade**. The preview is smaller; the download contains
   every output pixel.

The web UI uses the finish configured in `run.py`, including OpenRT golden-hour
lighting and final 2× upscaling. Input/output hard-coded paths in `run.py` are
replaced by the browser's selections. No Node, frontend build, GPU driver or
cloud account is needed. Keep the terminal open. Refreshing the browser preserves
a running job and completed download. Stop cancels the worker; retry creates a
new output folder. Results, masks, metadata and logs stay in `web_runs/<run-id>/`.
Interrupted uploads are reported and must be selected again. One job runs at a
time; original files and earlier results are not overwritten. Folder drops can
include subfolders; duplicate names are stored separately. Uploads stream to disk,
with two photo uploads at a time. Defaults allow 8 GB per file and 10,000 photos;
the image processor's output-pixel and available-memory limits still apply.

```sh
bash run.sh --no-browser             # Print the local URL without opening it.
bash run.sh --port 0                 # Pick an available port.
bash run.sh --setup-only             # Install/build, then exit.
MAPREPAIR_INSTALL_SYSTEM=0 bash run.sh  # Fail clearly instead of installing OS packages.
```

`MAPREPAIR_PYTHON=/path/to/python3` selects an interpreter; `OpenCV_DIR` can
point at a custom OpenCV 4 CMake directory. The environment and OpenRT build
caches include the platform and folder location, so copied Mac caches are not
executed on Linux. To keep the copy small, omit `.runtime/`, `build/`,
`__pycache__/`, `web_runs/`, `validation/`, `results/` and `output/`. Keep all
source files plus `web_ui/`, `openrt/` and `third_party/OpenRT/` (including its
license). Those caches rebuild on the destination.

For an environment that is already set up, `python web.py` launches directly.
The web layer uses Python's standard library and plain HTML/CSS/JavaScript.
It binds only to loopback; all files stay on the computer running the server.
Chrome was tested with real TIFF/file and directory drops, invalid-path recovery,
Stop/retry, refresh while running, desktop/mobile layouts and full JPEG download.
The 94-photo TIFF browser run completed processing in **31.81 s**, producing a
9358 × 7882 JPEG with the original alpha mask and geographic bounds preserved.
Thirty automated tests pass. Details: `validation/web_ui/`. The interface cannot
guarantee success for corrupt inputs, missing system dependencies or exhausted
RAM/disk; these errors are shown with a downloadable log.

The launcher was validated inside a fresh **Ubuntu 24.04 x86-64** container:
automatic dependency setup, GCC/OpenCV 4.6 compilation, repeat setup without
reinstalling, a directory name containing spaces, HTTP UI startup, and all
30 tests including a real TIFF-to-JPEG request. The x86-64 container ran under
emulation on this Mac; no native Linux speed claim is made. The same tests pass
on macOS arm64. Fedora/Arch install branches and Windows are not yet tested.
Linux setup/build/test records are in `validation/launcher_linux/`.

## Run

Requires Python 3.10+, the packages in `requirements.txt`, and a C++17 compiler
for the compact native backend. The default OpenRT ray-tracing backend additionally
needs CMake, C++20 and OpenCV 4 development headers/libraries; see
[OpenRT setup](openrt/README.md).
Use Xcode Command Line Tools on macOS, GCC/Clang on Linux, or a Visual Studio
C++ developer terminal on Windows. It uses CPU threads and requires no GPU API.
This version is tested on macOS/Apple silicon and Ubuntu 24.04 x86-64 in a
container. Windows builds have not been executed here. Devices need a compatible Python/OpenCV installation and
enough storage for the input/output map.

Latest saved run: `results/siyi_odm_openrt_golden_hour/final_2x.jpg` —
1754 × 1356 JPEG from the supplied `odm.png` and all 40 SIYI photos, **14.18 s**.
Photo checks took 13.45 s; OpenRT lighting took 0.100 s. Initial C++ compilation
is excluded. Repair checking ran; no patch passed all evidence gates (19 photos
consistent, 4 flagged, 17 unresolved). This does not certify the map error-free.
`comparison.jpg` shows input / restored previous finish / OpenRT golden hour.
The exact original alpha footprint is retained in `final_2x_alpha.tif` and
composited onto white in the JPEG. This PNG has no georeferencing sidecar.
Twenty tests passed; reports are in `validation/openrt_golden_hour/`.
The source, bundled C++/OpenRT code, build files, documentation and this saved
result are together inside `MapRepair/`. Input photo/map paths remain editable
at the top of `run.py`; Python and the documented build dependencies are required.
The original result copy under ADM's `output/` is also retained. Saved run reports
record the original execution paths.

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
  a new/empty output folder. Defaults now use the 40-photo `101SIYI_IMG` folder
  and `Downloads/odm.png` (877 × 678 RGBA).
- `SHARPEN_PROFILE='siyi_low_light'` — restored previous positive detail .65,
  fine sharpen .15 and CLAHE blend .35, with bounded chroma cleanup.
- `CINEMATIC_LOOK=None`, `COLOR_PRESET='pro_camera'` — restored previous color
  recipe. The experimental exposure/filmic treatments remain optional under
  `'natural_cinema'` and `'warm_documentary'`.
- `GRADE_SHARPEN=False` — keep the grader's color recipe, but disable its second
  pair of unsharp masks because the preceding stage already sharpens the map.
- `TIFF_VALUE_RANGE=None` — byte TIFFs use their original values; unsigned
  higher-bit TIFFs use their full type range. Set `(0,4095)` for 12-bit color
  stored in uint16, for example. Float/signed TIFFs require an explicit range.
- `JPEG_QUALITY=95`, `JPEG_BACKGROUND=(255,255,255)`, `SAVE_ALPHA_MASK=True` —
  JPEG delivery, white transparent areas, and a lossless TIFF mask sidecar.
- `RAYTRACE`, `RAYTRACE_BACKEND='openrt'`, `MESH_OBJ`, `RAY_STRENGTH`, sun
  elevation/azimuth — lighting. `'native'` selects the original compact tracer.
- `OPENRT_LIGHTING='golden_hour_5pm'`, `RAY_STRENGTH=.65`, sun elevation 15° /
  azimuth 255° — warm direct sun and cooler sky fill, calibrated to neutral
  luminance. This is an artistic 5 pm look, not a date/location solar calculation.
  `'neutral'` preserves the flat-plane pixels exactly during ray tracing.
- `HIGHLIGHT_BLOOM=.02` — restored previous subtle highlight glow.
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
   source with its fitted homography into each connected patch and surrounding
   context. Match its slow color/illumination difference against intact overlap
   **outside all repairs**, rejecting color outliers and capping corrections at
   24 byte levels per channel. Match excessive source texture with a bounded
   0–2 pixel blur only when intact overlap shows a mismatch; never add sharpness
   here. Blend the detail band inward over 32 native pixels and broad color over
   64 pixels using smooth ramps. Read all appearance references from the original
   map, making the result independent of patch processing order. Alpha and
   pixels outside selected repair tiles remain exact at this stage.
7. ISM-derived finishing: bounded chroma cleanup → configured Lanczos scale **1** →
   positive detail → blended Lab CLAHE → fine sharpen → contrast. Then apply
   the restored `pro_camera` grade (without its duplicate sharpening). Optional
   experimental cinematic finishing instead uses bright-neutral white balance bounded to ±4%,
   virtual exposures at −0.65 / 0 / +0.65 stops with soft local weights, restrained
   filmic contrast/highlight compression, warm highlights/cool shadows, selective
   green saturation, highlight diffusion and subtle warm halation. Exposure
   adjustments are capped at ±.055 (about 14 byte levels). Derived exposures
   redistribute the TIFF's existing tones; they are not new captures or recovered
   HDR information. The old supplied color presets remain available unchanged.
8. Actual OpenRT C++ ray tracing with warm sun/cool sky (or configured compact
   native backend), then subtle highlight bloom. Warm relighting fades near
   white to preserve highlights; it does not blur or resample image detail.
9. C++ **FSR1 EASU → noise-aware RCAS**, at 2× dimensions, as the last image
   processing stage. Preserve transparency and update the pixel-to-world
   transform for the new pixel centers. Crop back to the exact input raster
   extent, apply its original alpha/validity mask at 2×, and export JPEG,
   a lossless alpha TIFF sidecar, geographic metadata and timings.

The accumulator in the pasted script is replaced by local source selection:
overlapping inconsistent photos are not averaged throughout their footprints.
The distance-transform mask has an explicit zero border, so feathering really
falls to zero at image edges. Inputs and unrelated map pixels are preserved
during the geometry-repair stage. The later finishing stages intentionally
change the whole map's presentation.

## GPS and georeferencing

- Reads standard EXIF GPS and the SIYI XMP fields used by this batch.
- Reads embedded GeoTIFF CRS and affine transforms, including rotation/shear.
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

The current default is the [OpenRT integration](openrt/README.md). It uses a
compiled upstream library, not a renamed copy of the compact tracer. OpenRT's
sun/sky lighting is normalized against a **neutral-lit** reference plane, so the
golden-hour sun's color remains visible on the flat map. Source RGB colors are
luminance-normalized and direct sun/sky weights are .65/.35. The adapter handles
OpenRT's built-in sky pi scaling and exports a linear RGB field explicitly.
The warm treatment is blended at .65 strength, with highlight headroom protection.
The `'neutral'` preset still leaves a flat plane unchanged. With registered geometry the
adapter supports diffuse sun/sky occlusion. It does not synthesize terrain from
image intensity. The following describes the retained `'native'` backend:

`native.cpp` builds a triangle BVH and traces orthographic camera rays, sun
visibility and cosine-weighted sky rays, with up to two diffuse bounces. This is
actual triangle intersection code, independent of Metal hardware ray tracing.

The default scene is the previously selected **flat map plane**. The native
backend's illumination is the neutral reference, so it preserves the photograph exactly. It cannot
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
- `patch_blend.py` — intact-context color/detail matching and smooth blending.
- `map_io.py` — TIFF color-band/alpha/validity reading, extent and JPEG export.
- `presentation.py`, `cinematic_grade.py` — copied/adapted ISM CPU recipes.
- `cinematic_finish.py` — virtual exposure blending, restrained cinematic looks.
- `openrt_backend.py`, `openrt/`, `third_party/OpenRT/` — actual upstream C++ ray
  tracing integration, build files, pinned source and BSD license.
- `native.cpp`, `native.py` — threaded FSR, triangle BVH, C ABI and local build.
- `tests/`, `validation/` — checks and measurements; no production map data.

Outputs include `geometry_report.json`, `repair_tiles.png`, optional intermediate
PNGs, `raytrace_report.json`, `final_2x.jpg`, `preview.jpg`, `final_2x_alpha.tif`, `final_geometry.json`,
world/projection sidecars when available, and `timing.json`.
`repair_tiles.png` is a coarse uint16 source-index grid (zero means unpatched),
not a full-resolution image; its transform/tile size is in the geometry report.

### TIFF bounds, alpha and JPEG

`MAP_IMAGE` / `--map` accepts `.tif` / `.tiff`. Rasterio/GDAL reads tagged RGB or
gray bands and combines dataset validity (internal masks / nodata) with an
explicit alpha band. An extra NIR/undefined band is **not** treated as alpha.
Opaque black remains valid; partial alpha is retained without multiplying it
twice. Images with ambiguous color-band roles require a tagged RGB/gray export.

The delivered rectangle is exactly the input raster's rectangle, with the same
geographic outer corners and 2× pixel dimensions. It does not tighten the crop
around the nontransparent pixels or apply a new lasso. Interior holes and the
original irregular footprint remain excluded using the input mask. JPEG has no
alpha channel: excluded pixels become `JPEG_BACKGROUND` (white), and partially
transparent pixels composite against it. `final_2x_alpha.tif` preserves the exact
mask separately at output resolution. `.jgw` / `.prj` sidecars georeference the
JPEG; the mask TIFF carries its own georeference when the input has one. JPEG
is quality 95, 4:4:4 color, and is still a lossy presentation export. Intermediate
PNG stages remain available for inspection when `SAVE_STAGES=True`.

Photo features are cached at reduced resolution. Full source colors are decoded
again only for selected repairs. Warps are bounded to affected boxes. Chroma
scratch works in strips. Final FSR intermediates use disk-backed arrays; allow
about eight bytes per final pixel for its two temporary BGRA files, plus PNG
exports. Those temporary files are removed after a successful export. Large
maps still require RAM for image codecs, the map and OpenCV/Pillow filters.

## Validation

The current suite has **18 passing tests**, including actual OpenRT plane/shadow
and thread-repeatability checks, cinematic strip/alpha/hidden-color parity,
monotone gray ramps and appearance mismatch,
unchanged outside pixels, patch order independence, TIFF alpha plus internal
mask, nodata, higher-bit color conversion, a non-alpha fourth band, rotated
bounds, JPEG dimensions and the lossless alpha sidecar. On the known shifted
region, the current smoother blending reduces mean RGB error **28.37 → 4.19**.
The earlier narrow blend reached 1.85 on that fixture; the wider transition
trades some correction near the repair perimeter for a less visible seam.
Real-flight geometric accuracy has no independent ground truth here.

Historical initial implementation measurements follow; they used PNG export
and the grader's extra sharpening. Current results are listed below them.

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

### Initial 94-photo check and finishing run (before blending/JPEG changes)

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

### Blending and JPEG run before OpenRT/cinematic update

The revised tool completed the same 94-photo check and finishing job in
**43.78 s**, including intermediate saves and JPEG/mask export. Input was the
saved `final_test/map_neutral.png` (6500×5978); TIFF input was also exercised
end-to-end on a controlled fixture, with rotated georeferencing and masked
holes. This timing excludes the original map reconstruction and used an
already compiled native library.

| Stage | Seconds |
|---|---:|
| Scan and independent source checks | 22.31 |
| Repair and appearance matching | 3.19 |
| ISM sharpening | 5.65 |
| Color grade without extra sharpening | 1.67 |
| Flat-plane ray tracing | 0.003 |
| Highlight glow | 0.58 |
| Final 2× EASU + RCAS | 2.49 |
| JPEG, mask, preview and metadata export | 3.12 |
| Total including load and intermediate exports | **43.78** |

The same **42 tiles / 29 source photos** were selected as in the original run.
The smoother inward blend changes 1,048,649 pixels; original alpha and every
pixel outside the selected repair tiles remain byte-exact before finishing.
The exported mask was verified pixel-for-pixel against the original mask
enlarged 2×. The JPEG is **13000×11956**, quality 95 with no chroma subsampling.

- [Current JPEG](../output/maprepair94_blended_jpeg/final_2x.jpg)
- [Before/after detail comparison](../output/maprepair94_blended_jpeg/detail_comparison.jpg)
- [Preview](../output/maprepair94_blended_jpeg/preview.jpg)
- [Lossless alpha mask](../output/maprepair94_blended_jpeg/final_2x_alpha.tif)
- [Validation report](validation/blending/real94_summary.json)
- [Test log](validation/blending/tests.log)

Close-ups show reduced rectangular transitions and sharpening halos. Existing
stitching/parallax defects can still be visible; appearance matching does not
establish ground-truth geometry. No changes were made to ISM or LeanADM.

### Supplied South 1st Place TIFF run

Repair remains enabled. The supplied 4679×3941 RGBA GeoTIFF and the 94 `mapv2`
photos completed in **31.83 seconds**, including repair, finishing and export.
**21 tiles from 14 source photos** were patched. Unresolved/flagged regions
remain reported; the result is not a certification of the map's geometry.

The original alpha footprint and geographic outer bounds were verified exact.
Its valid-data bounding rectangle already covers the full input raster, so
there are no completely empty outer rows/columns to trim. Final **9358×7882**
JPEG uses white outside that footprint; the exact mask is saved separately.

- [JPEG](../output/south_1st_place_repaired/final_2x.jpg)
- [Preview](../output/south_1st_place_repaired/preview.jpg)
- [Alpha mask](../output/south_1st_place_repaired/final_2x_alpha.tif)
- [Validation and timings](validation/south_1st_place/summary.json)

### OpenRT, cinematic filters and virtual exposures

The full supplied-TIFF run with actual OpenRT, repairs and the new
`warm_documentary` finish completed in **34.08 seconds**. OpenRT's upstream
objects were already compiled; its incremental build check took .51 s.
Repair/check time was 23.38 s, gentle sharpening 2.72 s, exposure/cinematic
finishing 2.31 s, OpenRT tracing .087 s, FSR 1.29 s and final export 1.44 s.
Total includes map load and intermediate image saves.

Repair selections and the repaired image are identical to the preceding TIFF
run: **21 tiles / 14 source photos**. OpenRT's unoccluded plane returned exactly
1.0 illumination everywhere, and the ray-tracing image pass was byte-identical.
A separate controlled wall scene produced shadow values down to .4125 and
repeatable results across thread counts. Visible differences in this TIFF are
from gentler sharpening and cinematic/exposure treatment.

On this map the three virtual exposures changed individual color channels by at
most **13.48 byte levels** before the subsequent grade. Bright-neutral white
balance gains were RGB **1.040 / .991 / .974**. The final JPEG remains
**9358×7882**, with the exact original alpha sidecar and geographic footprint.
The 18-test suite and full-output mask/bounds checks pass. This is a presentation
experiment; original noise, blur and stitching errors can remain visible.

- [Cinematic JPEG](../output/south_1st_place_openrt_cinematic/final_2x.jpg)
- [Previous/current comparison](../output/south_1st_place_openrt_cinematic/comparison.jpg)
- [Preview](../output/south_1st_place_openrt_cinematic/preview.jpg)
- [Measured report](validation/openrt_cinematic/summary.json)
- [OpenRT build and implementation](openrt/README.md)

To reproduce the previous finish, set `CINEMATIC_LOOK=None`,
`SHARPEN_PROFILE='siyi_low_light'`, `RAYTRACE_BACKEND='native'`, and
`HIGHLIGHT_BLOOM=.02`; keep `GRADE_SHARPEN=False`.
