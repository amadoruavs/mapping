# MapRepair

Standalone folder. **No runtime imports of ADM or ISM.** Portable Python 3.10+ and a C++17 compiler. OpenRT path needs CMake, C++20, and OpenCV 4 headers.

User-facing README: [MapRepair/README.md](../MapRepair/README.md).

## What it is for

You already have a map (ODM PNG/GeoTIFF, ISM PNG, whatever). You also have the photos that went into it. MapRepair:

1. Registers those photos to the map.
2. Looks for tiles where another photo disagrees and the geometry is well supported.
3. Optionally patches those tiles with a slow, feathered blend.
4. Finishes appearance (adaptive cinema by default).
5. Optionally traces a flat plane (or a registered OBJ) in OpenRT.
6. Upscales 2× with FSR1 EASU + RCAS.
7. Writes JPEG/PNG, an exact alpha TIFF, and world-file sidecars.

It will **not** build a map from photos alone. If you point it at a bad mosaic, you get a finished bad mosaic plus a report.

## How to run

```sh
cd MapRepair
bash run.sh                  # web UI on loopback
# or
python run.py --map MAP --photos PHOTOS --output EMPTY_DIR
python -m unittest discover -s tests -v
```

Edit the constants at the top of `run.py` if you run it with no flags. They still name `/Users/avneh/Downloads/...`.

`bash run.sh` creates a private `.runtime` venv keyed to platform + install path + interpreter. Copied Mac build directories will not run on Linux.

## Geometry rules that matter

Implemented in `map_geo_checker.py`, `georeference.py`, `patch_blend.py`, `map_io.py`.

- SIFT by default (ORB selectable). Cache features.
- GPS ROI is used only when the map actually has a world file / CRS. Never treat world-file metres as lon/lat.
- Never treat source-GPS extents as verified map bounds.
- EXIF + SIYI XMP are read. `NO GPS connected` means the GPS is untrusted.
- RANSAC keeps 20% of matches held out.
- A tile is patched only if **another source** independently agrees at that location (direct transform + measured tie-point hull).
- One source per tile, chosen by fit / sharpness / support / track scatter.
- Warp only the affected bbox. Zero-border distance transform + inward feather.
- **No exposure/color matching in the geometry stage.** That is a later finish problem.
- Weak fits stay flagged. GPS-only mismatch stays flagged.
- `patch_blend.py` estimates a robust slow RGB offset from intact overlap (clamped ±24 levels) and feathers 32 px (detail) / 64 px (color). The original map is the immutable reference. Source order must not change the result.

`map_io.py` uses rasterio/GDAL. A fourth band is **not** assumed to be alpha. Extra NIR is not a mask. Valid black is valid. UInt16 uses the full numeric range unless `TIFF_VALUE_RANGE` is set.

The export crop is the **original raster rectangle at 2×**, not a tight alpha bbox. JPEG composites transparency on white; `final_2x_alpha.tif` keeps the mask.

## Finish modes

| Mode | How |
|---|---|
| `FINISH_MODE='adaptive'` (default) | `adaptive_finish.py`: scene proxy, Haar noise, monotone tone, noise-gated luma detail, edge protection, gamut compression. No grain/vignette/bloom. |
| `FINISH_MODE='legacy'` | Earlier sharpen + `pro_camera` grade + optional cinematic looks. |

OpenRT (`RAYTRACE_BACKEND='openrt'`) is real upstream at `ef6205f8ca51fd123da4a2f0b7b7a6ad27809b80`, slightly patched for GCC (`std::sqrt` / `std::pow`). Default lighting `golden_hour_5pm` is artistic (sun elevation 15°, azimuth 255°), not a computed solar position. With no `MESH_OBJ` the geometry is a plane: warmth, not new tree shadows.

FSR 2× means spatial width/height ×2, **not** temporal FSR2/FSR3.

## Web UI

`web.py` + `web_ui/` + `web_worker.py`. Loopback only. CSRF-ish POST token. One active job. Runs land in `web_runs/<uuid>/` (that directory is gitignored and was not uploaded).

After a run, 52 grading sliders operate on a cached ≤1920×1080 BGRA proxy. Full export reads `grade_master.bgra`. Auto cinema derives editable settings from the finished master at half strength. Detail inspector is a lossless 640×480 native crop that must match the full export including spatial halos.

PNG download is lossless and has alpha but **no embedded CRS**. Keep/rename `.jgw` → `.pgw`.

## Tests that existed when this was packed

The README and MEMORY.md record 45 tests for adaptive cinema, 38 for 1080p grading, 30 for the web/launcher path, plus geometry/FSR/OpenRT checks. Run them. Do not assume the numbers in MEMORY.md still match if you edit kernels.

## Linux share archive

`MapRepair-linux.tar.gz` (SHA-256 in the sibling `.sha256` file) is a source-only tarball: no venv, no build, no images. First Linux GCC build needed three upstream OpenRT math-namespace patches; those are already in the tree.

## Things MapRepair is honest about

- Zero patches can meet the gates. That is a valid result (the SIYI ODM PNG run: 19 consistent, 4 flagged, 17 unresolved, 0 patches).
- Remaining stitch/parallax/exposure seams in the *input* map are not MapRepair's job unless a tile passes every evidence gate.
- Sharpening cannot recover grass blades that were never resolved.
- No surveyed accuracy claim.
