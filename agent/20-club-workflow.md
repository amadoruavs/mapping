# Suggested club workflow

This is how the pieces were meant to be used together, not a promise that every flight will succeed.

## Fast map on a Mac (ISM)

1. Copy the photo folder off the SIYI camera. Do not rename out of numeric order if you need sequence neighbors.
2. Set AGL to the **flown** height above the field, not EXIF altitude. 170 ft → 51.816 m is what ISM still has hardcoded.
3. `python ISM/run.py --input PHOTOS --output a-new-folder`
4. If it dies on `GPS camera positions contradict the visual mosaic`, open `geographic_alignment.json`. If the median is ~30 m, you likely collapsed — see [12-known-failures.md](12-known-failures.md). Do not "fix" it by raising the tolerance unless an officer wants a sloppy map.
5. If it succeeds: `map_neutral.png` is geometry; `map_graded.png` is the pretty one (lasso already applied).

## Better geometry from ODM (when you have time)

Use upstream ODM or the Fedora/macOS native port (`notes/odm-fedora/`). Expect minutes, not 20 seconds. Every-sixth selection is a speed hack, not quality parity.

## Finish / check (MapRepair)

1. Take the ODM or ISM raster plus the **same** photo folder.
2. `bash MapRepair/run.sh` on the laptop that will do the work (Mac or Linux).
3. Drop the map, drop the photos, run.
4. Read the report. Zero patches can still be a good outcome.
5. Grade in the browser if you want; download PNG + keep the world file.

## Do not

- Run ISM on Linux. There is no Metal backend.
- Run MapRepair and expect a mosaic from photos only.
- Commit `web_runs/` or 8k JPEGs.
- Publish a map as survey-grade.

## Hardware we never validated

- Windows.
- Fedora/Arch MapRepair install branches (code exists, not executed).
- Radeon iGPU / XDNA2 NPU (kernels exist as host math).
- Full 564-photo E2E.
