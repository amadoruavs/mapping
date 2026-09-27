# Do not claim

A checklist. If a sentence is not backed by a file in this repo (or a new measurement you just took), do not write it.

## Geometry / accuracy

- Do not call ISM/ADM/LeanADM an orthophoto, SfM, or photogrammetric reconstruction.
- Do not call GPS residuals "survey accuracy" or "GCP error."
- Do not say the 5 m gate "verifies" the map. It is a consistency check against the same GPS used in the fit.
- Do not equate image-center residuals with camera-center residuals.
- Do not treat `NO GPS connected` files as a calibrated PPK/RTK track.
- Do not say yaw from the SIYI gimbal is a heading truth. It disagreed by ~180° on the 94-photo set.
- Do not present the collapsed 0.0028 px median as good alignment.

## Speed

- Do not say "10× faster than ODM" without naming planar vs SfM+MVS and Docker vs native.
- Do not reuse a presentation-only time (grade, FSR, lasso) as an end-to-end mapping time.
- Do not reuse 83.2 s as a timing for ISM's current SIYI/grader/lasso defaults. That number is LeanADM, 306 photos, width 640, no FSR, no grade.
- Do not reuse 21 s as a timing for today's ISM defaults if the GUIDE says that 94-photo pair predates grader + lasso.
- Do not claim 564-photo E2E success. It was never finished.
- Do not claim <300 s / 250 photos. That target was unmet.
- Do not claim HIP/XDNA2 speedups. No device run exists in this archive.
- Do not claim Ubuntu container times are native Linux laptop times (the container was emulated on the Mac).

## Quality

- Do not say FSR recovered extra ground detail or "true GSD."
- Do not say OpenRT recovered buildings/trees unless a registered mesh was supplied.
- Do not say adaptive cinema / CLAHE / unsharp recovered blades of grass.
- Do not say MapRepair certified a map error-free when zero patches passed the gates.
- Do not treat SSIM vs ODM as a quality score.
- Do not treat Sobel/Laplacian ratios as "times sharper."
- Do not treat source-transition-fraction drops as overall image-quality wins.
- Do not claim FSR rasters are bit-identical across runs.
- Do not claim the Fedora every-sixth app matches WebODM quality.

## Process

- Do not overwrite output folders.
- Do not silently lower SIFT thresholds on the *main* graph (recovery is a separate, reported path).
- Do not add GPS-only edges.
- Do not "help" by inventing EXIF yaw/AGL.
- Do not describe lighting as physically relit photography when the geometry is a flat plane.
- Do not put NPU/Metal/ROCm in a marketing sentence if the pixel path still runs on CPU.

## Attribution

- AMD FSR is AMD's. OpenRT is Project-10's. OpenSfM residuals are OpenSfM's. Say so.
- The cinematic `pro_camera` numbers were supplied by the user. Do not "improve" them quietly.
