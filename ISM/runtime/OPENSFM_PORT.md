# OpenSfM geographic convergence in ISM

The optional `geographic_solver: "opensfm_planar"` combines ISM's visual matches
with a port of OpenSfM geographic residuals, using Metal and MPS. This is a
**fixed-ground-plane adaptation**, not the complete ODM/OpenSfM reconstruction
pipeline or a call to its CPU Ceres solver at runtime.

## Exact upstream source

ODM checkout: `1a7134170e6100cade3236486a279d6241f11f99`.
Its `SuperBuild/cmake/External-OpenSfM.cmake` pins OpenDroneMap/OpenSfM revision
`c5328439465e6ace011f39077d1077d7b1cdd65d`.

- [GPS position residual: DataPriorError](https://github.com/OpenDroneMap/OpenSfM/blob/c5328439465e6ace011f39077d1077d7b1cdd65d/opensfm/src/bundle/error/prior_error.h)
- [Orientation residual: UpVectorError](https://github.com/OpenDroneMap/OpenSfM/blob/c5328439465e6ace011f39077d1077d7b1cdd65d/opensfm/src/bundle/error/absolute_motion_errors.h)
- [GPS and orientation loss wiring](https://github.com/OpenDroneMap/OpenSfM/blob/c5328439465e6ace011f39077d1077d7b1cdd65d/opensfm/src/bundle/src/bundle_adjuster.cc)
- [WGS84 and topocentric conversion](https://github.com/OpenDroneMap/OpenSfM/blob/c5328439465e6ace011f39077d1077d7b1cdd65d/opensfm/geo.py)
- [Bundle use of shot GPS accuracy](https://github.com/OpenDroneMap/OpenSfM/blob/c5328439465e6ace011f39077d1077d7b1cdd65d/opensfm/src/sfm/src/ba_helpers.cc)

OpenSfM contributors retain their upstream copyright. The pinned fork carries
GNU AGPL version 3; its complete license is included in `runtime/LICENSES.txt`.
The geographic module adapts the equations and supplies new Metal derivatives,
packing and solver integration. Validation includes the actual unmodified
upstream C++ header templates, not just a second implementation of our shader.

## Two methods, one joint refinement

1. Read actual raw photos, EXIF/XMP and optional sidecar overrides.
2. Keep ISM's existing GPS candidate search, SIFT, MPS descriptor matching,
   ratio test, RANSAC and disconnected-graph recovery unchanged.
3. Run the verified symmetric visual alignment. Image-space transfer residuals
   cannot be reduced by shrinking the mosaic. This supplies initialization.
4. Establish initial geographic orientation and scale with a robust float64
   similarity. Decompose each homography into a camera centre and an orthogonal
   rotation using the configured intrinsics and positive-height plane solution.
5. Optimize physical camera poses with **both** residual methods below in every
   iteration. Positions, rotations, Jacobians, robust weights, normal equations,
   trial acceptance and convergence live on the GPU. MPS Cholesky solves each
   joint system. Geographic constraints participate before convergence; there
   is no second GPS similarity after this joint solve.
6. Render real-photo regions through those camera-to-ground projections, then
   use the existing sharpening and gentle SIYI grade. Both benchmark variants
   use the same presentation settings, no synthetic views, and no ray lighting.

### Visual residual

ISM retains mean bidirectional soft-L1 transfer error in **registration-image
pixels**: predict a matched location in the other source image through the
common ground plane. Each direction has half weight. This differs from ODM's
3D point reprojection residual; we do not claim equal full bundle solutions.

### Geographic residuals

- Position: `(camera_position - observed_position) / sigma`, the linear
  `DataPriorError` used for OpenSfM position priors, with its ordinary squared
  loss. GPS bias is fixed to identity; uncertainty is fixed during the solve.
- Orientation: OpenSfM's `UpVectorError`, normalized by angular uncertainty,
  with its block Cauchy loss `0.5 * log(1 + squared_residual_norm)`.
  Apply it to world up **and north** using the recorded camera rotation.
  The second axis is an explicit ISM adaptation to retain camera heading at
  nadir. OpenSfM's pan/roll angle residuals are degenerate at exact nadir.
- GPS horizontal sigma defaults to ODM's 10 m metadata fallback. Positive
  recorded horizontal accuracy or a sidecar override replaces it. Configured
  AGL uses its own 10 m sigma. Recorded axes default to 10 degrees; configured
  orientation assumptions use a weaker 45-degree prior.

Yaw, pitch and roll form a full rotation matrix. Optimization uses a local
axis-angle rotation, which avoids the yaw/roll ambiguity of nadir Euler angles.
CPU camera rotation initialization is existing ISM camera-axis convention;
flight yaw is never silently substituted for absolute camera yaw.

## GPU organization

`geo_camera_model` computes each camera's projection matrix, analytic derivatives
and 8x8 prior block once per evaluation. Six active pose parameters occupy eight
slots to reuse the existing alignment block layout; two slots are fixed. Arrays
are packed and resident across iterations. `geo_visual_linearize` evaluates
correspondences in parallel with float vectors and small matrix products;
`geo_assemble` combines visual and geographic blocks. Existing MPS Cholesky,
cost reduction and device damping handle the solve. No Python iteration loop or
CPU numerical optimizer performs the joint refinement. Float32 Metal arithmetic
is checked against float64 CPU calculations and unmodified OpenSfM functors.

Convergence uses a 600-iteration maximum, checks every four encoded iterations,
and stops early. Relative objective tolerance is 1e-5; step/predicted-gain
checks also require low damping. A heavily damped stall is not convergence.

One `ISM.metal` contains **91 entry points**, compiled into one shared library.
It is not a single GPU dispatch: MPS dispatches and the host runtime are still
required. File I/O, codecs, metadata, SIFT/RANSAC, double precision WGS84 and
initial pose decomposition remain host work.

## Geographic assumptions and limits

WGS84 latitude/longitude are converted through ECEF into local east/north
coordinates in float64 before GPU upload; global ECEF floats would lose local
precision. The ground plane is local z=0. AGL is configured as **51.816 m** on
the 94-image test, with **81-degree HFOV** and **1280-pixel registration width**.
EXIF sea-level/ellipsoid altitude is logged, never treated as AGL.

This port does not include triangulated scene points, depth maps, DEM/mesh
orthorectification, intrinsic/lens-distortion optimization, rolling-shutter
calibration, ground-control-point adjustment or GPS bias estimation. Those are
separate parts of ODM and require a larger 3D reconstruction implementation.
Flat ground cannot correct raised objects or all road seams.

The mapv2 metadata records GPS coordinates and `NO GPS connected`; all 94
photos lack measured GPS accuracy. Recorded headings also disagree with visual
image orientation, including roughly 180-degree differences. Initializing
directly from those recorded rotations failed. Visual initialization plus
robust orientation loss converged without forcing the bad headings onto images.
GPS fit residuals use the same GPS values that participate in optimization;
they are not independent surveyed accuracy.

## Validation and benchmark

See `ISM/benchmarks/odm_geo94/RESULTS.md`, the per-run JSON reports, and
`upstream_parity.json`. `scripts/opensfm_prior_reference.cpp` compiles the
unmodified `DataPriorError`, `UpVectorError` and `ceres::CauchyLoss` with Ceres.
`scripts/validate_geographic.py` compares 128 fixtures and the upstream WGS84
implementation. The portable tests also check physical projection, camera and
visual Jacobians, prior Hessians, invalid horizons/heights and noisy-camera
convergence. This validation does not assert full ODM algorithm parity.
