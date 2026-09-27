# FSR frame generation and final presentation

## Standalone ISM package

Current portable implementation and usage are in [ISM/GUIDE.md](../ISM/GUIDE.md).
It contains one 84-kernel `ISM.metal` shared by ADM and the actual AMD FSR passes,
plus the necessary host runtime, timed `run.py` and separate `run_visual.py`.
The user's current run.py defaults use the gentler SIYI photo profile. The saved
standard ISM306 run took116.628s; that timing predates SIYI defaults and subsequent
matching-recovery/Gaussian corrections. The sections below retain prior evidence.

## Current revision: preserve detail, natural late-day grade

1. Register all 306 original photos with the lean Metal/MPS pipeline.
2. Actual AMD FSR frame generation: two levels, 915 unique generated views,
   1221 compositor inputs; original photographs supply the registration.
3. Coherent mapping to 6500x4394.
4. **No NLM. No final FSR spatial upscale.** Lanczos 2x, positive detail **1.3**,
   existing Lab CLAHE, fine sharpening and contrast.
5. Run the bounded Metal triangle ray tracer described below.
6. One fused cinematic Metal grade, `pro_camera`; export 13000x8788.

Whites and vibrance now live in `PRESETS['pro_camera']` and its CPU reference
`pro_camera_grade`, not in an after-the-fact processing pass. The Metal kernel
`post_cinematic_grade` applies the same equations. Gentle luma contrast and a
continuous highlight shoulder preserve separation in bright pavement. The .3
white control reduces the shoulder's end slope by 30%, with white .9475 before
exposure; it does not map white to gray .7. Vibrance reduces muted chroma by up
to 10%. Tint is weaker on neutral surfaces. Warm highlights/cool shadows are
restrained, exposure -.06 stops. Extra grade sharpening, grain and vignette are
zero. Existing clipped source detail cannot be recovered.

### Ray tracing and its actual limitation

`cinematic_path_trace` uses Metal's triangle intersector with an actual triangle
acceleration structure, camera rays, solar-disk visibility rays, cosine-weighted
diffuse directions and at most two diffuse bounces. Deterministic quadrature
avoids temporal noise/history caches. `adm.raytracing.trace` accepts registered
triangle vertices and per-triangle diffuse reflectance. Limits: 512x512 light
grid, 1..64 samples, 1M triangles, 512 MiB acceleration structure plus build scratch.

This benchmark contains only a planar map, so the run traces its two flat-plane
triangles at 128x87, 8 samples. The 3-degree sun is an artistic low-sun assumption,
not a calculation of the original capture time. A flat plane gives uniform
irradiance: the ray stage preserves image pixels and original shadows. Real
building/terrain shadows require registered measured geometry. No depth is
fabricated from brightness. Relighting already-lit RGB imagery with a bounded
irradiance ratio is an approximation, not a physical recovery of scene radiance.

Implementation follows the intersection model in [Apple's Metal ray-tracing
documentation](https://developer.apple.com/documentation/metal/ray-tracing-with-acceleration-structures)
and the diffuse light-transport formulation in [PBRT](https://www.pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Simple_Path_Tracer).

### Validation and timing

Shared Objective-C bridge and MSL rebuilt; 58 regression tests passed. Grade is
pixel-identical to CPU in the test fixture. Updated sharpen differs by mean
.04078/255, max7/255 in BGRA versus the OpenCV reference. Ray tests verify a real
roof changes the sun shadow with direction, repeated execution is deterministic,
and a flat plane preserves pixels. All runtime source/config hashes are recorded
in `fsr_experiments/sunset_source_manifest.json`.

Presentation-only preview took 3.97762 s from a previously saved neutral map.
The fresh benchmark `metal_fsr3_sunset_e2e` completed in **109.70606s** full-process
wall time on Apple M4 Pro: **10.29394s under two minutes**. All 306 originals were
retained, 915 virtual views generated, 1221 views composed. No match-cache reuse.
All original transforms exactly match LeanADM and42 frozen source/config hashes
were unchanged through the run. 58 regression tests passed before timing.

| Stage | Seconds |
|---|---:|
| Input preparation |15.063|
| Feature matching |50.615|
| Alignment |7.425|
| FSR generation, mapping, neutral export |32.466|
| Lanczos/detail/CLAHE/fine sharpen and report |0.658|
| Ray tracing and application |0.030|
| Cinematic grade and PNG export |3.039|
| Process startup/remaining overhead |0.410|

The fused color kernel takes .03239 s; most grade/export time is PNG encoding.
FSR generation itself takes 22.48977 s and is included in 32.466 s above. These are
separate full-run observations; the entire improvement over 120.36548 s must not
be attributed solely to code changes because matching also varies between runs.
Maximum process RSS 1.976 GiB is not a measurement of all unified GPU memory.

The actual result is `map_graded.png` (13000x8788), with `overview.png` and
`road_comparison.png`. Audit in `quality_audit/` replays the complete map and
reproduces final pixels exactly. On the same 137961 bright-road pixels, native-
scale texture RMS is LeanADM 2.31898, previous version .49926, new version 2.44360.
This is local texture contrast, not a measure of newly captured resolution.
The preserved pre-grade global contrast still clips 8.70% of the bright-road mask
in at least one channel; grading cannot recover those clipped values.
Separate fresh FSR neutral renders differ slightly despite identical transforms
(mean absolute BGRA difference .05104/255, maximum 201); exact FSR raster
repeatability is not established. See `fsr_experiments/sunset_e2e_summary.json`
for the detailed comparison and validation results.

Active config: `fsr_experiments/natural_sunset.json` / `full_pipeline.json`.

### Previous full benchmark and diagnosed failure

The previous no-final-FSR version completed 306 originals in 120.36548 s full
process wall time, with no matching cache reuse; internal 119.79328 s excludes
startup/overhead. Its stage replay was pixel-identical to the benchmark output.
On a fixed bright-road mask, NLM reduced texture RMS 2.0865 -> 1.6429 (-21.3%).
The old after-grade white curve reduced RMS 1.9458 -> .4993 (-74.3%). These are
local texture/contrast proxies, not independent resolution measurements. The
former white -.3 curve was too aggressive; it is replaced in the current preset.
See `metal_fsr3_lanczos_toned_e2e/quality_audit/` for that historical audit.

## Historical update: FSR upscale removed

The active configuration now keeps **FSR frame generation**, then mapping,
NLM denoise, **Lanczos 2x**, detail/CLAHE/fine sharpening, and color grading.
`presentation_fsr_passes=0`; the final FSR spatial upscale is removed.
The requested whites -30% and vibrance -10% are applied after `pro_camera`.
White reduction is a monotone highlight roll-off above luminance .5 whose neutral
white endpoint becomes .7. Vibrance is a saturation-aware reduction in chroma.
Both adjustments execute on Metal and preserve transparency; these are defined
ADM controls rather than a claim of matching a particular photo editor's sliders.

Updated output: `output/killerofmaps/metal_fsr3_lanczos_toned/map_graded.png`
(13000x8788), with `comparison.png` and `road_detail_comparison.png` alongside it.
This reuses the saved FSR neutral map; registration/frame generation were not rerun.
Presentation plus export took **9.12687 s**, with **4.22061 s** sharpening
(3.71980 s NLM) and **4.48939 s** grade/export. The earlier full-pipeline
143.59494 s timing does not describe this new configuration.

The active `full_pipeline.json` and `no_fsr_upscale.json` contain this recipe.
The earlier config is preserved as `full_pipeline_fsr_upscale_reference.json`;
the original benchmark's complete settings also remain in its `resolved_config.json`.
Five presentation hardware tests pass, including CPU parity, monotonic highlight
response, alpha preservation, and optimized-versus-original NLM parity.

The remainder describes the earlier FSR-upscale experiment and its measurements.

## Current sequence

1. Select and register the 306 original photos with the lean Metal/MPS pipeline.
2. Generate midpoints and quarter frames with AMD FSR Frame Interpolation 1.1.3,
   translated from the unmodified SDK 1.1.4 GLSL to SPIR-V and Metal 4.
   Two levels across 305 adjacent pairs produce **915 unique generated views**,
   giving **1,221 compositor inputs**. Synthetic views add no matching constraints.
3. Compose the map using coherent source selection. Registration and the map
   grid still come from the original photographs.
4. Enlarge the completed map with AMD FSR1 EASU.
5. Apply the user's recipe: NLM (2/2, patch 7, search 15), Lanczos4, positive
   detail enhancement 1.8, Lab CLAHE (2.0, 8×8), fine sharpening (sigma .45,
   weights 1.35/-.35), and absolute contrast (1.05/-2).
6. Apply the supplied `pro_camera` grade, including clarity and fine sharpening.
7. Export the neutral map and the final graded presentation, with their geometry.

The first timed configuration uses one FSR 2× enlargement and Lanczos scale 1.
Setting the latter to 2 makes the final image 4× wider and taller than the map.
The adapter limits final presentation to 160 million pixels before allocating it.

## What is actually Metal

The AMD reconstruction, game-vector field, mip pyramids, disocclusion,
interpolation and inpainting shaders execute on Metal. The adapter supplies
motion from the registered planar homographies and assumes flat ground at the
configured altitude. Optical-flow assistance is disabled. This is the actual
AMD game-motion-vector path, with no game-engine depth truth or temporal
super-resolution. Mapping and FSR spatial upscaling also run on Metal.

The presentation recipe's color conversions, NLM, Lanczos, Gaussian filtering,
CLAHE, contrast, and the global `pro_camera` grade run on Metal. Small coefficient
tables are prepared on the CPU. Other grading presets and regional grading
retain their NumPy/Pillow implementation. Codecs, SIFT, RANSAC, geographic fitting,
and the frame-generation adapter's motion/alpha preparation still use the CPU.

Generation keeps two decoded originals and three generated images in its working
cache. The compositor traverses the images twice, so it executes **1,830 generation
calls for 915 unique views**. It saves no synthetic-photo files or match cache.
There are no repeated upscale/sharpen passes during registration or mapping in
this configuration. The later, more specific requested order governs the run.

## Numerical checks and optimization

The first Metal NLM implementation remains available as
`PresentationMetal.denoise(..., optimized=False)`. The optimized version reuses
horizontal patch-distance sums, saturates only distances whose weights are
already zero, and accumulates independent pixels in vector registers. Textured,
smooth and constant fixtures produce exactly the same output before and after
this optimization. A same-size Lanczos resize is skipped because it is identity.

On `IMG_0014.jpg` enlarged to 1280×1280 with FSR:

| Component | First Metal version | Optimized Metal |
|---|---:|---:|
| NLM | 1.32546 s | 0.20624 s |
| Entire sharpening recipe | 1.33511 s | 0.21302 s |
| `pro_camera` grade | 0.00777 s | 0.00705 s |

The sharpening recipe differs from the OpenCV reference by 0.07584/255 per
color channel on average, p99 2, maximum 9. After grading, mean difference is
0.20615/255, p99 4, maximum 30. OpenCV fixed-point Gaussian rounding and rare
color-table rounding differences can be amplified by CLAHE and sharpening.
This is close numerical agreement, **not bit-identical CPU parity**. The
optimization preserves the first Metal implementation's results. Pillow's
fractional-box blur and `pro_camera` grade match exactly in the hardware fixture.

The saved frozen-registration FSR render took 33.7542 s for frame generation and
composition, or 34.5663 s including PNG export. This excludes registration and
the final presentation stages; do not describe it as full-pipeline timing.

## Build and run

The frame-generation translation currently requires macOS 26 / Metal 4 for
device-coherent image access. The SDK is pinned to
`c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`. The port is experimental, not an
AMD-provided Metal backend. SPIR-V optimization is necessary to remove unused
read access and fit Metal's limit of eight read/write textures.

```sh
brew install glslang spirv-cross spirv-tools
.venv/bin/python scripts/build_fsr3.py
.venv/bin/python scripts/benchmark_gpu_revision.py killerofmaps \
  --config output/killerofmaps/fsr_experiments/full_pipeline.json \
  --output output/killerofmaps/metal_fsr3_full --max-output-mb 768
```

The benchmark command requires an empty output folder. The exact shader
translation manifest is `build/fsr3/manifest.json`; attribution is in
`third_party/PRESENTATION_NOTICES.txt` and the vendored AMD license files.

## Artifacts and interpretation

The fresh full 306-photo run completed in **143.59494 seconds** (not under two
minutes), with 11,254 accepted edges and 915 generated views. Original geometry
is 6500×4394; the graded presentation is 13000×8788. Preparation took 15.352 s,
matching 66.896 s, alignment 7.552 s, and composition/presentation/exports
53.275 s. Within presentation, FSR enlargement took .094 s, sharpening 14.148 s
(NLM 13.744 s), and grading/export 4.717 s. Saved output size is 274,814,233 bytes.
See `metal_fsr3_full_timing.json` and the per-stage reports for the measured run.

Visual review of matching crops shows stronger edges but excessive local contrast,
bright halos and lost texture in bright pavement. This is not a demonstrated
quality improvement over LeanADM. The exact requested strong-detail/CLAHE/fine
sharpen recipe is followed by another clarity/sharpen pass in `pro_camera`.
The comparison shows that combined effect and does not isolate FSR's contribution.
Comparison images are in `fsr_experiments/comparison_lean_fsr3/`.

- Lean baseline: `output/killerofmaps/metal_lean/map_neutral.png`.
- FSR frame-generation render with saved registration:
  `output/killerofmaps/fsr_experiments/fsr3_render/map_neutral.png`.
- Full pipeline output folder: `output/killerofmaps/metal_fsr3_full/`.
- `map_geometry.json` and `image_to_map.npy` describe the neutral map.
- `presentation_geometry.json` and `image_to_presentation.npy` describe the
  enlarged final presentation. Pixel-center offsets are included.
- `interpolation_report.json` records all parents, cameras, counts and assumptions.
- `presentation_report.json` and `grading_report.json` record settings and timing.

More pixels and sharper edges do not establish more recovered scene detail.
Generated views reuse the original observations and can introduce interpolation
artifacts. This benchmark consists of rotated map crops with supplied centers;
it does not validate 3D geometry, surveyed accuracy, or unseen detail recovery.
The custom planar interpolation experiments in `adm/virtual_views.py` are
historical ablations and are not the production FSR frame-generation path.

The independent non-FSR backup is `LeanADM/`. It was copied without rerunning it,
as requested. Its saved 83.2055 s measurement describes the original lean run.
