# 11 — The other golden hour: capturing at low sun, and what it does to photogrammetry

Status: research note, 2026-09-13.

## 1. Why golden-hour captures are hard for SfM/MVS/texturing

Physics of a sun at 0–10° elevation, per the capture:

- **Long, dark cast shadows**: shadow length = height / tan(elevation); a 10 m tree at 6°
  throws a 95 m shadow. Large regions of the scene become low-texture and low-SNR; the
  sky-lit shadow is blue and ~1/10–1/30 the radiance of the lit part (clear sky).
- **Moving shadows during the flight**: the sun moves 0.25°/min in azimuth-ish terms; over a
  30-min flight shadow edges move by ~7° ⇒ a 95 m shadow tip moves ~12 m. Features on shadow
  edges are inconsistent between overlapping images (violates the static-scene assumption),
  which SfM sees as outliers and texturing sees as seams.
- **Colour shift between images**: sun colour temperature changes quickly (sky/sun ratio,
  Rayleigh reddening); auto white balance and auto exposure amplify the drift.
- **Glare / flare / low contrast**: near-horizontal sun into oblique/facade images; haze;
  large dynamic range (lit roof vs shadow) exceeding the sensor's ⇒ clipped highlights or
  crushed shadows.
- **Vegetation self-shadowing** increases; sparse point clouds under canopies.

ODM's own docs are unambiguous for stockpile surveys: "always prefer the flights during the
midday, when the sun is at the nadir so everything is consistently illuminated", "in order to
avoid shadows which on one side of the stockpile can obstruct feature detection and lessen
the number of resulting points", and "best results are obtained with evenly overcast or
clear skies" ([ODM tutorials](https://docs.opendronemap.org/tutorials/)). Recommended overlap
75–80 % front / 65–70 % side, more at higher altitude. Community threads report strong
October low-sun shadows degrading forest reconstructions
([community: forest orthos](https://community.opendronemap.org/t/forest-orthos-produced-from-same-test-images-show-different-results/693))
— with the counter-note that low-sun shadows on snow can *add* texture for matching.

Photography sites recommend the opposite (20–60 min after sunrise/before sunset for
aesthetics — e.g. [Aerotimelapse](https://aerotimelapse.com/best-time-to-fly-a-drone/)),
which is precisely the tension behind this project: shoot at midday for geometry, then
*synthesise* golden hour (docs 07–10).

## 2. If you must capture at golden hour

Capture-side mitigations (standard practice, not ODM-specific):

- Fix exposure and white balance manually (no AWB/AE), shoot RAW/DNG if the drone allows
  and develop with one setting; a grey card / reflectance panel in the first images helps.
- Fly perpendicular to the sun azimuth to keep flare out of the frame; nadir only; avoid
  obliques into the sun.
- Shorten the mission (shadows move) or split into blocks processed with `split-merge`
  (`--split`, `--split-overlap`) so each block has near-constant shadows — **hypothesis**,
  untested.
- Increase overlap (85/75) to give matching more chances in low-contrast shadow.

## 3. ODM flags that help (from `docs.opendronemap.org/arguments` and `opendm/config.py`)

| Flag | Default | Why it matters at low sun |
|---|---|---|
| `--feature-quality ultra\|high\|medium\|low\|lowest` | `high` | `ultra` extracts on full-res images; recovers features in dark shadow areas at memory/time cost. |
| `--feature-type sift\|dspsift\|hahog\|akaze\|orb` | `dspsift` | SIFT-family (dspsift/sift) is the most illumination-robust; AKAZE also decent; ORB is worst in low contrast. |
| `--min-num-features` | 10000 | Raise (20–30 k) so shadow regions still contribute keypoints. |
| `--matcher-type flann\|bow\|bruteforce` | `flann` | `bruteforce` is the most robust "at significant speed cost"; FLANN "enhanced stability"; BOW faster but can miss matches. |
| `--matcher-neighbors` | 0 (triangulation) | Set 8–12 to force matching against GPS neighbours when triangulation drops shadowy images. |
| `--texturing-skip-global-seam-leveling` | False | Skips "normalization of colors across all images — useful when processing radiometric data". Global leveling *hides* exposure drift by solving per-vertex colour offsets (Waechter et al. 2014 extended Lempitsky–Ivanov); with strong drift it can also *skew* colours ([mvs-texturing issue #63](https://github.com/nmoehrle/mvs-texturing/issues/63)). For de-lighting (doc 07) keeping it **on** gives a smoother, more "single illuminant" texture; for radiometric truth turn it **off**. |
| `--texturing-skip-local-seam-leveling` | deprecated (always on) | Local Poisson blending near seams; `opendm/config.py` lists the flag under `DEPRECATED`. |
| `--texturing-keep-unseen-faces` | False | Irrelevant to lighting; avoids holes. |
| `--texturing-single-material` | False | One atlas — simpler for viewers/de-lighting. |
| `--radiometric-calibration camera\|camera+sun` | none | Multispectral only (MicaSense/DLS); `opendm/dls.py` computes sun angle with pysolar for the `camera+sun` mode; the community thread asking how it behaves without a DLS went unanswered ([thread 11330](https://community.opendronemap.org/t/automatic-detection-of-sun-irradiance-and-sun-angle-data-by-exif-xmp-tags/11330)). Not usable for RGB. |
| `--sky-removal` / `--bg-removal` | off | ONNX masks via onnxruntime (`opendm/skyremoval`, `opendm/bgfilter`); sky removal helps obliques at golden hour where the bright sky wrecks matching and texturing. |
| `--auto-boundary` | off | Crops to the flown area — removes the long-shadow fringe outside the survey. |
| `--pc-quality`, `--mesh-size` | | Not lighting-related; unchanged. |

There is **no** ODM flag that removes shadows or equalises illumination physically;
mvs-texturing's tone mapping options are only `none` and `gamma` (`libs/tex/settings.h`),
and ODM passes `none`.

## 4. Radiometric / colour normalisation approaches

### 4.1 Inside the texturing step

- **mvs-texturing** (Waechter, Moehrle, Goesele, ECCV 2014 "Let There Be Color!", BSD-3):
  view selection (data term GMI = gradient magnitude, photo-consistency outlier removal),
  then *global seam leveling* (per-vertex additive colour correction solved as a sparse
  least-squares over seams) and *local seam leveling* (Poisson editing on strips along
  seams). This equalises *between views* but never estimates albedo; it also spreads a
  shadow-edge colour discontinuity into neighbouring patches when a seam happens to cross a
  shadow boundary.
- **OpenMVS `TextureMesh`** implements the same Waechter pipeline with `bGlobalSeamLeveling`
  / `bLocalSeamLeveling` options ([TextureMesh.cpp](https://github.com/cdcseacave/openMVS/blob/master/apps/TextureMesh/TextureMesh.cpp));
  the term "color balancing" in OpenMVS docs refers to this same global leveling (**check the
  current `--global-seam-leveling` / `--local-seam-leveling` option names in your build**).
  Not used by ODM (ODM uses OpenMVS for densification only).

### 4.2 Before SfM (pre-processing the input images)

- **Gray-world white balance** (per image: scale channels so mean R=G=B) — cheap, removes
  the warm cast, but also removes *real* golden colour and is biased by large uniform
  surfaces (a field, a roof). Variants: shades-of-gray, max-RGB, Cheng's PCA. Overview:
  [Manansala, white-balancing methods](https://jmanansala.medium.com/image-processing-with-python-color-correction-using-white-balancing-6c6c749886de).
- **Histogram matching / colour transfer to a reference image** (Reinhard lαβ transfer, or
  per-channel CDF matching) — used in orthomosaic pipelines before stitching; a survey of
  drone mapping pipelines finds "color transfer and histogram matching … more efficient"
  than AWB for inter-image consistency ([Towards Robust Drone Vision in the Wild](https://arxiv.org/abs/2208.12655)
  discusses illumination shifts; general practice described in
  [JOUAV orthomosaic overview](https://www.jouav.com/blog/orthomosaic.html)).
- **Learned multi-image colour balancing** for aerial blocks (e.g. "Aerial Image Color
  Balancing Based on Rank-Deficient Free Network",
  [ResearchGate](https://www.researchgate.net/publication/354124677_Aerial_Image_Color_Balancing_Based_on_Rank-Deficient_Free_Network))
  — research code, not drop-in.
- Important caveat: SfM feature extraction (SIFT/DSP-SIFT) works on **grayscale** gradients
  and is largely invariant to global colour/exposure changes; white balancing helps the
  *texture*, not the *geometry*. What hurts geometry is contrast loss in shadows and moving
  shadow edges — pre-processing cannot fix moving shadows.

### 4.3 Does a GPU pre-processing shader make sense?

- For a batch of 500–2000 × 20–48 MP JPEGs, a Metal/WebGPU compute pass doing
  gray-world/histogram-match + mild shadow lift (local tone mapping, CLAHE on luminance)
  runs in a few seconds — well worth it *if* the inputs are RAW/16-bit; on 8-bit JPEGs
  lifting shadows mostly amplifies noise and compression blocks.
- CPU NumPy/OpenCV does the same in ~1–3 min for 1000 images — the GPU is not the bottleneck;
  the bottleneck is deciding the *reference*. Recommended design: pick the reference as the
  median-exposure nadir image, match all others to it (per-channel CDF on the overlap
  region only, to avoid content bias), write to a `images_normalized/` folder and run ODM on
  that. Keep the originals; ODM reads EXIF from the JPEGs, so preserve EXIF (`exiftool
  -TagsFromFile`).
- A **shadow-aware** pre-process (lift only the detected shadow region toward the lit
  statistics, keep a soft penumbra) is where an ONNX shadow detector (S-EO U-Net, DSSDNet,
  SARU; see doc 07) pays off; it directly addresses "low contrast in shadows" for matching.
  Do it as `--sky-removal`-style ODM stage: onnxruntime, 384–1024 px tiles, mask written to
  `images/*_shadow.png` for later de-lighting as well.

## 5. Synthetic golden-hour training data for UAV detection

The owner also generates UAV detection datasets; golden hour is a legitimately hard domain
(long shadows that look like objects, low contrast, orange cast, lens flare). Rendering
sun-consistent shadows is straightforward with any physically based renderer because a sun
is a directional light: pick lat/lon/date/time → az/el (pysolar / Sun Position / UE Sun-Sky
actor), and the renderer produces geometrically consistent shadows for free.

Generators and what they offer:

| Generator | Engine | Lighting control | Annotations | Licence |
|---|---|---|---|---|
| [SynDroneVision](https://arxiv.org/abs/2411.05633) (WACV 2025) | Unreal Engine 5.0 + Colosseum | "Sun and Sky Actor" with **solar time** for a geographic location + Post Process Volume | 2D boxes; 140 038 RGB images at 2560×1489 ([Zenodo](https://zenodo.org/records/13360116)) | dataset on Zenodo (check record licence) |
| [Sim2Air](https://arxiv.org/abs/2110.05145) / [larics/synthetic-UAV](https://github.com/larics/synthetic-UAV) | Blender Cycles | daylight/partly cloudy/twilight, texture randomisation | boxes | see repo |
| [BlenderProc](https://github.com/DLR-RM/BlenderProc) | Blender Cycles | `bproc.types.Light(type='SUN')`, HDRI worlds, random sun elevation | COCO/boxes/segmentation/depth/normals | **GPL-3.0** |
| [Kubric](https://github.com/google-research/kubric) | Blender Cycles + PyBullet | `DirectionalLight`, HDRI domes | boxes, segmentation, depth, flow | **Apache-2.0** |
| Colosseum (AirSim fork) | UE5 | UE sun/sky | boxes via API | repo licence not exposed via API — check |
| ODM itself (this project) | Blender Cycles on ODM meshes (doc 10) | Nishita sky at chosen time | render + paste drone models → boxes | ODM AGPL-3.0; textures are yours |

Design notes for sun-consistent synthetic golden hour on **real ODM sites**:

1. Use the de-lit albedo mesh (doc 07) as the background; if not de-lit, restrict the
   synthetic sun to a cone around the *capture* sun (±10° azimuth, same elevation) so the
   baked and rendered shadows agree — this is the cheapest sun-consistency rule.
2. Insert UAV models (with shadow casting) above the DSM; their shadows fall on real
   geometry at the correct length: shadow length = altitude / tan(elevation) — at 5° a UAV
   at 30 m AGL has its shadow 340 m away, likely outside the frame, which is itself a
   realistic golden-hour cue for detectors (shadow decoupled from the object).
3. Randomise: elevation 0–15°, dust 1–5 (haze), exposure ±1 EV, white balance 3500–6500 K,
   flare on/off, motion blur; keep the sky model physically based (Nishita) rather than an
   HDRI so the sun disc and shadow direction agree.
4. Labels: boxes from the renderer's object index pass (Blender `Object Index` /
   BlenderProc `render_segmap`), plus a per-image JSON with sun az/el so the detector can
   be evaluated by sun elevation bin.
5. 2D augmentation alternatives when 3D is too slow: shadow-aware copy-paste and shadow
   synthesis networks (ShadowGAN, G2R-ShadowNet; instance shadow detection
   [arXiv 2207.04614](https://arxiv.org/abs/2207.04614)); physics-guided aerial shadow
   synthesis with penumbra modelling in [AeroDeshadow / AeroDS](https://arxiv.org/abs/2604.15903).
   These lack true sun consistency but are cheap.

## 5b. Recipe: capture-time sun vector from a dataset (feeds docs 07–10)

```python
# uses only what ODM already installs: exifread/PIL, pysolar==0.13 (GPL-3.0), numpy
from datetime import datetime, timezone
from pysolar.solar import get_altitude, get_azimuth
def sun_for(lat, lon, dt_utc):                       # degrees; azimuth 0=N, clockwise
    return get_altitude(lat, lon, dt_utc), get_azimuth(lat, lon, dt_utc)
# 1. read DateTimeOriginal + OffsetTimeOriginal (or GPSDateStamp/GPSTimeStamp = UTC) per image
# 2. lat/lon from GPSLatitude/GPSLongitude (or reference_lla.json for the site centroid)
# 3. compute (alt, az) per image, take the median; record min/max spread
# 4. write <project>/odm_report/sun.json = {utc_median, alt_deg, az_deg, alt_spread, az_spread}
```

Sanity checks: `alt_spread` > 5° means shadows moved noticeably during the flight (expect
seam artifacts); a wrong time zone shows up as a 15°/h azimuth error against the baked
shadow direction in the orthophoto (doc 08 §8 validation). `opendm/dls.py` contains the same
pysolar calls plus a NED conversion (`ned_from_pysolar`) that can be reused.

## 5c. Measurement protocol for any pre-processing or flag change

Run the same dataset twice (baseline vs candidate) and compare, from ODM's own outputs:

| Metric | Where | What to look for |
|---|---|---|
| Features per image, matches per pair | `opensfm/reports/*.json`, `odm_report/report.pdf` | candidate should not lose > 5 % features in shadowed images |
| Reconstruction: registered images, reprojection error | `opensfm/stats/stats.json` | all images registered; RMS error unchanged |
| Dense points under shadow | `odm_filterpoints/point_cloud.ply` clipped by the shadow mask | density ratio shadow/lit should rise toward 1 |
| Texture seams | visual diff of `odm_texturing/*.jpg` at shadow-edge seams | fewer colour discontinuities |
| Orthophoto colour consistency | per-tile mean/std of `odm_orthophoto.tif` along flight lines | reduced stripe pattern |

Keep `--rerun-from opensfm` (or `dataset`) in the loop so only the affected stages re-run;
a 500-image test set on the M4 Pro is a 20–40 min iteration.

## 6. Recommendations

- Capture for geometry (midday or overcast), synthesise golden hour for looks and for
  training data — the two goals are compatible only if de-lighting works.
- If golden-hour captures already exist: `--feature-quality ultra --feature-type dspsift
  --min-num-features 24000 --matcher-type bruteforce --sky-removal` and histogram-matched
  inputs; expect noisier point clouds under long shadows regardless.
- Do not build a GPU white-balance shader as a first step; build the reference-matching
  pre-process in NumPy/OpenCV first, measure its effect on `--feature-quality` match counts,
  and port to GPU only if it becomes a throughput problem.

## Sources

- ODM tutorials (capture guidance): https://docs.opendronemap.org/tutorials/ ; ODM options: https://docs.opendronemap.org/arguments/ ; `--texturing-skip-global-seam-leveling` https://docs.opendronemap.org/arguments/texturing-skip-global-seam-leveling/ ; community thread on its influence https://community.opendronemap.org/t/influence-of-texturing-skip-global-seam-leveling-parameter/26876
- ODM community: forest orthos / shadows https://community.opendronemap.org/t/forest-orthos-produced-from-same-test-images-show-different-results/693 ; DLS/sun EXIF thread https://community.opendronemap.org/t/automatic-detection-of-sun-irradiance-and-sun-angle-data-by-exif-xmp-tags/11330
- mvs-texturing: https://github.com/nmoehrle/mvs-texturing ; issue #63 colour skew https://github.com/nmoehrle/mvs-texturing/issues/63 ; Waechter et al. 2014 https://download.mmag.hrz.tu-darmstadt.de/media/FB20/GCC/paper/Waechter-2014-LTB.pdf ; OpenMVS TextureMesh https://github.com/cdcseacave/openMVS/blob/master/apps/TextureMesh/TextureMesh.cpp
- White balance / colour transfer: https://jmanansala.medium.com/image-processing-with-python-color-correction-using-white-balancing-6c6c749886de ; https://arxiv.org/abs/2208.12655 ; https://www.jouav.com/blog/orthomosaic.html ; https://www.researchgate.net/publication/354124677_Aerial_Image_Color_Balancing_Based_on_Rank-Deficient_Free_Network
- Drone timing (aesthetics): https://aerotimelapse.com/best-time-to-fly-a-drone/ ; https://drone.courses/blog/golden-hour-drone-photography/
- Synthetic data: SynDroneVision https://arxiv.org/abs/2411.05633 , https://zenodo.org/records/13360116 ; Sim2Air https://arxiv.org/abs/2110.05145 , https://github.com/larics/synthetic-UAV ; BlenderProc https://github.com/DLR-RM/BlenderProc ; Kubric https://github.com/google-research/kubric , https://arxiv.org/abs/2203.03570 ; review https://link.springer.com/article/10.1007/s10462-025-11431-3
- Shadow synthesis/augmentation: https://arxiv.org/abs/2207.04614 ; AeroDeshadow https://arxiv.org/abs/2604.15903
