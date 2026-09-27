# Golden hour shaders for ODM outputs — overview and recommended plan

Written 2026-09-13. Synthesises files `01`–`12` in this folder. Research only; no code changed.

## The problem in one sentence

ODM's textured mesh and orthophoto already contain the sun that was shining when the photos
were taken, so a "golden hour shader" is really **two shaders**: one that *removes* the baked
capture-time light (de-lighting), and one that *adds* a synthetic low warm sun with long
shadows, warm sky ambient, and haze (relighting), followed by a colour grade.

## What ODM gives us (facts verified in the code)

| Asset | Flag | Notes for relighting |
|---|---|---|
| `odm_textured_model_geo.obj` + JPEG atlases | default | Atlases up to 8192 px, **24576 px** if inputs > 8000 px (`stages/mvstex.py`); tone mapping `none`; seams levelled. |
| `odm_textured_model_geo.glb` | `--gltf` | Declares **`KHR_materials_unlit` as *required*** (`opendm/gltf.py`) so viewers render it unlit; a relighting viewer must replace the material. Z-up with `CESIUM_RTC` offset. |
| Cesium 3D Tiles | `--3d-tiles` | Via Obj2Tiles; whether the unlit extension survives is unverified. |
| `odm_orthophoto.tif` | default | cm/px resolution flag, metre geotransform. |
| `dsm.tif`, `dtm.tif` | `--dsm`, `--dtm` | metres, nodata -9999; the heightfield for 2.5D shadows. |
| Sun vector at capture time | already computable | `opendm/dls.py` uses pysolar; **caveat**: `opendm/photo.py` stamps EXIF local wall-clock as UTC, so the capture-time sun can be wrong by the UTC offset unless corrected (`02-sun-position-from-metadata.md` §2.3). |

## Recommended pipeline (cheapest thing that looks right)

1. **Sun vectors** — capture-time sun from EXIF (after fixing the UTC issue) and target
   golden-hour sun by solving elevation = 4–6° for the same lat/lon/date (`02`).
2. **De-light (2.5D first)** — from the DSM compute normals and a ray-marched hard-shadow mask
   for the *capture* sun; divide the orthophoto by the estimated irradiance (inverse Lambert +
   sky ambient) to get a pseudo-albedo. Optional: run a licence-clean aerial shadow-detection
   ONNX model through onnxruntime exactly like ODM's sky filter does (`07`). Mesh version uses
   the same idea with a shadow map from the capture sun.
3. **Relight** — Lambert/Oren-Nayar terrain shading with the golden-hour sun colour from an
   analytic sky (Hosek–Wilkie or the compact GLSL function in `03`), cascaded PCF shadows for
   the mesh or DSM ray-march soft shadows (0.53° sun disc) for the ortho, analytic height fog
   with a warm in-scatter term (`04`, `08`).
4. **Grade** — CCT-tinted light (not white balance), AgX or Khronos PBR Neutral tone map, mild
   split toning, small bloom on the sun disc, optional `.cube` LUT; defaults table in `05`.
5. **Deliver** — three.js r186 (`SunLight` + `Sky`) or WebGPU/TSL for the GLB and
   3DTilesRendererJS for tiles (`09`); Metal port notes for a native viewer in `06`; Blender
   Cycles with the Nishita sky for hero stills/video (`10`).

## Where the GPU work actually is

* Fragment shaders for relighting are trivially cheap on an M4 Pro.
* The DSM ray-march (shadow + 8-azimuth sky visibility) is the only "heavy" pass; it maps
  directly to a Metal compute kernel (`06` has the threadgroup layout) and to a WebGPU
  compute pass. For a 10k × 10k DSM it is seconds, not minutes.
* De-lighting via a neural model is the only step with real GPU cost, and it can reuse ODM's
  onnxruntime path (CoreML EP on Apple, see `../gpu-acceleration/metal/03-*.md`).

## What not to do first

* Don't try NeRF/3DGS relighting for a first version (`07` surveys them: heavy, immature for
  km-scale nadir data).
* Don't white-balance the *inputs* to fix golden-hour captures; ODM's seam levelling already
  handles most of it and shooting time matters more (`11`).
* Don't write the sky model from scratch; port Hosek–Wilkie (BSD-3) or use three.js `Sky`.

## Reading order

`01` physics → `02` sun position → `07` de-lighting → `08` ortho/DSM → `04` GLSL → `05`
grading → `03` sky models → `09` viewer → `06` Metal → `10` Blender → `11` capture-side →
`12` bibliography.

## Open questions carried forward

* Is code available for Song & Qin 2024 (arXiv 2409.03032)? It would replace the heuristic
  de-lighting.
* Does Obj2Tiles preserve `KHR_materials_unlit`?
* Fix or work around ODM's EXIF-as-UTC timestamp before trusting capture-time sun vectors.
* Which aerial shadow detector has both a permissive licence and an ONNX export.
