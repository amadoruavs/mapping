# 12 — Annotated bibliography for golden-hour shading of ODM outputs

Status: 2026-09-13. Licences are as reported by the GitHub API / project pages on that date;
"none" means no LICENSE file was detected — treat as all-rights-reserved. Items marked (?)
are uncertain.

## A. De-lighting and intrinsic decomposition

1. **Song & Qin — "A general albedo recovery approach for aerial photogrammetric images
   through inverse rendering"**, ISPRS J. Photogramm. 218 (2024). https://arxiv.org/abs/2409.03032
   — Physics-based inverse rendering with known sun + photogrammetric geometry; the paper
   that matches ODM's data exactly. Code: not stated (?).
2. **Song & Qin — "A novel intrinsic image decomposition method to recover albedo for aerial
   images in photogrammetry processing"**, ISPRS Annals 2022. https://arxiv.org/abs/2204.04142
   — Earlier version; shows matching improves after albedo recovery.
3. **Careaga & Aksoy — Intrinsic Image Decomposition via Ordinal Shading (TOG 2023) /
   Colorful Diffuse IID in the Wild (TOG 2024)**. https://github.com/compphoto/Intrinsic ,
   https://arxiv.org/abs/2409.13690 — Best general-purpose albedo/shading CNN;
   **academic-only licence**.
4. **Marigold IID (Appearance / Lighting)** — https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1 ,
   paper https://arxiv.org/abs/2505.09358 — diffusion-based albedo/shading; code Apache-2.0,
   weights CreativeML Open RAIL++-M; too heavy for atlases.
5. **LibreMarigoldV2b-albedo** — https://huggingface.co/LibreYOLO/LibreMarigoldV2b-albedo —
   Apache-2.0 albedo model on Qwen-Image-Edit; no ONNX export.
6. **Unity DeLightingTool** — https://github.com/Unity-Technologies/DeLightingTool — MIT;
   AO + object-space normal + bent-normal driven de-lighting; unmaintained since 2020.
   Walkthrough: https://sketchfab.com/blogs/community/unity-photogrammetry-de-lighting-workflow/
7. **Agisoft Metashape "Remove Lighting"** — https://agisoft.freshdesk.com/support/solutions/articles/31000158885-remove-lighting-from-texture-in-agisoft-metashape-professional
   — proprietary; "does not support delighting in strong shadows".
8. **Agisoft Texture De-Lighter (standalone, free)** — https://www.agisoft.com/index.php?id=71
   — brush-stroke lit/shadow seeds → illumination map; targets cast shadows + AO.
9. **Substance 3D Sampler — Delight (AI powered)** — https://helpx.adobe.com/substance-3d-sampler/filters/tools/delight-ai-powered.html
   — proprietary image-space delighter.
10. **Texture-aware IID with model- and learning-based priors (2025)** —
    https://arxiv.org/abs/2509.09352 — recent IID; check code/licence (?).

## B. Shadow detection / removal (aerial first)

11. **AISD — Aerial Imagery dataset for Shadow Detection** (Luo, Li, Shen, ISPRS J. 2020) —
    https://github.com/RSrscoder/AISD — 514 images; licence none; Baidu download.
12. **S-EO dataset (CVPRW 2025)** — https://centreborelli.github.io/shadow-eo/ ,
    https://arxiv.org/abs/2504.06920 — 702 georeferenced USGS tiles, geometry-aware shadow
    masks, U-Net detector; strongest candidate for orthophoto masks.
13. **SARU — Shadow-Aware and Removal Unified framework for RS images** —
    https://github.com/AeroVILab-AHU/SARU , https://arxiv.org/abs/2604.25432 — pretrained
    weights on Google Drive; licence check needed (?).
14. **AeroDeshadow / AeroDS (2026)** — https://arxiv.org/abs/2604.15903 — physics-guided
    shadow synthesis with penumbra; code "will be released".
15. **MAMNet full-scale shadow detection for RS (Remote Sensing 2024)** — https://doi.org/10.3390/rs16244789
16. **ShadowFormer (AAAI 2023)** — https://github.com/GuoLanqing/ShadowFormer — MIT; ISTD/SRD
    pretrained; ground-level.
17. **ShadowDiffusion (CVPR 2023)** — https://github.com/GuoLanqing/ShadowDiffusion — licence none.
18. **DC-ShadowNet (ICCV 2021)** — https://github.com/jinyeying/DC-ShadowNet-Hard-and-Soft-Shadow-Removal
    — unsupervised; licence none.
19. **Unveiling Deep Shadows — survey & benchmark** — https://arxiv.org/abs/2409.02108 ,
    https://github.com/xw-hu/Unveiling-Deep-Shadows
20. **Awesome-Shadow-Removal** — https://github.com/GuoLanqing/Awesome-Shadow-Removal
21. **Instance Shadow Detection with a single-stage detector** — https://arxiv.org/abs/2207.04614
    — shadow-aware copy-paste augmentation idea.

## C. Neural relighting of outdoor scenes

22. **GaRe — Relightable 3DGS for outdoor scenes from unconstrained photo collections
    (ICCV 2025)** — https://arxiv.org/abs/2507.20512 — sun visibility extraction, sky/indirect
    split, ray-traced shadows.
23. **LumiGauss (WACV 2025)** — https://arxiv.org/abs/2408.04474 , code
    https://github.com/joaxkal/lumigauss — 2DGS + SH lighting with shadows.
24. **The Sky's the Limit (ECCV 2024)** — https://arxiv.org/abs/2311.16937 — sky pixels as
    illumination observations; outside-in visibility.
25. **SOL-NeRF (SIGGRAPH Asia 2023)** — http://geometrylearning.com/SOL-NeRF/ — sun as one
    spherical Gaussian + sky as low-order SH: a good parametric model to copy.
26. **ROSGS (2025)** — https://arxiv.org/abs/2509.11275 ; **SRUG (2026)** — https://arxiv.org/abs/2605.24700
27. **OutCast — Outdoor single-image relighting with cast shadows (EG 2022)** —
    https://arxiv.org/abs/2204.09341 — learned image-space ray marching over a depth map.
28. **Fast sun-aligned outdoor relighting with TensoRF (WACV 2024)** — https://arxiv.org/abs/2311.03965
29. **Awesome-Relighting** — https://github.com/tandaily/Awesome-Relighting

## D. Heightfield shadows, horizons, sky-view factor

30. **Timonen & Westerholm — Scalable Height Field Self-Shadowing (CGF 2010)** —
    http://wili.cc/research/hfshadow/hfshadow.pdf — exact horizons for N azimuths, linear
    time; unifies hard/soft shadows on GPU.
31. **Timonen — Line-Sweep Ambient Obscurance (CGF 2013)** — https://onlinelibrary.wiley.com/doi/10.1111/cgf.12155
32. **Optimally fast soft shadows on curved terrain with DP and maximum mipmaps** —
    https://arxiv.org/abs/2005.06671
33. **Iñigo Quílez — soft shadows in raymarched SDFs** — https://iquilezles.org/articles/rmshadows/
    — penumbra estimator `k·h/t`; **terrain marching** — https://iquilezles.org/articles/terrainmarching/
34. **vterrain.org — Terrain Lighting** — http://vterrain.org/Performance/lighting.html —
    spherical-cap / smoothstep soft shadow trick.
35. **Shadertoy: "Raymarching plane height map"** https://www.shadertoy.com/view/WsByzK ;
    **"Soft Shadow Variation"** https://www.shadertoy.com/view/lsKcDD ;
    **"Raymarching - Soft Shadows"** https://www.shadertoy.com/view/tlXBRl — Shadertoy
    default licence is CC BY-NC-SA 3.0 unless stated otherwise.
36. **rvt-py (Relief Visualization Toolbox)** — https://github.com/EarthObservation/RVT_py ,
    docs https://rvt-py.readthedocs.io/ — Apache-2.0; `hillshade`, `sky_view_factor`,
    `shadow_horizon`, `sky_illumination`.
37. **GRASS r.sun / r.horizon / r.skyview / r.sunmask** — https://grass.osgeo.org/grass-stable/manuals/r.sun.html ,
    https://grass.osgeo.org/grass-stable/manuals/r.horizon.html ,
    https://grass.osgeo.org/grass78/manuals/addons/r.skyview.html ,
    https://grass.osgeo.org/grass-stable/manuals/r.sunmask.html — GPL-2.0+.
38. **SAGA GIS ta_lighting (Sky View Factor, Topographic Openness, Analytical Hillshading)** —
    https://saga-gis.sourceforge.io/saga_tool_doc/9.5.2/ta_lighting_5.html — GPL.
39. **gdaldem** — https://gdal.org/en/stable/programs/gdaldem.html ; `gdal raster hillshade`
    (3.11) — https://gdal.org/en/stable/programs/gdal_raster_hillshade.html ; USGS OF 92-422
    (multidirectional formula) — http://pubs.usgs.gov/of/1992/of92-422/of92-422.pdf ;
    Igor PR — https://github.com/OSGeo/gdal/pull/1339 — MIT/X.
40. **Robert Simmon — A gentle introduction to GDAL part 5: shaded relief** —
    https://medium.com/@robsimmon/a-gentle-introduction-to-gdal-part-5-shaded-relief-ec29601db654

## E. Raster delivery to the browser

41. **geotiff.js** — https://geotiffjs.github.io/geotiff.js/ — MIT; `fromUrl`, `readRasters`, `Pool`.
42. **georaster-layer-for-leaflet** — https://github.com/GeoTIFF/georaster-layer-for-leaflet — 2D preview, no WebGL.
43. **rio-cogeo / rio-tiler / TiTiler** — https://github.com/cogeotiff/rio-cogeo ,
    https://github.com/cogeotiff/rio-tiler , https://developmentseed.org/titiler/ — MIT/BSD.
44. **WebGPU spec, limits** — https://www.w3.org/TR/webgpu/#limits — default
    `maxTextureDimension2D = 8192`.
45. **webgl2fundamentals — cross-platform issues (MAX_TEXTURE_SIZE)** —
    https://webgl2fundamentals.org/webgl/lessons/webgl-cross-platform-issues.html

## F. Web viewers

46. **three.js r186** — Sky https://threejs.org/docs/pages/Sky.html ; SkyMesh
    https://threejs.org/docs/pages/SkyMesh.html ; CSM https://threejs.org/docs/pages/CSM.html ;
    CSMShadowNode https://threejs.org/docs/pages/CSMShadowNode.html ; **SunLight**
    https://threejs.org/docs/pages/SunLight.html (PR https://github.com/mrdoob/three.js/pull/34476,
    merged 2026-09-08) ; TSL https://threejs.org/docs/pages/TSL.html ; KTX2Loader
    https://threejs.org/docs/pages/KTX2Loader.html — MIT.
47. **three-csm** — https://github.com/StrandedKitty/three-csm — MIT; origin of the CSM addon.
48. **pmndrs/sky** — https://github.com/pmndrs/sky — WebGPU Hillaire-style sky (better at
    grazing sun than Preetham).
49. **Maxime Heckel — Field guide to TSL and WebGPU** — https://blog.maximeheckel.com/posts/field-guide-to-tsl-and-webgpu/
50. **3DTilesRendererJS v0.5.2** — https://github.com/NASA-AMMOS/3DTilesRendererJS — Apache-2.0;
    WebGPU plugin gap issue https://github.com/NASA-AMMOS/3DTilesRendererJS/issues/1380
51. **Obj2Tiles** — https://github.com/OpenDroneMap/Obj2Tiles — AGPL-3.0; ODM `--3d-tiles`.
52. **CesiumJS 1.145** — SunLight https://cesium.com/learn/cesiumjs/ref-doc/SunLight.html ;
    DirectionalLight https://cesium.com/learn/cesiumjs/ref-doc/DirectionalLight.html ;
    ShadowMap https://cesium.com/learn/cesiumjs/ref-doc/ShadowMap.html ; Viewer
    https://cesium.com/learn/cesiumjs/ref-doc/Viewer.html ; ImageBasedLighting
    https://cesium.com/learn/cesiumjs/ref-doc/ImageBasedLighting.html — Apache-2.0.
53. **Cesium community — sunset/sunrise lighting** — https://community.cesium.com/t/sunset-sunrise-night-lighting-for-terrains-and-models/4610 ;
    **lightColor on tilesets** — https://community.cesium.com/t/usage-of-lightcolor-on-cesium3dtileset-to-brighten-models/8164
54. **Potree 1.8.2** — https://github.com/potree/potree — BSD-style (read LICENSE);
    **potree-core** — https://github.com/tentone/potree-core
55. **WebODM 3D viewer docs** — https://docs.webodm.org/tutorials/potree-3d-viewer/ ;
    vendor dir https://github.com/OpenDroneMap/WebODM/tree/master/app/static/app/js/vendor — AGPL-3.0.
56. **glTF-Transform** — https://gltf-transform.dev/cli — MIT; `etc1s`, `uastc`, `resize`,
    `draco`, `meshopt`.
57. **KTX-Software (toktx)** — https://github.com/KhronosGroup/KTX-Software — Apache-2.0 (per repo).
58. **Safari 26 WebGPU** — https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/ ;
    **Chrome WebGPU release** — https://developer.chrome.com/blog/webgpu-release ;
    **gpuweb implementation status** — https://github.com/gpuweb/gpuweb/wiki/Implementation-Status
59. **onnxruntime-web WebGPU EP** — https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html

## G. Blender

60. **Sky Texture node (Nishita) — 4.2 manual** — https://docs.blender.org/manual/en/4.2/render/shader_nodes/textures/sky.html
61. **Sun Position extension v4.5.0** — https://extensions.blender.org/add-ons/sun-position/ ,
    source https://projects.blender.org/pioverfour/sun_position — GPL-3.0-or-later; NOAA/Meeus.
62. **Cycles GPU rendering (Metal)** — https://docs.blender.org/manual/en/4.2/render/cycles/gpu_rendering.html ;
    Metal AMD/Intel removal in 4.3 — https://devtalk.blender.org/t/cycles-remove-support-of-metal-with-amd-intel-gpus-for-4-3-and-onwards/35098
63. **EEVEE Next (4.2 release notes)** — https://developer.blender.org/docs/release_notes/4.2/eevee/
64. **AgX (4.0 release notes)** — https://developer.blender.org/docs/release_notes/4.0/color_management/ ;
    https://github.com/EaryChow/AgX
65. **Render baking (AO)** — https://docs.blender.org/manual/en/latest/render/cycles/baking.html ;
    headless AO bake gist https://gist.github.com/AndrewRayCode/760c4634a77551827de41ed67585064b
66. **Import operators** — https://docs.blender.org/api/current/bpy.ops.import_scene.html

## H. Capture-side, colour normalisation, synthetic data

67. **ODM docs — tutorials (midday guidance, overlap)** — https://docs.opendronemap.org/tutorials/ ;
    **options** — https://docs.opendronemap.org/arguments/
68. **mvs-texturing (Waechter et al. ECCV 2014)** — https://github.com/nmoehrle/mvs-texturing —
    BSD-3; paper https://download.mmag.hrz.tu-darmstadt.de/media/FB20/GCC/paper/Waechter-2014-LTB.pdf ;
    colour skew issue https://github.com/nmoehrle/mvs-texturing/issues/63
69. **OpenMVS TextureMesh** — https://github.com/cdcseacave/openMVS/blob/master/apps/TextureMesh/TextureMesh.cpp — AGPL-3.0.
70. **pysolar** — https://github.com/pingswept/pysolar — GPL-3.0 (already an ODM dependency, `pysolar==0.13`).
71. **White balancing methods in Python** — https://jmanansala.medium.com/image-processing-with-python-color-correction-using-white-balancing-6c6c749886de
72. **SynDroneVision (WACV 2025)** — https://arxiv.org/abs/2411.05633 , data https://zenodo.org/records/13360116
73. **Sim2Air / larics synthetic-UAV** — https://arxiv.org/abs/2110.05145 , https://github.com/larics/synthetic-UAV
74. **BlenderProc** — https://github.com/DLR-RM/BlenderProc — GPL-3.0.
75. **Kubric** — https://github.com/google-research/kubric — Apache-2.0; paper https://arxiv.org/abs/2203.03570
76. **Synthetic 3D-detection dataset generation review (AI Review 2025)** — https://link.springer.com/article/10.1007/s10462-025-11431-3

## I. ODM source files consulted (read-only, 3.6.2 tree)

- `stages/mvstex.py` — texrecon call: `-t none`, `-d gmi`, `-o gauss_clamping`,
  `--max_texture_size` 8192 (×3 if largest photo dim > 8000), `--skip_global_seam_leveling`
  passthrough, `--nadir_mode` for the 2.5D run.
- `opendm/gltf.py` — GLB writer: `KHR_materials_unlit` required, `CESIUM_RTC`, Draco via
  `draco_transcoder -qt 16 -qp 16`.
- `opendm/ogctiles.py` — Obj2Tiles invocation (`--divisions`, `--lat/--lon/--alt`) and
  Entwine point-cloud tiles in EPSG:4978.
- `opendm/dls.py` — pysolar sun altitude/azimuth (used for multispectral DLS).
- `opendm/dem/commands.py` — DSM nodata −9999, `gdal_fillnodata`.
- `opendm/config.py` — flag defaults (`--dem-resolution 5`, `--orthophoto-resolution 5`,
  feature/matcher flags, deprecated `--texturing-skip-local-seam-leveling`).
- `opendm/skyremoval/skyfilter.py` — onnxruntime session pattern (CUDA or CPU EP, 384² input).
- `pixi.toml` — pins: gdal 3.11.1, onnxruntime 1.22.2, pysolar 0.13, rasterio 1.4.3,
  pdal 2.8.4, pygltflib 1.16.5, python 3.12.9.
- `contrib/blender/{common.py,odm_photo.py,odm_video.py,README.md}` — 2.7x-era Blender scripts.
