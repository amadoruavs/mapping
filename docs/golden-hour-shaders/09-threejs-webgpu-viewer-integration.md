# 09 — Web viewer integration: three.js / WebGPU, 3D Tiles, Cesium, Potree

Status: research note, 2026-09-13. Documentation only; code below is illustrative.
Versions checked via GitHub API on 2026-09-13: three.js **r186** (2026-09-08),
3DTilesRendererJS **v0.5.2** (2026-08-24), CesiumJS **1.145** (2026-09-01),
Potree **1.8.2** (2023-12-12, last release).

## 1. What ODM hands the viewer

| Output | Flag | Format details (from ODM 3.6.2 source) |
|---|---|---|
| `odm_texturing/odm_textured_model_geo.obj` + `.mtl` + `*.jpg` | default | Multiple materials/atlases unless `--texturing-single-material`. Atlas size: `max_texture_size = 8192`, **×3 = 24576** when the largest input photo dimension > 8000 px (`stages/mvstex.py`). |
| `odm_textured_model_geo.glb` | `--gltf` | `opendm/gltf.py`: one mesh, one primitive per material, `POSITION/TEXCOORD_0/NORMAL`, JPEG textures embedded, material `metallicFactor=0, roughnessFactor=1` **and `KHR_materials_unlit` in `extensionsRequired`**, `CESIUM_RTC` centre when georeferenced, then `draco_transcoder -qt 16 -qp 16` (falls back to uncompressed on error). |
| `3d_tiles/model/tileset.json` | `--3d-tiles` | Obj2Tiles (AGPL-3.0) splits/decimates the OBJ into LODs (`divisions` from area, `--lat/--lon/--alt` from `reference_lla.json`), `opendm/ogctiles.py`. |
| `3d_tiles/pointcloud/` | `--3d-tiles` | Entwine (EPT) build in EPSG:4978 → 3D Tiles point cloud (`build_pointcloud`). |
| `odm_georeferenced_model.laz` | default | For Potree/EPT. |

Two consequences for relighting:

1. **The GLB is unlit by contract.** three.js `GLTFLoader` honours `KHR_materials_unlit` and
   creates `MeshBasicMaterial`; Cesium likewise renders unlit. To relight, walk the scene and
   replace each material with `MeshStandardMaterial({ map, roughness: 1, metalness: 0 })`
   (or a TSL `MeshStandardNodeMaterial`) keeping `map.colorSpace = SRGBColorSpace`.
   3DTilesRendererJS tiles from Obj2Tiles are ordinary glTF with lit PBR materials
   (**verify** — Obj2Tiles writes its own glTF; check for the unlit extension in a sample).
2. **Texture size.** 8192² JPEG atlases fit under the WebGPU default limit of 8192 exactly;
   the 24576² case exceeds every browser limit (WebGPU default 8192, Apple adapters
   16384 for WebGL `MAX_TEXTURE_SIZE`). Either run ODM with smaller input images, post-process
   with `gltf-transform resize --width 8192 --height 8192`, or split the atlas. See §7.

## 2. three.js (WebGL and WebGPU) — the pieces

### 2.1 Sky

- WebGL: `import { Sky } from 'three/addons/objects/Sky.js'` — Preetham analytic model,
  uniforms `turbidity, rayleigh, mieCoefficient, mieDirectionalG, sunPosition, showSunDisc`
  ([docs](https://threejs.org/docs/pages/Sky.html)).
- WebGPU: `SkyMesh` (same model, TSL implementation; uniforms are `sky.turbidity.value` etc.)
  — [docs](https://threejs.org/docs/pages/SkyMesh.html), example
  [webgpu_sky](https://threejs.org/examples/webgpu_sky.html).
- Preetham at very low elevations (< 3°) gets unphysically saturated; clamp elevation ≥ 2°
  or switch to a Hillaire-style sky ([pmndrs/sky](https://github.com/pmndrs/sky), WebGPU,
  "based on the Unreal Engine sky by Sébastien Hillaire") for hero shots.
- Use the sky as the environment map (render to a `PMREMGenerator` cube / `scene.environment`)
  so shadowed sides pick up blue sky light; hide the sun disc while baking the env map (the
  Sky docs show `showSunDisc.value = false` for this).

### 2.2 Sun + cascaded shadows

- WebGL: `CSM` addon (`three/addons/csm/CSM.js`, originally StrandedKitty/three-csm, MIT):
  `new CSM({ camera, parent: scene, cascades: 3, maxFar: 100000, mode: 'practical',
  shadowMapSize: 2048, lightDirection, lightIntensity: 3, lightMargin: 200, fade })`;
  call `csm.setupMaterial(material)` on every lit material and `csm.update()` each frame
  ([docs](https://threejs.org/docs/pages/CSM.html)).
- WebGPU: `CSMShadowNode` — `light.shadow.shadowNode = new CSMShadowNode(light, { cascades,
  mode, maxFar, lightMargin, fade })`, then `updateFrustums()` when the camera changes
  ([docs](https://threejs.org/docs/pages/CSMShadowNode.html)).
- **New in r186: `SunLight`** (`three/addons/lights/SunLight.js`): a directional light with
  camera-fitted cascaded shadows via `SunLightShadow`; direction is defined by its position;
  WebGPU needs `renderer.library.addLight(SunLightNode, SunLight)`
  ([docs](https://threejs.org/docs/pages/SunLight.html)). PR
  [mrdoob/three.js#34476](https://github.com/mrdoob/three.js/pull/34476) (merged
  2026-09-08) switched 13 WebGPU examples to it, "replacing manual shadow bounds with
  camera-fitted cascades". This is the simplest path for a golden-hour sun over a
  kilometre-scale mesh.
- Shadow acne at grazing sun angles: use `shadow.normalBias` (0.02–0.1 in scene units) and
  PCF-soft or VSM; at 5–10° elevation shadow maps stretch badly — cascades + a high
  `shadowMapSize` (4096) on the M4 Pro is fine.

### 2.3 Custom lighting with TSL (WebGPU node materials)

The point of TSL here is to implement the doc 07/08 model in the material: albedo ×
(sun·V·n·l + sky·SVF + AO). Example
[webgpu_lights_custom](https://threejs.org/examples/webgpu_lights_custom.html) shows a
custom `LightingModel`; the field guide by Maxime Heckel is a good primer
([blog](https://blog.maximeheckel.com/posts/field-guide-to-tsl-and-webgpu/)); TSL reference
[docs](https://threejs.org/docs/pages/TSL.html).

```js
import * as THREE from 'three/webgpu';
import { texture, uv, normalWorld, uniform, max, dot, float, vec3, mix } from 'three/tsl';

const sunDir   = uniform(new THREE.Vector3());       // updated from time-of-day
const sunColor = uniform(new THREE.Color(1.0, 0.72, 0.42));
const skyColor = uniform(new THREE.Color(0.35, 0.45, 0.70));
const albedo   = texture(albedoTex, uv());           // de-lit atlas if available, else ODM atlas
const aoTex    = texture(aoOrSvfTex, uv());          // baked AO / SVF, optional

const mat = new THREE.MeshStandardNodeMaterial();
mat.colorNode = albedo;                              // let three's PBR + SunLight shadows do sun·V·n·l
mat.aoNode    = aoTex.r;                             // sky-visibility term
mat.roughnessNode = float(1.0); mat.metalnessNode = float(0.0);
// Scene: sun = new SunLight(sunColor, 3); scene.environment = skyEnvFromSkyMesh;
```

For a *fully* custom model (e.g. divide-out of the capture-time shading on the GPU), write a
`LightingModel` subclass or compute `colorNode` from `albedo.div(captureShade)` where
`captureShade` is a uniform-driven `max(0, dot(normalWorld, captureSun))`-based node — only
valid where a shadow mask texture for the capture sun exists (doc 07 §4).

### 2.4 Time-of-day → sun vector

Port the NOAA/Meeus solar position (what Blender's Sun Position add-on and pysolar use) to
JS, or precompute a table server-side with pysolar for the site's lat/lon and the chosen
date; feed `sunPosition` (sky) and `SunLight.position` from the same vector. Golden hour is
sun elevation between roughly −4° and +6° (photographers' convention); shadows lengthen
as `1/tan(elevation)`.

## 3. 3DTilesRendererJS (NASA-AMMOS) for ODM's `--3d-tiles`

- `npm i 3d-tiles-renderer`; Apache-2.0; three.js, Babylon.js and R3F front-ends
  ([README](https://github.com/NASA-AMMOS/3DTilesRendererJS)).
- Minimal three.js use:

```js
import { TilesRenderer } from '3d-tiles-renderer';
import { GLTFExtensionsPlugin, TilesFadePlugin } from '3d-tiles-renderer/plugins';
const tiles = new TilesRenderer('/3d_tiles/model/tileset.json');
tiles.registerPlugin(new GLTFExtensionsPlugin({ dracoLoader, ktx2Loader }));
tiles.registerPlugin(new TilesFadePlugin());
tiles.setCamera(camera); tiles.setResolutionFromRenderer(camera, renderer);
tiles.addEventListener('load-model', ({ scene }) => scene.traverse(o => {
  if (o.isMesh) { o.castShadow = o.receiveShadow = true;
    if (o.material.isMeshBasicMaterial) o.material = new THREE.MeshStandardMaterial({ map: o.material.map, roughness: 1, metalness: 0 }); }
}));
scene.add(tiles.group);  // per frame: tiles.update()
```

- Obj2Tiles tilesets are georeferenced (ECEF via `--lat/--lon/--alt`); for a local-frame
  viewer, re-orient `tiles.group` with the inverse of the root transform or use the
  library's `ReorientationPlugin`/`GlobeControls` (names per current README plugin list —
  **verify** exact plugin names against v0.5.x).
- **WebGPU status**: the maintainer stated the core renderer is WebGPU-ready, but issue
  [#1380](https://github.com/NASA-AMMOS/3DTilesRendererJS/issues/1380) (open, 2025-11,
  v0.4.18 / r181) shows `ImageOverlayPlugin` uses `ShaderMaterial` and fails under
  `WebGPURenderer`. For a golden-hour viewer that only needs meshes + custom materials this
  is acceptable: swap materials in `load-model` as above and avoid the overlay plugin.
- LRU cache: "once the LRU cache reaches its maximum, no additional tiles load" — raise
  `tiles.lruCache.maxSize`/`minSize` for a 32 GB M4 Pro.

## 4. CesiumJS with ODM 3D Tiles

- Load: `const ts = await Cesium.Cesium3DTileset.fromUrl('/3d_tiles/model/tileset.json'); viewer.scene.primitives.add(ts);`
- Lighting: `scene.light` defaults to `new SunLight()` (`color: WHITE`, `intensity: 2.0`;
  [SunLight docs](https://cesium.com/learn/cesiumjs/ref-doc/SunLight.html), and
  `Scene.js` line ~768 `this.light = new SunLight()`). Replace with a
  [`DirectionalLight`](https://cesium.com/learn/cesiumjs/ref-doc/DirectionalLight.html)
  for an arbitrary sun, or simply set `viewer.clock.currentTime = JulianDate.fromIso8601('2026-06-21T17:40:00Z')`
  and Cesium computes the real sun; `viewer.clock.shouldAnimate = true; multiplier = 600`
  animates golden hour.
- Shadows: `new Cesium.Viewer(div, { shadows: true, terrainShadows: Cesium.ShadowMode.ENABLED })`;
  `viewer.shadowMap.size = 4096; softShadows = true; darkness = 0.3; maximumDistance = 5000;
  fadingEnabled = true` (shadows fade near the horizon — **set `fadingEnabled = false` for
  golden hour** or they vanish exactly when wanted); "only one or four cascades"
  ([ShadowMap docs](https://cesium.com/learn/cesiumjs/ref-doc/ShadowMap.html),
  [Viewer docs](https://cesium.com/learn/cesiumjs/ref-doc/Viewer.html)). `ShadowMode`:
  `DISABLED=0, ENABLED=1, CAST_ONLY=2, RECEIVE_ONLY=3` (`ShadowMode.js`).
- Tileset material: `ts.lightColor = new Cesium.Cartesian3(1.0, 0.75, 0.5)` scales the
  directional light; `ts.imageBasedLighting = new Cesium.ImageBasedLighting({ imageBasedLightingFactor, luminanceAtZenith, sphericalHarmonicCoefficients, specularEnvironmentMaps })`
  (since 1.94; [issue #10059](https://github.com/CesiumGS/cesium/issues/10059)) — set SH
  coefficients from a golden-hour sky HDR for blue-tinted shadows. Globe: `scene.globe.enableLighting = true`,
  `dynamicAtmosphereLighting` for the terrain.
- Cesium's built-in shading is Phong/PBR with a single sun — no de-lighting; the same
  double-shadow problem applies unless the tiles carry a de-lit albedo. Custom shaders on
  tilesets are possible via `Cesium.CustomShader` (fragment stage with `material.diffuse`),
  which is where a per-pixel `1/captureShade` correction could go.

## 5. Point clouds: Potree / Potree 2 / Cesium

- **Potree 1.8.2** (Dec 2023) is the last tagged release; the "Potree 2.0" WebGPU rewrite
  exists only as forks/experiments on GitHub — no official release found. Potree is
  BSD-2-Clause-style (repo `LICENSE`, "NOASSERTION" on GitHub API; read the file).
- Potree has EDL (eye-dome lighting) and per-point normals are optional; sun shadows on point
  clouds are not supported. A golden-hour look for point clouds = RGB grade + EDL + fog; if
  shadows are needed, render the DSM/mesh instead.
- For a three.js-native path: [tentone/potree-core](https://github.com/tentone/potree-core)
  (uses shiukaheng's potree-loader; since 2.0) exposes Potree as a three.js object so the
  same `SunLight` scene can host it; ODM's EPT output loads directly.
- Cesium: `Cesium3DTileset` point clouds support `pointCloudShading` (EDL, attenuation) and
  the same `lightColor`.

## 6. What WebODM ships today

WebODM (`OpenDroneMap/WebODM`, AGPL-3.0) vendors under `app/static/app/js/vendor/`:
`potree/` (a patched Potree 1.x build with `build/potree/` + `shaders/`), three.js loaders
(`OBJLoader.js`, `GLTFLoader.js`, `MTLLoader.js`, `DRACOLoader.js`, `draco/`), `leaflet`,
`pannellum` (360° photos), and `threedgs.umd.min.js` — a 3D Gaussian Splatting viewer
(**inference from filename**; not verified in code). Its "3D model" tab shows the OBJ/GLB via
three.js inside the Potree viewer, with EDL and measurement tools
([WebODM docs](https://docs.webodm.org/tutorials/potree-3d-viewer/)). So a golden-hour
plugin for WebODM would live in the Potree/three.js scene: add a `Sky`, a `DirectionalLight`/
`CSM`, swap the loaded textured model's `MeshBasicMaterial`s for `MeshStandardMaterial`s,
and enable shadow maps — all WebGL 2, no WebGPU, given the vendored three.js revision.

## 7. Textures: limits and KTX2/Basis

- ODM atlases are JPEG; decoding a 8192² JPEG on the main thread stalls ~0.5–1 s and costs
  256 MB of RGBA in GPU memory *per atlas*; a city block with 6–10 atlases is 1.5–2.5 GB.
  Compress to **KTX2**: `gltf-transform etc1s model.glb out.glb` (small, lower quality) or
  `gltf-transform uastc model.glb out.glb --level 2 --rdo` (higher quality; 4:1 in GPU
  memory, ~8 bpp on disk with RDO) — [glTF-Transform CLI](https://gltf-transform.dev/cli),
  MIT, requires KTX-Software's `toktx` on PATH. Load with `KTX2Loader` + `GLTFLoader`
  (`KHR_texture_basisu`); the loader transcodes to ASTC on Apple GPUs
  ([KTX2Loader docs](https://threejs.org/docs/pages/KTX2Loader.html)).
- KTX2 is limited to the same max dimensions; resize first: `gltf-transform resize in.glb out.glb --width 8192 --height 8192`.
- Draco (ODM applies `-qt 16 -qp 16`) or `gltf-transform meshopt` for geometry; Draco needs
  `DRACOLoader` (WASM) on the client — WebODM already vendors it.

## 8. Performance budget on an M4 Pro laptop (Safari 26 / Chrome)

- **WebGPU availability**: Safari 26 (macOS Tahoe 26, Sept 2025) ships WebGPU by default and
  Apple says it "supersedes WebGL … and is preferred for new sites"
  ([WebKit blog](https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/));
  Chrome has shipped WebGPU on macOS via Dawn→Metal since 113 (2023)
  ([Chrome blog](https://developer.chrome.com/blog/webgpu-release)). Both browsers on an M4
  Pro therefore run three.js `WebGPURenderer`; keep a `WebGLRenderer` fallback for older
  Safari.
- Targets: 60 fps at 2560×1600 (native 3024×1964 is fine for stills). Budget per frame
  ≈ 16 ms: 3–4 cascades × 4096² shadow maps ≈ 3–5 ms for a 5 M-triangle scene; sky env
  update when the time slider moves (throttle to 10 Hz); post-processing (tonemap, bloom,
  haze) ≈ 1–2 ms at 4K.
- Memory: unified memory (24–64 GB) makes GPU memory generous, but Safari caps per-tab GPU
  allocations; keep total textures < 2 GB and use KTX2.
- Triangle budget: ODM meshes default to 200 k faces for the 3D model (`--mesh-size`)
  but Obj2Tiles LODs and the 2.5D mesh push more; 5–10 M triangles with cascaded shadows
  is comfortable on an M4 Pro at 1440p (**estimate**, not measured).
- onnxruntime-web has a WebGPU EP (`import * as ort from 'onnxruntime-web/webgpu'`,
  `executionProviders: ['webgpu']`) if a shadow-mask model must run client-side; its docs
  still describe Safari support as "Technology Preview" (stale relative to Safari 26)
  ([ORT docs](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)).

## 9. Minimal reference architecture

```
/viewer
  index.html            three r186 (webgpu build, WebGL fallback), SunLight, SkyMesh, CSMShadowNode
  tiles.js              3DTilesRendererJS v0.5.x  -> ODM /3d_tiles/model/tileset.json
  glb.js                GLTFLoader+DRACOLoader+KTX2Loader -> odm_textured_model_geo.glb (unlit→standard swap)
  ortho2p5d.js          DSM R32F + ortho tiles -> displaced grid + fragment relight (doc 08)
  sun.js                NOAA solar position (lat/lon from reference_lla.json), golden-hour slider
  post.js               AgX-like tonemap, haze, bloom, vignette, grain
```

## Sources

- three.js docs: Sky https://threejs.org/docs/pages/Sky.html ; SkyMesh https://threejs.org/docs/pages/SkyMesh.html ; CSM https://threejs.org/docs/pages/CSM.html ; CSMShadowNode https://threejs.org/docs/pages/CSMShadowNode.html ; SunLight https://threejs.org/docs/pages/SunLight.html ; TSL https://threejs.org/docs/pages/TSL.html ; KTX2Loader https://threejs.org/docs/pages/KTX2Loader.html ; example webgpu_sky https://threejs.org/examples/webgpu_sky.html ; webgpu_lights_custom https://threejs.org/examples/webgpu_lights_custom.html ; SunLight PR https://github.com/mrdoob/three.js/pull/34476
- three-csm: https://github.com/StrandedKitty/three-csm ; CSM on WebGPU thread https://discourse.threejs.org/t/cascaded-shadow-maps-csm-on-webgpu/84235 ; pmndrs/sky https://github.com/pmndrs/sky ; TSL field guide https://blog.maximeheckel.com/posts/field-guide-to-tsl-and-webgpu/
- 3DTilesRendererJS: https://github.com/NASA-AMMOS/3DTilesRendererJS ; WebGPU issue https://github.com/NASA-AMMOS/3DTilesRendererJS/issues/1380
- Obj2Tiles: https://github.com/OpenDroneMap/Obj2Tiles ; ODM source `opendm/ogctiles.py`, `opendm/gltf.py`, `stages/mvstex.py`
- CesiumJS: SunLight https://cesium.com/learn/cesiumjs/ref-doc/SunLight.html ; DirectionalLight https://cesium.com/learn/cesiumjs/ref-doc/DirectionalLight.html ; ShadowMap https://cesium.com/learn/cesiumjs/ref-doc/ShadowMap.html ; Viewer https://cesium.com/learn/cesiumjs/ref-doc/Viewer.html ; ImageBasedLighting https://cesium.com/learn/cesiumjs/ref-doc/ImageBasedLighting.html ; IBL bundling https://github.com/CesiumGS/cesium/issues/10059 ; lightColor thread https://community.cesium.com/t/usage-of-lightcolor-on-cesium3dtileset-to-brighten-models/8164 ; sunset lighting thread https://community.cesium.com/t/sunset-sunrise-night-lighting-for-terrains-and-models/4610
- Potree: https://github.com/potree/potree ; potree-core https://github.com/tentone/potree-core ; WebODM 3D viewer https://docs.webodm.org/tutorials/potree-3d-viewer/ ; WebODM vendor dir https://github.com/OpenDroneMap/WebODM/tree/master/app/static/app/js/vendor
- glTF-Transform: https://gltf-transform.dev/cli ; KTX2/Basis in three.js https://threejs.org/examples/jsm/libs/basis/
- WebGPU: Safari 26 https://webkit.org/blog/16993/news-from-wwdc25-web-technology-coming-this-fall-in-safari-26-beta/ ; Chrome https://developer.chrome.com/blog/webgpu-release ; spec limits https://www.w3.org/TR/webgpu/#limits ; onnxruntime-web WebGPU https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
