# 08 — 2.5D relighting of the orthophoto with the DSM

Status: research note, 2026-09-13. Documentation only.

## 1. Inputs ODM gives us

| File | Producer | Units / format | Notes |
|---|---|---|---|
| `odm_orthophoto/odm_orthophoto.tif` | odm_orthophoto stage | RGB(A) 8-bit, **cm/px** resolution (`--orthophoto-resolution`, default 5 cm/px, capped by GSD) | 4th band is alpha; `--cog` makes it a Cloud Optimized GeoTIFF; `--tiles` writes XYZ PNG tiles. |
| `odm_dem/dsm.tif` | odm_dem stage (PDAL) | Float32 **metres**, resolution `--dem-resolution` default 5 cm/px | nodata default −9999 (`opendm/dem/commands.py`, honouring the file's nodata if set); gaps filled with `gdal_fillnodata`. Requires `--dsm`. |
| `odm_dem/dtm.tif` | same, `--dtm` | Float32 metres | ground only — use for "terrain-only" shadows or to isolate object heights (`dsm − dtm`). |
| CRS | both in the project's UTM zone (metres) | | Same grid *if* both resolutions are equal; otherwise resample the DSM to the orthophoto grid (bilinear) or vice versa. |

Resolution/units traps:

- `--dem-resolution` and `--orthophoto-resolution` are in **cm/px**; the GeoTIFF geotransform
  is in **metres** (UTM). A 5 cm/px DSM has pixel size 0.05 m. Slope needs `dz/dx` with both
  axes in metres: `gdaldem` handles this with `-s` (scale) — 1 for projected metres, 111120
  for lat/long degrees; GDAL 3.11 adds `-xscale/-yscale` and auto-detection
  ([gdaldem docs](https://gdal.org/en/stable/programs/gdaldem.html)). ODM's `pixi.toml` pins
  `gdal==3.11.1`, so the new `gdal raster hillshade` CLI is also available
  ([docs](https://gdal.org/en/stable/programs/gdal_raster_hillshade.html)).
- Nodata −9999 must be masked *before* any gradient or ray-march, or every DSM edge becomes a
  9 km cliff. `gdaldem` writes 0 as the hillshade nodata; use `-compute_edges` to avoid a
  one-pixel nodata ring.
- The orthophoto is rendered from ODM's 2.5D mesh (unless `--use-3dmesh`), the DSM from the
  point cloud; they align well but tree crowns and roof edges can differ by a few pixels.
  Feather shadow masks by 2–3 px.
- Vertical datum: irrelevant for shading (only relative heights matter), but matters if you
  hand the DSM to Cesium terrain.

## 2. Static hillshade with GDAL (baseline, CPU)

```
gdaldem hillshade dsm.tif hs.tif -az 315 -alt 45 -z 1 -s 1 -compute_edges -alg Horn
gdaldem hillshade dsm.tif hs_multi.tif -multidirectional -compute_edges     # 225/270/315/360°, USGS OF 92-422
gdaldem hillshade dsm.tif hs_igor.tif -igor -compute_edges                   # Maperitive "Igor", low-contrast, no -alt
gdaldem hillshade dsm.tif hs_comb.tif -combined -alt 45 -az 315              # slope × oblique
```

- `-az` light azimuth (0 = north, 90 = east; default 315), `-alt` altitude (90 overhead,
  0 raking), `-z` vertical exaggeration, `-alg Horn|ZevenbergenThorne` (Zevenbergen-Thorne
  "more suited to smooth landscapes", Horn "better on rougher terrain"). Output is 8-bit.
- For golden hour set `-alt` to the target sun elevation (≈5–15°) and `-az` to the sun
  azimuth. Hillshade is *Lambert cos(θ)* only — **no cast shadows**. At 10° elevation the
  lack of cast shadows is the first thing a viewer notices, so hillshade alone is only a
  preview.
- `-multidirectional` and `-igor` are cartographic (they *reduce* directional drama);
  useful as an ambient/SVF-like base layer, not for the sun term.
  Background on the modes: [Robert Simmon's GDAL shaded relief article](https://medium.com/@robsimmon/a-gentle-introduction-to-gdal-part-5-shaded-relief-ec29601db654),
  [GDAL PR #1339 adding -igor](https://github.com/OSGeo/gdal/pull/1339).

## 3. Horizon / sky-view based shadows (CPU, GIS toolboxes)

### 3.1 rvt-py (Relief Visualization Toolbox, Apache-2.0)

Package [EarthObservation/RVT_py](https://github.com/EarthObservation/RVT_py);
API [rvt.vis](https://rvt-py.readthedocs.io/en/latest/rvt.vis.html):

- `rvt.vis.hillshade(dem, resolution_x, resolution_y, sun_azimuth, sun_elevation, ve_factor)`
- `rvt.vis.sky_view_factor(dem, resolution, compute_svf, compute_asvf, compute_opns, svf_n_dir=16, svf_r_max=10, svf_noise=0, asvf_dir, asvf_level)` → SVF, anisotropic SVF, positive openness.
  Doc: [sky-view factor](https://rvt-py.readthedocs.io/en/latest/listofvis_svf.html)
  (recommended display stretch 0.65–1.0; anisotropic SVF assumes "the sky is brighter in some
  directions").
- `rvt.vis.shadow_horizon(dem, resolution, shadow_az, shadow_el, ...)` → **binary cast-shadow
  map + horizon** for a given sun azimuth/elevation.
- `rvt.vis.sky_illumination(dem, resolution, sky_model='overcast'|'uniform', compute_shadow, shadow_az, shadow_el, ...)` → diffuse sky illumination with optional directional shadow.
- `rvt.vis.local_dominance`, `msrm` — cartographic extras.

This is the shortest path to a *correct* golden-hour 2.5D shading in Python:
`shade = V·max(0, n·l)·k + SVF`, exactly the model of doc 07 §4.

### 3.2 GRASS GIS `r.horizon` / `r.sun` / `r.skyview`

- [r.horizon](https://grass.osgeo.org/grass-stable/manuals/r.horizon.html): horizon angle
  raster per azimuth step ("starts at a very shallow angle and walks along the line of sight").
- [r.sun](https://grass.osgeo.org/grass-stable/manuals/r.sun.html): mode 1 gives
  instantaneous incidence angle (`incidout`) and beam/diffuse/reflected irradiance for a given
  `day`, `time`, Linke turbidity and albedo; shadowing is computed from the DEM by default
  (`-p` disables it) or from `horizon_basename` rasters (much faster); "areas with NULL values
  are shadowed" in `incidout`. This gives a physically scaled W/m² sun+sky split for free.
- [r.skyview](https://grass.osgeo.org/grass78/manuals/addons/r.skyview.html): SVF from
  r.horizon.

### 3.3 SAGA GIS `ta_lighting`

- Sky View Factor (module 3) and Topographic Openness (module 5) — Yokoyama et al. 2002
  ([SAGA tool docs](https://saga-gis.sourceforge.io/saga_tool_doc/9.5.2/ta_lighting_5.html)).
  Also "Analytical Hillshading" with a *shadow* option in the same library.

## 4. GPU heightfield ray-march (interactive sun)

For a live golden-hour slider the shadow term has to be computed in the viewer. The DSM is
a heightfield, so a fragment shader can march toward the sun:

```glsl
// uDsm: R32F texture (metres), uPx: metres per texel, uSun: unit vector (x east, y north, z up)
float sunVisibility(vec2 uv) {
  float h0 = texture(uDsm, uv).r;
  if (h0 <= NODATA_THRESHOLD) return 1.0;
  vec2  dir  = normalize(uSun.xy) * uTexel;      // one texel per step in uv space
  float rise = uPx * uSun.z / length(uSun.xy);   // metres gained per texel of travel
  float z = h0 + 0.05;                           // bias to avoid self-shadow acne
  for (int i = 1; i < MAX_STEPS; ++i) {
    vec2 p = uv + dir * float(i);
    if (any(lessThan(p, vec2(0))) || any(greaterThan(p, vec2(1)))) break;
    z += rise;
    float h = texture(uDsm, p).r;
    if (h > z) return 0.0;                       // hard shadow
  }
  return 1.0;
}
```

- **Step length / max distance**: at 10° elevation a 30 m building throws a 170 m shadow =
  3400 texels at 5 cm/px. Marching that per pixel is too slow; the standard fixes are
  (a) march on a coarser DSM mip (e.g. 25 cm/px) and refine the last few steps on the fine
  level, (b) max-mipmap acceleration (ray skips whole quadtree cells whose max height is
  below the ray) — [arXiv 2005.06671](https://arxiv.org/abs/2005.06671), or (c) precompute
  **horizon angles** for N azimuths (Timonen & Westerholm 2010: "exact horizon for a set of
  azimuthal directions in time linear in height field size", 23 fps at 1024² × 64 directions
  on a 2008 GPU — [PDF](http://wili.cc/research/hfshadow/hfshadow.pdf)) and then shadow test
  is just `sunElevation < horizon[azimuthBin]` with interpolation between bins.
- **Soft shadows**: the sun subtends ≈0.53°. With horizon maps: penumbra fraction =
  `smoothstep(horizon − 0.265°, horizon + 0.265°, sunElevation)`. With ray-marching: track the
  minimum angular clearance `k·(z − h)/dist` along the ray (iq's soft-shadow estimator,
  [rmshadows](https://iquilezles.org/articles/rmshadows/)); vterrain.org describes the
  same "spherical cap" smoothstep trick for terrain
  ([Terrain Lighting](http://vterrain.org/Performance/lighting.html)). Real golden-hour
  penumbrae from a 30 m building are ~30 cm wide — 6 px at 5 cm/px — so a 1-texel feather is
  *not* enough for hero shots; use the angular approach.
- **Normals** from the DSM in the same shader: central differences scaled by `uPx`
  (Horn kernel if you want to match gdaldem).

## 5. Combining into a golden-hour composite

Sky/sun colours for the look (empirical, tune by eye; physically based sky models give
these for free — three.js `Sky` (Preetham) or Blender Nishita, see docs 09/10):

- Sun at 5–10° elevation: colour ≈ (1.0, 0.75, 0.45) linear, intensity falls ~×0.3 vs noon.
- Sky (ambient) at golden hour: ≈ (0.35, 0.45, 0.7) linear, dim; shadows go blue.

Pass order (all fragment shaders on full-screen quads / tiles):

1. **Load** DSM (R32F) and orthophoto (RGBA8, decode sRGB→linear), mask nodata/alpha.
2. **Normals** N from DSM.
3. **De-light (optional, doc 07)**: `S_cap = V_cap·max(0,N·l_cap)·k + SVF`; `A = I / S_cap`.
   Without this step, treat `I` as albedo and accept residual baked shadows.
4. **Sun visibility** `V_new` for the target sun (ray-march or horizon map; soft).
5. **Shade**: `L = A · (sunColor·V_new·max(0,N·l_new) + skyColor·SVF + AO_term)`.
6. **Atmosphere**: distance/height haze `mix(L, hazeColor, 1 − exp(−d·β))`, warm haze for
   golden hour; optional light bloom on lit edges.
7. **Grade**: exposure, AgX/ACES-like tone map, slight lift of blacks, colour temperature
   shift, film grain; encode to sRGB.

A cheap **look-only** variant (no geometry): steps 1, 6, 7 plus a hillshade multiply — this
is what most "golden hour filters" are, and it is honest to label it as such.

## 6. Getting GeoTIFFs into WebGL/WebGPU textures

### 6.1 Client-side decode with geotiff.js

[geotiff.js](https://geotiffjs.github.io/geotiff.js/) reads COGs over HTTP range requests:

```js
import { fromUrl, Pool } from 'geotiff';
const tiff  = await fromUrl('/odm_dem/dsm.tif');           // must be a COG for ranged reads
const img   = await tiff.getImage();                       // level 0; getImage(i) for overviews
const [ox, oy] = img.getOrigin(); const [rx, ry] = img.getResolution();
const nodata = img.getGDALNoData();                        // -9999 for ODM
const pool  = new Pool();                                  // web workers for decoding
const [dsm] = await img.readRasters({ window: [x0, y0, x1, y1], samples: [0], pool }); // Float32Array
// upload: WebGL2 -> gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, dsm)
// three.js -> new THREE.DataTexture(dsm, w, h, THREE.RedFormat, THREE.FloatType)
// WebGPU  -> device.createTexture({format:'r32float', ...}); queue.writeTexture(...)
```

- Filtering R32F textures in WebGL2 needs `OES_texture_float_linear`; on WebGPU `r32float`
  is *not* filterable by default (sample with `textureLoad` or convert to `r16float`,
  which is filterable, at 0.03 m precision for a 1000 m range — check the site's height
  range). Rendering to float targets in WebGL2 needs `EXT_color_buffer_float`.
- ODM's orthophoto with `--cog` and `--tiles` already gives both a COG and XYZ PNG tiles.
  For the DSM, produce a COG with overviews (`gdal_translate -of COG -co OVERVIEWS=IGNORE_EXISTING`
  or `rio cogeo create`) so the viewer can read the pyramid level it needs.
- [georaster-layer-for-leaflet](https://github.com/GeoTIFF/georaster-layer-for-leaflet)
  wraps geotiff.js for Leaflet without WebGL — fine for a 2D preview, not for shaders.

### 6.2 Server-side tiling for large rasters

- [rio-cogeo](https://github.com/cogeotiff/rio-cogeo) creates COGs; [rio-tiler](https://github.com/cogeotiff/rio-tiler)
  + [TiTiler](https://developmentseed.org/titiler/) serve dynamic XYZ tiles and can encode a
  float DSM into Mapbox **Terrain-RGB** (`h = −10000 + (R·65536 + G·256 + B)·0.1`) so the
  viewer can use ordinary 8-bit PNG tiles and decode height in the shader — 0.1 m
  quantisation is acceptable for shading but *not* for fine ray-marched shadows on 5 cm data
  (a 10 cm step at 10° elevation is a 57 cm horizontal error). Prefer float COG tiles for
  hero quality; use Terrain-RGB for the overview zoom levels.
- Tile size: 512² or 1024² tiles with a 1-px overlap for gradients; shadow ray-marching
  needs neighbours — either march on a coarse global DSM (e.g. 1 m/px whole-site texture,
  ≤ 8192² fits everywhere) and only use fine tiles for normals, or use horizon maps
  precomputed server-side (rvt-py / Timonen) and served as 8-bit tiles per azimuth bin.

### 6.3 Size limits

- WebGPU default `maxTextureDimension2D` is **8192**; higher must be requested via
  `requestDevice({ requiredLimits })` and only up to the adapter's limit
  ([WebGPU spec, limits](https://www.w3.org/TR/webgpu/#limits)). Apple GPUs report 16384
  for WebGL `MAX_TEXTURE_SIZE` in Safari/Chrome (community reports; verify on the M4 Pro with
  [webglreport.com](https://webglreport.com)).
- A 5 cm/px orthophoto of a 1 km² site is 20000² px — never a single texture; always tiled.

## 7. Proposed shader pass order (concrete)

```
[CPU / build step, once per project]
  dsm.tif ─► COG + overviews (float)          ─► /tiles/dsm/{z}/{x}/{y}.tif or .bin
  dsm.tif ─► rvt: SVF (16 dir, 10 m), horizon[N=32 az] ─► 8-bit tiles
  ortho   ─► --cog / --tiles                   ─► /tiles/ortho/{z}/{x}/{y}.png
  EXIF    ─► capture sun az/el (JSON)
[GPU per frame, per visible tile]
  P0 normals + slope  (from DSM tile, R32F)
  P1 V_new            (horizon lookup + soft penumbra, or ray-march on coarse DSM)
  P2 delight          (optional; divide by S_cap using V_cap from the same horizon maps)
  P3 shade + haze     (sun/sky colours from Preetham/Nishita for chosen time)
  P4 grade + tonemap  (AgX-like curve, temperature, vignette, grain)
```

Everything in P0–P4 is a handful of texture reads per pixel; at 4K on an M4 Pro this is
well under a millisecond per pass. The only expensive thing is P1 if done by brute-force
marching — that is why the horizon-map precompute is recommended.

## 8. Validation ideas

- Render the capture-time sun (`V_cap`) and overlay on the orthophoto: shadow edges must
  coincide with the baked shadows. Any offset reveals DSM/ortho misregistration or a wrong
  time zone in EXIF (a 1 h error moves the sun ~15° in azimuth).
- Compare `gdaldem hillshade -az -alt` against the shader's `max(0, N·l)` on the same DSM
  — should match to 8-bit precision.
- Compare `rvt.vis.shadow_horizon` against the GPU `V_new` for the same az/el.

## Sources

- gdaldem: https://gdal.org/en/stable/programs/gdaldem.html ; gdal raster hillshade (3.11): https://gdal.org/en/stable/programs/gdal_raster_hillshade.html ; -igor PR: https://github.com/OSGeo/gdal/pull/1339 ; USGS OF 92-422 (multidirectional): http://pubs.usgs.gov/of/1992/of92-422/of92-422.pdf ; Simmon: https://medium.com/@robsimmon/a-gentle-introduction-to-gdal-part-5-shaded-relief-ec29601db654
- rvt-py: https://github.com/EarthObservation/RVT_py ; https://rvt-py.readthedocs.io/en/latest/rvt.vis.html ; https://rvt-py.readthedocs.io/en/latest/listofvis_svf.html
- GRASS: https://grass.osgeo.org/grass-stable/manuals/r.sun.html ; https://grass.osgeo.org/grass-stable/manuals/r.horizon.html ; https://grass.osgeo.org/grass78/manuals/addons/r.skyview.html
- SAGA ta_lighting: https://saga-gis.sourceforge.io/saga_tool_doc/9.5.2/ta_lighting_5.html ; https://saga-gis.sourceforge.io/saga_tool_doc/7.0.0/ta_lighting.html
- Timonen & Westerholm 2010: http://wili.cc/research/hfshadow/hfshadow.pdf ; max-mipmap soft terrain shadows: https://arxiv.org/abs/2005.06671 ; iq soft shadows: https://iquilezles.org/articles/rmshadows/ ; vterrain lighting: http://vterrain.org/Performance/lighting.html
- geotiff.js: https://geotiffjs.github.io/geotiff.js/ ; georaster-layer-for-leaflet: https://github.com/GeoTIFF/georaster-layer-for-leaflet
- rio-cogeo / rio-tiler / TiTiler: https://github.com/cogeotiff/rio-cogeo ; https://github.com/cogeotiff/rio-tiler ; https://developmentseed.org/titiler/
- WebGPU limits: https://www.w3.org/TR/webgpu/#limits ; WebGL cross-platform limits: https://webgl2fundamentals.org/webgl/lessons/webgl-cross-platform-issues.html
- ODM docs, options: https://docs.opendronemap.org/arguments/ ; ODM source `opendm/dem/commands.py`, `opendm/config.py`.
