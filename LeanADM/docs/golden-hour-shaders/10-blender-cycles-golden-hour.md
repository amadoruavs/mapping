# 10 — Offline hero renders and video: Blender 4.x, Nishita sky, Sun Position, Cycles/EEVEE on Apple Silicon

Status: research note, 2026-09-13. Script outline only — nothing written into `contrib/`.

## 1. What already exists in ODM `contrib/blender/`

Files: `common.py`, `odm_photo.py`, `odm_video.py`, `photo_360.blend`, `photo_vr.blend`, `README.md`.

- `common.loadMesh()` imports `odm_texturing/odm_textured_model_geo.obj` with
  `bpy.ops.import_scene.obj(axis_forward='Y', axis_up='Z')`, calls
  `bpy.utils.register_module('materials_utils')` and `bpy.ops.xps_tools.convert_to_cycles_all()`,
  re-centres the model on Z, then **replaces every material's Diffuse BSDF with an Emission
  shader fed by the diffuse texture** — i.e. it deliberately renders the ODM texture unlit,
  which is the right call for a texture with baked lighting and exactly what we must *undo*
  for a golden-hour render.
- `odm_photo.py` renders a still (equirectangular 360 or stereo VR) and writes XMP pano tags
  with ExifTool; `odm_video.py` builds a NURBS `CameraPath` from an `.xyz` waypoint file,
  adds a `FOLLOW_PATH` constraint, animates it over N frames and renders FFMPEG MP4 at
  4096×2048 (README notes Python 2.7 and the Google spatial-media injector).
- **These scripts target Blender 2.7x**: `bpy.ops.import_scene.obj` (removed in 4.0 —
  use `bpy.ops.wm.obj_import`), `scene.objects.link` (2.8+: `collection.objects.link`),
  `bpy.utils.register_module` (gone), `bpy.context.copy()` operator overrides (4.x uses
  `context.temp_override`). A golden-hour script must be rewritten, keeping the same
  CLI contract (`blender -b file.blend --python script.py -- <project-path> ...`).

## 2. Blender 4.x building blocks

### 2.1 Sky Texture node (Nishita)

Manual (4.2): [Sky Texture Node](https://docs.blender.org/manual/en/4.2/render/shader_nodes/textures/sky.html).
Sky types: Preetham (1999), Hosek/Wilkie (2012), **Nishita** ("improved version of the 1993
model by Nishita et al."). Nishita inputs:

| Property | Meaning / range (manual) |
|---|---|
| Sun Disc (Cycles only) | enable sun disc lighting |
| Sun Size | angular diameter in degrees (real sun ≈ 0.53°) |
| Sun Intensity | multiplier for sun disc |
| Sun Elevation | degrees above horizon |
| Sun Rotation | degrees around zenith |
| Altitude | camera height above sea level, ≤ 60 km ("the model only accounts for the first two layers of the atmosphere") — set it to the site's elevation, *not* the drone altitude, because the sky is evaluated for the camera |
| Air | 0 none · 1 clear day · 2 highly polluted |
| Dust | 0 none · 1 clear · 5 city · 10 hazy ("darkens the lower part of the atmosphere") |
| Ozone | 0 none · 1 clear · 2 city ("makes the sky appear bluer") |

Golden-hour presets that read well: elevation 3–8°, dust 2–4 (warm haze), ozone 1–1.5,
air 1. EEVEE renders the Nishita sky colour but **the sun disc is Cycles-only**; in EEVEE add
a Sun light object (EEVEE Next in 4.2 also "automatically extract[s] sun light contribution
from HDRI lighting" for world textures —
[4.2 release notes, EEVEE](https://developer.blender.org/docs/release_notes/4.2/eevee/)).

### 2.2 Sun Position add-on (extension)

- [Sun Position on extensions.blender.org](https://extensions.blender.org/add-ons/sun-position/),
  by Damien Picard (pioverfour), v4.5.0, `blender_version_min = "4.2.0"`, licence
  **GPL-3.0-or-later** (from `blender_manifest.toml`); bundled with Blender ≤ 4.1, an
  extension since 4.2 ([4.1 manual](https://docs.blender.org/manual/en/4.1/addons/lighting/sun_position.html)).
- "It uses physical characteristics to position the Sun in the scene: geographic location,
  time and date. It is based on the NOAA online calculator, which relies on calculations
  from Jean Meeus' *Astronomical Algorithms*." Two modes: *Sun Object Mode* (drives a Sun
  light, the Sky Texture and a visualisation collection from lat/lon/date/time/UTC/north
  offset) and *Environment Mode* (locks an HDRI to a sun light).
- Same inputs ODM has: `reference_lla.json` (lat/lon/alt) and EXIF times. Its Python
  properties live on `scene.sun_pos_properties` (`latitude, longitude, month, day, year,
  time, UTC_zone, north_offset, sun_object, sky_texture, use_daylight_savings`) — property
  names from the add-on source; **verify against v4.5.0** before scripting. Alternative
  without the add-on: compute az/el with pysolar (ODM's `opendm/dls.py` already does) and set
  the Sun light rotation directly: `rot = (radians(90 − el), 0, radians(−az))` in Blender's
  Z-up frame, plus `sky.sun_elevation/sun_rotation`.

### 2.3 Cycles vs EEVEE Next on Apple Silicon

- Cycles Metal backend: "Metal is supported on Apple computers with Apple Silicon, AMD and
  Intel graphics cards. macOS 13.0 or newer… GPU accelerated denoising is available on Apple
  Silicon" ([4.2 manual, GPU rendering](https://docs.blender.org/manual/en/4.2/render/cycles/gpu_rendering.html));
  Blender 4.3+ drops Metal for AMD/Intel — Apple Silicon only
  ([devtalk announcement](https://devtalk.blender.org/t/cycles-remove-support-of-metal-with-amd-intel-gpus-for-4-3-and-onwards/35098)).
  Enable via Preferences ▸ System ▸ Cycles Render Devices ▸ Metal, then Render ▸ Device = GPU
  (scriptable: `bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'`).
- **MetalRT** (hardware ray tracing on M3/M4-class GPUs) is exposed as a Cycles device
  preference (`use_metalrt`) — **not verified on the 4.2 manual page fetched**; check
  Preferences ▸ System on the M4 Pro; the general expectation is a speed-up on
  high-poly scenes, but Apple GPUs "will likely not perform similarly to high-end RTX GPUs"
  ([devtalk Metal feedback](https://devtalk.blender.org/t/cycles-apple-metal-device-feedback/21868)).
- EEVEE Next (4.2): "Shadows are now rendered using Virtual Shadow Maps… Light visibility is
  now computed using Shadow Map Ray Tracing, providing plausible soft shadows"; sun light
  extracted from HDRI; ray-traced GI option. Multiple bug reports of EEVEE Next stalling on
  Apple Silicon in 2024 ([#124137](https://projects.blender.org/blender/blender/issues/124137),
  [#127033](https://projects.blender.org/blender/blender/issues/127033)); re-test on a
  current 4.x/5.x build before relying on it for batch video.
- Recommendation: **Cycles + Metal for hero stills** (sun disc, correct sky radiance, path
  traced soft shadows from a 0.53° sun); **EEVEE Next for video previews and long camera
  paths**, then Cycles at 64–256 samples with OIDN GPU denoise for finals.

### 2.4 Colour management

Blender 4.0 made **AgX** the default view transform ("replaces Filmic… bright colors go
towards white, similar to real cameras" — [4.0 release notes](https://developer.blender.org/docs/release_notes/4.0/color_management/)).
AgX handles the saturated orange sun and sky far better than Filmic; use
`scene.view_settings.view_transform = 'AgX'`, look `'AgX - Medium High Contrast'`, exposure
+0.5…+1.0 EV at golden hour, and keep the ODM JPEG textures tagged **sRGB** (image
`colorspace_settings.name = 'sRGB'`) so they are linearised before shading.

## 3. Importing ODM output with materials

- OBJ: `bpy.ops.wm.obj_import(filepath=..., forward_axis='Y', up_axis='Z')` — the 4.x
  importer reads the MTL and creates Principled BSDF materials with the JPEG atlases wired
  to Base Color. ODM's OBJ is in the local georeferenced frame (metres, offset stored in
  `odm_georeferencing/odm_georeferencing_model_geo.txt` / `coords.txt`); the scene stays
  near the origin, which is what Blender wants.
- GLB: `bpy.ops.import_scene.gltf(filepath=...)` — imports `KHR_materials_unlit` as an
  emission-style "unlit" node group ([glTF-Blender-IO](https://github.com/KhronosGroup/glTF-Blender-IO));
  you must rebuild the material as Principled with the texture in Base Color to receive
  light. Draco-compressed GLBs decode fine in the importer.
- **De-lit albedo**: if doc 07 produced an albedo atlas (same UV layout, same file names),
  simply point each Image Texture node at the albedo JPEG/PNG; if only an albedo orthophoto
  exists, add a second UV projection (`UV Project` modifier from a top-down camera) or use
  Generated/Object coordinates mapped to the ortho extent. Optional AO map → multiply into
  Base Color or feed a mix with the sky term.
- Point cloud alternative: Blender 4.x imports PLY/CSV via Geometry Nodes; LAZ needs a
  converter (PDAL → PLY). Points render as instanced spheres — fine for stills, heavy for
  video.

## 4. Camera paths (extending the `odm_video.py` pattern)

Keep the `.xyz` waypoint contract and the FOLLOW_PATH approach, but use 4.x API:

```python
curve = bpy.data.curves.new('CameraPath', 'CURVE'); curve.dimensions = '3D'
spl = curve.splines.new('NURBS'); spl.points.add(len(wps) - 1)
for p, (x, y, z) in zip(spl.points, wps): p.co = (x, y, z, 1.0)
spl.use_endpoint_u = True
path = bpy.data.objects.new('CameraPath', curve); scene.collection.objects.link(path)
con = cam.constraints.new('FOLLOW_PATH'); con.target = path; con.use_curve_follow = True
with bpy.context.temp_override(object=cam, constraint=con):
    bpy.ops.constraint.followpath_path_animate(constraint=con.name, frame_start=1, length=n_frames)
```

Nice golden-hour moves: low orbit (`TRACK_TO` an empty at the site centre) while the sun
elevation is keyframed from 12° down to 2° over the clip, so shadows visibly lengthen.

## 5. Documented script outline (not written into `contrib/`)

`odm_golden_hour.py` — `blender -b --python odm_golden_hour.py -- <project> [--time 17:45] [--date 2026-06-21] [--engine CYCLES|BLENDER_EEVEE_NEXT] [--albedo <dir>] [--waypoints path.xyz --frames 240]`

1. **Args**: parse after `--` (same convention as `odm_photo.py`).
2. **Site**: read `odm_georeferencing/coords.txt` for the UTM offset and
   `odm_report/…` / `reference_lla.json` (used by `opendm/ogctiles.py`) for lat/lon/alt.
3. **Capture time** (default): median of EXIF `DateTimeOriginal` via `exiftool -j` or
   `PIL` over `images/`; convert to UTC with the GPS-derived time zone.
4. **Scene reset**: `bpy.ops.wm.read_factory_settings(use_empty=True)`; engine; Metal
   device; `view_transform='AgX'`; resolution 3840×2160 (stills 7680×4320).
5. **Import**: `wm.obj_import` (or `import_scene.gltf`); join meshes; set
   `image.colorspace_settings.name='sRGB'`; if `--albedo`, swap image paths; convert any
   emission/unlit materials to Principled (roughness 0.9, specular 0.2 for a matte
   photogrammetry look).
6. **World**: Sky Texture (Nishita) → Background (strength 1.0); `altitude = site alt`,
   `dust`, `ozone` from args; plus a Sun light (`energy ≈ 3–5 W/m²` at golden hour vs 100 at
   noon-ish for physically plausible ratios if the sky strength is 1; **tune** — Blender's sun
   strength is in W/m² and the Nishita sun disc can replace the light in Cycles).
7. **Sun position**: try `scene.sun_pos_properties` (add-on enabled via
   `bpy.ops.preferences.addon_enable(module='bl_ext.blender_org.sun_position')` — module
   path per 4.2 extension naming, **verify**), else compute with pysolar and set rotation
   + `sky.sun_elevation/sun_rotation` directly. Keyframe elevation for time-lapse.
8. **Ground/horizon**: add a large plane at the DTM's median height with the orthophoto
   (or the albedo orthophoto) so shadows and haze extend beyond the mesh edge; optionally
   displace it by the DSM outside the mesh footprint.
9. **Camera**: from `--waypoints` (FOLLOW_PATH) or a default orbit; focal 35 mm; DoF off.
10. **Render settings**: Cycles 128 samples, adaptive, OIDN denoise (GPU), light tree on,
    `film_transparent=False`; EEVEE: shadows VSM resolution high, ray tracing on.
11. **Output**: stills PNG 16-bit (or EXR for grading), video FFMPEG H.264 / ProRes;
    write to `<project>/odm_golden_hour/`.
12. **Sidecar**: write a JSON with sun az/el, time, site lat/lon, sky params — the same
    numbers the web viewer (doc 09) needs, so both paths match.

Batch matrix for a "golden hour reel": elevations {10, 6, 3, 0.5}° × azimuth of the real
date, dust {1, 3}; 8 stills per site.

## 6. Performance expectations (M4 Pro, estimates — not benchmarked)

- Cycles Metal, 4K, 128 samples + OIDN, 2–5 M triangles, 8 × 8192² JPEG textures
  (~1.5 GB texture memory): a few minutes per still; ~10–30 s per 1080p video frame at 64
  samples. Texture memory is the constraint — resize atlases to 4096² for video.
- EEVEE Next: seconds per 4K frame once shaders compile; VSM at grazing sun needs
  `scene.eevee.shadow_ray_count` / `shadow_step_count` raised to avoid noise in long shadows.

## 7. Open questions

- Exact `sun_pos_properties` field names and the extension module id in Blender 4.2+/5.x.
- Whether MetalRT is on by default for M4 in the target Blender build.
- Whether to keep the existing `photo_360.blend` (2.7x-era file) or generate scenes from
  `read_factory_settings` — the latter avoids legacy node names (`'Color Mult'`,
  `'Diffuse Texture'`) that `common.py` depends on.

## Sources

- ODM `contrib/blender/{common.py,odm_photo.py,odm_video.py,README.md}` (read-only).
- Blender Sky Texture (4.2): https://docs.blender.org/manual/en/4.2/render/shader_nodes/textures/sky.html
- Sun Position extension: https://extensions.blender.org/add-ons/sun-position/ ; source/manifest https://projects.blender.org/pioverfour/sun_position ; 4.1 manual https://docs.blender.org/manual/en/4.1/addons/lighting/sun_position.html
- Cycles GPU rendering (Metal): https://docs.blender.org/manual/en/4.2/render/cycles/gpu_rendering.html ; Metal AMD/Intel removal https://devtalk.blender.org/t/cycles-remove-support-of-metal-with-amd-intel-gpus-for-4-3-and-onwards/35098 ; Metal device feedback https://devtalk.blender.org/t/cycles-apple-metal-device-feedback/21868 ; Metal viewport https://code.blender.org/2023/01/introducing-the-blender-metal-viewport/
- EEVEE Next 4.2 notes: https://developer.blender.org/docs/release_notes/4.2/eevee/ ; Apple Silicon issues https://projects.blender.org/blender/blender/issues/124137 , https://projects.blender.org/blender/blender/issues/127033
- AgX (4.0): https://developer.blender.org/docs/release_notes/4.0/color_management/ ; EaryChow/AgX https://github.com/EaryChow/AgX
- Import operators: https://docs.blender.org/api/current/bpy.ops.import_scene.html ; glTF-Blender-IO https://github.com/KhronosGroup/glTF-Blender-IO
- Baking: https://docs.blender.org/manual/en/latest/render/cycles/baking.html
- pysolar: https://github.com/pingswept/pysolar (GPL-3.0)
