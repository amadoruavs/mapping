# 07 — De-lighting photogrammetry textures (the core problem)

Status: research note, 2026-09-13. No code changes. Facts about ODM were read from the
ODM 3.6.2 tree under `/Users/avneh/Code/ADM/ODM` (read-only); everything else is cited.

## 1. Why a "golden hour shader" cannot just add a warm light

ODM's texturing stage (`stages/mvstex.py`) calls mvs-texturing / `texrecon` with
`-t none` (no tone mapping), the default data term `gmi`, `gauss_clamping` outlier removal,
and leaves both **global** and **local seam leveling** on (the CLI only exposes
`--texturing-skip-global-seam-leveling`; the local-skip flag is listed under `DEPRECATED`
in `opendm/config.py`). The result is that `odm_textured_model_geo.obj` (and the GLB written
by `opendm/gltf.py`) carries a texture in which:

- direct sunlight × cos(θ) shading is baked in (bright south-facing roofs, dark north walls),
- cast shadows from buildings, trees and terrain are baked in as dark pixels,
- ambient occlusion in courtyards / under canopies is baked in,
- the sky's blue tint in shadow and the sun's warm tint in lit areas are baked in,
- per-image exposure differences have been *blended*, not removed (seam leveling
  equalises colour between views; it does not know what the albedo is).

`opendm/gltf.py` additionally writes the material with `KHR_materials_unlit` in
`extensionsRequired`, i.e. the GLB explicitly tells a viewer *"do not light this"*. Any
relighting viewer must first replace that material (see doc 09).

Adding a low, warm directional light on top of this texture therefore produces double
shading: the new sun lights faces that are already dark from the old sun, old shadows stay
where they were (pointing the wrong way for a new sun), and the warm grade multiplies an
already-warm/blue-tinted texture. The only ways out are:

1. **De-light** the texture to an approximate albedo, then relight with a physically based
   sky + sun (best quality, most work), or
2. **Fake it in 2.5D** on the orthophoto/DSM with hillshade + ray-marched shadows, accepting
   residual baked shadows (doc 08), or
3. **Look-only grade** (colour temperature, gamma, haze, vignette) that never claims to move
   the sun — cheap, honest, often "good enough" for a demo.

This document is about option 1 and what can realistically run on an M4 Pro.

## 2. Terminology and the image formation model

For an opaque, mostly Lambertian outdoor surface at capture time:

```
I(x) = A(x) · [ E_sun · V(x) · max(0, n(x)·l_sun) + E_sky(x) · SVF(x) ] + I_indirect(x)
```

- `A` albedo (what we want), `n` surface normal (we have it: mesh or DSM),
- `l_sun` unit vector to the sun at capture time (we can compute it: EXIF `DateTimeOriginal`
  + GPS → pysolar, which ODM already bundles: `opendm/dls.py` uses
  `pysolar.get_altitude/get_azimuth`; `pixi.toml` pins `pysolar==0.13`),
- `V` sun visibility (0 in cast shadow, 1 lit, fractional in penumbra) — computable by
  ray-marching the DSM/mesh toward `l_sun`,
- `SVF` sky-view factor (fraction of the sky hemisphere visible) — computable from the DSM
  (rvt-py, SAGA, GRASS `r.skyview`),
- `E_sun`, `E_sky` are unknown scalars/colours per capture (or per image); the ratio
  `E_sun/E_sky` is ~5–10 for a clear midday sky, lower at golden hour.

Intrinsic image decomposition (IID) is the general name for solving `I = A·S`
(reflectance × shading). "De-lighting" in the photogrammetry industry means the same thing
applied to a mesh texture, usually with the geometry known, which makes the problem far
better posed than single-image IID.

Song & Qin state the aerial case precisely: textures "directly embed spatially and temporally
variant environmental lighting information, such as sun illumination and direction"
([arXiv 2204.04142](https://arxiv.org/abs/2204.04142), ISPRS Congress 2022) and, in the
follow-up, that "both sun illumination and scene geometry are estimable in aerial
photogrammetry" and can be used as direct model inputs
([arXiv 2409.03032](https://arxiv.org/abs/2409.03032), ISPRS J. Photogramm. 218, 2024).
That is exactly the ODM situation.

## 3. Survey of existing tools

### 3.1 Agisoft Metashape — "Remove Lighting" (built-in) and "De-Lighter" (standalone)

- **Remove Lighting** (Tools > Mesh > Remove Lighting, Metashape Pro): heuristic that
  "smooth[s] out dark and light sides of the object model", optionally using an ambient
  occlusion map to recover areas that are too dark. Modes: *Single color* (marble, rock) vs
  *Multi color*; presets *Default* / *Aggressive* (tuned for rocks, can cause artifacts).
  Agisoft's own doc says removing shadows while preserving texture features "is impossible"
  in general and that the tool "does not support delighting in strong shadows".
  Source: [Agisoft helpdesk article 31000158885](https://agisoft.freshdesk.com/support/solutions/articles/31000158885-remove-lighting-from-texture-in-agisoft-metashape-professional).
- **Agisoft Texture De-Lighter** (free standalone): "targeted at removing cast shadows and
  ambient occlusion from 3D models"; requires the user to paint rough *lit* / *shadow* brush
  strokes ("one stroke per material"), builds an illumination map by binary classification,
  works on 8-bit JPEG textures, has a 4× downscaled *Preview* mode, region exclusion, and a
  separate "Remove Shading and Ambient Occlusion" pass.
  Source: [Agisoft De-Lighter workflow](https://www.agisoft.com/index.php?id=71).
- Relevance to ODM: proprietary, GUI-driven, per-model manual strokes. Not automatable for
  an ODM pipeline, but the *design* (user-marked lit/shadow seeds → illumination map →
  divide) is the right mental model for a cheap semi-automatic tool.

### 3.2 Unity De-Lighting Tool (open source, MIT)

- Repo: [Unity-Technologies/DeLightingTool](https://github.com/Unity-Technologies/DeLightingTool)
  — MIT licence (GitHub API), last push 2020-05 (effectively unmaintained, Unity 2017.1+).
- Inputs: base colour, **ambient-occlusion map, object-space normal map, bent-normal map**
  (plus position map) baked in Knald/xNormal/Substance/Blender. The tool fits a low-order
  lighting environment consistent with AO/bent normals and divides it out. Sketchfab's
  walkthrough stresses overcast capture and that deep cast shadows are "an inherent
  limitation" ([Sketchfab blog](https://sketchfab.com/blogs/community/unity-photogrammetry-de-lighting-workflow/),
  [80.lv](https://80.lv/articles/photogrammetry-de-lighting-tool-for-unity)).
- Relevance: the *algorithmic recipe* — bake AO + bent normals, fit an SH lighting
  environment, divide — is directly portable to Python/NumPy or a compute shader. It does not
  model a directional sun with hard shadows, so it is a complement to, not a replacement
  for, the sun/shadow term in §5.

### 3.3 Adobe Substance 3D Sampler — "Delight (AI Powered)"

- Part of the *Image to Material* filter; "designed to remove lighting and shadow
  information from images" and outputs albedo/roughness/normal/height.
  Sources: [Adobe doc](https://helpx.adobe.com/substance-3d-sampler/filters/tools/delight-ai-powered.html),
  [CG Channel on Sampler 4.0](https://www.cgchannel.com/2023/02/adobe-releases-substance-3d-sampler-4-0/).
- Image-space, meant for flat material captures; proprietary; no API worth automating for
  gigapixel aerial atlases. Listed for completeness.

### 3.4 RealityScan / RealityCapture, Polycam, DJI Terra

- No first-party "de-lighting" feature was found for RealityScan 2.x, Polycam or DJI Terra in
  the search performed (RealityScan 2.0/2.1 release coverage mentions AI masking,
  alignment, UV/rendering improvements only). **Uncertain** — treat as "not a documented
  feature" rather than "confirmed absent".

### 3.5 Academic, aerial-specific (closest to ODM)

- **Song & Qin 2022** ("A novel intrinsic image decomposition method to recover albedo for
  aerial images in photogrammetry processing", ISPRS Annals V-2-2022; arXiv 2204.04142):
  inverse image-formation model using photogrammetric geometry (DSM/mesh) as initial
  geometry; demonstrated better feature/dense matching after albedo recovery.
- **Song & Qin 2024** ("A general albedo recovery approach for aerial photogrammetric images
  through inverse rendering", ISPRS J. 218; arXiv 2409.03032): physics-based inverse
  rendering with known sun position and geometry; "does not require additional input other
  than data acquired through typical drone-based photogrammetric collection"; recovered
  albedo "can facilitate model relighting and shading". **Code availability not stated on
  the abstract pages** — check the authors' group site before planning around it.
- These two papers are the strongest evidence that the cheap heuristic in §5 is viable:
  it is essentially their forward model with a simpler solver.

### 3.6 General IID / albedo networks (ground-level training, but usable)

| Model | What it outputs | Licence | Notes |
|---|---|---|---|
| [compphoto/Intrinsic](https://github.com/compphoto/Intrinsic) (Careaga & Aksoy, TOG 2023 "Ordinal Shading", TOG 2024 "Colorful Diffuse IID") | albedo, colourful diffuse shading, specular residual | **academic use only**; commercial via SFU licensing (README) | Very good on in-the-wild photos; CNN, exportable to ONNX in principle but licence blocks product use. |
| [Marigold IID](https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1) (prs-eth; diffusers pipeline) | albedo + material (Appearance) or albedo + diffuse shading + non-diffuse residual (Lighting) | code Apache-2.0; **weights CreativeML Open RAIL++-M** | SD2-based latent diffusion; heavy, slow at atlas scale, not practical to export to ONNX for ODM. |
| [LibreYOLO/LibreMarigoldV2b-albedo](https://huggingface.co/LibreYOLO/LibreMarigoldV2b-albedo) | linear-RGB albedo | **Apache-2.0** weights | Built on Qwen-Image-Edit-2509; model card says export/MPS not supported — GPU server only. |
| [RSrscoder/AISD](https://github.com/RSrscoder/AISD) + DSSDNet (Luo et al., ISPRS J. 2020) | aerial **shadow masks** | no licence file; dataset on Baidu | 514 aerial images; the reference aerial shadow-detection benchmark. |
| [S-EO dataset](https://centreborelli.github.io/shadow-eo/) (CVPRW 2025) | geometry-aware shadow masks, 702 georeferenced USGS tiles | public-domain source imagery; trained U-Net detector | Best fit for orthophoto shadow masks; check repo licence before use. |
| [AeroVILab-AHU/SARU](https://github.com/AeroVILab-AHU/SARU) | shadow detection + removal for remote sensing | check repo | Pretrained weights on Google Drive. |
| [AeroDeshadow](https://arxiv.org/abs/2604.15903) (2026) | physics-guided shadow synthesis + penumbra-aware removal for aerial imagery | "will be made publicly available" | Directly targets the umbra/penumbra structure of drone shadows. |
| [ShadowFormer](https://github.com/GuoLanqing/ShadowFormer) (AAAI 2023) | shadow-free image given image + mask | **MIT** | Trained on ISTD/SRD (ground-level); needs fine-tuning for nadir aerial. |
| [ShadowDiffusion](https://github.com/GuoLanqing/ShadowDiffusion) (CVPR 2023) | same | **no licence file** | Avoid for product use. |
| [DC-ShadowNet](https://github.com/jinyeying/DC-ShadowNet-Hard-and-Soft-Shadow-Removal) (ICCV 2021) | unsupervised hard/soft shadow removal | **no licence file** | Unpaired training — attractive for drone data, but licence unclear. |

Survey/benchmark to keep bookmarked: *Unveiling Deep Shadows*
([arXiv 2409.02108](https://arxiv.org/abs/2409.02108), repo
[xw-hu/Unveiling-Deep-Shadows](https://github.com/xw-hu/Unveiling-Deep-Shadows)) and the
curated list [GuoLanqing/Awesome-Shadow-Removal](https://github.com/GuoLanqing/Awesome-Shadow-Removal).

### 3.7 Neural relighting (NeRF / 3DGS) — what it teaches, why not for v1

- **GaRe** (ICCV 2025, [arXiv 2507.20512](https://arxiv.org/abs/2507.20512)): relightable 3DGS
  for outdoor scenes; "residual-based sun visibility extraction", region-based supervision,
  and "ray-tracing-based technique for realistic shadow simulation" — separates sunlight, sky
  radiance and indirect light per Gaussian.
- **LumiGauss** (WACV 2025, [arXiv 2408.04474](https://arxiv.org/abs/2408.04474), code
  [joaxkal/lumigauss](https://github.com/joaxkal/lumigauss)): 2DGS + SH environment lighting,
  explicit shadow handling in outdoor scenes.
- **The Sky's the Limit** (ECCV 2024, [arXiv 2311.16937](https://arxiv.org/abs/2311.16937)):
  sky pixels as direct illumination observations + neural sky-visibility.
- **SOL-NeRF** (SIGGRAPH Asia 2023, [project](http://geometrylearning.com/SOL-NeRF/)):
  sun as a single spherical Gaussian, sky as low-order SH — a good parametric lighting model
  to copy for the heuristic path.
- **ROSGS** ([arXiv 2509.11275](https://arxiv.org/abs/2509.11275)), **SRUG**
  ([arXiv 2605.24700](https://arxiv.org/abs/2605.24700)): 2025–26 follow-ups.
- **OutCast** (Eurographics 2022, [arXiv 2204.09341](https://arxiv.org/abs/2204.09341)):
  single-image outdoor relighting with cast shadows via learned image-space ray marching.

Why not for v1: these are research training pipelines (hours per scene on CUDA), they
replace ODM's mesh with a radiance field, and their outputs do not drop into a GLB/3D Tiles
viewer. Their *decomposition targets* (sun visibility, sky SVF, indirect) are however exactly
the layers the heuristic below computes analytically.

## 4. The cheap heuristic: geometric de-lighting with the known sun

Everything needed is already produced by ODM or trivially derived:

| Quantity | Where it comes from |
|---|---|
| Sun direction at capture | EXIF `DateTimeOriginal` (+ `OffsetTime` or GPS-derived UTC) and GPS lat/lon of each photo → `pysolar.get_altitude/get_azimuth` (already vendored in `opendm/dls.py`). Take the median over the flight; a 20-min survey moves the sun ~5°. |
| Normals | mesh normals (ODM's GLB writes `NORMAL`), or DSM gradient (Horn/Zevenbergen-Thorne, same kernels `gdaldem` uses). |
| Sun visibility `V` | ray-march the DSM (`dsm.tif`, default 5 cm/px, nodata −9999 per `opendm/dem/commands.py`) toward the sun; or `rvt.vis.shadow_horizon(dem, resolution, shadow_az, shadow_el)`; or GRASS `r.sun` mode 1 `incidout` (NULL = shadowed); or SAGA `ta_lighting`. |
| Sky-view factor | `rvt.vis.sky_view_factor` (svf/asvf/openness), GRASS `r.skyview`, SAGA "Sky View Factor". |
| Ambient occlusion on the 3D mesh | Blender bake (Cycles, AO pass; scriptable headless) or a GPU raster AO bake via xatlas UVs; see §6. |

### 4.1 Inverse-Lambert step

Per texel (or per orthophoto pixel), with `k = E_sun/E_sky` unknown:

```
S(x) = V(x) · max(0, n·l) · k + SVF(x)          # shading, up to a global scale
A(x) ≈ I(x) / S(x)
```

- Solve for `k` (and a per-channel sun/sky colour) by least squares on pixels that are
  clearly lit (`V=1`, `n·l>0.7`) vs clearly shadowed (`V=0`) on the *same material class*
  (roads, roofs) — this is the automatic version of Agisoft's brush strokes.
- Clamp `S` from below (e.g. 0.15) to avoid blowing up noise in deep shadow; blur the
  boundary of `V` by the sun's angular diameter (0.53°) plus DSM uncertainty (a 2–3 px
  feather at 5 cm/px is typical).
- Work in **linear** light: ODM's texture is sRGB JPEG; decode, divide, re-encode.
- Expect failure on: vegetation (non-Lambertian, self-shadowing at leaf scale), water,
  glass/metal roofs (specular), and anywhere the DSM is wrong (tree canopies look like
  blobs, thin poles vanish). This is why Agisoft says the general case "is impossible".

### 4.2 Shadow mask by DSM ray-marching

```
for each pixel p:  h0 = dsm[p]
  step along d = (cos(az), sin(az)) in pixel units, t = 1..T:
     z_ray = h0 + t·px_size·tan(elev)
     if dsm[p + t·d] > z_ray + eps: V[p] = 0; break
```

- Use a max-mipmap (quadtree of maxima) to skip empty space; this is the "maximum mipmaps"
  idea of [arXiv 2005.06671](https://arxiv.org/abs/2005.06671) (soft terrain shadows with
  dynamic programming) and the horizon approach of Timonen & Westerholm, *Scalable Height
  Field Self-Shadowing* (CGF 29(2), 2010, [PDF](http://wili.cc/research/hfshadow/hfshadow.pdf)),
  which computes exact horizons for N azimuths in O(N·pixels) and unifies hard/soft shadows.
- Soft shadows: sample the sun disc (0.53° ⇒ elevation ± 0.265°) with 3–5 rays, or convert
  the horizon angle into a penumbra fraction analytically; Iñigo Quílez's soft-shadow
  penumbra `k·h/t` formulation transfers directly
  ([iquilezles.org/articles/rmshadows](https://iquilezles.org/articles/rmshadows/)).
- Runs trivially on the GPU (WebGL2/WebGPU fragment or compute shader) — see doc 08 for a
  shader pass order; on CPU with NumPy/Numba for a 20k×20k DSM budget ~1–2 min.

### 4.3 Where the mask is wrong, fall back to learned shadow detection

Combine the geometric mask with a learned mask (S-EO U-Net, DSSDNet/AISD, SARU) via
logical OR on confidence — the geometric mask is precise where the DSM is good (buildings),
the network is better on trees. Both can be exported to ONNX and run through onnxruntime
exactly like `opendm/skyremoval/skyfilter.py` does (`ort.InferenceSession(model,
providers=[CUDAExecutionProvider|CPUExecutionProvider])`, 384×384 tiles). On the M4 Pro use
`CoreMLExecutionProvider` (onnxruntime 1.22.2 is pinned in `pixi.toml`) — **untested claim**;
CPU EP is the safe default.

## 5. Recommended pipeline for ODM outputs

```
 EXIF+GPS ──pysolar──► sun az/el (median)
 dsm.tif  ──rvt/GPU──► V (sun visibility), SVF, normals
 odm_orthophoto.tif ─► I (linear)            ┐
 odm_textured_model ─► texture atlas (linear)┤  A = I / S,  S = V·max(0,n·l)·k + SVF
 [optional] AO bake (mesh) ──────────────────┘  (k, sun/sky colour fitted from lit/shadow pairs)
 [optional] learned shadow mask (ONNX) ──OR──► refine V in vegetation
 ► albedo_ortho.tif / albedo atlas.jpg  ► viewer relights with new sun (doc 09) or Blender (doc 10)
```

Stages and cost on an M4 Pro (rough, **estimates**):

1. Sun vector: milliseconds.
2. DSM derivatives + SVF (rvt-py, 16 directions, 10 m radius) on a 10k×10k DSM: ~1–3 min
   CPU; on GPU: seconds.
3. Shadow mask: GPU seconds; CPU ~1–2 min with max-mipmap.
4. Inverse-Lambert divide on the orthophoto: seconds (NumPy/rasterio, tiled).
5. The mesh texture is the awkward part: the atlas is not in map space. Two options:
   (a) project the orthophoto-space `S` onto the atlas by rendering the mesh with UV as
   position and world XY as colour (one GPU pass), then divide per texel; or
   (b) skip the atlas and build the viewer's material from the *albedo orthophoto* projected
   top-down (works for 2.5D scenes, fails on facades).
6. Optional AO bake on the mesh (Blender headless, 64 samples): minutes to tens of minutes
   for a multi-million-triangle mesh.

Pros: fully automatic, physically motivated, uses only ODM outputs, GPU-friendly, degrades
gracefully (unlit shadows simply remain darker). Cons: wrong on vegetation/specular, needs
a clean DSM (`--dsm --dem-resolution 5` and `--pc-classify`/SMRF help), edge halos where the
DSM is offset from the orthophoto (both are produced from the same point cloud in ODM so
alignment is good, but the orthophoto is rendered from the 2.5D mesh unless `--use-3dmesh`).

## 6. Ambient occlusion baking

- AO is the shadow term that *does not* depend on the sun. Baking it once and re-using it in
  every relight is worth it: `albedo · (V·(n·l)·sun + SVF·sky)` looks flat without AO in
  alleys and under eaves.
- Blender Cycles AO bake is scriptable headless (`bpy.ops.object.bake(type='AO')`; the mesh
  needs a UV map — ODM's OBJ has one; example script pattern: [AndrewRayCode gist](https://gist.github.com/AndrewRayCode/760c4634a77551827de41ed67585064b);
  manual: [Render Baking](https://docs.blender.org/manual/en/latest/render/cycles/baking.html)).
- GPU alternative: rasterise the mesh into its UV atlas and shoot hemisphere rays against a
  BVH (DXR/Metal RT/compute); [Twinklebear/dxr-ao-bake](https://github.com/Twinklebear/dxr-ao-bake)
  shows the xatlas + inline-ray-tracing pattern.
- For 2.5D outputs, SVF *is* the AO (sky hemisphere visibility), so no bake is needed —
  `rvt.vis.sky_view_factor` returns it directly.

## 7. What runs where (GPU vs CPU)

| Step | CPU (Python) | GPU (web) | GPU (native) |
|---|---|---|---|
| Sun vector | pysolar | JS port (NOAA/Meeus) | — |
| Normals/slope | rasterio + NumPy | fragment shader from DSM texture | — |
| SVF / horizon | rvt-py (Apache-2.0), GRASS, SAGA | compute shader (Timonen) | Metal compute |
| Cast shadow | rvt `shadow_horizon`, r.sun | fragment/compute ray-march | Metal compute |
| Inverse Lambert | NumPy tiled | fragment shader | — |
| Learned shadow mask | onnxruntime CPU/CoreML | onnxruntime-web WebGPU EP | CoreML |
| AO bake | Blender Cycles (Metal) | — | Blender Cycles (Metal) |

## 8. Open questions

1. Is code available for Song & Qin 2024? If yes, it likely supersedes the heuristic.
2. Which learned aerial shadow detector has a usable licence *and* an ONNX export
   (S-EO U-Net is the best candidate; verify the repo licence)?
3. How to handle the mvs-texturing global seam leveling: for de-lighting, running ODM with
   `--texturing-skip-global-seam-leveling` may keep per-view radiometry closer to the
   physical model (ODM docs say the flag "is useful when processing radiometric data"), at
   the cost of visible seams — needs an A/B test.
4. Orthophoto vs atlas: do we need the atlas at all for the target viewer, or is a
   re-projected albedo orthophoto + DSM displacement enough (doc 08/09)?

## Sources

- ODM source (read-only): `stages/mvstex.py`, `opendm/gltf.py`, `opendm/dls.py`,
  `opendm/dem/commands.py`, `opendm/config.py`, `opendm/skyremoval/skyfilter.py`, `pixi.toml`.
- Agisoft Remove Lighting: https://agisoft.freshdesk.com/support/solutions/articles/31000158885-remove-lighting-from-texture-in-agisoft-metashape-professional
- Agisoft De-Lighter workflow: https://www.agisoft.com/index.php?id=71
- Unity DeLightingTool: https://github.com/Unity-Technologies/DeLightingTool ; Sketchfab walkthrough: https://sketchfab.com/blogs/community/unity-photogrammetry-de-lighting-workflow/
- Substance 3D Sampler Delight: https://helpx.adobe.com/substance-3d-sampler/filters/tools/delight-ai-powered.html
- Song & Qin 2022: https://arxiv.org/abs/2204.04142 ; 2024: https://arxiv.org/abs/2409.03032
- Careaga & Aksoy Intrinsic: https://github.com/compphoto/Intrinsic ; https://arxiv.org/abs/2409.13690
- Marigold IID: https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1 ; https://arxiv.org/abs/2505.09358
- LibreMarigold albedo: https://huggingface.co/LibreYOLO/LibreMarigoldV2b-albedo
- AISD: https://github.com/RSrscoder/AISD ; S-EO: https://centreborelli.github.io/shadow-eo/ ; SARU: https://github.com/AeroVILab-AHU/SARU ; AeroDeshadow: https://arxiv.org/abs/2604.15903
- ShadowFormer: https://github.com/GuoLanqing/ShadowFormer ; ShadowDiffusion: https://github.com/GuoLanqing/ShadowDiffusion ; DC-ShadowNet: https://github.com/jinyeying/DC-ShadowNet-Hard-and-Soft-Shadow-Removal
- Unveiling Deep Shadows survey: https://arxiv.org/abs/2409.02108
- GaRe: https://arxiv.org/abs/2507.20512 ; LumiGauss: https://arxiv.org/abs/2408.04474 ; The Sky's the Limit: https://arxiv.org/abs/2311.16937 ; SOL-NeRF: http://geometrylearning.com/SOL-NeRF/ ; OutCast: https://arxiv.org/abs/2204.09341
- Timonen & Westerholm 2010: http://wili.cc/research/hfshadow/hfshadow.pdf ; max-mipmap terrain shadows: https://arxiv.org/abs/2005.06671 ; iq soft shadows: https://iquilezles.org/articles/rmshadows/
- rvt-py: https://rvt-py.readthedocs.io/en/latest/rvt.vis.html ; GRASS r.sun: https://grass.osgeo.org/grass-stable/manuals/r.sun.html
- Blender baking: https://docs.blender.org/manual/en/latest/render/cycles/baking.html ; dxr-ao-bake: https://github.com/Twinklebear/dxr-ao-bake
- pysolar: https://pypi.org/project/pysolar/ (GPL-3.0)
