# 04 — GLSL relighting shaders for ODM outputs (golden hour)

Complete, documented GLSL ES 3.00 (WebGL2) sources for:

1. **Mesh pass** — relight the textured OBJ/GLB (`odm_texturing/odm_textured_model_geo.obj|glb`)
   under a low, warm sun with long shadows, sky ambient, warm haze, rim light.
2. **Shadow-map pass** — single or cascaded orthographic shadow maps sized for large terrain.
3. **2D orthophoto + DSM pass** — relight `odm_orthophoto/odm_orthophoto.tif` using
   `odm_dem/dsm.tif` as a heightfield: normals from the DSM, ray-marched sun shadows, sky
   visibility.

Everything is documentation; nothing here is wired into ODM. Coordinate frame throughout:
**ODM local UTM, Z-up** (see file 02; convert the sun vector, not the shaders, if you load the
model Y-up). The sky/sun colours come from file 03, grading from file 05.

---

## 1. Shadow-map pass

### 1.1 Single orthographic shadow map

For a directional sun, an orthographic projection looking along `-sunDir` that encloses the
model's bounding box. For an ODM model of ~500 m × 500 m at 2048² that is ~25 cm/texel — enough
for building shadows, not for grass. Cascades (§1.3) fix the near field.

```glsl
// ---- shadow_depth.vert (GLSL ES 3.00) ----
#version 300 es
layout(location = 0) in vec3 aPosition;
uniform mat4 uLightViewProj;          // ortho * lookAt(center + sunDir*R, center, up)
uniform mat4 uModel;
void main() { gl_Position = uLightViewProj * uModel * vec4(aPosition, 1.0); }
```

```glsl
// ---- shadow_depth.frag ----
#version 300 es
precision highp float;
void main() { /* depth only; attach a DEPTH_COMPONENT24/32F texture, no colour attachment */ }
```

Host-side polygon offset (`gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(factor, units)`)
is the hardware **slope-scale bias**: depth += `factor · max(|dz/dx|, |dz/dy|) + units · r`.
Typical starting values for a 24-bit map: `factor = 2.0, units = 4.0`. MJP's survey recommends
starting a *manual* constant bias around 0.0015 (normalised depth) and adding normal-offset
scale 1–2 texels ([A Sampling of Shadow Techniques](https://therealmjp.github.io/posts/shadow-maps/)).

Light matrix construction (host, pseudo-code): transform the 8 corners of the model AABB (or the
current cascade's frustum slice) into light view space, take min/max, build
`ortho(minX, maxX, minY, maxY, -maxZ - pad, -minZ + pad)`. Snap `minX/minY` to whole shadow texels
to stop shimmering when the camera moves ([LearnOpenGL CSM](https://learnopengl.com/Guest-Articles/2021/CSM)).

### 1.2 Cascades for large terrain

Practical split scheme (Zhang et al. 2006 PSSM, as used by MJP and LearnOpenGL): for N cascades,
`d_i = λ · near·(far/near)^(i/N) + (1−λ)·(near + (far−near)·i/N)`, λ ≈ 0.5–0.8. For a drone site
viewed from 50–2000 m, 3–4 cascades at 2048² in a `sampler2DArrayShadow` work well. LearnOpenGL's
bias for cascade `layer`: `bias = max(0.05 · (1 − N·L), 0.005); bias *= 1 / (cascadeFar[layer] · 0.5)`.

---

## 2. Mesh relighting pass

### 2.1 Inputs

| Uniform / attribute | Type | Source |
|---|---|---|
| `aPosition, aNormal, aUv` | attributes | OBJ (ODM writes normals? — mvs-texturing OBJ has `vn` only if requested; otherwise compute flat/smooth normals at load, or use `dFdx/dFdy` in the fragment shader) |
| `uAlbedo` | `sampler2D` (sRGB → decode) | texture atlas `odm_textured_model_geo_material0000_map_Kd.png` etc. |
| `uSunDir` | `vec3` (unit, toward sun, Z-up) | file 02 |
| `uSunColor` | `vec3` linear HDR | `sunColor(e,T) · E0 · exposure`, file 03 |
| `uSkyAmbientZenith, uSkyAmbientHorizon, uSkyAmbientGround` | `vec3` | hemisphere lighting from the sky model (or `skyAmbient()`), file 03 |
| `uShadowMap` | `sampler2DShadow` or `sampler2DArrayShadow` | §1 |
| `uLightViewProj[N]`, `uCascadeSplits` | `mat4[N]`, `vec4` | §1 |
| `uCameraPos` | `vec3` | |
| `uFog*` | see §2.4 | |
| `uRoughness` (Oren–Nayar σ) | `float` | 0.3–0.6 for terrain, 0 = Lambert |

### 2.2 Vertex shader

```glsl
#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;   // ODM local UTM metres, Z-up
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;

uniform mat4 uModel;                       // usually identity for ODM models
uniform mat4 uViewProj;
uniform mat3 uNormalMatrix;                // inverse-transpose of uModel (3x3)

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUv;

void main() {
    vec4 wp   = uModel * vec4(aPosition, 1.0);
    vWorldPos = wp.xyz;
    vNormal   = normalize(uNormalMatrix * aNormal);
    vUv       = aUv;
    gl_Position = uViewProj * wp;
}
```

### 2.3 Fragment shader — lighting

```glsl
#version 300 es
precision highp float;
precision highp sampler2DArrayShadow;

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUv;
out vec4 fragColor;

// ---------- material ----------
uniform sampler2D uAlbedo;                 // sRGB-encoded 8-bit atlas
uniform float     uRoughness;              // Oren-Nayar sigma in radians (0 = Lambert). 0.35 for grass/soil

// ---------- lights ----------
uniform vec3  uSunDir;                     // unit vector toward the sun, Z-up
uniform vec3  uSunColor;                   // linear HDR irradiance colour (already exposed)
uniform vec3  uSkyZenith, uSkyHorizon, uSkyGround;   // hemisphere ambient (linear, exposed)
uniform vec3  uCameraPos;

// ---------- shadows ----------
#define NUM_CASCADES 3
uniform sampler2DArrayShadow uShadowMap;   // depth compare enabled: TEXTURE_COMPARE_MODE = COMPARE_REF_TO_TEXTURE
uniform mat4  uLightViewProj[NUM_CASCADES];
uniform float uCascadeFar[NUM_CASCADES];   // view-space distances
uniform vec2  uShadowTexel;                // 1/width, 1/height
uniform float uShadowNormalOffset;         // world metres, ~1.5 texels: texelWorldSize*1.5
uniform mat4  uView;                       // to get view depth for cascade selection

// ---------- fog / aerial perspective ----------
uniform vec3  uFogColorSky;                // colour of haze away from the sun (cool)
uniform vec3  uFogColorSun;                // colour of haze toward the sun (warm, ~ uSunColor tint)
uniform float uFogDensity;                 // a: 1/m at ground level, e.g. 4e-4 (clear) .. 2e-3 (hazy)
uniform float uFogHeightFalloff;           // b: 1/m, e.g. 1/150 (haze concentrated in lowest ~150 m)
uniform float uFogGroundZ;                 // reference height (mesh min Z) so fog is relative to terrain
uniform float uFogSunPower;                // 8.0 (IQ default) ; lower = wider glow

// ---------- rim / backlight ----------
uniform float uRimStrength;                // 0.3
uniform float uRimPower;                   // 3.0

const float PI = 3.141592653589793;

// ---- sRGB decode (WebGL2 without EXT_sRGB textures) ----
vec3 srgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

// ---- Oren-Nayar (qualitative model, Oren & Nayar 1994; coefficients as on Wikipedia) ----
// L = (rho/pi) * E0 * cos(theta_i) * (A + B * max(0, cos(phi_i - phi_r)) * sin(alpha) * tan(beta))
// A = 1 - 0.5 s2/(s2+0.33),  B = 0.45 s2/(s2+0.09),  alpha = max(ti,tr), beta = min(ti,tr)
float orenNayar(vec3 n, vec3 l, vec3 v, float sigma) {
    float NdL = dot(n, l), NdV = dot(n, v);
    float s2 = sigma * sigma;
    float A = 1.0 - 0.5 * s2 / (s2 + 0.33);
    float B = 0.45 * s2 / (s2 + 0.09);
    float thetaI = acos(clamp(NdL, -1.0, 1.0)), thetaR = acos(clamp(NdV, -1.0, 1.0));
    float alpha = max(thetaI, thetaR), beta = min(thetaI, thetaR);
    vec3 lp = normalize(l - n * NdL);                              // projections onto the tangent plane
    vec3 vp = normalize(v - n * NdV);
    float cosPhiDiff = max(dot(lp, vp), 0.0);
    return max(NdL, 0.0) * (A + B * cosPhiDiff * sin(alpha) * tan(beta));
}

// ---- PCF with 5x5 kernel on a comparison sampler ----
float shadowPCF(int layer, vec3 shadowUVZ) {
    float sum = 0.0;
    for (int y = -2; y <= 2; ++y)
        for (int x = -2; x <= 2; ++x)
            sum += texture(uShadowMap, vec4(shadowUVZ.xy + vec2(x, y) * uShadowTexel, float(layer), shadowUVZ.z));
    return sum / 25.0;
}

// Returns 1 = lit, 0 = shadowed. Uses normal-offset + slope-scaled constant bias.
float sunShadow(vec3 worldPos, vec3 n, vec3 l) {
    float viewDepth = -(uView * vec4(worldPos, 1.0)).z;
    int layer = NUM_CASCADES - 1;
    for (int i = 0; i < NUM_CASCADES; ++i) { if (viewDepth < uCascadeFar[i]) { layer = i; break; } }

    float NdL = clamp(dot(n, l), 0.0, 1.0);
    float sinT = sqrt(1.0 - NdL * NdL);
    // normal offset: push the lookup position along the normal, more at grazing angles (Holbert, GDC 2011)
    vec3 offPos = worldPos + n * (uShadowNormalOffset * sinT * float(layer + 1));

    vec4 lp = uLightViewProj[layer] * vec4(offPos, 1.0);
    vec3 uvz = lp.xyz / lp.w * 0.5 + 0.5;
    if (uvz.z > 1.0) return 1.0;                                     // beyond the far plane: unshadowed
    // slope-scale bias in depth units (LearnOpenGL CSM form), scaled per cascade
    float bias = max(0.05 * (1.0 - NdL), 0.005) / (uCascadeFar[layer] * 0.5);
    uvz.z -= bias;
    float s = shadowPCF(layer, uvz);
    // fade the last cascade out at its far edge
    if (layer == NUM_CASCADES - 1) {
        float f = smoothstep(uCascadeFar[layer] * 0.9, uCascadeFar[layer], viewDepth);
        s = mix(s, 1.0, f);
    }
    return s;
}

// ---- height fog with analytic integral (Iñigo Quilez, "fog" article) ----
// density d(z) = a * exp(-b * (z - z0)); integral along the ray gives:
//   fogAmount = (a/b) * exp(-(ro.z - z0) * b) * (1 - exp(-t * rd.z * b)) / rd.z
float heightFogAmount(vec3 ro, vec3 rd, float t) {
    float a = uFogDensity, b = uFogHeightFalloff;
    float rz = max(abs(rd.z), 1e-4) * sign(rd.z + 1e-6);
    float f = (a / b) * exp(-(ro.z - uFogGroundZ) * b) * (1.0 - exp(-t * rz * b)) / rz;
    return 1.0 - exp(-f);                                             // Beer-Lambert transmittance -> amount
}

void main() {
    vec3 albedo = srgbToLinear(texture(uAlbedo, vUv).rgb);
    vec3 n = normalize(vNormal);
    vec3 v = normalize(uCameraPos - vWorldPos);
    vec3 l = normalize(uSunDir);
    if (!gl_FrontFacing) n = -n;

    // --- direct sun ---
    float diffuse = (uRoughness > 0.0) ? orenNayar(n, l, v, uRoughness) : max(dot(n, l), 0.0);
    float shadow  = sunShadow(vWorldPos, n, l);
    vec3 direct   = albedo * uSunColor * diffuse * shadow;             // Lambert already carries 1/pi in uSunColor scale

    // --- sky ambient: hemisphere lighting (zenith/horizon/ground) ---
    float up = n.z * 0.5 + 0.5;
    vec3 skyAmb = mix(uSkyGround, mix(uSkyHorizon, uSkyZenith, smoothstep(0.5, 1.0, up)), smoothstep(0.0, 0.5, up));
    // cheap directional occlusion: surfaces facing away from the sun see slightly less bright horizon
    skyAmb *= 0.85 + 0.15 * (dot(n, vec3(l.xy, 0.0)) * 0.5 + 0.5);
    vec3 ambient = albedo * skyAmb;

    // --- rim / backlight: the sun behind the object lights grazing silhouettes (translucent grass, dust) ---
    float VdL   = dot(v, l);                                            // >0 when looking toward the sun
    float fres  = pow(1.0 - max(dot(n, v), 0.0), uRimPower);
    float back  = smoothstep(0.0, 1.0, VdL) ;                           // only when sun is in front of the camera
    vec3 rim    = uSunColor * uRimStrength * fres * back * shadow;      // shadowed pixels get no rim
    // (no albedo multiply: rim reads as scattered sunlight, keep it warm)

    vec3 color = direct + ambient + rim;

    // --- aerial perspective / warm haze ---
    vec3  rd = -v;                                                      // camera -> fragment
    float t  = length(vWorldPos - uCameraPos);
    float fog = heightFogAmount(uCameraPos, rd, t);
    float sunAmount = pow(max(dot(rd, l), 0.0), uFogSunPower);         // glow toward the sun
    vec3  fogColor  = mix(uFogColorSky, uFogColorSun, sunAmount);
    color = mix(color, fogColor, fog);

    fragColor = vec4(color, 1.0);                                       // HDR linear; tone map in the post pass (file 05)
}
```

### 2.4 Parameter notes

- `uSunColor` should be **irradiance-scaled** so `albedo · uSunColor · 1.0` is the response of a
  white surface facing the sun. With file 03's `sunColor()` (≈ (0.35, 0.16, 0.05) at 5°, T = 2) and
  an exposure chosen so that the *zenith* sun maps white to ~0.8, the golden-hour frame is
  ~2.5 stops darker; the grading pass adds ~+1.5 EV to taste (photographers' cameras do this
  automatically).
- **Fog colours**: `uFogColorSky` ≈ 0.5 × horizon sky colour (cool blue-grey);
  `uFogColorSun` ≈ `uSunColor` × 1.5 (orange). Density: clear air a = 3–5 × 10⁻⁴ m⁻¹ with
  b = 1/300 m; hazy golden hour a = 1–2 × 10⁻³, b = 1/120 m (haze sits low). These
  are visual, not physical — Bruneton/Hillaire give the physical aerial perspective (file 03).
- **Rim**: physically this is forward scattering by translucent/rough surfaces; keep `uRimStrength`
  ≤ 0.4 or the terrain looks glowing.
- **Oren–Nayar**: σ = 0.35–0.5 for soil/grass (visibly flatter than Lambert at low sun, which is
  what real terrain does at golden hour); 0 for roofs and asphalt. The inter-reflection term L2 of
  the full model is omitted (Wikipedia gives it: `0.17 ρ²/π E0 cosθi σ²/(σ²+0.13) [1 − cos(φi−φr)(2β/π)²]`).
- **Contact hardening**: sun angular diameter ≈ 0.53°, so the penumbra width is ≈ d/108 (d =
  occluder-to-receiver distance). PCSS (Fernando 2005, [Semantic Scholar](https://www.semanticscholar.org/paper/Percentage-closer-soft-shadows-Fernando/ef8fac37a59334fc032f40abe1b1ea6016529465))
  does this in three steps: blocker search (average depth of occluders in a search region ≈ light
  size × (dReceiver − near)/dReceiver), penumbra estimate `w = (dReceiver − dBlocker)/dBlocker · lightSize`,
  then PCF with kernel radius `w`. For a directional light use `lightSize` = tan(0.265°) × (cascade
  depth range) in texels. NVIDIA's OpenGL sample is a good reference
  ([Soft Shadows Sample](https://archive.docs.nvidia.com/gameworks/content/gameworkslibrary/graphicssamples/opengl_samples/softshadowssample.htm));
  a 5×5 blocker search is usually sufficient. This is what makes 100 m golden-hour shadows read
  as sharp at the base and soft at the tip.
- **Normals**: if the OBJ lacks `vn`, derive per-face normals in the fragment shader:
  `n = normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)))` (flat), or precompute smooth normals
  on load. The DSM-derived normal (§4) is a good alternative for the 2.5D mesh.

### 2.5 WGSL translation notes (WebGPU)

| GLSL ES 3.00 | WGSL |
|---|---|
| `layout(location=0) in vec3 aPosition;` | `struct VSIn { @location(0) position: vec3<f32>, ... }` |
| `out vec3 vWorldPos; gl_Position = ...` | `struct VSOut { @builtin(position) clip: vec4<f32>, @location(0) worldPos: vec3<f32>, ... }` |
| `uniform mat4 uViewProj;` (loose uniforms) | `struct Frame { viewProj: mat4x4<f32>, ... } @group(0) @binding(0) var<uniform> frame: Frame;` (16-byte alignment; keep `vec3` → `vec4` or add padding, see file 06 table) |
| `uniform sampler2D uAlbedo; texture(uAlbedo, uv)` | `@group(1) @binding(0) var albedoTex: texture_2d<f32>; @group(1) @binding(1) var albedoSamp: sampler; textureSample(albedoTex, albedoSamp, uv)` |
| `sampler2DArrayShadow` + `texture(sm, vec4(uv, layer, ref))` | `var shadowTex: texture_depth_2d_array; var shadowSamp: sampler_comparison; textureSampleCompare(shadowTex, shadowSamp, uv, layer, ref)` — returns 0/1 (hardware PCF 2×2 with `filterable` comparison sampler) ([WGSL spec §textureSampleCompare](https://www.w3.org/TR/WGSL/#texturesamplecompare)) |
| `dFdx/dFdy` | `dpdx/dpdy` (fragment stage only) |
| `gl_FrontFacing` | `@builtin(front_facing) ff: bool` |
| `mix, clamp, smoothstep, pow` | same names; `mix` with a scalar third arg on vectors needs `mix(a, b, vec3(t))` |
| `sRGB decode in shader` | prefer an `rgba8unorm-srgb` texture format and let the sampler decode |
| Clip-space Z | WebGPU is 0..1 depth (like Metal/D3D), WebGL is −1..1: the `uvz.z` of the shadow lookup must **not** be remapped with `*0.5+0.5` in WGSL, only `xy` |
| Loops in PCF | unrolled fine; WGSL requires `var` loop counters (`for (var y = -2; y <= 2; y++)`) |
| `texture(...)` in non-uniform control flow | `textureSample` must be in uniform control flow in WGSL; use `textureSampleLevel` inside `if` |

Cascade selection via a loop over `uCascadeFar` and the `mat4` array become `array<mat4x4<f32>, 3>`
in the uniform struct (each element 64 B, naturally aligned). WebGPU samples on the shadow mapping
demo ([webgpu-samples](https://webgpu.github.io/webgpu-samples/samples/shadowMapping/)) show the
depth-texture + comparison-sampler binding setup.

---

## 3. Sky-dome pass

Draw a unit cube/sphere around the camera with `gl_Position.z = gl_Position.w` (far plane), and
in the fragment shader call `skyRadiance(normalize(worldDir), uSunDir, ...)` from file 03 (or
sample the Hosek/Hillaire sky-view LUT). Render **after** the mesh with `depthFunc(LEQUAL)` so it
only fills uncovered pixels. Apply the same fog colour toward the horizon so the mesh's haze and
the sky agree: `color = mix(color, fogColorAtHorizon, pow(1 − dir.z, 8))`.

---

## 4. 2D pass: orthophoto + DSM relighting

Inputs: `odm_orthophoto.tif` (RGBA, sRGB) and `dsm.tif` (float32 metres, nodata usually
−9999) resampled to the **same grid**; both are north-up with the same UTM CRS in ODM
(orthophoto and DEMs share the `--orthophoto-resolution`/`--dem-resolution` GSD if you set them
equal; otherwise resample on load). Load the DSM as an `R32F` texture (`gl.getExtension('EXT_color_buffer_float')`
not needed for sampling, only for rendering to float). The texture's pixel size in metres (`uGsd`)
comes from the GeoTIFF `GeoTransform`; the DSM row 0 is **north**, so `+y` in texture space =
south. The sun vector is the ENU vector from file 02; in texture space `sunTex.xy = (E, −N)`.

```glsl
#version 300 es
precision highp float;

in vec2 vUv;                                // 0..1 over the orthophoto; (0,0) = top-left (north-west)
out vec4 fragColor;

uniform sampler2D uOrtho;                   // sRGB RGBA
uniform sampler2D uDsm;                     // R32F metres; NEAREST or LINEAR filtering
uniform vec2  uTexSize;                     // DSM width,height in pixels
uniform float uGsd;                         // metres per pixel (assume square)
uniform float uNoData;                      // e.g. -9999.0
uniform vec3  uSunDirENU;                   // unit, toward sun, (E, N, U)
uniform vec3  uSunColor;                    // linear HDR, exposed
uniform vec3  uSkyZenith, uSkyHorizon;      // ambient
uniform float uSunAngularRadius;            // radians, 0.00465 (0.266 deg)
uniform float uMaxShadowDist;               // metres, e.g. 300.0 (5 deg sun: 11.4 * max height difference)
uniform int   uShadowSteps;                 // e.g. 128
uniform float uRoughness;                   // Oren-Nayar sigma (0 = Lambert)
uniform vec3  uFogColorSun, uFogColorSky;
uniform float uHazeStrength;                // 0..1, cheap "aerial perspective" for a top-down view: haze by height
uniform float uHazeRefZ, uHazeFalloff;      // metres

vec3 srgbToLinear(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }

float dsm(vec2 uv) {
    float z = texture(uDsm, uv).r;
    return z;                               // caller handles nodata
}
bool isNoData(float z) { return z <= uNoData + 0.5 || isnan(z); }

// ---- normal from DSM by central differences (Horn / Zevenbergen-Thorne style) ----
// ENU frame: x = east (uv.x +), y = north (uv.y -), z = up
vec3 dsmNormal(vec2 uv) {
    vec2 px = 1.0 / uTexSize;
    float zc = dsm(uv);
    float zl = dsm(uv - vec2(px.x, 0.0)), zr = dsm(uv + vec2(px.x, 0.0));
    float zu = dsm(uv - vec2(0.0, px.y)), zd = dsm(uv + vec2(0.0, px.y));
    if (isNoData(zl)) zl = zc; if (isNoData(zr)) zr = zc;
    if (isNoData(zu)) zu = zc; if (isNoData(zd)) zd = zc;
    float dzdx = (zr - zl) / (2.0 * uGsd);            // east
    float dzdy = (zu - zd) / (2.0 * uGsd);            // north (row up = north)
    return normalize(vec3(-dzdx, -dzdy, 1.0));
}

// ---- ray-marched heightfield shadow toward the sun ----
// Walk from the surface point along the sun vector; if the DSM is above the ray, we are in shadow.
// Soft version: track the minimum "clearance angle" and compare to the sun's angular radius (like IQ soft shadows).
float dsmShadow(vec2 uv, float z0, vec3 n) {
    vec3 s = uSunDirENU;
    if (s.z <= 0.0) return 0.0;                                     // sun below horizon
    float horiz = length(s.xy);
    if (horiz < 1e-4) return 1.0;                                   // sun at zenith: no cast shadow
    float slope = s.z / horiz;                                      // rise per horizontal metre = tan(elevation)

    // start slightly above the surface to avoid self-shadowing (bias ~ 1 texel of slope + 5 cm)
    float bias = uGsd * slope + 0.05;
    float minClear = 1e9;                                           // smallest (rayZ - dsmZ)/dist -> soft factor
    float dist = uGsd;                                              // horizontal metres travelled
    // exponential stepping: fine near the point (contact hardening), coarse far away
    float stepMul = pow(uMaxShadowDist / uGsd, 1.0 / float(uShadowSteps));
    for (int i = 0; i < 512; ++i) {
        if (i >= uShadowSteps || dist > uMaxShadowDist) break;
        // uv offset for `dist` horizontal metres along the sun azimuth (row 0 = north => -N maps to +v)
        vec2 uvS = uv + vec2(s.x, -s.y) / horiz * dist / (uTexSize * uGsd);
        if (any(lessThan(uvS, vec2(0.0))) || any(greaterThan(uvS, vec2(1.0)))) break;  // left the map: assume lit
        float zs = dsm(uvS);
        if (isNoData(zs)) { dist *= stepMul; continue; }
        float rayZ = z0 + bias + slope * dist;
        float clearance = rayZ - zs;                                // <0 => blocked
        if (clearance < 0.0) return 0.0;                            // hard shadow
        // soft: angular clearance relative to the sun's angular radius
        minClear = min(minClear, clearance / dist);                 // ~ tan(angle above the ray)
        dist *= stepMul;
    }
    // penumbra: clearance angle < sun radius -> partially occluded
    float k = clamp(minClear / uSunAngularRadius, 0.0, 1.0);
    return k * k * (3.0 - 2.0 * k);
}

// ---- sky visibility (horizon-based, 8 azimuths, short range) : cheap ambient occlusion for valleys/streets ----
float skyVisibility(vec2 uv, float z0) {
    float acc = 0.0;
    const int NAZ = 8;
    for (int a = 0; a < NAZ; ++a) {
        float ang = 6.2831853 * float(a) / float(NAZ);
        vec2 d = vec2(cos(ang), sin(ang));
        float maxTan = 0.0;
        float dist = uGsd;
        for (int i = 0; i < 16; ++i) {                              // ~16 samples, exponential to ~60 m
            vec2 uvS = uv + vec2(d.x, -d.y) * dist / (uTexSize * uGsd);
            float zs = dsm(uvS);
            if (!isNoData(zs)) maxTan = max(maxTan, (zs - z0) / dist);
            dist *= 1.3;
        }
        float horizonAngle = atan(maxTan);                          // 0 = flat
        acc += 1.0 - sin(horizonAngle);                             // fraction of the hemisphere above the horizon (cos-weighted approx)
    }
    return acc / float(NAZ);
}

// Oren-Nayar for a top-down viewer (v = +Z)
float diffuseTerm(vec3 n, vec3 l, float sigma) {
    float NdL = max(dot(n, l), 0.0);
    if (sigma <= 0.0) return NdL;
    vec3 v = vec3(0.0, 0.0, 1.0);
    float s2 = sigma * sigma;
    float A = 1.0 - 0.5 * s2 / (s2 + 0.33), B = 0.45 * s2 / (s2 + 0.09);
    float ti = acos(clamp(dot(n, l), -1.0, 1.0)), tr = acos(clamp(dot(n, v), -1.0, 1.0));
    float alpha = max(ti, tr), beta = min(ti, tr);
    vec3 lp = normalize(l - n * dot(n, l)), vp = normalize(v - n * dot(n, v));
    return NdL * (A + B * max(dot(lp, vp), 0.0) * sin(alpha) * tan(beta));
}

void main() {
    vec4 ortho = texture(uOrtho, vUv);
    float z0 = dsm(vUv);
    if (ortho.a < 0.01 || isNoData(z0)) { fragColor = vec4(0.0); return; }

    vec3 albedo = srgbToLinear(ortho.rgb);
    vec3 n = dsmNormal(vUv);
    vec3 l = uSunDirENU;

    float shadow = dsmShadow(vUv, z0, n);
    float skyVis = skyVisibility(vUv, z0);

    vec3 direct  = albedo * uSunColor * diffuseTerm(n, l, uRoughness) * shadow;
    vec3 skyAmb  = mix(uSkyHorizon, uSkyZenith, n.z * 0.5 + 0.5) * skyVis;
    vec3 ambient = albedo * skyAmb;

    vec3 color = direct + ambient;

    // top-down "haze": low ground is seen through more haze than high ground (very rough aerial perspective)
    float haze = uHazeStrength * (1.0 - exp(-max(uHazeRefZ - z0, 0.0) * uHazeFalloff));
    vec3 hazeColor = mix(uFogColorSky, uFogColorSun, 0.5);          // nadir view: no strong sun-direction term
    color = mix(color, hazeColor, haze);

    fragColor = vec4(color, ortho.a);                               // HDR linear -> post pass
}
```

Implementation notes for the 2D pass:

- **Step count vs. shadow length.** At 5° elevation and a 30 m building, shadows run 340 m; with a
  5 cm GSD that is 6,800 texels. The exponential stepping above covers `uMaxShadowDist` in
  `uShadowSteps` samples with fine steps near the origin (where contact hardening matters) and
  coarse steps far away (where the tall occluders are). A cleaner approach for production: build a
  **max-mip pyramid** of the DSM and march with mip-level ≈ log2(step); or precompute a
  **horizon map** (max elevation angle per azimuth, 8–16 azimuths) once per sun *azimuth* —
  Wikipedia "Horizon mapping" ([link](https://en.wikipedia.org/wiki/Horizon_mapping)), Snyder &
  Nowrouzezahrai's *Fast Soft Self-Shadowing on Dynamic Height Fields*
  ([PDF](https://cim.mcgill.ca/~derek/files/hfvisib.pdf)) for a GPU-friendly line-sweep. For a
  fixed sun the horizon-angle map along the sun azimuth is a **single 1D sweep per scanline** in
  the sun direction (O(N) total), which is what GDAL's `gdaldem hillshade` does *not* do (it has
  no cast shadows) and what rreusser's line-sweep terrain lighting notebook does
  ([link](https://rreusser.github.io/notebooks/line-sweep-terrain-lighting/)).
- **De-shadowing the capture-time sun**: run `dsmShadow` with the *capture-time* sun vector
  (file 02 §6) and lift shadowed texels by `1 / (diffuseFrac)` before relighting; the direct/diffuse
  table in file 01 gives the fraction.
- **Nodata and edges**: treat nodata as "no occluder" and rays leaving the raster as lit; the
  orthophoto alpha handles the cutline.
- Vertical units: ODM DSM is in metres (same as the horizontal UTM units), so no scale factor is
  needed; if you use a DSM in feet or a different vertical datum, scale `z` before use.
- Everything after `fragColor` (tone map, grading, bloom on the specular sun glints) is shared
  with the mesh pass (file 05).

---

## 5. Frame graph (WebGL2)

1. Shadow cascades (3 × 2048², depth only) — only when the sun or model moves.
2. Mesh pass → HDR `RGBA16F` colour target (needs `EXT_color_buffer_half_float` / `EXT_color_buffer_float`).
3. Sky-dome pass into the same target.
4. Post: bloom (file 05), tone map + grade + LUT, output to the canvas (sRGB).

For the 2D product the "frame" is a single full-screen quad running the §4 shader into an HDR
target followed by step 4; export via `readPixels` or `toBlob` after tone mapping.

---

## Sources

- LearnOpenGL — Cascaded Shadow Mapping (split scheme, bias, PCF, sampler2DArray): https://learnopengl.com/Guest-Articles/2021/CSM
- MJP — A Sampling of Shadow Techniques (bias values, normal offset, PSSM λ, gather-PCF): https://therealmjp.github.io/posts/shadow-maps/
- Khronos forum — CSM + slope-based bias + PCF: https://community.khronos.org/t/csm-slope-based-bias-pcf/64968
- Alex Tardif — Cascaded Shadow Maps with Soft Shadows: https://alextardif.com/shadowmapping.html
- Fernando 2005 — Percentage-Closer Soft Shadows (Semantic Scholar): https://www.semanticscholar.org/paper/Percentage-closer-soft-shadows-Fernando/ef8fac37a59334fc032f40abe1b1ea6016529465
- NVIDIA GameWorks — Soft Shadows Sample (PCSS): https://archive.docs.nvidia.com/gameworks/content/gameworkslibrary/graphicssamples/opengl_samples/softshadowssample.htm
- Wikipedia — Oren–Nayar reflectance model (A, B, L2 coefficients): https://en.wikipedia.org/wiki/Oren%E2%80%93Nayar_reflectance_model
- Iñigo Quilez — fog (analytic height fog integral, sun-direction fog colour): https://iquilezles.org/articles/fog/
- Unreal Engine — Exponential Height Fog: https://dev.epicgames.com/documentation/unreal-engine/exponential-height-fog-in-unreal-engine
- Wikipedia — Horizon mapping: https://en.wikipedia.org/wiki/Horizon_mapping
- Snyder & Nowrouzezahrai — Fast Soft Self-Shadowing on Dynamic Height Fields (PDF): https://cim.mcgill.ca/~derek/files/hfvisib.pdf
- rreusser — Line-sweep terrain lighting: https://rreusser.github.io/notebooks/line-sweep-terrain-lighting/
- vterrain.org — Terrain lighting survey: http://vterrain.org/Performance/lighting.html
- Shaderbits — Ray-marched heightmaps: https://shaderbits.com/blog/ray-marched-heightmaps
- WGSL spec — textureSampleCompare / sampler_comparison: https://www.w3.org/TR/WGSL/#texturesamplecompare
- WebGPU samples — Shadow mapping: https://webgpu.github.io/webgpu-samples/samples/shadowMapping/
- webgpufundamentals — Textures: https://webgpufundamentals.org/webgpu/lessons/webgpu-textures.html
- three.js Sky.js (sky-dome pass reference): https://github.com/mrdoob/three.js/blob/dev/examples/jsm/objects/Sky.js
