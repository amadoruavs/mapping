# 05 — Colour temperature, tone mapping and the "golden hour look"

Post-processing chain for the HDR output of the shaders in file 04:

`HDR linear → (exposure) → bloom add → tone map → split tone / grade → LUT → sRGB encode → grain, vignette`

All code is GLSL ES 3.00, documentation only.

---

## 1. Kelvin → linear RGB (Planckian locus)

Three approaches, from quick-and-dirty to colorimetrically correct. All produce a **normalised
chromaticity**; multiply by the intensity you want (from file 03) separately.

### 1.1 Tanner Helland (2012) — empirical fit to Mitchell Charity's blackbody table

Valid 1,000–40,000 K (best 1,500–15,000 K); output is **sRGB-encoded 0–255**, so decode to linear
before using it as a light colour. Source: [tannerhelland.com](https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html)
(CIE 1964 10° observer data).

```glsl
// Tanner Helland fit. Returns sRGB-encoded colour in 0..1 (NOT linear).
vec3 kelvinToSRGB_Helland(float kelvin) {
    float t = clamp(kelvin, 1000.0, 40000.0) / 100.0;
    float r, g, b;
    r = (t <= 66.0) ? 255.0 : 329.698727446 * pow(t - 60.0, -0.1332047592);
    g = (t <= 66.0) ? 99.4708025861 * log(t) - 161.1195681661
                    : 288.1221695283 * pow(t - 60.0, -0.0755148492);
    b = (t >= 66.0) ? 255.0 : (t <= 19.0 ? 0.0 : 138.5177312231 * log(t - 10.0) - 305.0447927307);
    return clamp(vec3(r, g, b) / 255.0, 0.0, 1.0);
}
vec3 srgbToLinear(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
vec3 kelvinToLinear_Helland(float k) { return srgbToLinear(kelvinToSRGB_Helland(k)); }
```

### 1.2 Krystek (1985) — rational fit of the Planckian locus in CIE 1960 (u, v)

The task brief mentions "Krystof Neumann's fit"; I could not find any source under that name.
The closest published rational fit is **M. Krystek, "An algorithm to calculate correlated colour
temperature", Color Res. Appl. 10(1):38–40, 1985** ([Wiley](https://onlinelibrary.wiley.com/doi/10.1002/col.5080100109)),
whose coefficients (quoted on Wikipedia's *Planckian locus* page,
[link](https://en.wikipedia.org/wiki/Planckian_locus)) appear verbatim in several Shadertoy
"blackbody" shaders. Valid **1,000 K < T < 15,000 K**.

```glsl
// Krystek 1985: Planckian locus in CIE 1960 UCS (u, v); valid 1000..15000 K
vec2 planckUV_Krystek(float T) {
    float T2 = T * T;
    float u = (0.860117757 + 1.54118254e-4 * T + 1.28641212e-7 * T2) /
              (1.0 + 8.42420235e-4 * T + 7.08145163e-7 * T2);
    float v = (0.317398726 + 4.22806245e-5 * T + 4.20481691e-8 * T2) /
              (1.0 - 2.89741816e-5 * T + 1.61456053e-7 * T2);
    return vec2(u, v);
}
// (u,v) -> (x,y) -> XYZ (Y=1) -> linear sRGB (D65)
vec3 uvToLinearSRGB(vec2 uv) {
    float x = 3.0 * uv.x / (2.0 * uv.x - 8.0 * uv.y + 4.0);
    float y = 2.0 * uv.y / (2.0 * uv.x - 8.0 * uv.y + 4.0);
    vec3 XYZ = vec3(x / y, 1.0, (1.0 - x - y) / y);
    const mat3 XYZ_TO_SRGB = mat3(  3.2406, -0.9689,  0.0557,     // column-major: columns are X,Y,Z contributions
                                   -1.5372,  1.8758, -0.2040,
                                   -0.4986,  0.0415,  1.0570);
    return max(XYZ_TO_SRGB * XYZ, 0.0);
}
vec3 kelvinToLinear_Krystek(float T) { return uvToLinearSRGB(planckUV_Krystek(clamp(T, 1000.0, 15000.0))); }
```

### 1.3 Kim et al. 2002 (a.k.a. "Kang 2002") — cubic spline in CIE 1931 (x, y)

Kim, Moon, Hong, Lee, Cho, Kim, "Design of advanced color temperature control system for HDTV
applications", *J. Korean Phys. Soc.* 41(6):865, 2002. `colour-science` exposes it as
`colour.temperature.CCT_to_xy_Kang2002` ([docs](https://colour.readthedocs.io/en/v0.3.7/colour.temperature.cct.html)).
Coefficients from Wikipedia *Planckian locus* ([link](https://en.wikipedia.org/wiki/Planckian_locus)).
Valid **1,667–25,000 K** — the widest of the three, and the one to use for the 2,000–5,500 K
golden-hour range.

```glsl
// Kim et al. 2002 cubic-spline Planckian locus, CIE 1931 xy; valid 1667..25000 K
vec2 planckXY_Kim2002(float T) {
    float t = clamp(T, 1667.0, 25000.0);
    float it = 1.0 / t, it2 = it * it, it3 = it2 * it;
    float x = (t <= 4000.0)
        ? -0.2661239e9 * it3 - 0.2343589e6 * it2 + 0.8776956e3 * it + 0.179910
        : -3.0258469e9 * it3 + 2.1070379e6 * it2 + 0.2226347e3 * it + 0.240390;
    float x2 = x * x, x3 = x2 * x;
    float y = (t <= 2222.0) ? -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
            : (t <= 4000.0) ? -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
            :                  3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483;
    return vec2(x, y);
}
vec3 xyToLinearSRGB(vec2 xy) {
    vec3 XYZ = vec3(xy.x / xy.y, 1.0, (1.0 - xy.x - xy.y) / xy.y);
    const mat3 XYZ_TO_SRGB = mat3( 3.2406, -0.9689,  0.0557,
                                  -1.5372,  1.8758, -0.2040,
                                  -0.4986,  0.0415,  1.0570);
    return max(XYZ_TO_SRGB * XYZ, 0.0);
}
vec3 kelvinToLinear_Kim2002(float T) { return xyToLinearSRGB(planckXY_Kim2002(T)); }
```

Normalise for use as a *light colour*: `c / max(c.r, max(c.g, c.b))` (keeps the brightest channel
at 1) or divide by luminance `dot(c, vec3(0.2126, 0.7152, 0.0722))` (keeps Y = 1, so exposure does
not change with CCT). Examples (Kim 2002, max-normalised, linear): 2,000 K ≈ (1.00, 0.30, 0.02);
3,000 K ≈ (1.00, 0.55, 0.20); 4,000 K ≈ (1.00, 0.76, 0.50); 5,500 K ≈ (1.00, 0.95, 0.88); 6,504 K
≈ (1, 1, 1) by construction of the sRGB white point. (Rough values from evaluating the formula;
recompute for exact numbers.)

**Which to use?** For *sunlight* the physically right answer is the transmittance-based
`sunColor()` in file 03 — the sun is not a blackbody at 2,500 K, it is a 5,800 K source seen
through a red filter, and the two differ in green. Kelvin→RGB is for the *artistic* knobs (white
balance, split-tone tints, "make it 3,200 K") and for matching photographer vocabulary.

---

## 2. White balance vs tinting the light

Two different operations that both make the image "warmer":

| | Tint the light (`uSunColor` warm) | White balance (camera-style) |
|---|---|---|
| What changes | The *lit* surfaces get warm; shadows stay sky-blue; highlights on white objects go orange. | Everything shifts: shadows go from blue toward neutral/warm too. |
| Physical analogue | The sun at 5° elevation. | Setting the camera's WB to "shade"/"cloudy" (7,000 K) or "daylight" (5,500 K). |
| Where in the chain | In the lighting shader (scene-referred). | After lighting, before/after tone mapping (a **von Kries** chromatic adaptation). |
| Implementation | multiply irradiance by the CCT colour | LMS scaling (below) |

Real golden-hour photos are usually shot at **daylight WB (~5,500 K)** so the warm light is
preserved; setting WB to the *sun's* CCT (auto-WB) neutralises the look. In shader terms: light
the scene with a warm sun and a blue sky, then apply a *mild* WB toward warm (e.g. +300–500 K
"temperature", slight magenta "tint") in grading rather than a full correction.

```glsl
// Von Kries white balance in LMS (Catlike Coding / Unity URP style). temperature, tint in [-1, 1].
// Positive temperature = warmer output; positive tint = greener.
const mat3 LIN_SRGB_TO_LMS = mat3(0.390405, 0.070841, 0.023108,     // column-major (columns: R,G,B)
                                  0.549941, 0.963172, 0.128021,
                                  0.008932, 0.001357, 0.936245);
const mat3 LMS_TO_LIN_SRGB = mat3( 2.858475, -0.210182, -0.041812,
                                  -1.628792,  1.158262, -0.118258,
                                  -0.024891,  0.000324,  1.068743);
vec3 whiteBalanceCoeffs(float temperature, float tint) {
    // Approximation of URP's ColorBalanceToLMSCoeffs: shift the D65 white point along the locus (x) and perpendicular (y)
    float t1 = temperature * 10.0 / 6.0, t2 = tint * 10.0 / 6.0;
    float x = 0.31271 - t1 * (t1 < 0.0 ? 0.1 : 0.05);
    float y = 2.87 * x - 3.0 * x * x - 0.27509507 + t2 * 0.05;
    vec3 w1 = vec3(0.949237, 1.03542, 1.08728);                   // D65 in LMS
    vec3 XYZ = vec3(x / y, 1.0, (1.0 - x - y) / y);
    vec3 w2 = LIN_SRGB_TO_LMS * xyToLinearSRGB(vec2(x, y));
    return w1 / w2;
}
vec3 applyWhiteBalance(vec3 linearRGB, vec3 coeffs) {
    return LMS_TO_LIN_SRGB * ((LIN_SRGB_TO_LMS * linearRGB) * coeffs);
}
```

(Matrices and the temperature/tint parameterisation follow Unity's URP `ColorUtils`, as described
by Catlike Coding's *Color Grading* tutorial: [link](https://catlikecoding.com/unity/tutorials/custom-srp/color-grading/).
The xy shift is an approximation; for exactness use a Bradford CAT between the source and target
white points via `colour-science` on the CPU and upload a 3×3.)

---

## 3. Tone mapping

The HDR frame has a sun disk at 10⁴–10⁵× the terrain, bright warm haze, and dark blue shadows.
The tone mapper decides how much of the warm saturation survives.

### 3.1 ACES filmic fits

**Narkowicz 2016** ([blog](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/)):
luminance-only rational fit with max error 0.0138, pre-exposed so 1.0 → ~0.8; "over-saturates
brights" (the author's own note), which for golden hour is actually pleasant.

```glsl
vec3 acesNarkowicz(vec3 x) {                     // input: linear, exposure applied; output 0..1 linear
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
```

**Stephen Hill fit** (MIT; in MJP's BakingLab [ACES.hlsl](https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl))
applies the RRT+ODT fit in AP1 with input/output matrices — hue-preserving-ish, the one three.js
ships as `ACESFilmicToneMapping` (three.js transposes the matrices for column-major GLSL and
pre-multiplies by 1/0.6):

```glsl
const mat3 ACESInputMat  = mat3(0.59719, 0.07600, 0.02840,    // column-major (three.js transposed form)
                                0.35458, 0.90834, 0.13383,
                                0.04823, 0.01566, 0.83777);
const mat3 ACESOutputMat = mat3( 1.60475, -0.10208, -0.00327,
                                -0.53108,  1.10813, -0.07276,
                                -0.07367, -0.00605,  1.07602);
vec3 RRTAndODTFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}
vec3 acesHill(vec3 color) {                      // linear sRGB in, 0..1 linear sRGB out
    color = ACESInputMat * (color / 0.6);        // three.js: divide by 0.6 to match the "1.0 -> 0.8" exposure convention
    color = RRTAndODTFit(color);
    return clamp(ACESOutputMat * color, 0.0, 1.0);
}
```

### 3.2 AgX (Troy Sobotka; Blender 4 default)

Designed to avoid the ACES "notorious six" hue skews (bright saturated colours collapsing to
yellow/cyan/magenta). The minimal analytic form is Benjamin Wrensch's
([Minimal AgX implementation](https://iolite-engine.com/blog_posts/minimal_agx_implementation)),
which Filament and three.js adopted with Rec.2020 inset/outset matrices
(three.js `tonemapping_pars_fragment.glsl.js`, [source](https://github.com/mrdoob/three.js/blob/dev/src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js);
Filament PR [#7236](https://github.com/google/filament/pull/7236)). Copied from three.js (MIT):

```glsl
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(0.6274, 0.0691, 0.0164,
                                                0.3293, 0.9195, 0.0880,
                                                0.0433, 0.0113, 0.8956);
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3( 1.6605, -0.1246, -0.0182,
                                                -0.5876,  1.1329, -0.1006,
                                                -0.0728, -0.0083,  1.1187);
// 6th-order polynomial fit of the AgX sigmoid, mean error^2 3.67e-6 (Wrensch)
vec3 agxDefaultContrastApprox(vec3 x) {
    vec3 x2 = x * x, x4 = x2 * x2;
    return + 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
vec3 agx(vec3 color) {                              // linear sRGB in, 0..1 linear sRGB out
    const mat3 AgXInsetMatrix  = mat3(0.856627153315983, 0.137318972929847, 0.11189821299995,
                                      0.0951212405381588, 0.761241990602591, 0.0767994186031903,
                                      0.0482516061458583, 0.101439036467562, 0.811302368396859);
    const mat3 AgXOutsetMatrix = mat3( 1.1271005818144368, -0.1413297634984383, -0.14132976349843826,
                                      -0.11060664309660323, 1.157823702216272, -0.11060664309660294,
                                      -0.016493938717834573, -0.016493938717834257, 1.2519364065950405);
    const float AgxMinEv = -12.47393;               // log2(2^-10 * 0.18)
    const float AgxMaxEv = 4.026069;                // log2(2^6.5 * 0.18)
    color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
    color = AgXInsetMatrix * color;
    color = max(color, 1e-10);
    color = (log2(color) - AgxMinEv) / (AgxMaxEv - AgxMinEv);
    color = clamp(color, 0.0, 1.0);
    color = agxDefaultContrastApprox(color);
    // optional "look" here, e.g. punchy: slope 1.0, power 1.35, saturation 1.4 (Wrensch's agxLook)
    color = AgXOutsetMatrix * color;
    color = pow(max(vec3(0.0), color), vec3(2.2));  // AgX's 2.2 "EOTF" back to linear
    return LINEAR_REC2020_TO_LINEAR_SRGB * color;
}
```

### 3.3 Khronos PBR Neutral (2024)

Spec + reference GLSL in [KhronosGroup/ToneMapping](https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/README.md);
"just 13 lines of GLSL" ([Khronos press release](https://www.khronos.org/news/press/khronos-pbr-neutral-tone-mapper-released-for-true-to-life-color-rendering-of-3d-products)).
Goals: exact base-colour reproduction under neutral light, no hue shifts, analytically
invertible; parameters `Ks = 0.8 − F90 (F90 = 0.04)` for compression start and `Kd = 0.15` for
desaturation speed. Verbatim reference implementation (`pbrNeutral.glsl`):

```glsl
// Input color is non-negative and resides in the Linear Rec. 709 color space.
// Output color is also Linear Rec. 709, but in the [0, 1] range.
vec3 PBRNeutralToneMapping( vec3 color ) {
  const float startCompression = 0.8 - 0.04;
  const float desaturation = 0.15;

  float x = min(color.r, min(color.g, color.b));
  float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
  color -= offset;

  float peak = max(color.r, max(color.g, color.b));
  if (peak < startCompression) return color;

  const float d = 1. - startCompression;
  float newPeak = 1. - d * d / (peak + d - startCompression);
  color *= newPeak / peak;

  float g = 1. - 1. / (desaturation * (peak - newPeak) + 1.);
  return mix(color, newPeak * vec3(1, 1, 1), g);
}
```

### 3.4 Comparison for the golden-hour use case

| | ACES (Hill / Narkowicz) | AgX | PBR Neutral |
|---|---|---|---|
| Highlight roll-off | strong filmic shoulder; sun aureole → white with yellow ring | smooth, desaturates toward white gracefully | mild compression above 0.76; bright warm haze may clip to white *fast* |
| Hue behaviour | skews (saturated orange → yellow) ; "over-saturates brights" (Narkowicz) | hue-stable, slightly lower saturation overall | hue-exact below the knee, desaturates above (`Kd`) |
| Mid-tone contrast | high ("punchy") | moderate, adjustable via look | 1:1 (none) |
| Best for | cinematic golden hour, if you accept the yellow shift | photographic golden hour, dynamic range of sun + shadows | product/e-commerce accuracy — *not* HDR skies (Khronos says so) |
| Cost | 2 mat3 + rational | 4 mat3 + log2 + poly | branchy scalar |

Recommendation: **AgX** as default (Blender's and three.js' current filmic), with the "punchy"
look; ACES Hill as the "cinema" preset. PBR Neutral only for the orthophoto product when colour
fidelity to the source imagery matters more than the look.

---

## 4. Split toning (warm highlights, cool shadows)

Physically, golden-hour shadows are lit by the blue sky and highlights by the orange sun, so the
lighting shader already produces the split. Grading pushes it further. Adobe/Lightroom-style
split toning works in gamma space; Catlike Coding documents the Unity URP form
([link](https://catlikecoding.com/unity/tutorials/custom-srp/color-grading/)):

```glsl
// Split toning in approximate gamma space. shadowTint / highlightTint are 0..1 colours (e.g. (0.55,0.65,0.9) and (1.0,0.8,0.55)).
// balance in [-1,1]: >0 pushes more pixels into the highlight tint.
vec3 softLight(vec3 base, vec3 blend) {           // Photoshop soft-light
    return mix(base - (1.0 - 2.0 * blend) * base * (1.0 - base),
               base + (2.0 * blend - 1.0) * (sqrt(base) - base),
               step(0.5, blend));
}
vec3 splitTone(vec3 c, vec3 shadowTint, vec3 highlightTint, float balance, float strength) {
    vec3 g = pow(max(c, 0.0), vec3(1.0 / 2.2));
    float t = clamp(dot(clamp(g, 0.0, 1.0), vec3(0.2126, 0.7152, 0.0722)) + balance, 0.0, 1.0);
    vec3 sh = mix(vec3(0.5), shadowTint,    (1.0 - t) * strength);
    vec3 hi = mix(vec3(0.5), highlightTint, t * strength);
    g = softLight(g, sh);
    g = softLight(g, hi);
    return pow(g, vec3(2.2));
}
```

A scene-referred alternative that plays better with tone mapping: multiply *before* the tone map
by `mix(coolTint, warmTint, smoothstep(0.02, 0.6, luminance))` with tints normalised to Y = 1 — this
is effectively "shadows lit by a bluer sky", and it keeps the tone mapper's shoulder intact.

---

## 5. Bloom / glare for the sun disk and haze

Use the Call of Duty: Advanced Warfare bloom (Jimenez, SIGGRAPH 2014
[slides](https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/)),
as documented with GLSL in LearnOpenGL's *Physically Based Bloom*
([link](https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom)): progressive 13-tap
downsample (bilinear, effectively 36 texels) into a 5–6 level `R11F_G11F_B10F` mip chain, then
3×3 tent-filter upsample-and-add, and finally `mix(hdr, bloom, 0.04)` — a *strong* bias toward the
HDR buffer. No threshold: the sun disk (radiance ≫ 1) dominates naturally.

```glsl
// 13-tap downsample (Jimenez 2014 / LearnOpenGL). uSrcTexel = 1/srcResolution.
vec3 downsample13(sampler2D src, vec2 uv, vec2 px) {
    vec3 a = texture(src, uv + px * vec2(-2, 2)).rgb, b = texture(src, uv + px * vec2(0, 2)).rgb, c = texture(src, uv + px * vec2(2, 2)).rgb;
    vec3 d = texture(src, uv + px * vec2(-2, 0)).rgb, e = texture(src, uv).rgb,                   f = texture(src, uv + px * vec2(2, 0)).rgb;
    vec3 g = texture(src, uv + px * vec2(-2,-2)).rgb, h = texture(src, uv + px * vec2(0,-2)).rgb, i = texture(src, uv + px * vec2(2,-2)).rgb;
    vec3 j = texture(src, uv + px * vec2(-1, 1)).rgb, k = texture(src, uv + px * vec2(1, 1)).rgb;
    vec3 l = texture(src, uv + px * vec2(-1,-1)).rgb, m = texture(src, uv + px * vec2(1,-1)).rgb;
    return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}
// 3x3 tent upsample; radius in uv units (e.g. 0.005). Accumulate into the next-larger mip with additive blending.
vec3 upsampleTent(sampler2D src, vec2 uv, vec2 r) {
    vec3 s = texture(src, uv + vec2(-r.x,  r.y)).rgb * 1.0 + texture(src, uv + vec2(0,  r.y)).rgb * 2.0 + texture(src, uv + vec2(r.x,  r.y)).rgb * 1.0
           + texture(src, uv + vec2(-r.x,  0  )).rgb * 2.0 + texture(src, uv).rgb * 4.0                  + texture(src, uv + vec2(r.x,  0  )).rgb * 2.0
           + texture(src, uv + vec2(-r.x, -r.y)).rgb * 1.0 + texture(src, uv + vec2(0, -r.y)).rgb * 2.0 + texture(src, uv + vec2(r.x, -r.y)).rgb * 1.0;
    return s / 16.0;
}
```

Golden-hour specifics: tint the bloom slightly warm (`bloom *= vec3(1.0, 0.92, 0.8)`) — real
lens veiling glare takes the colour of the sun — and add an optional anisotropic streak (1D blur
along a chosen angle) for the "sunstar". Keep the mix at 0.03–0.06; more looks like fog, not glare.

---

## 6. Film grain and vignette

```glsl
// Cheap, stable hash noise (Dave Hoskins style). Apply in display-referred (post tone map) space.
float hash12(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec3 filmGrain(vec3 c, vec2 fragCoord, float time, float amount /*0.02..0.05*/) {
    float n = hash12(fragCoord + fract(time) * 1000.0) - 0.5;
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return c + n * amount * (1.0 - lum * 0.7);            // less grain in highlights, like film
}
vec3 vignette(vec3 c, vec2 uv, float strength /*0.3*/, float radius /*0.75*/) {
    float d = distance(uv, vec2(0.5)) / 0.7071;
    return c * (1.0 - strength * smoothstep(radius, 1.0, d));
}
```

Use *temporal* grain (per-frame seed) for interactive views and *static* grain for exported
orthophotos; never grain the HDR buffer (it would bloom).

---

## 7. LUT (.cube) workflow

### 7.1 Format (Adobe Cube LUT Specification 1.0, 2013)

From the spec PDF ([mirror](https://kono.phpage.fr/images/a/a1/Adobe-cube-lut-specification-1.0.pdf)):

- Text file; `#` starts a comment line; optional `TITLE "text"`.
- `DOMAIN_MIN r g b` / `DOMAIN_MAX r g b` default `0 0 0` / `1 1 1`; input and output are **not**
  limited to 0–1 (HDR allowed, values within ±1e37).
- Either `LUT_1D_SIZE N` (N in [2, 65536], three independent 1D tables) **or** `LUT_3D_SIZE N`
  (N in [2, 256]; readers may not support 256 — 33 or 65 is standard).
- 3D data order: **red index changes fastest**, then green, then blue: line index
  `r + N·g + N·N·b`.
- Each data line: three floats = the *output* RGB for that lattice point. Values may be
  non-monotonic.

### 7.2 Baking a golden-hour LUT

The LUT should capture the **display-referred** part of the chain only (everything after tone
mapping: white balance nudge, split tone, saturation, curves), so that it can be applied by any
viewer (Cesium, three.js `LUTPass`, DaVinci Resolve, Lightroom) on top of a standard tone mapper.

```python
# Documentation-only: bake a 33^3 .cube by pushing the identity lattice through the grading chain.
import numpy as np
N = 33
r, g, b = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N), np.linspace(0, 1, N), indexing='ij')
rgb = np.stack([r, g, b], -1).reshape(-1, 3)             # index = r + N*g + N*N*b  ->  build with b outermost
rgb = rgb.reshape(N, N, N, 3).transpose(2, 1, 0, 3).reshape(-1, 3)  # now iterates r fastest
out = grade(rgb)                                          # numpy port of splitTone/whiteBalance/saturation (display-referred, sRGB-encoded in, out)
with open('golden_hour.cube', 'w') as f:
    f.write('TITLE "ODM golden hour"\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\nLUT_3D_SIZE %d\n' % N)
    for o in np.clip(out, 0, 1):
        f.write('%.6f %.6f %.6f\n' % tuple(o))
```

### 7.3 Applying a LUT in GLSL

WebGL2 has `sampler3D`: upload the cube as `RGB16F`/`RGB8` N×N×N with `LINEAR` filtering and
sample with a half-texel inset:

```glsl
uniform sampler3D uLut; uniform float uLutSize; uniform float uLutStrength;
vec3 applyLut(vec3 c) {                                   // c: display-referred, 0..1, sRGB-encoded (match how the LUT was baked)
    vec3 uvw = c * (uLutSize - 1.0) / uLutSize + 0.5 / uLutSize;
    return mix(c, texture(uLut, uvw).rgb, uLutStrength);
}
```

(For WebGL1/GLES2 without 3D textures, slice the cube into a 2D strip and lerp two slices — the
Defold tutorial shows the layout: [link](https://defold.com/tutorials/grading/).)
Order of operations must match the bake: `tonemap → sRGB encode → LUT → grain/vignette`.

---

## 8. "Golden hour look" defaults

| Parameter | Default | Range | Rationale |
|---|---|---|---|
| Sun elevation | 5° | 2–8° | Core of the band (file 01) |
| Sun CCT (for UI/preview) | 2,700 K (T=2), 2,000 K (T=4) | 1,800–3,500 K | file 01 table |
| Sun colour (linear, max-normalised) | (1.00, 0.46, 0.14) [≈ transmittance at 5°, T=2, normalised] | | file 03 `sunColor()` |
| Sun intensity vs zenith | 0.17 | 0.05–0.4 | file 01 (direct-beam luminance) |
| Turbidity | 2.5 | 2–6 | 2 = crisp; ≥4 = hazy pastel |
| Mie coefficient (three.js scale) | 0.006 | 0.003–0.02 | aureole size |
| HG anisotropy g | 0.8 | 0.7–0.85 | |
| Sky zenith ambient (linear) | (0.20, 0.32, 0.60) × exposure | | blue, ~1/5 of sun-lit white |
| Sky horizon ambient | (0.55, 0.45, 0.35) × exposure | | warm-grey horizon |
| Ground bounce | 0.3 × albedo-ish grey | | |
| Direct : diffuse on horizontal | ≈ 1 : 1 | 0.5–2 | file 01 |
| Exposure (EV100 target) | 12.5 | 11.5–14 | sunset EV 12, before sunset 12–14 |
| Oren–Nayar σ | 0.4 (terrain), 0 (roofs) | 0–0.6 | |
| Shadow PCF kernel | 5×5 (+PCSS) | 3×3–9×9 | |
| Height-fog a, b | 8e-4 m⁻¹, 1/150 m⁻¹ | 3e-4–2e-3 / 1/300–1/80 | warm haze in the lowest ~150 m |
| Fog colour toward sun | sunColor × 1.5 | | |
| Fog colour away | horizon sky × 0.5 | | |
| Fog sun power | 8 | 4–16 | width of the glow |
| Rim strength / power | 0.3 / 3 | 0–0.5 | |
| Tone mapper | AgX (punchy: slope 1.0, power 1.35, sat 1.4) | ACES Hill alt. | §3 |
| White balance | +0.15 temperature, +0.03 tint (URP scale) | | keep the warmth, don't neutralise |
| Split tone shadows | (0.55, 0.65, 0.90), strength 0.25 | | sky-blue |
| Split tone highlights | (1.00, 0.80, 0.55), strength 0.35 | | orange |
| Split tone balance | +0.1 | −0.3–+0.3 | favour highlights |
| Saturation | +10 % | 0–20 % | |
| Bloom mix / tint | 0.04 / (1.0, 0.92, 0.8) | 0.02–0.06 | |
| Vignette | 0.25 at radius 0.75 | 0–0.4 | |
| Grain | 0.025 (interactive), 0.015 (export) | | |
| LUT size | 33 | 17–65 | |

---

## Sources

- Tanner Helland — How to convert temperature (K) to RGB: https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html
- Krystek 1985, An algorithm to calculate correlated colour temperature, Color Res. Appl. 10:38–40: https://onlinelibrary.wiley.com/doi/10.1002/col.5080100109
- Wikipedia — Planckian locus (Kim 2002 and Krystek 1985 approximations, coefficients): https://en.wikipedia.org/wiki/Planckian_locus
- colour-science `colour.temperature` (CCT_to_xy_Kang2002 et al.): https://colour.readthedocs.io/en/v0.3.7/colour.temperature.cct.html
- Catlike Coding — Color Grading (white balance LMS, split toning, SMH): https://catlikecoding.com/unity/tutorials/custom-srp/color-grading/
- Narkowicz — ACES Filmic Tone Mapping Curve: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
- Stephen Hill ACES fit (MJP BakingLab, MIT): https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
- three.js tone mapping chunk (ACES, AgX, Neutral GLSL): https://github.com/mrdoob/three.js/blob/dev/src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js
- Wrensch — Minimal AgX implementation: https://iolite-engine.com/blog_posts/minimal_agx_implementation
- Filament AgX PR #7236: https://github.com/google/filament/pull/7236
- three.js issue #27362 (AgX adoption discussion): https://github.com/mrdoob/three.js/issues/27362
- Khronos PBR Neutral README + pbrNeutral.glsl: https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/README.md
- Khronos press release (PBR Neutral): https://www.khronos.org/news/press/khronos-pbr-neutral-tone-mapper-released-for-true-to-life-color-rendering-of-3d-products
- model-viewer tone-mapping comparison page: https://modelviewer.dev/examples/tone-mapping
- Jimenez 2014 — Next Generation Post Processing in Call of Duty: Advanced Warfare: https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/
- LearnOpenGL — Physically Based Bloom: https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom
- Adobe Cube LUT Specification 1.0 (PDF mirror): https://kono.phpage.fr/images/a/a1/Adobe-cube-lut-specification-1.0.pdf
- aras-p/smol-cube (binary .cube equivalent, format notes): https://github.com/aras-p/smol-cube
- Defold — Grading shader tutorial (2D LUT strip): https://defold.com/tutorials/grading/
- Wikipedia — Exposure value (EV table): https://en.wikipedia.org/wiki/Exposure_value
