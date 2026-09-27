# 03 — Sky and sun models for real-time golden hour

Which analytic/precomputed sky model to use for (a) the visible sky dome behind the ODM mesh,
(b) the **sun colour/intensity** that lights the mesh, and (c) the **ambient sky colour** used in
shadow. Ends with a compact, self-contained GLSL sun+sky function (documentation only).

---

## 1. The candidates

### 1.1 Preetham, Shirley & Smits 1999 — "A Practical Analytic Model for Daylight"

- Paper: SIGGRAPH 1999 ([PDF](https://courses.cs.duke.edu/fall01/cps124/resources/p91-preetham.pdf),
  [SIGGRAPH history page](https://history.siggraph.org/learning/a-practical-analytic-model-for-daylight-by-preetham-shirley-and-smits/)).
- Sky luminance uses the **Perez** five-coefficient formula (their eq. 3/4):
  `F(θ, γ) = (1 + A·e^(B/cos θ)) · (1 + C·e^(Dγ) + E·cos²γ)`, `Y = Yz · F(θ, γ) / F(0, θs)`,
  with θ = view zenith angle, γ = angle between view ray and sun, θs = sun zenith angle. The same
  form is fitted for CIE x and y chromaticity; A…E are **linear in turbidity T**, and the zenith
  luminance/chromaticity `Yz, xz, yz` are polynomials in T and θs (their Appendix).
- Sun: extraterrestrial spectrum attenuated by Rayleigh, aerosol (Ångström β = 0.04608T − 0.04586,
  α = 1.3), ozone, mixed gases, water vapour (Appendix; see file 01).
- Known weaknesses (Hosek & Wilkie 2012 §2.3.1, Zotti & Wilkie's *Critical Review*
  [slides](https://dspace5.zcu.cz/bitstream/11025/11160/4/Zotti_prezentace.pdf)): valid only for
  T ≈ 2–6; **poor sunsets**; can return **negative** luminance at low turbidity and near the horizon;
  no ground-albedo term.
- Cost: a handful of exp/cos per pixel — essentially free.
- Licence: formulas in a paper; the many public implementations carry their own licences.

### 1.2 Hosek & Wilkie 2012 (+ 2013 solar radiance extension)

- Paper: *An Analytic Model for Full Spectral Sky-Dome Radiance*, SIGGRAPH 2012
  ([project page](https://cgg.mff.cuni.cz/projects/SkylightModelling/),
  [preprint PDF](https://cgg.mff.cuni.cz/projects/SkylightModelling/HosekWilkie_SkylightModel_SIGGRAPH2012_Preprint_lowres.pdf)).
- Formula (their eq. 8–9), an extended Perez form with **nine** coefficients A…I and an
  anisotropic Mie-like term χ:

  ```
  F(θ, γ) = (1 + A·e^(B / (cos θ + 0.01))) · (C + D·e^(Eγ) + F·cos²γ + G·χ(H, γ) + I·√cos θ)
  χ(g, α) = (1 + cos²α) / (1 + g² − 2g·cos α)^(3/2)
  ```

  θ = view zenith angle, γ = angle to the sun. Radiance = `F(θ, γ) · L_M` where L_M is the mean
  spectral radiance of the upper hemisphere (their §5.5), per spectral band or per XYZ channel.
- Coefficients are **not linear in turbidity**: for each of 10 integer turbidities, 2 albedos
  (0, 1) and 9 parameters there are **6 Bézier control points** over solar elevation, interpolated
  along `(elevation/90°)^(1/3)`; linear interpolation in turbidity and albedo (§5.3). The
  reference dataset covers T = 1…10, albedo 0…1, 11 spectral bands (320–720 nm), 46 solar elevations
  from 0° to 90° (§5). **Solar elevations below 0° are excluded** (§5.2) — so no blue hour.
- Improvements over Preetham: sunsets, high turbidity, ground albedo, per-band fitting, no negative
  values at low T. A separate 2013 dataset gives the **solar disk radiance** consistent with the
  sky.
- Cost: ~30 % slower than Preetham per evaluation (Kol, [Analytical sky simulation](https://timothykol.com/project.php?id=3));
  the CPU-side coefficient evaluation is the expensive part, the per-pixel formula is cheap.
  Kol also observes an unphysical brightness *increase* at low solar elevations in some
  configurations — worth checking for the golden-hour range.
- Reference implementation: `ArHosekSkyModel.c/.h` v1.4a, **3-clause BSD**, © Hosek & Wilkie
  2012–2013 (e.g. as vendored in [pbrt-v3](https://github.com/mmp/pbrt-v3/blob/master/src/ext/ArHosekSkyModel.h)
  and [ebruneton/clear-sky-models](https://github.com/ebruneton/clear-sky-models/blob/master/atmosphere/model/hosek/ArHosekSkyModel.cc)).
  GLSL: Shadertoy "Hosek-Wilkie Skylight Model" ([wslfD7](https://www.shadertoy.com/view/wslfD7)),
  which embeds the RGB coefficient tables.

### 1.3 Nishita 1993/1996 → Bruneton & Neyret 2008 (precomputed single + multiple scattering)

- Nishita et al. 1993 introduced ray-marched Rayleigh + Mie single scattering with
  exponential density profiles; Bruneton & Neyret 2008 (*Precomputed Atmospheric Scattering*,
  EGSR 2008, [HAL](https://hal.science/inria-00288758), [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2008.01245.x))
  precompute transmittance (2D), scattering (4D packed in a 3D texture) and ground irradiance (2D)
  including multiple scattering, so that sky, sun, **aerial perspective for any point in the
  atmosphere** and light shafts are texture look-ups.
- 2017 re-implementation ([ebruneton.github.io](https://ebruneton.github.io/precomputed_atmospheric_scattering/),
  [GitHub](https://github.com/ebruneton/precomputed_atmospheric_scattering)): heavily commented
  GLSL, ozone layer, luminance vs spectral modes, "a fraction of the cost" of spectral rendering
  when precomputing luminance, **3-clause BSD**. The exposed GLSL API (in `model.h`, not
  re-verified by fetch here) is `GetSolarRadiance()`, `GetSkyRadiance(camera, viewRay, shadowLength, sunDir, out transmittance)`,
  `GetSkyRadianceToPoint(camera, point, shadowLength, sunDir, out transmittance)` (aerial perspective)
  and `GetSunAndSkyIrradiance(point, normal, sunDir, out skyIrradiance)` — the last one is exactly
  the sun+ambient pair the mesh shader needs. A three.js port exists ([jeantimex](https://github.com/jeantimex/precomputed_atmospheric_scattering)).
- Cost: a few seconds of precomputation (once per atmosphere), then 2–3 texture fetches per pixel.
  Handles below-horizon sun (blue hour) physically. Downsides: 3D/4D LUTs, complexity, LUT
  artefacts near the horizon that Hillaire's method was designed to remove.

### 1.4 Hillaire 2020 — "A Scalable and Production Ready Sky and Atmosphere Rendering Technique" (Unreal's Sky Atmosphere)

- EGSR 2020 / CGF 39(4) ([Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14050),
  [EG digital library](https://diglib.eg.org/items/8a3e5350-18b3-46bd-9274-3add5af88c75),
  reference code [sebh/UnrealEngineSkyAtmosphere](https://github.com/sebh/UnrealEngineSkyAtmosphere), **MIT**).
- Per-frame, small 2D LUTs instead of 4D ones: a **transmittance LUT** (256×64), a
  **multi-scattering LUT** (32×32) built with a novel "energy-conserving infinite scattering
  series" approximation, a low-res **sky-view LUT** (paper: 192×108, lat/long around the camera)
  and an **aerial-perspective volume** (32×32×32 froxels). (Sizes from the paper as I recall
  them; the UE docs confirm the LUT names and the 3D "AerialPerspectiveLUT" but not the
  resolutions: [UE Sky Atmosphere docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/sky-atmosphere-component-in-unreal-engine).)
- Earth defaults (UE docs + paper): ground radius 6360 km, atmosphere 100 km, Rayleigh scale
  height 8 km, Mie anisotropy g = 0.8; the paper's coefficients (per km) are Rayleigh scattering
  (5.802, 13.558, 33.1)×10⁻³, Mie scattering 3.996×10⁻³, Mie absorption 4.4×10⁻³, ozone absorption
  (0.650, 1.881, 0.085)×10⁻³ with a tent profile centred at 25 km (values are the standard ones
  used by Bruneton 2017 and Hillaire 2020 — quoted from memory of the paper, verify before use).
- Cost: ~0.3 ms class on console GPUs for all LUTs; the mesh pass then samples two textures.
  Dynamic atmosphere parameters per frame, ground-to-space, sun below horizon, works with
  arbitrary turbidity by editing the coefficients.

### 1.5 Cheap real-time approximations

- **three.js `Sky`** ([Sky.js](https://github.com/mrdoob/three.js/blob/dev/examples/jsm/objects/Sky.js)):
  despite the class doc citing Preetham, the shader is a *single-scattering Rayleigh + Mie
  analytic* (after Simon Wallner / Blender Artists "preetham sky HDR" thread) — no Perez
  coefficients. Per-vertex it computes `βR = totalRayleigh · rayleigh` with
  `totalRayleigh = (5.80e-6, 1.356e-5, 3.03e-5)` for λ = (680, 550, 450) nm, `βM = totalMie(T) · mieCoefficient`,
  and a sun intensity `EE·max(0, 1 − exp(−(cutoffAngle − acos(sunDir.y))/steepness))` (an
  "earth shadow hack" that dims the sun below ~1.6 rad zenith). Per-fragment: optical lengths
  `sR = 8.4 km / (cos z + 0.15(93.885 − z°)^−1.253)`, `sM = 1.25 km / (...)`, extinction
  `Fex = exp(−(βR·sR + βM·sM))`, in-scatter `Lin = (sunE · (βR·phaseR + βM·phaseHG)/(βR + βM) · (1 − Fex))^1.5`
  blended toward `Fex^0.5` near the horizon, plus a 66-arcsecond-cos sun disk and a night term.
  Uniforms: `turbidity` (2), `rayleigh` (1), `mieCoefficient` (0.005), `mieDirectionalG` (0.8),
  `sunPosition`. Output is HDR, meant to go through three.js tone mapping. MIT (three.js).
- **Shadertoy "atmospheric scattering" family** (Nishita-style ray marches with 8–16 view steps
  × 4–8 light steps): physically plausible sunsets, ~1 ms at 1080p; too slow to call per mesh
  fragment but fine for a sky dome or a 32×32 sky-view LUT.
- **Gradient skies / SH ambient**: zenith, horizon, ground colours in three lerps; used as
  fallback and for mobile.

### 1.6 Comparison

| Model | Quality at 0–8° elevation | Below 0°? | Turbidity range | Per-pixel cost | Setup | Licence |
|---|---|---|---|---|---|---|
| Preetham | mediocre (washed/negative near horizon) | no | 2–6 | trivial | none | paper |
| Hosek–Wilkie | very good; explicit sunset fit, albedo | no | 1–10 | trivial (+ CPU coefficient eval) | embed 10×2×9×6×3 float table | BSD-3 |
| Bruneton 2008/2017 | excellent, physically based | yes | via coefficients | 2–3 LUT fetches | offline/at-startup precompute (seconds) | BSD-3 |
| Hillaire 2020 | excellent, production | yes | via coefficients | 2 LUT fetches | per-frame LUTs, ~0.3 ms | MIT (ref code) |
| three.js Sky (single scatter) | decent, cheap; sun colour plausible | hack | any (unphysical) | ~20 ALU | none | MIT |
| Gradient | stylised | yes | n/a | trivial | none | — |

**Recommendation:** for the WebGL relighting demo use **Hosek–Wilkie** (RGB tables from the
Shadertoy port or ArHosekSkyModel's RGB dataset) evaluated on the CPU per frame into (a) a small
sky-view cubemap / lat-long texture and (b) 9 SH coefficients for ambient, plus the **direct-sun
transmittance formula** below for `uSunColor`. For WebGPU/Metal with time-of-day animation and
blue hour, implement **Hillaire 2020** (MIT reference code is HLSL; it ports to WGSL/MSL cleanly).

---

## 2. What the mesh shader needs from the sky model

1. `uSunColor` (linear RGB radiance or irradiance scale): sun disk radiance × solid angle
   ≈ `E_sun(λ) = E_0(λ) · τ(λ, m)`. With E_0 ≈ 1361 W/m² total, the visible-band irradiance on a
   surface normal to the sun at elevation e is ≈ 1361·τ·(visible fraction) — in "sunny-16" terms
   ~100 klux at zenith, ~10 klux at 5° (file 01).
2. `uSkyAmbient` or SH: hemispherical integral of sky radiance weighted by the cosine — with a
   Hosek/Bruneton sky, integrate numerically on the CPU (e.g. 32×16 samples) each time the sun moves.
   For a cheap analytic stand-in, sample the compact function below at zenith, 4 horizon
   directions and the sun/anti-sun horizon points, and average with cosine weights.
3. The **sky dome** itself, drawn last with depth = far.
4. Optionally **aerial perspective** (transmittance + in-scatter along the view ray to each mesh
   point): Bruneton `GetSkyRadianceToPoint` / Hillaire AP volume give it exactly; the analytic
   height-fog in file 04 approximates it.

---

## 3. Compact self-contained GLSL sun + sky (documentation only)

A single-scattering analytic sky in the spirit of three.js `Sky` but with the sun's
transmittance done with Preetham/Iqbal coefficients so that `sunColor()` is consistent with the
table in file 01. HDR output in arbitrary "sun = 1 at zenith" units; multiply by exposure. GLSL ES
3.00. ~40 ALU per call; call `skyRadiance()` per sky-dome fragment and `sunTransmittance()` once
per frame (or per vertex).

```glsl
#version 300 es
precision highp float;

// ---------- atmosphere constants (three.js Sky.js / Preetham) ----------
const vec3  LAMBDA_UM     = vec3(0.680, 0.550, 0.450);                  // RGB primaries, micrometres
const vec3  BETA_R        = vec3(5.804542996261093e-6, 1.3562911419845635e-5, 3.0265902468824876e-5); // Rayleigh, 1/m at sea level (three.js totalRayleigh)
const vec3  MIE_CONST     = vec3(1.8399918514433978e14, 2.7798023919660528e14, 4.0790479543861094e14); // three.js MieConst
const float RAYLEIGH_ZENITH_LENGTH = 8.4e3;                              // m, Rayleigh scale height (three.js)
const float MIE_ZENITH_LENGTH      = 1.25e3;                             // m, Mie scale height (three.js)
const float PI = 3.141592653589793;

// Relative optical air mass, Kasten & Young 1989 (elevation e in degrees)
float airMass(float elevDeg) {
    float z = 90.0 - elevDeg;
    return 1.0 / (cos(radians(z)) + 0.50572 * pow(6.07995 + elevDeg, -1.6364));
}

// Direct-beam transmittance per RGB, Preetham appendix (Iqbal):
//   tau_r = exp(-0.008735 * lambda^-4.08 * m),  tau_a = exp(-beta * lambda^-1.3 * m),
//   beta = 0.04608*T - 0.04586   (T = turbidity)
vec3 sunTransmittance(float elevDeg, float turbidity) {
    float m    = airMass(max(elevDeg, 0.0));
    float beta = 0.04608 * turbidity - 0.04586;
    vec3 tauR  = exp(-0.008735 * pow(LAMBDA_UM, vec3(-4.08)) * m);
    vec3 tauA  = exp(-beta     * pow(LAMBDA_UM, vec3(-1.30)) * m);
    return tauR * tauA;                                                   // ozone/water/gas omitted (small in visible)
}

// Sun colour reaching the ground, relative to the extraterrestrial spectrum (assumed white here).
// Multiply by an extraterrestrial RGB (e.g. D65-ish (1.0, 0.98, 0.93)) if you want the zenith sun slightly warm.
vec3 sunColor(float elevDeg, float turbidity) {
    // smooth fade as the disk sets (geometric horizon; refraction ~0.5 deg is ignored)
    float horizonFade = smoothstep(-0.833, 0.5, elevDeg);                // -0.833 = disc centre at apparent sunset
    return sunTransmittance(elevDeg, turbidity) * horizonFade;
}

float phaseRayleigh(float cosTheta) { return 3.0 / (16.0 * PI) * (1.0 + cosTheta * cosTheta); }
float phaseHG(float cosTheta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

// Mie extinction coefficient from turbidity (three.js totalMie): c = (0.2 T) * 10e-18 ; beta_M = 0.434 c MieConst
vec3 betaMie(float turbidity, float mieCoefficient /* ~0.005 */) {
    float c = (0.2 * turbidity) * 10.0e-18;
    return 0.434 * c * MIE_CONST * mieCoefficient;
}

// Sky radiance along unit view direction `dir` (Z-up, ENU) for unit sun direction `sunDir`.
// Returns HDR linear RGB, ~1.0 at the sun's aureole, includes a soft sun disk if `withSun`.
vec3 skyRadiance(vec3 dir, vec3 sunDir, float turbidity, float mieCoefficient, float g, bool withSun) {
    float sunElevDeg = degrees(asin(clamp(sunDir.z, -1.0, 1.0)));
    vec3  sunE       = sunColor(sunElevDeg, turbidity) * 20.0;           // sun irradiance scale (arbitrary HDR units)

    // optical depth along the view ray to the top of the atmosphere (view elevation clamped at horizon)
    float viewElevDeg = degrees(asin(clamp(dir.z, 0.0, 1.0)));
    float mv  = airMass(viewElevDeg);
    float sR  = RAYLEIGH_ZENITH_LENGTH * mv;
    float sM  = MIE_ZENITH_LENGTH * mv;

    vec3 bR = BETA_R;                                                     // could scale with a "rayleigh" uniform
    vec3 bM = betaMie(turbidity, mieCoefficient);

    vec3 Fex = exp(-(bR * sR + bM * sM));                                 // extinction along the view ray

    float cosT = dot(dir, sunDir);
    vec3 betaRTheta = bR * phaseRayleigh(cosT);
    vec3 betaMTheta = bM * phaseHG(cosT, g);

    // single in-scatter, three.js form (empirical 1.5 power + horizon blend)
    vec3 Lin = pow(sunE * ((betaRTheta + betaMTheta) / (bR + bM)) * (1.0 - Fex), vec3(1.5));
    Lin *= mix(vec3(1.0),
               pow(sunE * ((betaRTheta + betaMTheta) / (bR + bM)) * Fex, vec3(0.5)),
               clamp(pow(1.0 - sunDir.z, 5.0), 0.0, 1.0));

    vec3 L0 = vec3(0.1) * Fex;                                            // faint night/space term
    vec3 color = (Lin + L0) * 0.04 + vec3(0.0, 0.0003, 0.00075);

    if (withSun) {
        const float sunAngularDiameterCos = 0.999956676946448;            // cos(66 arcsec ... three.js value, ~0.53 deg disc uses 0.99995)
        float disc = clamp((cosT - sunAngularDiameterCos) * 50000.0, 0.0, 1.0);
        color += (760.0 * disc) * min(sunE * Fex, vec3(80.0)) * 0.04;
    }
    // below-horizon ground: darken toward a ground albedo * ambient
    color = mix(color * 0.2, color, smoothstep(-0.05, 0.02, dir.z));
    return color;
}

// Cheap hemispherical ambient: cosine-weighted average of 7 sky samples. Call on CPU or per frame.
vec3 skyAmbient(vec3 sunDir, vec3 n, float turbidity, float mie, float g) {
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    vec3 dirs[7] = vec3[7](vec3(0,0,1), vec3(1,0,0.15), vec3(-1,0,0.15), vec3(0,1,0.15), vec3(0,-1,0.15),
                           normalize(vec3(sunDir.xy, 0.15)), normalize(vec3(-sunDir.xy, 0.15)));
    for (int i = 0; i < 7; ++i) {
        vec3 d = normalize(dirs[i]);
        float w = max(dot(n, d), 0.0) * (i == 0 ? 2.0 : 1.0);
        acc += skyRadiance(d, sunDir, turbidity, mie, g, false) * w;
        wsum += w;
    }
    return acc / max(wsum, 1e-4) * PI;                                    // ~ irradiance/pi scale used by Lambert
}
```

Notes:

- The Rayleigh/Mie constants, the 1.5-power in-scatter and the horizon blend are lifted from
  three.js `Sky.js` (MIT; credits there to Simon Wallner and the Blender Artists thread). They
  are **empirical**, not a solution of the RTE; they look right for 0–90° elevation and give a
  usable orange aureole for golden hour. Hosek–Wilkie or Hillaire are strictly better for the
  sky *dome*; the point of this function is that it is 60 lines and needs no tables.
- `sunTransmittance()` *is* physically motivated (Preetham/Iqbal). At T = 2 it gives the
  T = 2 column of the CCT table in file 01: (0.35, 0.16, 0.05)-ish at 5° elevation, i.e. a
  ~2,500 K colour with ~17 % of zenith luminance.
- **Sun disk colour** = `sunColor(e, T) · E_0` (the disk is the direct beam; its *radiance* is the
  irradiance divided by the disk solid angle 6.8×10⁻⁵ sr — enormous, so clamp before bloom).
- **Ambient sky colour** in shadow = `skyAmbient(sunDir, N)`; at 5° it is blue-grey with a warm
  bias toward the sun's azimuth, exactly the "cool shadows with warm edges" of real golden hour.
- Units: the whole thing is relative; set `exposure` so that a Lambertian white lit by the zenith
  sun maps to ~0.8 after tone mapping, then golden hour will come out ~1–2 EV darker, which the
  grading pass in file 05 lifts back.

---

## 4. Deriving parameters from the sky model (checklist)

| Quantity | From Hosek–Wilkie | From Bruneton/Hillaire | From the compact function |
|---|---|---|---|
| Sun disk radiance | 2013 solar radiance dataset (`arhosekskymodel_solar_radiance`) | `GetSolarRadiance()` / transmittance LUT × solar irradiance | `sunColor()` × E₀ / Ω_sun |
| Sun irradiance on the ground (for `uSunColor`) | integrate disk radiance × Ω_sun ≈ radiance × 6.8e-5 sr | `GetSunAndSkyIrradiance()` | `sunColor()` × E₀ |
| Sky ambient / SH | numeric hemisphere integral of `arhosek_tristim_skymodel_radiance` | `GetSunAndSkyIrradiance()` sky term (already cosine-integrated) | `skyAmbient()` |
| Aerial perspective | not provided (sky only) | `GetSkyRadianceToPoint()` / AP volume | height fog, file 04 |
| Turbidity input | T = 1…10 | edit Mie coefficients (T≈2 ≙ Mie scattering 3.996e-3/km; scale ~linearly) | `turbidity`, `mieCoefficient` |
| Blue hour | not supported (elev ≥ 0 only) | supported | not supported (hack: fade to a fixed blue gradient) |

---

## Sources

- Preetham, Shirley, Smits 1999 (PDF): https://courses.cs.duke.edu/fall01/cps124/resources/p91-preetham.pdf
- SIGGRAPH history page for Preetham et al.: https://history.siggraph.org/learning/a-practical-analytic-model-for-daylight-by-preetham-shirley-and-smits/
- Zotti & Wilkie, A Critical Review of the Preetham Skylight Model (slides): https://dspace5.zcu.cz/bitstream/11025/11160/4/Zotti_prezentace.pdf
- Hosek & Wilkie project page: https://cgg.mff.cuni.cz/projects/SkylightModelling/
- Hosek & Wilkie 2012 preprint: https://cgg.mff.cuni.cz/projects/SkylightModelling/HosekWilkie_SkylightModel_SIGGRAPH2012_Preprint_lowres.pdf
- ArHosekSkyModel.h (BSD-3 header, pbrt-v3 vendored copy): https://github.com/mmp/pbrt-v3/blob/master/src/ext/ArHosekSkyModel.h
- Shadertoy — Hosek-Wilkie Skylight Model (GLSL): https://www.shadertoy.com/view/wslfD7
- Kol, Analytical sky simulation (Preetham vs Hosek–Wilkie notes): https://timothykol.com/project.php?id=3
- Bruneton & Neyret 2008, Precomputed Atmospheric Scattering (HAL): https://hal.science/inria-00288758
- Bruneton 2017 re-implementation page: https://ebruneton.github.io/precomputed_atmospheric_scattering/
- Bruneton GitHub (BSD-3): https://github.com/ebruneton/precomputed_atmospheric_scattering
- three.js port of Bruneton: https://github.com/jeantimex/precomputed_atmospheric_scattering
- Hillaire 2020, CGF 39(4): https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14050
- Hillaire 2020 reference implementation (MIT): https://github.com/sebh/UnrealEngineSkyAtmosphere
- Unreal Engine Sky Atmosphere docs: https://dev.epicgames.com/documentation/en-us/unreal-engine/sky-atmosphere-component-in-unreal-engine
- three.js Sky.js: https://github.com/mrdoob/three.js/blob/dev/examples/jsm/objects/Sky.js
- Simon Wallner, atmospheric scattering (credited by three.js): http://simonwallner.at/project/atmospheric-scattering/
- Kasten & Young 1989 air mass: https://opg.optica.org/ao/abstract.cfm?uri=ao-28-22-4735
