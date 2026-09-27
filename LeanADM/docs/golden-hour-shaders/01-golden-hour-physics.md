# 01 — What physically defines "golden hour"

Research notes for the ODM golden-hour shader project. Everything here is documentation only;
no code in `/ODM` is touched. Numbers marked **[computed]** were produced with the scripts
described in the "Reproducibility" section at the end, using published formulas; numbers marked
**[measured]** or **[source]** come from the cited reference.

---

## 1. Definitions by solar elevation

"Golden hour" is not a fixed clock interval; it is a band of **solar elevation angles** (angle of
the sun's centre above the geometric horizon). Different sources draw the band slightly
differently:

| Source | Golden hour band | Blue hour band | Notes |
|---|---|---|---|
| timeanddate.com ([link](https://www.timeanddate.com/astronomy/golden-hour.html)) | −4° to +6° | −4° to −6° | Their calculator's definition; band straddles sunrise/sunset. |
| PhotoPills ([link](http://www.photopills.com/articles/mastering-golden-hour-blue-hour-magic-hours-and-twilights)) | −4° to +6° | −4° to −6° | Widely used by photographers; PetaPixel repeats the same numbers ([link](https://petapixel.com/2014/06/11/understanding-golden-hour-blue-hour-twilights/)). |
| Wikipedia, *Golden hour (photography)* ([link](https://en.wikipedia.org/wiki/Golden_hour_(photography))) | "roughly 0° to 6°" while the sun is up; also notes the sun is at ~10–12° one hour after sunrise in Los Angeles | "significant depth below the horizon" | Also gives the common "first/last hour of daylight" heuristic. |
| Wikipedia, *Blue hour* ([link](https://en.wikipedia.org/wiki/Blue_hour)) | — | strongest at −4° to −8° | Attributes the blue to ozone Chappuis absorption, not Rayleigh alone. |

Twilight phases, for reference (PhotoPills, same link): **civil** 0° to −6°, **nautical** −6° to
−12°, **astronomical** −12° to −18°.

**Recommendation for the shader work:** treat the *shader's* golden hour as **+2° to +8°** solar
elevation (the sun is up, so there is a direct beam that casts long shadows), with a "sunset/sunrise
edge" preset at +0.5° to +2°, and a separate *blue hour* preset for −4° to −6° (no direct beam, only
a blue-dominated sky). The task brief's "~6° below to ~6° above" is the union of golden and blue hour
in the timeanddate/PhotoPills conventions.

### Duration depends on latitude and season

The sun's elevation rate near the horizon is roughly `0.25° per minute × cos(latitude)` (the
sun moves 15°/h along its diurnal circle; the vertical component shrinks with latitude and with
the sun's declination). timeanddate's page ([link](https://www.timeanddate.com/astronomy/golden-hour.html))
gives ~24 minutes at the equator and 2 h or more at 60° latitude in summer for the −4°…+6°
band. **[computed]** For San Francisco (37.77° N) on 2026-09-13 the pysolar scan in
`02-sun-position-from-metadata.md` shows the sun dropping from +5.6° to +0.1° in 30 minutes
(01:50 → 02:20 UTC), i.e. ~0.18°/min.

---

## 2. Why the light is warm: air mass + Rayleigh scattering

### 2.1 Air mass

The direct beam's path length through the atmosphere, relative to the zenith path, is the
**relative optical air mass** `m`. The plane-parallel approximation `m = sec(z)` diverges at the
horizon; the standard correction is Kasten & Young (1989), *Applied Optics* 28(22):4735
([abstract](https://opg.optica.org/ao/abstract.cfm?uri=ao-28-22-4735)):

```
m(z) = 1 / ( cos z + 0.50572 · (96.07995° − z)^(−1.6364) )      z = zenith angle in degrees
```

(equivalently, with elevation `e = 90° − z`: `1 / (cos z + 0.50572 · (6.07995 + e)^−1.6364)`).
Wikipedia's *Air mass (astronomy)* article ([link](https://en.wikipedia.org/wiki/Air_mass_(astronomy)))
notes this gives ~38 at the horizon. **[computed]** values from the formula:

| Elevation | 0° | 1° | 2° | 3° | 5° | 6° | 10° | 15° | 20° | 30° | 45° | 60° | 90° |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Air mass m | 37.9 | 26.3 | 19.4 | 15.2 | 10.3 | 8.8 | 5.6 | 3.8 | 2.9 | 2.0 | 1.41 | 1.15 | 1.00 |

So at 5° elevation the direct beam traverses **~10×** more atmosphere than at zenith. Preetham,
Shirley & Smits (1999) use the older Kasten form `m = 1/(cos θs + 0.15·(93.885 − θs)^−1.253)`
in their appendix ([paper PDF](https://courses.cs.duke.edu/fall01/cps124/resources/p91-preetham.pdf)),
which is what the three.js Sky shader also hard-codes ([Sky.js](https://github.com/mrdoob/three.js/blob/dev/examples/jsm/objects/Sky.js)).

### 2.2 Rayleigh vs Mie

- **Rayleigh scattering** (molecules, particle ≪ λ): scattered intensity ∝ λ⁻⁴. Blue at 400 nm
  scatters (700/400)⁴ ≈ **9.4×** more than red at 700 nm
  ([Wikipedia: Rayleigh scattering](https://en.wikipedia.org/wiki/Rayleigh_scattering)). Over a
  long path the blue end is removed from the direct beam and appears in the sky instead — hence a
  red-orange sun and a blue sky. NOAA/SPC's Corfidi essay ([link](https://www.spc.noaa.gov/publications/corfidi/sunset/))
  notes violet is scattered "three to four times more effectively" than the longer visible
  wavelengths.
- **Mie scattering** (aerosols, particle ≳ λ): only weakly wavelength dependent, strongly
  forward-peaked. Aerosol haze therefore **whitens/greys** the sky and **mutes** sunset colour;
  clean air gives the saturated oranges/reds, dust and pollution give "pale yellows and pinks"
  (Corfidi, same link; also NASA GISS aerosol notes [link](https://www.giss.nasa.gov/edu/icp/research/ppa/1997/reyes/)).
  Volcanic *stratospheric* aerosols are the exception that produces vivid afterglows.

In shader terms: Rayleigh gives you the hue shift of the sun and the blue sky; Mie gives you the
bright glow (aureole) around the sun, the milky horizon, and the "warm haze" of aerial perspective.
The Henyey–Greenstein phase function with asymmetry `g ≈ 0.76–0.8` is the usual real-time Mie
stand-in (three.js uses `mieDirectionalG = 0.8`; Unreal's Sky Atmosphere default anisotropy is 0.8,
per the [UE docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/sky-atmosphere-component-in-unreal-engine)).

### 2.3 Turbidity

Preetham et al. define turbidity `T` as the ratio of the vertical optical thickness of the
*haze + molecules* atmosphere to that of *molecules alone* (`T = (τm + τh)/τm`), quoting values at
550 nm ([paper](https://courses.cs.duke.edu/fall01/cps124/resources/p91-preetham.pdf), §2.1 and Fig. 3).
Their Figure 3 (meteorological range vs turbidity, after McCartney 1976) reads approximately:

| Turbidity T | ~1 | ~2 | ~3 | ~4–5 | ~8 | ~16 | ~32–64 |
|---|---|---|---|---|---|---|---|
| Sky | pure air | exceptionally clear | very clear | clear | light haze | haze | thin fog |

(read from the figure's log axes; treat as ±1 step). Hosek & Wilkie note the Preetham fit is only
valid for T ≈ 2–6, and their own model covers T = 1–10
([Hosek–Wilkie 2012 preprint](https://cgg.mff.cuni.cz/projects/SkylightModelling/HosekWilkie_SkylightModel_SIGGRAPH2012_Preprint_lowres.pdf), §2.3.1, §5).

Preetham's appendix (from Iqbal) gives the direct-beam transmittance pieces that we use below:

```
τ_rayleigh(λ) = exp( −0.008735 · λ^−4.08 · m )                 λ in µm
τ_aerosol(λ)  = exp( −β · λ^−α · m ),  α = 1.3,  β = 0.04608·T − 0.04586
(+ ozone, mixed gas, water vapour terms, minor in the visible)
```

---

## 3. Colour temperature of sunlight vs elevation

### 3.1 Published anchor values

- Wikipedia *Golden hour*: ≈ **2,000 K** shortly after sunrise/before sunset, ≈ **3,500 K**
  during golden hour, ≈ **5,500 K** at midday ([link](https://en.wikipedia.org/wiki/Golden_hour_(photography))).
- Wikipedia *Color temperature* table: candle flame / sunset / sunrise **1,850 K**; horizon
  daylight **5,000 K**; vertical daylight **5,500–6,000 K**; overcast **6,500 K**; clear blue
  poleward sky **15,000–27,000 K**; D65 = 6,504 K ([link](https://en.wikipedia.org/wiki/Color_temperature)).
  ("Sunset 1,850 K" there refers to the disk *at* the horizon.)
- rp-photonics: ~5,900 K outside the atmosphere, ~5,600 K with the sun overhead
  ([link](https://www.rp-photonics.com/color_temperature.html)).
- Peyvandi, Hernández-Andrés et al. 2016 (JOSA A 33(6):1049) simulated 5.6 M spectra with SBDART:
  *hemispheric daylight* (sun + sky on a horizontal plane) has an interquartile CCT of
  **5,712–7,757 K** and a minimum of about **3,600 K** with aerosols present
  ([abstract](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-33-6-1049)). Note this is
  global irradiance, not the direct beam — the direct beam alone goes much lower.
- Hernández-Andrés et al. 2001 (JOSA A 18:1325): 2,600 Granada daylight spectra, **5,700 K**
  typifies daylight; chromaticities lie far above the Planckian locus at high CCT
  ([abstract](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-18-6-1325)).

I could not retrieve a peer-reviewed table of *direct-sun* CCT vs elevation (ResearchGate and
Springer blocked the fetches). The table below is therefore **computed** from the Preetham/Iqbal
attenuation above applied to a 5,778 K blackbody, integrated with the CIE 1931 2° observer and
converted to CCT with the Hernández-Andrés 1999 xy→CCT formula (`colour-science`). It reproduces
the Wikipedia anchors to within ~10–15 % but reads slightly *low* at high elevations (5.1 kK vs.
the measured ~5.5–5.8 kK) because the crude blackbody/attenuation model over-reddens. Use the
**shape**, and rescale to your chosen zenith value if you want absolute numbers. Below ~1,500 K
the CCT concept stops being meaningful (the xy point leaves the range of the fit; the sun is a
deep red disk that is mostly extinguished anyway).

### 3.2 Table: elevation → air mass → CCT → direct/diffuse

Direct/diffuse ratio = (direct normal irradiance × sin e) / diffuse horizontal irradiance, i.e.
the ratio of *sun-on-horizontal* to *sky-on-horizontal*. **[computed]** with pvlib 0.15's
Ineichen clear-sky model (Linke turbidity 3, sea level) and Bird (AOD500 = 0.1, PW = 1.5 cm)
at 37.8° N on 2026-09-13; see the "Reproducibility" section. ODM's own `opendm/dls.py`
comments assume "for clear sky, I_direct/I_diffuse ~ 6" (for the irradiance sensor) — consistent
with the 45° row.

| Sun elev. | Air mass | CCT direct sun, T=2 [computed] | CCT, T=4 [computed] | Direct-beam luminance rel. to zenith (T=2) | Direct-horiz / Diffuse (Ineichen TL3) | Same (Bird) | Look |
|---|---|---|---|---|---|---|---|
| 0.5° | 31 | ~700 K (out of range) | — | 0.005 | ≈0 | ≈0 | disk on the horizon, dark red; sky lights the scene |
| 1° | 26 | ~1,000 K (out of range) | — | 0.010 | ~0.2 | ~0.5 | |
| 2° | 19.4 | ~1,500 K | ~700 K (o.r.) | 0.03 | 0.39 | 0.83 | deep orange-red; shadows softer than sun |
| 3° | 15.2 | ~1,900 K | ~1,200 K | 0.07 | ~0.6 | ~0.9 | classic "sunset" light |
| 5° | 10.3 | ~2,500 K | ~1,900 K | 0.17 | 0.99 | 0.92 | **golden hour core**: sun ≈ sky on horizontal |
| 8° | 6.9 | ~3,100 K | ~2,500 K | 0.32 | 1.6 | 1.3 | golden, long shadows |
| 10° | 5.6 | ~3,400 K | ~2,900 K | 0.41 | ~2.0 | ~1.6 | late golden hour |
| 15° | 3.8 | ~3,900 K | ~3,400 K | 0.58 | 3.0 | 2.2 | warm afternoon |
| 20° | 2.9 | ~4,200 K | ~3,800 K | 0.69 | 3.8 | 2.9 | |
| 30° | 2.0 | ~4,600 K | ~4,200 K | 0.82 | 5.2 | 4.1 | |
| 45° | 1.41 | ~4,900 K | ~4,500 K | 0.92 | 6.8 | 5.7 | ODM's "~6" |
| 90° | 1.00 | ~5,100 K (model) / 5,500–5,800 K measured | ~4,800 K | 1.00 | — | — | |

Take-aways for the shader:

1. Between 2° and 8° the direct beam's **luminance drops by an order of magnitude** while its
   **CCT falls from ~3,000 K to ~1,500–2,000 K**. Exposure must rise (or the tonemapper must
   compress less) to keep the ground readable, which is exactly what a camera's auto-exposure does.
2. The sky's horizontal irradiance falls much more slowly than the sun's, so the **key-to-fill
   ratio collapses** from ~6:1 at midday to ~1:1 at 5°. That is the physical reason golden-hour
   light is "low contrast" even though shadows are long and crisp.
3. Because the sky remains blue-ish while the sun goes orange, lit surfaces and shadowed surfaces
   have **opposite colour casts** (warm/cool) — the basis of the split-toning look in
   `05-color-grading-and-tonemapping.md`.

### 3.3 Absolute light levels

Wikipedia's exposure-value table ([link](https://en.wikipedia.org/wiki/Exposure_value)) gives
EV100 = 15 for a sunny scene, 12–14 just before sunset, 12 at sunset, 9–11 just after sunset,
and the incident-light relation `E = 2.5 · 2^EV lux`. That is ≈ 82,000 lux (EV 15) → ≈ 10,000 lux
(EV 12) → 1,300–5,000 lux (EV 9–11): the scene gets **8× darker** from midday to sunset and
another 2–8× through civil twilight. If the renderer uses physical units, expose for
EV ≈ 12–13 in golden hour.

---

## 4. Shadow length

A vertical object of height `h` casts a ground shadow of length

```
L = h / tan(e)              e = solar elevation
```

(the shadow points along the *anti-solar* azimuth, i.e. away from the sun). **[computed]**:

| e | 2° | 5° | 10° | 20° | 30° | 45° | 60° |
|---|---|---|---|---|---|---|---|
| L / h | 28.6 | 11.4 | 5.7 | 2.75 | 1.73 | 1.00 | 0.58 |

So at 5° a 10 m tree throws a 114 m shadow, and a 1 m tall wall shadows 11 m of ground. On a DSM
of GSD 5 cm that is ~230 texels of shadow per metre of height — the ray-marched DSM shadow pass in
`04-glsl-relighting-shaders.md` must be able to walk that far (use a coarse-to-fine or mip walk,
not a fixed 64-step loop).

Also relevant: with elevation `e`, the *penumbra* width from the sun's 0.53° angular diameter is
`w ≈ d · tan(0.53°) ≈ d/108` where `d` is the occluder–receiver distance along the ray — for a
114 m shadow tip that is ~1 m of soft edge, which is why long golden-hour shadows look sharp at
the base and soft at the tip (the "contact hardening" note in file 04).

---

## 5. Blue hour vs golden hour

| | Golden hour (≈ +2° … +8°) | Blue hour (≈ −4° … −6°/−8°) |
|---|---|---|
| Direct beam | present, orange, low (CCT 1.5–3.5 kK) | none at ground level |
| Sky | blue zenith, warm bright horizon around sun, pink anti-solar "Belt of Venus" possible | deep saturated blue; red is *absorbed* by ozone (Chappuis band, ~500–700 nm) as the light grazes the stratosphere ([Wikipedia: Blue hour](https://en.wikipedia.org/wiki/Blue_hour)) |
| Shadows | very long, warm-lit / blue-filled | none (only ambient occlusion) |
| Contrast | low key-to-fill (~1:1 – 2:1) | very low; essentially a single large blue area light |
| Typical CCT of illumination | 2–4 kK on lit surfaces, 8–12 kK in shadow (sky-lit) | ~10–15 kK overall (sky only); search results quote 10,000 K+ for twilight sky but I found no rigorous table — treat as approximate |
| EV100 | ~12–14 | ~7–10 |

For the shader this means the blue hour preset is simply: `sunIntensity = 0`, sky colour from the
sky model evaluated at a *negative* elevation (Hosek–Wilkie does not go below 0°, see file 03;
use a hand-tuned gradient or Hillaire/Bruneton), and much stronger exposure.

---

## 6. Haze and turbidity in the look

| Effect | Clean air (T ≈ 2) | Hazy (T ≈ 5–6) |
|---|---|---|
| Sun colour at 5° | orange, still bright | deep red, dim; disk may be visible to the naked eye |
| Aureole around sun | small | large, bright, whitish |
| Horizon sky | light blue/white, sharp horizon | milky yellow-grey, horizon "dissolves" |
| Aerial perspective | distant objects shift to blue | distant objects shift to warm grey (haze lit by orange sun) |
| Colour saturation of sunset | high | pastel (Corfidi) |
| Direct/diffuse ratio | higher | lower — more light arrives as scattered "fill" |

In the shader this is one parameter (`turbidity` / `mieCoefficient`) that (a) raises Mie extinction
and inscatter, (b) reddens the sun further via `β(T)`, and (c) raises the fog/aerial-perspective
density in the mesh pass. Preetham's "half hour before sunset, turbidity 6" figure is the canonical
reference image for a hazy golden hour ([paper](https://courses.cs.duke.edu/fall01/cps124/resources/p91-preetham.pdf), Fig. 7).

---

## 7. Summary of the physical recipe (for the shader authors)

1. Compute solar elevation `e` and azimuth from the ODM photos' GPS + time (file 02).
2. Air mass `m(e)` (Kasten–Young). Direct-beam transmittance per RGB channel via Rayleigh
   `exp(−0.008735 λ^−4.08 m)` × aerosol `exp(−β λ^−1.3 m)` with λ = {0.68, 0.55, 0.45} µm (the
   three.js primaries) — this alone gives a very plausible sun colour and intensity (file 03).
3. Sky radiance from an analytic model (Hosek–Wilkie for quality, or the compact single-scatter
   function in file 03); integrate over the hemisphere for ambient.
4. Light the mesh: Lambert/Oren–Nayar with long shadows, sky ambient in shadow (cool), warm
   height-fog/aerial perspective whose colour is the in-scattered sun (file 04).
5. Expose for EV ≈ 12–13, tone map, split-tone, bloom the sun disk (file 05).

---

## Reproducibility

Both computed tables were produced with a throw-away virtualenv (`pysolar`, `pvlib 0.15.2`,
`colour-science`) — nothing in the ODM tree was modified. The scripts, verbatim:

```python
# air mass (Kasten–Young 1989) and shadow length
import math
for e in [0,1,2,3,5,6,10,15,20,30,45,60,90]:
    z = 90 - e
    m = 1/(math.cos(math.radians(z)) + 0.50572*(6.07995 + e)**-1.6364)
    print(e, round(m, 2), None if e == 0 else round(1/math.tan(math.radians(e)), 2))
```

```python
# direct-sun CCT vs elevation, Preetham/Iqbal attenuation of a 5778 K blackbody
import numpy as np, colour, math
lam = np.arange(380, 781, 5)
sun = colour.sd_blackbody(5778, colour.SpectralShape(380, 780, 5)).values
cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy().align(colour.SpectralShape(380, 780, 5))
def cct_for(elev_deg, T):
    zs = 90 - elev_deg
    m = 1.0/(math.cos(math.radians(zs)) + 0.15*(93.885 - zs)**-1.253)
    lam_um = lam/1000.0
    beta = 0.04608*T - 0.04586
    tau = np.exp(-0.008735*lam_um**-4.08*m) * np.exp(-beta*lam_um**-1.3*m)
    XYZ = colour.sd_to_XYZ(colour.SpectralDistribution(dict(zip(lam, sun*tau))), cmfs)
    return colour.xy_to_CCT(colour.XYZ_to_xy(XYZ), method='Hernandez 1999'), XYZ[1]
```

```python
# direct/diffuse ratio vs elevation with pvlib clear-sky models
import pvlib, pandas as pd, math
times = pd.date_range('2026-09-13', '2026-09-14', freq='1min', tz='UTC')
loc = pvlib.location.Location(37.7749, -122.4194, altitude=10)
sp = loc.get_solarposition(times)
am = pvlib.atmosphere.get_relative_airmass(sp['apparent_zenith'])
cs = pvlib.clearsky.ineichen(sp['apparent_zenith'], am, linke_turbidity=3.0, altitude=10, dni_extra=1361)
el = 90 - sp['apparent_zenith']
# ratio = cs['dni']*sin(el)/cs['dhi'] sampled at el ≈ 2,5,8,15,20,30,45
```

---

## Sources

- Wikipedia — Golden hour (photography): https://en.wikipedia.org/wiki/Golden_hour_(photography)
- Wikipedia — Blue hour: https://en.wikipedia.org/wiki/Blue_hour
- Wikipedia — Color temperature: https://en.wikipedia.org/wiki/Color_temperature
- Wikipedia — Air mass (astronomy): https://en.wikipedia.org/wiki/Air_mass_(astronomy)
- Wikipedia — Rayleigh scattering: https://en.wikipedia.org/wiki/Rayleigh_scattering
- Wikipedia — Exposure value: https://en.wikipedia.org/wiki/Exposure_value
- timeanddate — Golden hour: https://www.timeanddate.com/astronomy/golden-hour.html
- PhotoPills — Mastering golden hour, blue hour and twilights: http://www.photopills.com/articles/mastering-golden-hour-blue-hour-magic-hours-and-twilights
- PetaPixel — Understanding golden hour, blue hour and twilights: https://petapixel.com/2014/06/11/understanding-golden-hour-blue-hour-twilights/
- Kasten & Young 1989, Revised optical air mass tables and approximation formula, Appl. Opt. 28(22): https://opg.optica.org/ao/abstract.cfm?uri=ao-28-22-4735
- Preetham, Shirley, Smits 1999, A Practical Analytic Model for Daylight (PDF): https://courses.cs.duke.edu/fall01/cps124/resources/p91-preetham.pdf
- Hosek & Wilkie 2012, An Analytic Model for Full Spectral Sky-Dome Radiance (preprint PDF): https://cgg.mff.cuni.cz/projects/SkylightModelling/HosekWilkie_SkylightModel_SIGGRAPH2012_Preprint_lowres.pdf
- Corfidi (NOAA/SPC) — The colors of twilight and sunset: https://www.spc.noaa.gov/publications/corfidi/sunset/
- NASA GISS — Aerosols in the atmosphere: https://www.giss.nasa.gov/edu/icp/research/ppa/1997/reyes/
- rp-photonics — Color temperature: https://www.rp-photonics.com/color_temperature.html
- Peyvandi et al. 2016, Colorimetric analysis of outdoor illumination across varieties of atmospheric conditions, JOSA A 33(6):1049: https://opg.optica.org/josaa/abstract.cfm?uri=josaa-33-6-1049
- Hernández-Andrés et al. 2001, Color and spectral analysis of daylight in southern Europe, JOSA A 18:1325: https://opg.optica.org/josaa/abstract.cfm?uri=josaa-18-6-1325
- Spitschan et al. 2016, Variation of outdoor illumination as a function of solar elevation and light pollution, Sci. Rep.: https://pmc.ncbi.nlm.nih.gov/articles/PMC4895134/
- Unreal Engine — Sky Atmosphere component: https://dev.epicgames.com/documentation/en-us/unreal-engine/sky-atmosphere-component-in-unreal-engine
- three.js Sky.js source: https://github.com/mrdoob/three.js/blob/dev/examples/jsm/objects/Sky.js
- pvlib clear-sky models (Ineichen, Bird): https://pvlib-python.readthedocs.io/
- ODM `opendm/dls.py` (local clone, "I_direct/I_diffuse ~ 6" comment)
