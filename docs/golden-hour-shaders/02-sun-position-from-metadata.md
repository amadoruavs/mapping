# 02 — Sun direction from ODM photo metadata

Goal: turn the EXIF GPS position + capture time that every ODM input photo carries into a
**sun direction vector in the frame of the ODM outputs**, and also *synthesise* a golden-hour
sun for the same site (same date, different time) so the shader can relight the scene as if it
had been flown at +5° elevation. Documentation only — no ODM code is modified.

---

## 1. What ODM already does (read-only survey of `/ODM` 3.6.2)

| Where | What | Notes for us |
|---|---|---|
| `opendm/photo.py` (lines ~271–288) | Parses `EXIF DateTimeOriginal` (+ `SubSecTime`) with `datetime.strptime(..., "%Y:%m:%d %H:%M:%S")` and then **`replace(tzinfo=timezone.utc)`**; stores `photo.utc_time` in **milliseconds** since epoch. | **Caveat:** EXIF `DateTimeOriginal` is the *camera's local wall-clock time* with no zone. ODM labels it UTC as-is. If the drone's clock was set to local time (DJI defaults to local time), ODM's "UTC" is off by the local UTC offset — enough to move the sun by up to 7 h × 15°/h. See §2.3. |
| `opendm/photo.py` `find_mean_utc_time(photos)` | Mean of `utc_time/1000` across photos. | Good single timestamp for a whole dataset. |
| `opendm/dls.py` `compute_sun_angle(position, pose, utc_datetime, sensor_orientation)` | Calls `pysolar.get_altitude(lat, lon, utc)` and `get_azimuth(...)`, converts to radians, wraps azimuth to [0, 2π), and builds a **NED** sun vector `ned_from_pysolar(az, alt) = (cos az·cos alt, sin az·cos alt, −sin alt)`. | The legacy branch `azimuth = 180 - pysolar.GetAzimuth(...)` exists for pysolar 0.6's old south-based convention; modern pysolar (≥0.7) returns north-based clockwise azimuth, so no correction is applied. |
| `opendm/multispectral.py` (line ~141) | Uses `dls.compute_sun_angle([photo.latitude, photo.longitude], ...)` for irradiance normalisation. | Confirms `photo.latitude/longitude` are decimal degrees WGS84. |
| `opendm/location.py` `extract_utm_coords()` | Picks UTM zone/hemisphere from the **first** photo's lon/lat (`zone = floor((lon+180)/6) % 60 + 1`), converts every photo to UTM with pyproj, averages E/N, floors to integers, and writes `coords.txt`. | Header + offset format below. |
| `odm_georeferencing/coords.txt` | Line 1: `WGS84 UTM <zone><N|S>` ; line 2: `<east_offset> <north_offset>` (integers, metres); following lines: per-photo `(E−dx, N−dy, alt)`. Parsed by `ODM_GeoRef.FromCoordsFile` in `opendm/types.py`. | **The textured OBJ/GLB vertices are UTM easting/northing minus this offset, Z = altitude in metres, Z-up.** |
| `opendm/gltf.py` `obj2glb(..., rtc=reconstruction.get_proj_offset())` | Writes the GLB with the same vertex array as the OBJ (no axis swap) plus a `CESIUM_RTC` extension `{center: [east_offset, north_offset, 0]}`. | So the GLB is **Z-up in a local UTM frame**, *not* the glTF-standard Y-up. Viewers that assume Y-up (three.js `GLTFLoader`) will show the model lying on its side unless rotated −90° about X. |
| `opendm/ogctiles.py` | Runs `Obj2Tiles ... --lat --lon --alt` from `reference_lla.json`. | 3D Tiles get an ENU→ECEF placement from that lat/lon; the tile content stays local ENU. |
| `stages/mvstex.py` | Texturing with `toneMapping: 'none'`. | Baked sunlight of the original flight is in the albedo — see §6. |

---

## 2. pysolar (already an ODM dependency)

### 2.1 API (verified against `pysolar/solar.py` on master and the 0.8rc docs)

```python
from pysolar import solar
solar.get_altitude(latitude_deg, longitude_deg, when, elevation=0,
                   temperature=288.15, pressure=101325.0)   # -> degrees above horizon, refraction-corrected
solar.get_azimuth(latitude_deg, longitude_deg, when, elevation=0) # -> degrees, 0 = North, clockwise (E = 90)
solar.get_altitude_fast(lat, lon, when); solar.get_azimuth_fast(lat, lon, when)  # spherical-trig approximations
from pysolar import radiation
radiation.get_radiation_direct(when, altitude_deg)          # W/m^2, clear-sky direct normal (not zero at night!)
```

- **Azimuth convention** (docs, [pysolar.readthedocs.io](https://pysolar.readthedocs.io/en/latest/)):
  "Azimuth is reckoned with zero corresponding to north. Positive azimuth estimates correspond to
  estimates east of north; negative estimates, or estimates larger than 180 are west of north."
  The source (`get_topocentric_azimuth_angle`, "West is negative, East is positive, Masters p. 395")
  returns values that ODM wraps with `% (2π)`. Always wrap to [0, 360).
- **Altitude** = topocentric elevation + atmospheric refraction correction (pressure/temperature
  parameters), i.e. *apparent* elevation. For shading you generally want apparent elevation near
  the horizon (that is where the sun *looks*); refraction adds ~0.5° at the horizon.
- **Timezone-aware datetimes are mandatory.** A naive datetime raises
  `pysolar.tzinfo_check.NoTimeZoneInfoError: ... should be made timezone-aware` (verified with
  pysolar from PyPI, Sept 2026). The docs example uses `tzinfo=datetime.timezone.utc`.
- pysolar prints a *leap seconds* warning for dates beyond its table; ODM silences it with
  `warnings.simplefilter("ignore")` in `dls.py`. Harmless for our purposes (sub-arc-second effect).
- pysolar implements the NREL SPA (Reda & Andreas) steps; the docs example gives
  `get_altitude(42.206, -71.382, 2007-02-18 15:13:01.130320 UTC) = 30.914…°` and
  `get_azimuth(...) = 149.248…°`, which you can use as a unit-test vector.

### 2.2 NREL SPA in one paragraph

The Solar Position Algorithm (Reda & Andreas, NREL/TP-560-34302, *Solar Energy* 2004, corrigendum
2007) computes the sun's topocentric zenith and azimuth with a stated uncertainty of **±0.0003°**
for years **−2000 to 6000** ([OSTI record](https://www.osti.gov/biblio/15003974)). Steps: Julian
Day/Ephemeris Day → Earth heliocentric longitude/latitude/radius (VSOP87 series) → geocentric
sun position → nutation (Δψ, Δε) and true obliquity → apparent sun longitude → apparent sidereal
time → geocentric right ascension/declination → observer local hour angle → **topocentric**
correction for parallax (needs observer lat/lon/elevation) → topocentric zenith with **refraction**
(needs pressure, temperature) → azimuth measured eastward from north. Inputs therefore are: UTC
time, ΔT (TT−UT1, ~69 s today), latitude, longitude, elevation, pressure, temperature. pvlib's
`solarposition.spa_python` exposes exactly these and returns `apparent_zenith, zenith,
apparent_elevation, elevation, azimuth, equation_of_time`
([pvlib docs](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.spa_python.html)).
pysolar's `get_altitude/get_azimuth` are the same pipeline with defaults.

### 2.3 Time zone pitfalls (the thing most likely to be wrong)

1. `EXIF DateTimeOriginal` has **no time zone**; EXIF 2.31 added `OffsetTimeOriginal`
   (e.g. `"-07:00"`) but many drones do not write it. (From the EXIF 2.31 spec as I remember it;
   not re-verified here.)
2. `EXIF GPSTimeStamp` + `GPSDateStamp` **are UTC** (from the GNSS receiver) when present — the
   most reliable source. DJI writes them on most models; check with `exiftool -G1 -a -s`.
3. Otherwise derive the zone from lat/lon (`timezonefinder` + `zoneinfo`) and localise
   `DateTimeOriginal` before converting to UTC.
4. Sanity check: the sun must be above the horizon at capture time for a normal daytime flight.
   If `get_altitude()` returns a negative number for a daylight flight, the timestamp is in local
   time being treated as UTC (ODM's default behaviour).

```python
# Documentation-only example: robust UTC capture time for one photo
import datetime as dt
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder   # pip install timezonefinder

def capture_time_utc(tags, lat, lon):
    # 1) Prefer GPS date/time (UTC)
    if 'GPS GPSDateStamp' in tags and 'GPS GPSTimeStamp' in tags:
        y, m, d = map(int, str(tags['GPS GPSDateStamp'].values).split(':'))
        h, mi, s = [float(v.num) / float(v.den) for v in tags['GPS GPSTimeStamp'].values]
        return dt.datetime(y, m, d, int(h), int(mi), int(s), tzinfo=dt.timezone.utc)
    local = dt.datetime.strptime(str(tags['EXIF DateTimeOriginal'].values), "%Y:%m:%d %H:%M:%S")
    # 2) EXIF 2.31 offset if present
    if 'EXIF OffsetTimeOriginal' in tags:
        off = str(tags['EXIF OffsetTimeOriginal'].values)      # "+02:00"
        sign = 1 if off[0] == '+' else -1
        hh, mm = int(off[1:3]), int(off[4:6])
        tz = dt.timezone(sign * dt.timedelta(hours=hh, minutes=mm))
        return local.replace(tzinfo=tz).astimezone(dt.timezone.utc)
    # 3) Zone from position
    zone = TimezoneFinder().timezone_at(lat=lat, lng=lon) or 'UTC'
    return local.replace(tzinfo=ZoneInfo(zone)).astimezone(dt.timezone.utc)
```

---

## 3. From (elevation, azimuth) to a direction vector

Let `e` = elevation (radians, from `get_altitude`), `a` = azimuth (radians, clockwise from north,
from `get_azimuth`). The **unit vector pointing from the scene toward the sun** is

```
ENU (x=East, y=North, z=Up):        S = ( cos e · sin a ,  cos e · cos a ,  sin e )
NED (x=North, y=East, z=Down):      S = ( cos e · cos a ,  cos e · sin a , −sin e )      (= ODM dls.ned_from_pysolar)
```

(Wikipedia's *Solar azimuth angle* page gives the same ENU components and `azimuth = atan2(S_x, S_y)`
for the north-clockwise convention: [link](https://en.wikipedia.org/wiki/Solar_azimuth_angle).)

Engine frames:

| Frame | Axes | Sun vector from ENU `(E, N, U)` |
|---|---|---|
| ODM OBJ / GLB (local UTM, Z-up) | x = easting, y = northing, z = up | `(E, N, U)` — identity |
| glTF standard / three.js / Babylon (Y-up, right-handed; glTF spec: +Y up, +Z front, metres — [spec](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)) | x = east, y = up, z = **south** (−north) | `(E, U, −N)` |
| Cesium 3D Tiles / CesiumJS internal (Z-up ENU per tile, ECEF globally; Cesium rotates Y-up glTF to Z-up: [Cesium Transforms](https://cesium.com/learn/cesiumjs/ref-doc/Transforms.html)) | ENU at tile origin | `(E, N, U)` then `eastNorthUpToFixedFrame` if you light in ECEF |
| Unreal (Z-up, left-handed, X forward) | x = north?, y = east, z = up (project-dependent) | `(N, E, U)` typically |

A Y-up conversion of the whole ODM model is `M = rotX(−90°)`, which maps `(x, y, z)_zup → (x, z, −y)_yup`
— apply the *same* rotation to the sun vector, or just use the table above.

**Light direction vs sun direction:** most shaders want `L = normalize(sunPos − p)` = the vector
*toward* the sun (what the formulas above give). Some engines (three.js `DirectionalLight`) define
the light by a position and target; set `light.position = S * bigDistance`.

### 3.1 Grid convergence (UTM "north" ≠ true north)

pysolar azimuth is relative to **true** north; ODM's y axis is **UTM grid north**. They differ by
the grid convergence angle, approximately `γ ≈ (λ − λ0) · sin φ` (λ0 = zone central meridian;
up to ≈ ±3° at zone edges, 0 on the central meridian). Rotate the ENU sun vector about Z by `−γ`
(or ask pyproj: `Proj(...).get_factors(lon, lat).meridian_convergence`). For golden-hour looks a
3° azimuth error is invisible, but for matching *existing* baked shadows it matters. (Formula is
the standard first-order approximation; verify against pyproj if precision is required.)

### 3.2 GLSL / WGSL snippets (documentation only)

```glsl
// GLSL ES 3.00 — sun direction from elevation/azimuth (radians), ENU/Z-up
vec3 sunDirENU(float elev, float azim) {
    float ce = cos(elev);
    return vec3(ce * sin(azim), ce * cos(azim), sin(elev));   // (E, N, U)
}
vec3 sunDirYUp(float elev, float azim) {                       // glTF / three.js
    vec3 s = sunDirENU(elev, azim);
    return vec3(s.x, s.z, -s.y);                               // (E, U, -N)
}
```

```wgsl
fn sunDirENU(elev: f32, azim: f32) -> vec3<f32> {
    let ce = cos(elev);
    return vec3<f32>(ce * sin(azim), ce * cos(azim), sin(elev));
}
```

In practice compute the vector once on the CPU and upload it as a uniform (`uSunDir`), together
with `uSunColor` (file 03) — the trig in the shader is only useful for animated time-of-day.

---

## 4. Synthesising a golden-hour time

Given the site's mean lat/lon (ODM: average of photo positions, as `location.py` does) and the
capture **date**, find the instant when the apparent elevation equals a target (e.g. +5°) in the
evening (or morning). Elevation is monotonic between solar noon and sunset, so a bisection on
`get_altitude` is enough:

```python
# Documentation-only. Solve for the evening instant when solar elevation == target_deg.
import datetime as dt
from pysolar import solar

def solar_noon_utc(lat, lon, date_utc):
    # coarse: scan the day at 1-min steps for max altitude
    best = None
    for m in range(0, 24 * 60):
        t = dt.datetime.combine(date_utc, dt.time(), tzinfo=dt.timezone.utc) + dt.timedelta(minutes=m)
        a = solar.get_altitude(lat, lon, t)
        if best is None or a > best[1]:
            best = (t, a)
    return best[0]

def golden_hour_time(lat, lon, date_utc, target_deg=5.0, evening=True, iters=40):
    noon = solar_noon_utc(lat, lon, date_utc)
    a, b = (noon, noon + dt.timedelta(hours=12)) if evening else (noon - dt.timedelta(hours=12), noon)
    for _ in range(iters):                       # bisection; altitude decreases from a to b (evening)
        mid = a + (b - a) / 2
        if (solar.get_altitude(lat, lon, mid) > target_deg) == evening:
            a = mid
        else:
            b = mid
    t = a + (b - a) / 2
    return t, solar.get_altitude(lat, lon, t), solar.get_azimuth(lat, lon, t) % 360.0
```

**[computed]** For San Francisco (37.7749, −122.4194) on 2026-09-13 this returns
`2026-09-14 01:51:35 UTC` (= 18:51 PDT), elevation 5.000°, azimuth 270.6° (sun almost due west),
so `S_ENU = (cos 5°·sin 270.6°, cos 5°·cos 270.6°, sin 5°) = (−0.996, +0.011, +0.087)` and the shadows
point toward azimuth 90.6° (east). The morning solution is ≈ 14:19 UTC (07:19 PDT), azimuth ≈ 89°.

Edge cases: at high latitudes in summer the sun may never reach 5° *below*, or in winter never
reach 5° *above*; check `max altitude ≥ target` before bisecting. Near the poles use pvlib's
vectorised `spa_python` and `numpy.argmin(|elev − target|)` on a 1-minute grid instead.

Alternatively let the artist pick elevation/azimuth directly (a "sun dial" UI) and keep the
real capture date only for the *azimuth of the true sun* (needed to reason about baked shadows, §6).

---

## 5. Putting it together (Python, documentation only)

```python
import json, math, datetime as dt, numpy as np
from pysolar import solar

def sun_for_odm_project(project_dir, target_elev_deg=None):
    """Return dict with sun elevation/azimuth (deg) and ENU + Y-up vectors for an ODM project.
    Uses the same per-photo lat/lon as ODM and the mean capture time (see §2.3 for the tz caveat)."""
    # ODM keeps per-shot metadata in odm_report/ or opensfm/ ; simplest: re-read EXIF via exifread
    # (documentation sketch: assume `photos` is a list of (lat, lon, utc_datetime))
    lat = float(np.mean([p[0] for p in photos])); lon = float(np.mean([p[1] for p in photos]))
    t = min(photos, key=lambda p: p[2])[2] + (max(p[2] for p in photos) - min(p[2] for p in photos)) / 2
    if target_elev_deg is not None:
        t, elev, azim = golden_hour_time(lat, lon, t.date(), target_elev_deg)
    else:
        elev = solar.get_altitude(lat, lon, t); azim = solar.get_azimuth(lat, lon, t) % 360
    e, a = math.radians(elev), math.radians(azim)
    enu = np.array([math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)])
    return {"time_utc": t.isoformat(), "elevation_deg": elev, "azimuth_deg": azim,
            "sun_dir_enu_zup": enu.tolist(), "sun_dir_yup": [enu[0], enu[2], -enu[1]]}
```

Where to get `photos` without re-reading EXIF: `opensfm/exif/*.exif` JSON files (written by
OpenSfM inside the ODM project) contain `gps.latitude/longitude` and `capture_time` (epoch seconds,
same UTC caveat), and `odm_report/shots.geojson` contains per-shot positions. Both are outputs
of the pipeline, so reading them does not touch ODM code.

---

## 6. Baked lighting in the ODM textures (why the real sun vector still matters)

mvs-texturing runs with `-t none`, so the texture atlas contains the **original** sun shading and
shadows. Relighting to golden hour therefore composes *two* suns unless the baked one is removed.
Practical options, in increasing effort:

1. **Ignore** — works surprisingly well for nadir imagery flown near midday (short shadows, flat
   lighting). Add the new long shadows on top.
2. **De-shadow with the true sun**: compute the *capture-time* sun vector (§3), ray-march the DSM
   along it to find pixels that were shadowed at capture time, and brighten them by the estimated
   shadow/lit ratio (≈ diffuse/(direct + diffuse) at that elevation, table in file 01) before
   relighting. This is the reason the `azimuth_deg` at capture time must be right (§2.3, §3.1).
3. **Delight**: divide the albedo by a Lambert term `max(N·S_capture, ε)·k + ambient` using the
   DSM normal, then relight. Cheap and usually good enough for terrain; fails on vertical faces.

All of these need the DSM (`odm_dem/dsm.tif`), whose georeferencing (`GeoTransform`) is in the same
UTM zone as `coords.txt`; the DSM pixel `(col,row)` maps to `E = x0 + col·gsd`, `N = y0 − row·gsd`,
and the mesh vertex `(x,y)` corresponds to `(E − east_offset, N − north_offset)`.

---

## Sources

- pysolar documentation: https://pysolar.readthedocs.io/en/latest/
- pysolar `solar.py` source: https://github.com/pingswept/pysolar/blob/master/pysolar/solar.py
- pysolar "Strange azimuth values" list thread (convention history): http://lists.pysolar.org/2015-05/0000000.html
- Reda & Andreas, Solar Position Algorithm for Solar Radiation Applications, NREL/TP-560-34302 (OSTI): https://www.osti.gov/biblio/15003974
- pvlib `solarposition.spa_python`: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.spa_python.html
- Wikipedia — Solar azimuth angle (conventions, ENU vector): https://en.wikipedia.org/wiki/Solar_azimuth_angle
- glTF 2.0 specification (coordinate system, +Y up, metres): https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Cesium `Transforms.eastNorthUpToFixedFrame`: https://cesium.com/learn/cesiumjs/ref-doc/Transforms.html
- Cesium 3D Tiles glTF content / up-axis discussion: https://github.com/CesiumGS/3d-tiles/issues/175
- ODM Obj2Tiles (used by `--3d-tiles`): https://github.com/OpenDroneMap/Obj2Tiles
- ODM outputs documentation: https://docs.opendronemap.org/outputs/
- Local read-only sources in the clone: `opendm/dls.py`, `opendm/photo.py`, `opendm/location.py`, `opendm/types.py` (`ODM_GeoRef`), `opendm/gltf.py`, `opendm/ogctiles.py`, `stages/mvstex.py`
