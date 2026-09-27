## Native Mac ODM port (2026-09-16)

Added ODM-Fedora/install-macos.sh + run-macos.sh; installed locked Pixi0.77.1
osx-arm64 dependencies in ODM-Fedora/.pixi; built actual OpenSfM/OpenMVS,
texturing/filter/orthophoto/DEM tools. Smoke imports/CLI/native binaries pass.
Qt integration passes. Web UI elapsed timer added. MapRepair launcher now finds
keg-only Homebrew OpenCV before attempting unnecessary brew installs.
Launched Mac ODM localhost8767 and updated MapRepair8768 with web_runs_mac;
older MapRepair8765/8766 sessions left untouched. User changed prefilled94folder
to36photo upload and completed6selected ultra reconstruction in77.298s; output
folder is output/odm_macos_ultra94_20260916 despite actually containing36->6.
Do NOT report this as the94-image benchmark. Dedicated94benchmark uses separate
output/odm_macos_ultra94_benchmark and _meta/timing.json. Check report status.

# 2026-09-16 update: MapRepair grading and ODM-Fedora quality

MapRepair adds advanced_grade.py and 13 web sliders after completion, explicit
pixel-scale crop preview/full JPEG export via protected POST /grade, lossless
BGRA master retained on new runs, old-JPEG fallback. Neutral identity, alpha,
strip parity, input validation and non-destructive export tests pass. Full33
MapRepair tests pass; real odm.png + 40 SIYI photos browser test23.46s,
1754x1356 graded JPEG, mobile/noJSerrors. Results under validation/advanced_grade.

ODM-Fedora (NOT sibling ODM) keeps dedup->every-sixth per explicit correction.
Default2.3cm/pixel per latest user instruction. DSPSIFT/ultra24k, pc ultra,
mesh500k/depth12; reconstruction CPU. Native dependency CMake flags target host
CPU, 8GiB reserve/3GiB worker heuristic max8. FSR1 tiled lossless GeoTIFF export
via quality_run.py, 512px halo tiles, SIMD RGB RCAS, optional gfx1150 HIP source
and target validation. Auto HIP requires CPU parity; no HIP hardware/compiler
available here, Fedora install and full ultra reconstruction NOT validated.
C++ build/Qt integration pass onMac (Linux cleanup skips); CPU FSR local877x678
->1754x1356 ~.045s, exact source-reference pixels/alpha/tile seams/geography.
Interleaved median baseline.04599s/SIMD.04555s (~1%, not Ryzen speed claim).
QUALITY_UPDATE.md documents these limits and ROCm sources. Source archives
MapRepair-linux.tar.gz and ODM-Fedora.tar.gz rebuilt with this update; runtime
builds, image datasets, caches and validation screenshots excluded.

# ADM implementation memory

## ODM Fedora native flight-line app (2026-09-15)

User selected standard WebODM/ODM quality with native Python orchestration allowed,
no Docker/Metal, Fedora x86_64, then a localhost website with whole-folder input,
exact deduplication and every-sixth-image selection starting at the first.
Implementation lives in ODM/fedora (C++17 Qt network server/process runner and
optional desktop window; static offline HTML/JS). Original ODM 3.6.2 standard
DSPSIFT/high/10000-feature + medium dense reconstruction retained, skip-3dmodel
only. CPU parallelism, no AMD GPU compute backend. Inputs sorted numerically,
SHA256 duplicate removal, then unique photos 1,7,13,...; this reduces overlap,
so full-flight WebODM quality parity is not established.

install-fedora.sh uses local checksum-pinned Pixi 0.77.1 and existing pixi.lock;
builds required orthophoto targets only, skips optional RPM recommendations,
reuses completed native engine, supports --offline recheck (initial installation
still requires internet). OpenSfM ExternalProject now honors build job limit
instead of using every core. C++ engine environment sets ODM_OFFLINE=1 and
PROJ_NETWORK=OFF; opendm/net.py rejects downloads with ODM_OFFLINE=1.
run-fedora.sh defaults to --serve on http://127.0.0.1:8765. Server validates
localhost host/origin and POST token, streams uploads/downloads, refuses incomplete
folder imports, shows progress/logs, and supports cancel/resume. Project manifest
guards against changed input/engine and preserves originals. Legacy experimental
native renderer archived under ignored ODM/experiments; not in Fedora package.

Validation: C++ builds on macOS Qt6.11; native process + HTTP tests pass (Linux
/proc descendant cleanup test skips here). Chrome folder upload14 -> duplicate1
-> selected3 -> mock map download, refresh and mobile checks pass, no external
requests or JS errors. These are mock-engine integration tests, NOT actual
Fedora install, actual orthophoto quality tests, or Ryzen speed measurements.
Fedora machine not available. User reported errors but did not provide error log
as of packaging. Final artifact ADM/ODM-Fedora.tar.gz and .sha256; setup guide
ODM/README_FEDORA.md. Browser/package reports under ODM/fedora/validation.

## MapRepair local web UI and bootstrap launcher (2026-09-15)

Added web.py stdlib loopback HTTP server, web_worker.py isolated subprocess,
and web_ui/{index.html,style.css,app.js}. User can drop a TIFF/PNG/JPEG, drop
or choose a source folder, Run, and download the full 2x JPEG. Optional absolute
folder path avoids copying source photos. Directory drag traverses all entry
batches/subfolders; browser streams map then two photos concurrently. Server
validates manifests, size/path, gates start on every required file, stores
duplicate names separately. Input folders and map originals remain untouched.
Job UUID output folders under MapRepair/web_runs; persisted states/log tails,
upload progress, stage/timer, Stop/retry, download log, refresh recovery.
One active job; exclusive runs-directory lock prevents another server resetting
live jobs. Failed/interrupted uploads cleaned; startup reports interrupted jobs.
Worker uses run.py appearance settings, skips diagnostic stage PNGs, preflights
size/build before photo scan. Changed end-of-run SHA256 to stream 1MiB chunks.

30 tests pass (10 web tests include complete actual TIFF->HTTP JPEG + alpha/CRS,
folder upload completeness/concurrency/duplicate names, cancellation, partial uploads,
retry, persistence, localhost request checks). Headless installed Chrome actual
TIFF drop + real94photo directory drop, Stop/retry, reload during/after run,
9358x7882 JPEG download exact to disk file, alpha/bounds exact, mobile no overflow,
zero JS errors. Processing31.8133s; worker32.1034s; browser run/upload/download
34.2336s. Reports/screenshots/logs in validation/web_ui/. Initial browser test
was interrupted by a CSP-incompatible Playwright wait expression; replaced
test wait with locator assertions. Successful rerun validates full folder flow.

run.sh added: whole-folder one-command setup, apt/dnf/pacman missing OS deps,
private .runtime venv keyed to platform/install path/base interpreter, hashed
requirements installs, run.py --compile-only then exec web.py with CLI flags.
--setup-only, --no-browser, --port, MAPREPAIR_PYTHON and install opt-out supported.
OS package setup may require sudo; no global pip installation. Default apt
avoids recommended extras. Mac setup and cached repeat succeeded. OpenRT build
directory now includes OS/architecture/C++ runtime/install path/env identity;
copied Mac binaries and absolute CMake caches cannot be reused on Linux.
Old build/openrt remains intact; new build/openrt-<platform>-<arch>-<id> holds
the worker and build.log. README documents bootstrapping and which caches/results
to omit when sharing.

Fresh Ubuntu24.04 x86-64 Docker validation (emulated on this Mac) installed OS
packages and Python dependencies, compiled native C++ and OpenRT, repeated setup
without reinstalling, launched HTTP200 UI, and passed all30 tests including a
real TIFF->JPEG request. Install folder '/work with spaces/MapRepair' verifies
shell/CMake quoting. First Linux attempt revealed upstream std::sqrtf/std::powf
not exposed in GCC's std namespace; fixed three upstream files: BxDF.h explicitly
includes cmath and calls std::sqrt(float); BRDFPhong.cpp and BRDFBlinn.cpp call
std::pow(float,float). These portability changes are documented in provenance;
upstream is now minimally patched, superseding earlier 'unmodified' statements.
Lighting math is equivalent. Build failures now include the log tail directly.
Final Linux30tests17.497s, Mac30tests27.420s including rebuild; these are test
suite times, not map benchmarks. Linux OpenCV C++4.6, PythonOpenCV5.0.0.93.
Logs/summary in validation/launcher_linux; first failed container auto-removed
before its detailed build log was copied, so a second clean setup reproduced
the error and preserved diagnostics. Reused that environment after the fix.
run.sh's apt branch uses --no-install-recommends. Fedora/Arch/Windows untested.
Source-only share archive ADM/MapRepair-linux.tar.gz excludes caches, virtualenvs,
images/results and validation artifacts; retains all code, UI, OpenRT and licenses.

## Previous finish restored, OpenRT golden hour on SIYI ODM PNG (2026-09-15)

Packaging follow-up: copied the complete latest result into
MapRepair/results/siyi_odm_openrt_golden_hour/ so code, bundled OpenRT, docs,
validation and saved output reside under MapRepair. Root output copy retained;
historical reports still record original execution paths. README points to
the packaged result. Inputs remain external editable paths; dependencies required.

User requested reverting the cinematic/exposure experiment, adding a 5 pm
golden-hour OpenRT look, then switched the run to Downloads/odm.png and its
SIYI photos. MapRepair/run.py now defaults to Downloads/101SIYI_IMG (40 JPGs,
IMG_0007 through IMG_0046) and Downloads/odm.png (877x678 RGBA, no georeference
sidecar). Visually confirmed the field in these photos matches the map.
Restored SHARPEN_PROFILE=siyi_low_light, CINEMATIC_LOOK=None, pro_camera,
GRADE_SHARPEN=False, HIGHLIGHT_BLOOM=.02, FSR_SHARPNESS=.2. Recomputing old
South 1st Place saved 01_repaired through sharpen/grade matches its saved
02_sharpened and 03_graded byte-for-byte. No ISM/LeanADM changes.

OpenRT backend remains actual pinned upstream. New OPENRT_LIGHTING=
golden_hour_5pm uses warm sun RGB(1.38,.98,.60), cool sky(.92,1,1.12), each
Rec.709 luminance-normalized, direct/sky weights .65/.35. Reference remains
neutral-lit so warm color survives normalization. Fixed double-pi scaling:
CLightSky constructor already multiplies by pi. Exports explicit linear RGB
from OpenRT BGR. RAY_STRENGTH=.65, sun elevation15/azimuth255 (artistic look,
not calculated 5 pm solar position). Linear-light multiplier fades near white
to preserve highlight texture. No invented terrain or new tree/building shadows;
flat-plane warmth keeps photographed shadows. Neutral preset remains exact
plane no-op; optional OBJ gives occlusion. Upstream remains unmodified.

20 tests pass: added golden RGB ordering, luminance, thread parity, alpha,
zero strength identity, monotone highlight ramp; all previous tests retained.
Source fingerprint22b3f051af7c098208bfa3d253969254f5fbb8e5a0f4abf20622a916f44a4a70.
Full PNG+40-photo scan/repair/finish run output/siyi_odm_openrt_golden_hour/
completed14.178139s (initial C++ build excluded): scan13.45198s, repair.04322s,
sharpen.15558s, grade.04960s, OpenRT.09990s, bloom.01105s, FSR.08747s,
JPEG/mask export.18506s. 19consistent,4flagged,17unresolved; zero patches met
all repair gates. 01_repaired identical to input. No claim of zero map errors.
Final1754x1356 JPEG, exact input alpha at2x saved as lossless alphaTIFF; all
stage alphas/crop exact; no geographic accuracy claim for ungeoreferenced PNG.
InputSHA51b30dc66915f72ad09d69ccdd40f127ea0dbd9d51aae6e19983270f71fd0cc5
unchanged. Mean OpenRT fieldRGB1.18919,.96685,.77116; mean visible RGB code
change against restored previous finish +6.08,-1.28,-8.38 beforeFSR.
Saved final_2x.jpg, previous_finish_2x.jpg, input_preview.jpg, comparison.jpg,
preview.jpg, stage PNGs and reports. Validation/tests/logs/revert verification
in MapRepair/validation/openrt_golden_hour/. Code before this revision saved
in its source_before subfolder. Previous South 1st outputs remain intact.

## OpenRT plus cinematic virtual exposure blending (2026-09-15)

User requested https://github.com/Project-10/OpenRT plus more cinematic filters,
then delegated trying rough/partial exposures. Implemented actual upstream
OpenRT core in MapRepair/third_party/OpenRT, sparse pinned checkout at
ef6205f8ca51fd123da4a2f0b7b7a6ad27809b80, unmodified BSD-3 sources. Custom
MapRepair/openrt CMake+trace.cpp build C++20 worker with existing Homebrew
opencv@4 4.14.0 dev library (Python cv2 remains5 in separate process). No system
install. Source pin/license/build docs/provenance included. Compiles successfully;
first adapter compile missing Ray.h was fixed. build/openrt ~22MB, source<1MB.
openrt_backend.py source-hash caches executable; explicit build failures, no
silent substitution. Raytrace backend setting 'native' keeps previous C++17.

Adapter actually uses OpenRT Scene/BSP/PrimTriangle/ShaderDiffuse, distant omni
sun and 16 deterministic sky samples. Orthographic lighting samples normalized
against identical unoccluded ground plane, float32 field. It evaluates direct
sun+sky occlusion, not recursive GI. Original flat-plane default yields1.0
everywhere and image byte-exact no-op. Optional registeredOBJ remains supported;
wallfixture minimum.412463, parallel/singlethread byteexact. No image-derived
terrain, invented tree/building shadows, or measuredHDR claims.

Added cinematic_finish.py, looks natural_cinema/warm_documentary. Premultiplied
alpha-safe downsampled fields; neutralWB from bright low-chroma pixels only
(Y .60..88, avoiding grass-as-gray-card), gains bounded.96..1.04. Virtual copies
at -.65/0/+.65 stops in linear RGB, smooth local well-exposed weights + neutral
bias; blend strength.55 warm/.45 natural, maxchannel adjustment.055 beforegrade.
Filmic tone/highlight compression, warm highlights/cool shadows, restrained
greens, very subtle diffusion/warmhalation. No blur of base detail or addedgrain.
Preserveoriginalalphageometry. These redistribute existing tones, notextraHDR.
New natural_detail sharpenprofile detail.25 fine.05 CLAHEblend.10 chroma.7.
run.py defaults CINEMATIC_LOOK='warm_documentary', SHARPEN_PROFILE='natural_detail',
RAYTRACE_BACKEND='openrt', HIGHLIGHT_BLOOM=0 (newfinishalreadyglows). Legacy
COLOR_PRESET remains available when CINEMATIC_LOOK=None; GRADE_SHARPEN=False.

18 tests pass, including OpenRTactualplane/shadow/threadrepeat, cinematic
stripalpha/hiddenRGBparity and monotonegrayramp, priorgeometry/FSR/TIFFfullpipeline.
Full actualTIFF+94photos rerun output/south_1st_place_openrt_cinematic/34.08272s.
Scan21.9032s repair1.4737s sharpen2.7167s newfinish2.3086s OpenRTtrace.08693s,
FSR1.2859s export1.4408s plusload/intermediates/incrementalOpenRTbuildcheck.5073s.
Initialupstreambuildexcluded. Same21tiles14sources andbyte-identicalrepairedmap
vs precedingrun. NativeJPEG9358x7882 ~18MB, whiteoutsidealpha. Masksidecarexact,
originalbounds/centertransformexact,inputSHA79e151e6...unchanged. MeasuredRGBWB
1.040/.9905/.9736,maxexposurechannelshift13.48bytecodes; raypassno-op verified.
comparison.jpg/preview.jpg/final_2x.jpg inoutputfolder. Allreports/testlog/runlog
MapRepair/validation/openrt_cinematic/. Earlier unselectedtriallooks saved under
output/south_1st_place_cinematic_trials (before restricting WB tobrightneutrals).
ISM/LeanADM untouched; remaining original map blur/noise/stitching still visible.

## South 1st Place TIFF, repairs enabled (2026-09-15)

Latest request keeps MapRepair's repair step enabled and runs supplied TIFF:
/Users/avneh/Downloads/South-1st-Place-9-15-2026-orthophoto (1).tif.
run.py MAP_IMAGE now defaults to this file; PHOTOS_FOLDER still Downloads/mapv2
(94 photos). No removal of scan/repair was applied. Original TIFF 4679x3941,
RGBA8, EPSG:32615, alpha=dataset validity exactly. Its nonzero-alpha bounding
box already spans the whole raster [0,0,4679,3941]. Preserve its exact irregular
footprint and outer coordinates; no new lasso or invented crop.

Completed output/south_1st_place_repaired/ in31.831958s: checks21.5348s,
repair1.3937s,sharpen2.6226s,grade.7934s,ray.0017s,bloom.2823s,FSR1.1999s,
export1.4770s plusload/intermediates. 21tiles/14sources/270071affectedpixels;
statuses32consistent,35flagged,14patched,13unresolved. Final9358x7882 JPEG,
white outside original alpha, plus exact final_2x_alpha.tif and .jgw/.prj.
Verified original alpha, unrelated pixels before finishing, 2x alpha sidecar,
original world bounds and JPEG sidecar center transform. InputSHA256
79e151e6b428de9a6068c466eccbe7c84eeb30483ec3d2e05757c92a55018619 unchanged.
Saved validation/south_1st_place/{run.log,summary.json} under MapRepair.
This times repair/finishing of an existing map, not raw-photo reconstruction.

## MapRepair blending, TIFF input and JPEG delivery (2026-09-15)

User flagged sharp/out-of-place rectangular repairs, then requested original
TIFF bounds and alpha-aware cropping with final JPEG delivery. Changes confined
to MapRepair; ISM/LeanADM unchanged. Source_before copies for original checker,
launcher and README in MapRepair/validation/blending/source_before/.

New patch_blend.py: connected repair components use intact overlap outside all
selected tiles to estimate robust slow RGB offsets (bounded +/-24 byte levels),
optional measured excess-texture matching with 0..2px source blur, no sharpening.
32px smooth inward detail blend, 64px color blend; immutable original map used
as every reference, source ordering does not change results. H/selection same.
Outside selected repair pixels and original alpha remain exact before finishing.
run.py GRADE_SHARPEN=False skips the color grader's duplicate clarity/unsharp
passes; dedicated siyi_low_light sharpening stays. Preset definitions unchanged.

New map_io.py uses rasterio/GDAL for tagged TIFF RGB/gray, correct tagged alpha,
internal validity masks/nodata; never assumes extra fourth/NIR band is alpha or
that valid black is empty. Partial alpha combines by min, avoiding double
multiplication. UInt16 defaults to full numeric range; TIFF_VALUE_RANGE optional,
mandatory for float/signed color. Embedded TIFF CRS/rotated affine supported.
Crop/export is exact original raster rectangle at2x with original mask nearest2x,
not an automatic tight alpha boundingbox or lasso. Final JPEG quality95/4:4:4,
transparent regions composited white (JPEG_BACKGROUND); exact alpha saved as
final_2x_alpha.tif. final_2x.jgw/.prj georeference JPEG; mask TIFF embeds CRS.
Installed rasterio1.5.1 and dependencies in existing .venv, requirements updated.

14 tests passed, including complete TIFF->repair->finish->JPEG/mask fixture,
partial alpha, internal holes, rotated corners, nodata, UInt16 display, extra
non-alpha band, source-order independence. Known shifted-region RGB error now
28.3743->4.19235; previous narrow blend1.8504, so gentler perimeter trades some
fixture correction for less visible transitions. No real surveyed accuracy.

Full same94-photo scan+repair+finish rerun: output/maprepair94_blended_jpeg/,
43.77855s CPU with native alreadycompiled. Scan22.306s, repair3.189s, sharpen5.652s,
grade1.672s, ray.0030s,bloom.582s,FSR2.493s,export3.124s plusload/intermediates.
Input existing6500x5978 PNG neutralmap SHA b55ff5f3... unchanged; this is not a
raw-photo map reconstruction benchmark. Same42tiles29sources as old report,
1048649affectedpixels; alpha and all unselectedpixels beforefinishing byteexact.
Final13000x11956 JPEG ~46MB, mask272KB verified exact against input alpha at2x.
detail_comparison.jpg and preview.jpg alongside, code reports/testlogs at
MapRepair/validation/blending/. Trial replay map moved outside code folder to
output/maprepair94_blend_trials/replayed_repaired.png. Remaining original map
stitch/parallax problems not fixed by appearance changes.

## Standalone MapRepair Python/C++ tool (2026-09-15)

User requested a small separate folder using their map_geo_checker concept,
hardcoded photo-folder/map paths, smart geometry patches, ISM sharpening/grade,
ray tracing, extra polish, then final FSR2x; portable Python/C/C++.
Created MapRepair/ (no imports of ADM/ISM at runtime). run.py top paths default
Downloads/mapv2, final_test/map_neutral.png, MapRepair/output (new/empty required).
CPU implementation using OpenCV/NumPy/Pillow/pyproj and bundled standard C++17
native.cpp; auto builds with Clang/GCC/MSVC, CMake also supplied. Tested macOS
AppleM4Pro only; Linux/Windows source/build paths provided but not executed.
Installed pyproj3.8.0+certifi into existing .venv with uv (venv has no pip).

Geometry uses cached SIFT (ORB selectable), GPS ROI with explicit world/.prj,
manual truebounds, or ISM local-origin georeference. Never treats world-file
metres as lon/lat or source-GPS extents as verified map bounds. EXIF+SIYIXMP,
NO GPS connected untrusted. RANSAC with 20% held-out matches, spatial hull
support and homography plausibility, local forward/backwardLK+NCC discrepancy
tiles. Independent overlapping source pairs constrain actual tie points in
native map pixels. A tile needs another source's agreeing direct transform and
measured tie-point hull at that location. One source pertile selected byfit,
sharpness/support/trackscatter; warp onlyaffected boundingbox, properzero-border
source distance transform + inwardpatchfeather. No exposure/color matching in
geometry stage. Weakfits, GPS-only mismatch and unsupported changes remain
flagged; not surveyedaccuracy or fullSfM. Reports map ownerindices to sourcepaths.

Finishing: copied ISM siyi_low_light sharpening, halo-striped chroma byteparity;
original supplied pro_camera grade -> C++BVHraytrace(two diffusebounces) -> subtle
highlightbloom.02 -> final2xFSR1EASU+noise-awareRCASstrength.2. FSR2x interpreted
2xspatialwidth/height, not temporalFSR2. AMDlicense included. NativeC++striped,
threaded,diskbacked FSRstorage, noGPUrequired. Defaultflatmap plane produces
neutral irradiance, pixel-exact no-op; optionalregisteredOBJ inmap-pixelXYZ
supportsrealocclusion. No intensity-derivedterrain/buildings. C++FSRvsISMMetal
max1codeafterEASU,2afterRCAS (mean<.001), thread/chunkrepeatbyteexact.

8portabletests pass; syntheticshiftedregionerror28.3743->1.8504 (0..255RGB),11tiles,
fullsynthetic720x960->1440x1920output. Actualcompleted94photo checker+finishing
run output/maprepair94_20260915/:52.620558sCPU (originalmappingexcluded, native
alreadycompiled), scan/pairs24.3458s,repair4.6508s,sharpen6.1884s,grade3.6859s,
ray.0040s,bloom.5717s,FSR2.8354s,finalexport4.9888s,plusload/intermediateexports.
42tiles using29photos,1072000affectedpixels. Statuscounts29patched,22consistent,
32flagged,11unresolved;145pairs=90consistent,25disagree,30unconfirmed. Visible
source/exposureseamsremain; no measuredreal-groundtruthqualityclaim.
Final13000x11956PNG withgeorefsidecars andfullinputalphafootprint(noaddedlasso).
InputmapSHA b55ff5f3c094e23981d7f172bc43f49a3895a6094636dffdeffd9ab980cf8d9a,
unchanged; ISMshader7f4ed7... unchanged. Outsidepatchedtilesallgeometry-stage
pixelsandalpha verifiedexact. Sourcefolderonlycode/tests/reports; largerPNG
outputs storedoutsideunderoutput/. README containsusage,limits,measuredstages;
MapRepair/validation/{tests.log,ism_parity.json,real94_summary.json,real94.log}.


## Supplied cinematic grader and gentler lasso (2026-09-15)

Latest user asked to replace the custom grade with the pasted cinematic script,
reduce lasso aggression, move practical processing into Metal, compile; user
will validate the flight. Restored all seven supplied global presets (including
original pro_camera: contrast.35,sat.95,vignette.08,grain0,blacks.05,shadows.02,
highlights.06,whites-.06,clarity35,sharpness90). Removed custom SIYI/sunset grade
and its post_cinematic_grade kernel. run.py now defaults color_grade=pro_camera;
separate presentation_profile=siyi_low_light remains. Original grading's own
clarity/sharpness follow the separate presentation sharpening pass. Old nonzero
white/vibrance overrides fail with migration guidance instead of being ignored.

Generic Metal grade now supports all global recipes and all three regional
recipes: four HSV scores packed as uchar4, Pillow-compatible feathering, per-region
hue rotation/grades, shared vignette/grain/clarity/sharpness. Grain presets retain
NumPy default_rng(seed) CPU noise generation (uploaded float32); all pixel
operations run Metal. Default pro_camera has no grain. Blur uses two ping-pong
buffers and batched encoding; default grade16dispatches,1command submission.
Alpha and geographic sampling preserved, transparent-border sharpening protected,
split-tone luma bounded to avoid fractional-power NaNs. CPU remains codecs,
metadata,SIFT/RANSAC,host orchestration, small tables and optional noise stream.

Lasso remains final image operation; inset.01->.005, mask opening.09->.06,
angular radius18->12. These are top-level SETTINGS/config fields, GPU validated.
Neutral-only fallback uses the same settings. run_visual hooks still observe
actual outputs. Photo preview now respects presentation_raytrace=False.

Packaged/recompiled ISM.metal92entrypoints, SHA256
7f4ed7b3d2ab40cecb880c30d239c9c2cc0712e66737c7c24c34531b532937ff.
50 portable tests pass, zero skips, including all7global/3regionalparity,
seed/alpha/repeat,old/newlasso,finalcropintegration,visualhooks,geographicchecks.
New report ISM/build/validation.json, log build/grader_validation.log.

Saved-map presentation check uses CURRENT final_test/map_neutral.png,94inputs,
6500x5978, sourceSHA256b55ff5f3c094e23981d7f172bc43f49a3895a6094636dffdeffd9ab980cf8d9a.
Earlier ISM/benchmarks/odm_geo94 and final_lasso_94 folders are now absent from
this checkout; recheck artifacts rather than assuming earlier memory paths exist.
New script scripts/validate_cinematic_update.py and artifacts under
ISM/benchmarks/grader_update/: fullmap,preview,geometry,cropreport,validation,
source_before copy of every edited implementation source and launcher/guide.
Read final_test only; did not alter its outputs or run raw photos.

Measured: sharpen.173444s, Metalgrade.238735/.388204s two samples, unbatchedgrade
.251576s, CPUreference3.650623s,newlasso.025811s inclupload, PNG+preview1.244727s.
Excludesdecode/GPUinit; no claimed totalE2E timing or batching speedwin from
these samples. Batched/unbatched GPU and repeatgrade byteexact. CPU/Metal before
unsharp max1code; aftertwo thresholdedunsharp max5, meanRGBerror4.8211e-6,
99.9997178%channelsidentical. AllregionalfixturemeanRGBerrors<.004,max4.
On sameNEWgradedmap, oldcrop77.4485% vsnew85.8944% coverage; newretained26752840
of31146214coveredpixels. AllretainedBGRApixelsandmap-to-worldexact. Existing
source/alignment seams remain. GUIDE describes current defaults, historical
benchmarks are explicitly not timings for this new presentation. No raw94/306
mapping rerun; user will validate.


## Final lasso after presentation (2026-09-15)

User requested lasso crop as the final step in the Python runner. Enabled
`SETTINGS['lasso_crop']=True` in ISM/run.py. Changed development and packaged
composition to defer lasso until after sharpening, optional lighting and color
grade, immediately before the single final PNG write. `map_graded.png` is cropped;
neutral map/support remain before presentation. With no grade, crops the enhanced
output; with no presentation, crops neutral as before. Reports identify which
image is affected, final-image stage, crop timing and unchanged map-to-world.
Presentation geometry covered_pixels reflects the cropped alpha footprint.
Visual runner records a Final lasso crop stage. No shader changes (91 kernels).

Portable suite now45tests passed, including a new final-image integration check
with1.5x scale, both graded and enhanced-only outputs: retained pixels, dimensions
and geographic transforms match the uncropped presentation exactly. Real saved
94-photo graded map crop check: retained85.3% coverage, crop0.037s, crop+PNGexport
0.914695s (excludes inputdecode and GPUinitialization), saved under
ISM/benchmarks/final_lasso_94/. No raw-flight/mapping rerun. Earlier94E2Etimings
predate final lasso; future launcher wall timers include it. GUIDE updated.


## OpenSfM geographic port and 94-image comparison (2026-09-15)

User explicitly authorized cleaning ISM, porting ODM geographic convergence into
Metal/MPS while retaining ISM matching, and old/new E2E tests on mapv2 (94 photos,
not C or 306). Completed. ISM now compiles 91 entry points into one ISM.metal
library, with optional `geographic_solver='opensfm_planar'` and CLI
`--geographic-solver opensfm_planar`. Default remains `visual` because the new
method is a tradeoff rather than a demonstrated quality winner. run.py now uses
symmetric=True, GPS mismatch bypass=False, interpolation_levels=0. Preserved
user input/output paths, 51.816m AGL, registration1280, SIYI grade, scale1,
raytraceFalse. Cleanup intentionally selects verified real-photo rendering.

Exact OpenSfM revision c5328439465e6ace011f39077d1077d7b1cdd65d is pinned by local
ODM SuperBuild. Downloaded reference at third_party/opensfm_reference. Ported
DataPriorError position residual, UpVectorError+Cauchy orientation residual and
float64 WGS84/ECEF/topocentric equations. GPU geographic model uses 6 DOF physical
camera poses with fixed intrinsics/groundz0; two world axes constrain rotation
including nadir heading (second axis is an ISM adaptation). GPS/AGL uncertainties
10m, measured camera axes10degrees; measured XMP/sidecar horizontal accuracy
supported. Configured AGL is separate from EXIF altitude. Native MPS solves
combined visual+geographic normal equations, with analytic derivatives and
GPU resident costs/priors, no Python numerical loop. World-coordinate conversion,
initial pose decomposition, SIFT/RANSAC and codecs remain CPU. This is NOT the
complete ODM/OpenSfM 3D reconstruction or runtime Ceres solver: no triangulation,
DEM/mesh, lens calibration, GCP or rolling-shutter/bias estimation.

Direct metadata initialization failed (new/ trial40.955s, no map): all94carry
'NO GPS connected' despite coordinates; no accuracy measurements; recordedyaw
can disagree~180degrees with image orientation. Fixed by first using the verified
visual solve, geographic similarity, and physical homography decomposition,
then robust joint refinement. New first successful run: visual init2.877885s,
geographic refinement0.708207s,15active/16encodediterations (visual61/64).
True camera-centre GPS median3.51674/p956.18954m passes5mmedian withoutbypass.

Preserved old integrated package: ISM/benchmarks/odm_geo94/source_old (51source
hashes verified); original development sources in development_before. LeanADM
untouched. Successful results old/, new_visual_init/, reversed-order repeats
old_repeat/, new_repeat/. All fresh workers, raw94, no matchcache reuse, same
508edges/60875matches and presentation. Oldwall22.426728/20.741724s,
new20.854751/21.614501s; medians21.584226 vs21.234626s. No clear speedwin;
order reversal changes which is faster. Both under120s. Firstold/newalignment
3.333337/3.608981s; added geographic work is not the source of apparent total
speed difference. Old6500x5978/new6500x6021, peakchildRSS~1.03–1.17GiB.
Each method's neutral PNG, graded PNG and transforms are byteidentical across
its repeated runs. Rawinputs verified by SHA256.

Common image-centre-to-GPS proxy: old median2.381587/p956.828591m,
new2.140594/6.084231m. This is the same near-nadir proxy for both, NOT true camera
centre or survey accuracy; GPS is also used in fitting. Do NOT compare new
truecamera3.51674m to oldimage-centre2.38159m as interchangeable definitions.
Visual transfer median/p95 increases1.099238/3.190113px ->1.230074/3.637949px.
Road/source/exposure seams remain, no overall quality/resolved-detail win shown.

44portable tests pass, including analytic camera/visual Jacobians and prior
Hessian againstfloat64, physical projection, noisycamera convergence,horizon
andcollapsedheight rejection, existingGPU parity/relocation. Also compiled actual
unmodified OpenSfM C++ DataPriorError,UpVectorError,ceres::CauchyLoss againstalready
installed Ceres/Eigen:128fixtures,37validgroundfacingpositionGPUcases, maxMetal
orientationcosterror4.317e-7,CPUposition1.11e-16,WGS84difference1.244e-9m.
These prove residual/cost parity, NOT completeCeresbundle solution parity.

Run with .venv/bin/python ISM/run.py --input /Users/avneh/Downloads/mapv2
--output <new-empty-folder> [--geographic-solver opensfm_planar]. GUIDE.md rewritten
for current defaults. Detailed technical scope/provenance in docs/OPENSFM_PORT.md
and portable runtime/OPENSFM_PORT.md; upstream AGPL included in LICENSES.txt.
Benchmark results, source/inputhashes, reports, comparisonimages are under
ISM/benchmarks/odm_geo94/RESULTS.md, comparison.json, overview_comparison.png,
road_comparison.png. Scripts benchmark_geographic.py, report_geographic_benchmark.py,
validate_geographic.py and opensfm_prior_reference.cpp reproduce development
checks. No mapping process remains running.


## Python GPS chunk-stack experiment (2026-09-15)

User requested10photos cut into50x50pixel chunks, estimate chunk lat/long and
stack/composite without the disconnected visual graph. Implemented independent
CPU script `experiments/gps_chunk_stack/run.py`, README and5geometry/parity tests.
Actual raw first10/94mapv2photos IMG_0006..0060:33880tiles(1200partialedges),
51.816m configuredAGL/81degreeHFOV, recordedgimbalyaw/pitch/roll, flatgroundz0.
Saves realRGBtiles in254.1MBmemmap, centerlat/long+ENU/corners/sourceindices,
interactiveHTMLphoto-layerstack, selected-source and weightedmean composites.
No visual registration, shaders, grading or recoveredterrain. Layers are photo
indices, not elevation. Metadata has GPScoordinates AND 'NO GPS connected' tag
inall10; logged unverified. Visualresults still have roadjumps/source seams;
mean composite doubles edges. Tilecutting itself doesnotfix camera pose error.

Finaloutput `output/gps_chunk_stack_10_verified/`:2494x4096map,.03764m/pxassumed,
6.159894s fullsubprocess CPU includingtilewrites+exports,peakRSS984907776bytes.
Stages .115512smetadata/projection/export,3.432259sdecode/tiles/composite,
2.184063smap+viewerexport,.275709sreportmetrics. All10 contribute,9381058covered
pixels,7765982overlap,max6observations. Initialrun6.577601s in
`output/gps_chunk_stack_10`; bounded diagnostic scratch inrepeat reducedpeakmemory.
Bothmaps/supportPNGs and entiretilearray byteidentical acrossruns. Five focused
tests pass including tiled-vs-whole projective warp exactparity and partialedges;
30real-sourcechunks checked exact withpaddingexcluded. LeanADM27sourcehashes
unchanged. No full94/306run or Metalport; do not extrapolate accuracy or speed.

## LeanADM on current mapv2 batch (2026-09-15)

User requested running the original LeanADM backup instead of ISM. Executed from
LeanADM with the parent Python environment, Metal/GPU optimization, GPS centers,
PNG export, registration width640, coarse budget128MiB, skip0 and flight AGL51.816m.
Input `/Users/avneh/Downloads/mapv2` contains94photos. Failed after9.455410s wall
at GPU alignment initialization: disconnected match graph despite512verified pairs
of2994candidates. No map produced; this is failed-run timing, not mapping speed.
Reports/log/config/source hashes in `output/leanadm_mapv2_20260915/`; all27hashed
backup source files verified unchanged after execution. Native build output was
created normally in LeanADM/build. No ISM recovery or solver changes imported.

## FSR rendering artifact fix (2026-09-15)

User showed severe duplicated/warped road pieces and requested a software fix;
then explicitly requested GPS mismatch bypass (implemented below). Continued
render diagnosis with preserved prior symmetric-transfer alignment rather than
the user's now-disabled symmetric setting. Original output had279virtualviews,
93pairs,558generation calls,183.996s E2E. Real-photo-only recompose of exactly
the same alignment removed the large FSR artifacts in4.865s.

Added `adm/fsr_geometry.py`: gate each recursive FSR pair on rotation<=15degrees,
95th-percentile motion<=quarter max image dimension, finite non-horizon midpoint,
and half-motion camera consistency<=1native source pixel on9x9grid. Reject the
whole adjacent pair if any recursive child fails. Actual93pairs:73accepted,
20skipped,219virtualviews. `fsr_validate_generated` Metal kernel now permits only
pixels seen in BOTH parents, parent channel difference<=48/255, generated channel
within parent range+/-20/255. Replaces old union-of-alpha support that admitted
FSR disocclusion/inpainting artifacts. Invalid pixels transparent; originals remain.
Compositor now flushes progress every25inputs in both traversals.

Validated recompose: `ISM/benchmarks/render_fix_94/filtered_fsr/`,127.777s
render/export only; no matching/alignment or grading. 6,931,081pixels(22.2513%of
coverage) select verified generated views; large road fragments are absent in
visual inspection. Comparison `ISM/benchmarks/render_fix_94/road_before_after.png`.
Real-only4.865s is faster; no new full-E2E timing claim. Compiled87kernels;
38package tests pass. The render used saved alignment_symmetric_transfer=True;
user's current run.py remainsFalse, interpolation_levels=1, raytraceFalse,
allow_gps_mismatch=True. Do not claim their currentFalse alignment is the tested
good geometry, or that GPS bypass repairs it. No user mapping process was running
at the compilation/recompose checks. runtime.zip remains the historical rollback.

## Explicit GPS mismatch bypass (2026-09-15 local)

User explicitly requested bypass of `GPS camera positions contradict the visual
mosaic` after changing run.py alignment_symmetric_transfer=False, interpolation
levels=1 and presentation_raytrace=False. Preserve those edits. Added boolean
allow_gps_mismatch (genericFalse, ISMrun.pyTrue) which only bypasses the final
GPS-center residual threshold. Reports preserve residual/tolerance and identify
failed fit and bypass. Other validation remains. Four focused geographic-fit
tests passed. This does not repair geometry or the reported rendering artifacts.
Rendering diagnosis is preserved in ISM/benchmarks/render_fix_94, including the
prior94-photo map, transforms, metadata and FSR report (279virtualviews,130.74s
generation). FSR/current midpoint adapter is under investigation; no rendering
fix has been applied yet.

## GPS-fit failure fix after rollback (2026-09-14 local)

User reported the restored pipeline failing `align_centers_to_geography`. Their
94-photo output was cleared during investigation. Reproduced matching/alignment
only in `ISM/benchmarks/gps_fit_94`: same508edges/60875matches,271iterations,
visual residual .09235px but median GPS discrepancy30.0948m. Mosaic-space objective
collapsed median photo area to .00845 of nominal registration-frame area.

Added optional `alignment_symmetric_transfer` (generic defaultFalse, ISMrun.pyTrue).
Two Metal kernels measure i->j and j->i transfer residuals in original resized
photo pixels, half-weighted, using analytical inverse/projection derivatives.
Matching and GPS/yaw behavior are otherwise the restored versions: no GPS/yaw
constraints inside optimization, final geographic similarity and5m limit retained.
Legacy error mode remains available; legacy synthetic60vs600parity still passes.
Symmetric convergence requires low damping (<.01); objective tolerance1e-5 is its
float32 floor, with a predicted-gain check for precision-limited cost oscillations.
Heavy damping cannot trigger its objective stop. Reports name the error definition
and threshold. Failed GPS fit now saves geographic_alignment.json before raising.

Final saved-match recheck `ISM/benchmarks/gps_fit_94_transfer_v2`:61active/64encoded
iterations,2.934255s alignment, median/p95 source-pixel transfer1.09924/3.19011px;
median/p95 GPS fit2.38159/6.82859m (passes unchanged median5m gate). All94 retained.
This measures alignment only, not full E2E time or surveyed map accuracy. No full
mapping run launched. Compiled86kernels;32self-tests passed, including derivative
finite differences, cost invariance under uniform100xshrink, noisy scale recovery,
heavy-damping rejection, GPS report-on-failure, config/dispatch and relocation.
The canceled motion-blur/RAM changes and GPS/yaw priors were not resumed.

## User-requested rollback (2026-09-14 23:20 local)

User explicitly canceled the later alignment changes and requested the code matching
`ISM/ISM.metal`. Its SHA-256 is
`c610ad5fc0a456f6978ead738a9aeb073695faa16b1f2436ec0be7283e0cc2c2`, exactly matching
the manifest in the user's `ISM/runtime.zip` (latest source timestamp 22:35:32).
Restored all 34 runtime source/manifest/license files byte-for-byte from that archive.
Restored matching development registration/config/pipeline/GPU solver and native/MSL
alignment sources so packaging will not reintroduce the rejected GPS/yaw priors.
Removed those six settings from `ISM/run.py` and archived the added geo_priors module
and its tests. Current paths and other user settings remain as edited by the user.
The active shader was already the requested version and was preserved byte-for-byte.
Backup and detailed restore manifest: `backups/ism_before_rollback_20260914_232001/`.
Verified all development shader hashes against the restored manifest; compiled 84
Metal kernels on Apple M4 Pro; all 26 package tests passed (2.573 s), including
relocation. Launcher settings validate with its actual Metal/PNG flags. No flight
run was started and final_test outputs were not modified. The old alignment behavior,
including its previously documented limitations, is restored. Do not resume the
canceled GPS/yaw, motion-blur, or RAM changes without a new user request.

## ISM standalone package and SIYI settings (2026-09-15 UTC)

CRITICAL latest600-iteration full-flight audit: user's `final_test` failed after
1094.049629s wall,913.194584s alignment,600iterations(401accepted/199rejected).
Finalobjective13140.9336; finalacceptedgain11.80859 vs threshold.0131527(897.8x).
Finalstep.004898 vs3e-7(16326.7x). Reportedmedian.0028486px/p95.01239065px
MUST NOT be presented as good alignment accuracy: inspection of transforms.npy
shows523/564projected footprints below1% of original registration-image area;
medianarea ratio2.92155e-6, no horizon folds. Most images collapsed in scale,
artificially reducing the forward mapped-coordinate objective. Rejected trace
steps include NaNs/huge norms. More iterations alone cannot establish valid map
geometry. Need scale/shape constraints or a non-collapsing residual/initialization
before accepting this solution. No map generated; no fixes/reruns made in audit.

Live diagnostics update: user asked to see current worker36589's activity. macOS
sample showed `adm_align -> adm_sync -> MTLCommandBuffer waitUntilCompleted`;
IORegistry reported100%device utilization. This establishes alignment/GPU activity,
not exact iterations or guaranteed forward progress. Added native stdout+flush
progress after existing4-iteration sync:active/budget/encoded counts,objective,
accepted/rejected,damping,elapsed. No extra synchronization/changed math.
Source update is for future processes; the already-running worker keeps its old
loaded dylib. Sample saved `ISM/build/worker_36589_sample.txt`. No restart requested.

Latest change: user requested a600-iteration cap with early stopping. Added
`alignment_max_iterations` (generic Config default60, ISMrun.py600, validated
integer1..600), passed pipeline→registration→GPUsolver; previous200 API ceiling
raised to600. Existing MSL convergence equations and four-iteration host batches
are unchanged. Self-test26tests passed. The noisy3-image alignment regression
converges after5 active/8encoded iterations with cap600, with bit-identical
transforms to cap60. No new full mapping run was launched.
The user's previous `final_test` run failed after283.199s at60iterations:
all564connected,19076edges,541282768planned scratchbytes,median.2792px/p951.0052px.
Last accepted objective306397.28→285305.91(6.9% improvement), followed by a
rejected trial: it had not plateaued. More iterations permit further optimization;
full-flight convergence with600 has not yet been verified.

Final handoff: user stopped further runs and will test themselves. Nothing is
running. `maps_recovery_v3` connected all564photos with11verifiedbridges after
lowering ONLY recovery SIFT detection contrast to.01;266–447inliers/bridge.
It then hit512MiB alignment workspace guard at193.877s. Added configurable
alignment_budget_mb(default512generic,1024inISMrun.py;128..2048validated),
propagated through pipeline/registration/GPUsolver. Syntax checked after this
last change; no further benchmark launched. Full564E2E remains unverified.
User independently changed run.py registration_width to1280; PRESERVED that edit.
Faint-texture recovery test repeats with identical matches/transforms (5CPU
graph tests pass; prior23-testfullsuite predates finaldetector/budgetupdates).

`ISM/` contains one `ISM.metal` with 84 exported kernels and a portable host
runtime. All ten translated FSR passes share that Metal 4 library with ADM.
The entire folder is required; a shader cannot perform file IO or CPU SIFT.
`run.py` exposes paths, SETTINGS and a full-process timer. Preserve the user's
edits: currently input `/Users/avneh/Downloads/maps`, output `final_test`, AGL
51.816 m (170 ft). The user chose the IMG_0556 settings as run.py defaults.
`run_visual.py` is separate, with VISUAL_PROCESS=True/False and a saved/live HTML
gallery. It shares SETTINGS but has separate path defaults. Its presentation
observation hooks are pixel-identical in tests; full visual flight smoke pending.

- SIYI profile: chroma-only edge-aware cleanup .7 (preserves per-pixel luma),
  Lanczos2x, positive detail .65, CLAHE blend .35, fine gain .15, final contrast
  1/0. Grade exposure+.08, shadow lift .035, luma contrast .02, RGB balance
  1.05/1/.94, tiny warm highlights. No NLM or final FSR spatial upscale.
- Standard profile remains available: detail1.3, CLAHE blend.75, fine.35,
  contrast1.02/-1; pro_camera exposure0, contrast.04, shadow lift.025 and weaker
  tints. Both use the nondestructive white shoulder .3/vibrance.1 inside grade.
- Gaussian now matches OpenCV's error-diffused fixed-point weights with a
  uint16 temporary (half prior bytes). This fixed CLAHE amplification of blur
  rounding: SIYI synthetic full recipe mean .032/255, max3, versus former max25.
  Chroma and grade fixtures exact. 23 package tests passed after these changes,
  including real FSR identity/translation, ray tracing, standalone relocation,
  connection recovery, invalid pair rejection and visual observation parity.
- Saved standard ISM306 benchmark: `ISM/benchmarks/e2e_306_pleasant`, full wall
  116.628144875s;306originals/915generated/1221inputs; matching55.4188s,
  alignment7.43384s, map/presentation/export38.2085s; RSS2049507328bytes.
  This PREDATES SIYI defaults, fixed-point Gaussian and match recovery. Its
  shader SHA252a4db148e8af4c36fb13188e60ba4fd8fd0f76ac298e186d0bfcd4b0a3ee16.
- `ISM/benchmarks/siyi_photo`: full3840x2160sample to7680x4320 presentation
  1.122878s inclPNG, FSR fullsize identity .383071s exact. Not a flight/mapping
  benchmark; predates Gaussian correction. Original folder renamed to `maps`.
- User reported disconnected graph on5644Kphotos. Added `adm/match_graph.py`:
  only disconnected graphs retry verified cross-group GPS/sequence pairs at1280,
  retaining .70ratio/30inliers/25%ratio/spatialcoverage requirements; convert
  matches back to original640grid. Bounded256attempts/16imagefeatureLRU. Reports
  filenames/components and never creates GPS-only edges or silently drops frames.
  Singular/horizon-crossing pair transforms now rejected before projection.
  Reproduction before that guard also hit `Projection intersects horizon` after
  86.395s; `/maps` retained564 and proposed109279pairs. Corrected full run is
  `ISM/benchmarks/maps_recovery_v2`; do not assume success until its report exists.

See `ISM/GUIDE.md` for current operation and qualified benchmarks. `LeanADM/`
remains unchanged. Old sections below describe their saved historical revisions.

## Latest revision: natural sunset, no NLM (2026-09-15 UTC)

User explicitly removed NLM, reduced positive detail to 1.3, moved white/vibrance
controls into `pro_camera`, requested natural late-sunset grading and ray tracing,
then a fresh timed run. Current sequence: register original photos -> actual FSR
frame generation -> mapping -> Lanczos 2x -> positive detail 1.3 -> CLAHE -> fine
sharpen / contrast -> triangle ray tracing -> fused cinematic `pro_camera` grade.
The separate `LeanADM/` backup is unchanged. Final FSR spatial upscale stays off.

- `adm/cinematic_grade.py::pro_camera_grade` is now the CPU reference for the
  revised preset; `post_cinematic_grade` is the fused Metal version. Exposure
  -.06 stops, luma contrast .12, mild cool shadows / warm highlights with reduced
  tint on neutral colors, white reduction .3 and vibrance reduction .1 inside the
  preset. White reduction controls shoulder slope (white maps to .9475 before
  exposure), rather than the earlier destructive .7 endpoint. Hue-preserving
  gamut compression. No extra grade sharpening, vignette, bloom or grain.
- NLM remains available for historical numerical tests but is not called by
  `enhance` or the current benchmark. Likewise `finish_tone` is historical;
  `write_presentation` no longer calls it. Legacy config white/vibrance knobs
  override the pro_camera preset rather than adding a second pass.
- Real ray tracing: `adm/raytracing.py`, `shaders/raytracing.metal`, and the
  `adm_trace` native bridge build a Metal triangle acceleration structure and
  trace primary, solar-disk visibility and cosine-weighted diffuse bounce rays.
  At most two diffuse bounces; deterministic samples, bounded grid <=512x512,
  max 1M triangles / 512 MiB acceleration scratch+structure. The Python `trace`
  API accepts explicit registered triangle soup/materials. This dataset has no
  measured elevation: `relight_map` supplies only two flat-plane triangles,
  128x87 grid, 8 samples, artistic sun elevation 3 / azimuth 255. Flat-plane lighting
  preserves pixels exactly; it adds no building/terrain shadows or new detail.
  Do not describe it as recovered 3D geometry or physically relit photography.
- Updated shared bridge fixes `adm_device` string lifetime (device name had
  sometimes been empty); all shaders recompiled. 58 regression tests passed,
  including CPU/Metal grade parity (exact test fixture), revised OpenCV sharpen
  parity (mean .04078/255, max7 BGRA fixture), directional occlusion against real
  test geometry, repeated-ray determinism and flat-map pixel preservation.
- Presentation-only preview `metal_fsr3_sunset_preview` took 3.97762 s including
  decode/export from a saved neutral map. This is not the full-process timing.
  Active config `fsr_experiments/natural_sunset.json`, also `full_pipeline.json`.
  Fresh 306-image benchmark completed in `metal_fsr3_sunset_e2e`:
  **109.706055375 s full-process wall**, 10.29394 s under 120; no matching cache.
  All 306 retained, 915 virtual /1221 inputs, 305 adjacent pairs, no skipped pairs.
  Prep15.0628s, matching50.6148s, alignment7.4252s,
  FSR+mapping+neutral export32.4658s, sharpening/reports.6581s (kernels.6561s),
  ray tracing.03041s, grade.03239s /grade+PNG3.03868s, startup overhead.41035s.
  Peak process RSS 1.976 GiB (not total GPU memory), saved pipeline bytes248606214.
  Original306 transforms exactly equal LeanADM; all 42 frozen source/config
  files unchanged during the benchmark. Source/config hash manifest:
  `fsr_experiments/sunset_source_manifest.json`; validation `sunset_validation.json`.
- `fsr_experiments/sunset_e2e_summary.json` consolidates timings and checks.
  `metal_fsr3_sunset_e2e/{map_graded.png,overview.png,road_comparison.png}` are
  the actual fresh-run result and previews. The post-benchmark stage replay
  reproduced its full graded map exactly (SHA25684cfe7a037235696d5172265e4f4147e4a803636e4f1fac0f52f859722236c61).
  Same137961-pixel road mask, area-filtered to native scale: texture RMS
  LeanADM 2.31898 /previous grade.49926 /new grade2.44360; gradients20.0651 /
  3.22221 /22.0631. Local contrast proxy, not added real resolution.
  Remaining limitation: preserved global contrast clips 8.70% of bright-road
  pixels in at least one channel before grading; grade cannot recover this.
  FSR neutral map across separate runs is not bit-identical despite identical
  transforms (mean BGRA abs difference .05104/255,max201); do not claim exact
  FSR raster repeatability. Final full-map stage replay itself is exact.

## Completed previous full run and detail-loss audit

`metal_fsr3_lanczos_toned_e2e` completed all 306 originals in **120.36548s** wall
time, 0.36548 s over two minutes (internal 119.79328 s excludes process overhead).
915 unique generated views / 1221 compositor inputs, all 305 pairs, no match-cache
reuse. Original transforms exactly match saved LeanADM. Prep15.429s,
matching53.147s, alignment7.402s, composition/presentation/export43.815s.
NLM/sharpen4.324s; grade/export4.864s. Peak process RSS1.974GiB is not total GPU
unified memory. Frozen41-file source/config hashes were unchanged during that run.

Its post-benchmark full-map stage replay reproduced the saved final pixels
exactly. Same fixed bright-road crop/mask: neutral texture RMS 2.0865 -> NLM1.6429
(-21.3%), pro_camera1.9458 -> old white/vibrance finish.4993 (-74.3%). Contrast
before grade clipped8.65% of pixels in at least one RGB channel on that mask.
These are local contrast proxies, not recovered-detail/overall quality scores.
The assistant's former white endpoint .7 implementation caused severe highlight
flattening; it was not equivalent to a conventional Whites -30 slider.
Reports: `quality_audit/stage_audit.json` and `quality_audit/road_stages.png` in
that old run. `fsr_experiments/e2e_lanczos_summary.json` contains all checks/timing.

## Historical presentation change: no final FSR upscale

User requested whites -30%, vibrance -10%, then explicitly removed FSR **before
sharpening**. FSR frame generation is retained. Current order: original registration
-> FSR interpolation -> mapping -> NLM -> Lanczos 2x -> detail/CLAHE/fine sharpening
-> pro_camera grade -> white/vibrance reduction. Current presentation defaults:
FSR passes 0, Lanczos scale 2. Grading adjustments default to zero generally;
the active benchmark configs set .3/.1.

`output/killerofmaps/metal_fsr3_lanczos_toned/map_graded.png` is the new 13000x8788
result, generated from `metal_fsr3_full/map_neutral.png`. It took **9.12687 s**
for presentation and export only, with **4.22061 s** sharpening (NLM 3.71980)
and 4.48939 s grading/export. No full registration/frame-generation benchmark
was rerun, so do not call this a 9-second end-to-end pipeline or reuse 143.6 s
as a measurement of the new settings.

White reduction uses a monotone highlight curve above .5 luminance, mapping
neutral white to .7 at reduction .3; it does not recover already clipped detail.
Vibrance scales chroma by 1-reduction*(1-HSV saturation). Both have Metal and
CPU-reference implementations. Five presentation hardware tests pass, including
the new tone parity/monotonicity/alpha test. The changed NLM reflection arithmetic
also passed parity tests. All image dimensions, projection and transparency are
retained with a separately correct presentation pixel transform.

Current config: `fsr_experiments/full_pipeline.json` and `no_fsr_upscale.json`.
Earlier config preserved as `full_pipeline_fsr_upscale_reference.json` and in
the original run's resolved config. Current previews: `comparison.png` (LeanADM,
previous enhanced, updated) and `road_detail_comparison.png` beside the new map.
`adm.presentation.write_presentation` now supports reusing an existing neutral
map; `experiment_fsr.py presentation_map --config ... --output ...` runs that path.


## Current FSR experiment (2026-09-15 UTC)

The lean baseline below remains the validated non-FSR reference. The user asked
for this experimental order: originals -> registration + FSR interpolation ->
mapping -> FSR upscale -> supplied NLM/Lanczos/detail/CLAHE/sharpen recipe ->
supplied cinematic grade. See `docs/FSR_PIPELINE.md` for implementation and evidence.

- Actual AMD Frame Interpolation 1.1.3 from SDK v1.1.4, revision
  `c6efa6bf7f2027b3ec94f28578bb5965eabb9e55`, now executes on Metal. Unmodified
  GLSL is compiled by glslang, optimized by spirv-opt, translated by SPIRV-Cross
  to MSL 4.0, and compiled at runtime. Requires macOS 26 for coherent textures.
  It uses game motion derived from ADM homographies and assumed constant flat
  depth (60 m in this run); optical-flow assistance is disabled. Never call the
  earlier custom `virtual_views.py` blend AMD FSR frame generation.
- `FSRViews` runs after original registration. For the 306-image benchmark all
  305 adjacent pairs pass; levels=2 creates 915 unique generated views and 1,221
  compositor inputs. Zero synthetic matching constraints. A 2-real/3-generated
  image cache means the two compositor traversals execute 1,830 generation calls.
  No generated-photo files or match cache are saved.
- Final upscale uses FSR1 EASU, followed by the user's supplied recipe. The first
  complete run uses FSR 2x then Lanczos scale 1 (2x total). An optional question
  about 2x versus 4x total was unanswered; 2x was stated as the first timed-run
  assumption. Do not claim the user explicitly chose it. Final presentation is
  bounded to 160 million pixels.
- NLM, fixed-point Lab conversions, Lanczos, CLAHE, Gaussian/detail/contrast and
  global `pro_camera` grading run on Metal. Other presets/regions remain CPU.
  Codecs, SIFT/RANSAC/geography and FSR adapter motion/alpha preparation still use
  CPU; this is not an entirely GPU-only program. `presentation_geometry.json`
  describes the enlarged graded map separately from neutral `map_geometry.json`.
- NLM optimization reuses horizontal distance sums; textured/smooth/flat tests
  are pixel-identical to the retained first Metal implementation. On a real
  1280x1280 FSR-upscaled photo NLM improved 1.32546 -> .20624 s (6.4x); entire
  sharpening .21302 s, grading .00705 s. CPU-reference sharpening difference
  mean .07584/255, p99 2, maximum 9; after grade mean .20615, p99 4, maximum 30.
  Therefore close CPU agreement, not bit-identical CPU parity. Hardware tests
  also verify actual FSR identity/translation and Pillow blur/grade parity.
- Completed fresh run: `output/killerofmaps/metal_fsr3_full`, **143.59494 s** for
  all 306 photos, 11,254 accepted edges, 915 virtual views. Preparation 15.352 s,
  matching 66.896 s, alignment 7.552 s, composition/presentation/export 53.275 s.
  Neutral map 6500x4394, final graded map 13000x8788. FSR final upscale .094 s;
  sharpening 14.148 s (NLM 13.744); grade/export 4.717 s. It missed 120 s.
- The user then asked to see this result compared to saved LeanADM. Side-by-side
  overview and matching detail crops are at
  `output/killerofmaps/fsr_experiments/comparison_lean_fsr3/{overview,detail_crops}.png`.
  Visually the requested combined enhancement is over-sharpened/high-contrast,
  with halos and loss of bright pavement texture. No overall quality win proven;
  do not attribute the combined grade/sharpen change specifically to FSR.
- `LeanADM/` is a separate clean non-FSR backup containing source, config and
  the saved 83.2055 s 306-photo benchmark map/reports. User explicitly said no
  need to rerun this backup; it was not separately validated.
- Remaining optimization investigation: 30-pair FSR profile .57348 s, split
  upload .01575, CPU motion .17354, shader dispatch/read .16895, CPU alpha .21528.
  No native motion/alpha adapter kernels implemented yet. A cheap exact reflection
  arithmetic optimization was added to NLM after the 143.6 s run; that run's
  timing does not include the new change. No further full benchmark completed.


## Current revision: lean pipeline (2026-09-14 evening)

This section supersedes the historical inspection below. See `docs/LEAN_PIPELINE.md` for implementation details, measurements, and artifact links.

- CLI now defaults to `png`: one neutral map, geometry/transforms and small reports. Match caching is optional via `--cache-matches` or `--resume`; diagnostic profile retains it. Corner dumps, support PNGs, lighting, GPS leave-one-out/heading diagnostics, and post-registration footprint-index exports are omitted from the normal PNG path. Full-pose corner checks used for actual matching are retained.
- Vector exports actually bypass compositing and save `project.json`. Config() still defaults to diagnostic for programmatic compatibility; CLI overrides it. Normal coherent output uses 8 rather than 12 primary full-resolution bytes/pixel by omitting support arrays. Diagnostic rendering still exposes coverage/source IDs.
- Matching removes per-descriptor Python DMatch construction, unused square roots and GPS-mode native-coordinate projections. Match settings/candidates are unchanged. `matching_report.json` separates features/upload, descriptor matching, verification/bookkeeping, and cache/diagnostic export time.
- Solver remains one C ABI operation with GPU numerical work, but checks convergence between four-iteration command batches. Actual encoded iterations are reported separately from the maximum budget. At most three frozen iterations remain within the final batch. Cache identity now includes bridge and registration source.
- Fresh full-process timings: SIYI 9.55985 s (previous 30.00013); Archive 9.11155 s (previous 33.63460); 306-image 128 MiB profile 83.20550 s (previous 119.48827). All neutral BGRA pixels and image-to-map transforms are exactly identical to saved baselines. New directories are `metal_lean`, with sibling `metal_lean_timing.json` and `evaluation_lean/comparison.json`.
- New output sizes: SIYI 64.31 MiB, Archive 47.37 MiB, 306 images 52.73 MiB. 306-image internal times: preparation 15.604 s; SIFT/upload 6.996 s; descriptor matching 43.173 s; verification/bookkeeping 7.175 s; alignment/geography 7.398 s; composition/exports 2.395 s. Alignment encoded 12 iterations (9 active) rather than 60.
- Crispness comparison now area-filters minification before geographic warping and uses the exact shared eroded coverage mask. ADM/ODM mean-gradient ratios: SIYI .7841 at 5.00 cm/pixel, Archive 1.0663 at 21.64 cm/pixel. Laplacian-variance ratios .4478 and 1.3401. These are edge/texture proxies, not overall quality or resolved-detail ratios. Earlier bilinear-only, separate-footprint sharpness figures overstate ADM and are not interchangeable. Crop figures are in each dataset's `evaluation_lean/crispness_crops.png`.
- Validation: 38 pipeline/GPU tests passed after implementation; two additional comparison tests verify alias filtering and shared-mask/parity behavior. All 40 pass. Preserved source before trimming is `/private/tmp/adm-before-trim-u0369xhm` (temporary, not guaranteed durable). No saved reference datasets or old outputs were deleted.

## Historical inspection before the lean changes

Inspected 2026-09-14 from local source and saved artifacts. The ADM root has no Git repository. The legacy source checkpoint is `checkpoints/full_metal_86s/`; the separate `ODM/` repository is at `1a7134170e6100cade3236486a279d6241f11f99`.

Source snapshot SHA-256: `aacfc4f06387ed28a46cb26be0ac612f5cab0cdc56ed4794c8dea04917009529`. Computed over sorted `adm/*.py`, `shaders/*.m`, and `shaders/*.metal`, concatenating each relative path, NUL, file bytes, NUL.

## Identity and current behavior

- `metal_optimized` is an output directory/profile name, not a module. `--backend metal --gpu-optimized` sets `gpu_alignment`, `coherent_composite`, `compact_outputs`, and `metal_recompute_warps` true.
- ADM is a planar photomosaic with one homography (eight projective parameters) per photograph. It does not reconstruct cameras in 3D, triangulate scene points, estimate depth, or build a mesh. Its optional input heightmap affects lighting, not registration or orthorectification.
- The optimized profile retains visual feature matching. Do not confuse it with the old `metal_fast` GPS-only preview experiment.
- CPU work: selection, image decoding, metadata, OpenCV SIFT/RANSAC, graph setup, geographic fitting, codecs, and reports. Metal/MPS handles exact descriptor matching, iterative homography alignment, coherent source selection, sampling, and supported lighting/cropping.
- The latest source changes include lasso cropping, output budgets/export profiles, buffer lifetime safeguards, and configurable coarse-compositor memory. Older README/GPU documents describe earlier behavior.

## Selection, matching, and geography

Sources: `adm/selection.py`, `metadata.py`, `geometry.py`, `vocabulary.py`, `registration.py`, and `pipeline.py`.

1. Sort supported input image filenames. Hash inputs and exclude configured initial frames, explicit exclusions, unreadable images, exact duplicates, detected blue annotation labels, and near duplicates. Defaults: skip first 7; thumbnail MAE threshold 1.5. Saved comparison configurations set skip-first to zero.
2. Read EXIF/XMP and optional explicit sidecar overrides. Altitude AGL comes from configuration/sidecar, never automatically from EXIF sea-level altitude. Default AGL 30 m is a placeholder; HFOV 81 degrees, nadir pitch, and square-pixel VFOV derivation are assumptions. Unsupported EXIF orientation is rejected.
3. Saved optimized runs use `--gps-centers`: a KD-tree selects potentially overlapping camera-center circles from AGL/FOV plus GPS tolerance (default 5 m). Recorded yaw does not place images. Without this flag, full camera pose predicts footprints and permits geographic corner verification. Explicit local mode uses sequence neighbors (default 6).
4. Resize each image to registration width (default 1400; Archive/killerofmaps use 640). CPU OpenCV SIFT requests 7000 features with contrast threshold .025. Keep descriptors resident on GPU. MPS computes batched descriptor dot products (512 query rows); MSL norms and 32-lane top-two reductions recover exact squared L2 distances with stable index ties.
5. Apply the .70 nearest/second-nearest ratio test (squared threshold .49 on GPU). Require at least 35 ratio matches. CPU RANSAC fits a homography with 2.5 px tolerance; require at least 30 inliers, 25% inlier fraction, and 3.5% source-image convex-hull coverage. Full-pose mode additionally checks geographic disagreement. Refit accepted inliers and retain at most 160 evenly indexed correspondence pairs per edge; edge `n` retains the original inlier count.
6. Four 50x50 mean-centered corner vocabularies are a separate path: predicted-pose rectification, NCC search radius 24, std >=3, NCC >=.8, delta RMSE <=.02, uniqueness margin >.02, geographic support when available. Four accepted corners can supply an edge. In GPS-center mode these checks are diagnostic-only, after SIFT; non-diagnostic profiles skip them. The latest 306-image runs have zero evaluated corners and all 11,254 edges labeled SIFT fallback.
7. After visual alignment, convert registration-space homographies to native image coordinates. GPS-center fitting uses a CPU float64 robust four-parameter similarity to local WGS84 east/north coordinates, with a vertical-axis flip, soft-L1 scale .75 m, and leave-one-out diagnostics. Requires >=3 images, pitch within 5 degrees of nadir, >=3 m center spread, and median residual <= configured GPS tolerance. GPS establishes scale/north; these residuals are not independent survey accuracy.

## GPU alignment

Sources: `adm/gpu_alignment.py`, `shaders/alignment.metal`, `shaders/ADMMetal.m::adm_align`.

- Anchor image is index N//2. Initialize all homographies by greedily extending the connected graph using the largest-inlier-count crossing edge. Reject disconnected graphs instead of dropping frames.
- Normalize coordinates by registration width. Upload eight float32 homography parameters per image, correspondences, ownership, edge lookup, and incidence arrays. Anchor parameters remain fixed through the normal equations.
- Default 60 iterations, allowed 1..200. One C ABI call encodes the whole budget into one command submission: analytical Jacobians/coordinate-wise soft-L1 weights; per-edge Hessian/gradient blocks; normal-matrix assembly; MPS Cholesky factorization/solve; trial cost reductions; device damping/acceptance/commit.
- Objective is the sum of `sqrt(1+r_px^2)-1` per residual coordinate. Initial damping .001; accepted cost-decreasing steps multiply damping by .35 (floor 1e-7); rejected steps multiply by 8 (ceiling 1e12). Diagonal regularization is damping times max(diagonal,1e-8), plus 1e-9.
- Objective convergence: accepted improvement <1e-6*max(old objective,1). Step convergence: squared relative step <=9e-14, successful factorization, damping <.01, and either a prior accepted step or an already exact solution. Trial projective denominators must exceed .05. After convergence parameters freeze, but all encoded stages still execute; this is not early termination of the command buffer.
- Download final parameters/state; convert to float64 for diagnostics/geography. Nonfinite results or nonconvergence block compositing.
- Jacobian uses 32 floats per correspondence; normal/factor matrices are each (8N)^2 float32 entries. Planned scratch = observations*156 + edges*1088 + (8N)^2*8 + (8N)*12 bytes, limited to 512 MiB, excluding inputs and MPS internals. This is not constant-memory in image/edge count.

## Coherent compositor

Sources: `adm/composite.py`, `adm/coherent_composite.py`, `shaders/coherent.metal`.

- Reject homographies whose image footprint crosses the projective horizon. Fit all footprints into a canvas with longest side `max_output_dimension` (default 6500), a small margin, and north-up orientation when geographic. Output pixel spacing follows this canvas sizing, not a requested physical GSD.
- Coarse scale is min(1, 384/max(W,H), sqrt(budget/(N*8*W*H))). Round coarse dimensions, with minimum 2 per axis. `coherent_budget_mb` defaults to 32 MiB and targets observations+confidence+unary storage; rounding and other buffers mean it is not an exact total-allocation cap.
- First source pass: decode one image at a time, inverse-warp samples onto the coarse grid. Trim normalized edge distance below .02 (about 1% of image dimensions). Confidence is squared normalized edge distance. Store uchar4 observations and half confidence; form confidence-weighted mean colors.
- Unary source cost: `3*(1-sqrt(confidence)) + photometric*min(RGB_squared_distance/(3*22^2),4) + .0001*source_index`. Photometric weight is .15 only if weighted mean per-channel variance <225; otherwise zero to avoid voting displaced foreground objects away. The 22 scale is hardcoded here, not read from `selection_sigma_rgb`.
- Initialize each coarse label from minimum unary cost, then do 32 red/black relaxation sweeps. Four-neighbor Potts penalties favor matching labels, with weight `2.5/(1+neighbor_mean_RGB_squared_distance/(3*30^2))`. Preserve existing labels on ties. This is local label relaxation, not a globally optimal graph cut.
- Second source pass: compute per-source low-frequency exposure bias against the mean, excluding variance >=900. Clamp offsets to +/-32 in 8-bit channels, confidence-weight them, blur numerator/weight with MPS Gaussian sigma 6 coarse pixels, then interpolate/normalize/clamp correction at rendering time.
- Render each source footprint ROI directly from the original decoded photograph. Score = normalized edge distance +2 when that source matches the coarse label. A valid labeled source therefore wins; where unavailable, choose the best geometric confidence. Count coverage independently. Each output pixel has one source's detail plus smooth exposure adjustment; no stochastic switching or high-frequency averaging between photographs.
- Full-resolution primary buffers total 12 bytes/pixel: best float32 (4), BGRA uint8 (4), coverage uint16 (2), source ID uint16 (2). There are additional coarse buffers, decoded source/upload, filtering scratch, CPU masks, and codec allocations. At 6500x4394 the primary buffers alone are 342,732,000 bytes. No output-tile scheduler exists.
- Source IDs are zero-based; 65535 means uncovered. Footprint vectors reference original source images; index lookup returns image support with tile ID -1, not the legacy raw tile API.
- A flat homography cannot correct general 3D parallax. Coherent labels can still cut elevated objects. Seed controls other pipeline operations but this compositor does not sample random source labels.

## Runtime and optional stages

- `adm/metal.py` builds an Objective-C dylib using xcrun clang, keyed by bridge-source hash. The bridge compiles concatenated pipeline/alignment/coherent/lasso MSL at runtime in safe math mode. Shared Metal buffers expose NumPy views whose owner retains the allocation; CPU reads require synchronization. Backend is thread-local and restricted to its creating thread.
- MPS product cache clears at 64 shapes; filtering scratch cache at 4 shapes. Gaussian convolution uses reflected halos and three private RGBA32F textures per cached shape, at most (2048+2r)x(256+2r). Every exported bridge function has an autorelease pool.
- Python Buffer allocation checks live device allocation against 60% physical RAM, overridable with `ADM_METAL_MEMORY_LIMIT_MB`. Direct C solver scratch and private MPS textures are not individually guarded by that Python check; the solver has its separate planned-scratch limit. Device max-buffer size and end-of-run allocated bytes are not peak memory measurements.
- Optional lasso (`adm/lasso.py`, `shaders/lasso.metal`) builds an opaque/nonblack coarse mask, opens the footprint, casts 512 inward boundary rays, reduces to 64 polygon anchors, and refines eight times. Default inset 1%; reject missing central support or removal of >half covered pixels. Retained colors and canvas/world transform stay unchanged; removed coverage/IDs clear. It does not repair internal registration.
- Diagnostic compact flat lighting uses a fused sRGB decode/warm treatment/encode kernel and omits HDR/float lighting exports. Supplied height fields enable the other terrain-lighting path. Optional shadows use contrast-derived relief and three short screen-space rays; they are artistic, separate from neutral mapping, and do not reconstruct geometry.

## Export/cache discrepancies: preserve these distinctions

- CLI defaults to `--export-profile vectors`, while `Config()` defaults to `diagnostic`. Programmatic `run()` calls without an export_profile argument can therefore behave differently from CLI runs.
- Despite CLI help, **current `compose()` always computes the raster and writes `map_neutral.png`, `coverage.png`, `source_ids.png`, transforms, and geometry under all profiles**. `vectors` is not yet a vector-only execution path; `png` is not yet a single-raster export path.
- `adm/project.py::write_recipe()` exists but has no pipeline call site. Current runs do not write `project.json`, despite the lighting summary referring to a saved render recipe.
- Profiles vectors/png skip the entire lighting stage, including requested `--stylized-shadows`. Historical SIYI/Archive optimized runs predate this default and include golden/shadow PNGs. Their old benchmark command now has different behavior unless diagnostic is explicit.
- Explicit CLI profiles activate output quotas: 32 MiB default for vectors, 256 MiB otherwise; `--max-output-mb` overrides. Managed exports use guarded atomic writes, conservative raw-size PNG preflight, and a 512 MiB free-disk reserve. Current raster exports can defeat the nominal vector budget. Existing files count toward the cap.
- Compact mode omits raw `tiles/`, SQLite tiles, `warps/`, and full linear lighting exports, **but still saves `registration.npz` even without --resume**, plus metadata/transforms/support images. Diagnostic mode also saves corner vocabulary caches. No automatic cleanup of previous outputs is performed.
- Resume only reuses accepted match edges when the fingerprint agrees; alignment and rendering rerun. Fingerprint includes selection/input hashes, metadata, configuration, GPS mode, OpenCV, and backend fingerprint. Backend fingerprint hashes MSL files plus `metal.py` and OpenCV; contrary to older wording it does not directly hash `ADMMetal.m` or all registration/pipeline code. Dylib build hashing is separate.

## Saved measurements and storage (inspected, not newly benchmarked)

| Run | Images | Full process wall | Alignment + geography | Match edges / observations |
|---|---:|---:|---:|---:|
| SIYI optimized | 29 | 30.000 s | .459 s | 187 / 26,555 |
| Archive optimized | 63 | 33.635 s | 2.800 s | 1,551 / 241,429 |
| killerofmaps optimized, 32 MiB coarse | 306 | 121.529 s | 35.337 s | 11,254 / 1,747,493 |
| killerofmaps optimized_128mb | 306 | 119.488 s | 35.222 s | 11,254 / 1,747,493 |

Latest completed variant is `output/killerofmaps/metal_optimized_128mb` (Sept 14 00:42 local). AGL 60 m, registration width 640, skip-first 0, canvas 6500x4394, coarse grid 285x193 versus 142x96 at 32 MiB. Both large runs use vectors and explicitly raised output cap to 2048 MiB; both save neutral/support rasters but no golden/shadow output. Baseline 306-image residual median/p95 .22791/.68777 registration pixels; planned solver scratch 332,824,268 bytes. Its matching takes 67.889 s and reports 9 executed / 60 encoded iterations.

Measured logical file sizes below are MiB, exclude input photographs, and do not deduplicate hard links across separate output directories:

| Run | Match cache | Vocabulary cache | Maps/previews | Support rasters | Metadata/transforms | Total |
|---|---:|---:|---:|---:|---:|---:|
| SIYI optimized | .39 | 2.48 | 193.27 | .62 | .31 | 197.07 |
| Archive optimized | 3.28 | 4.37 | 142.31 | .92 | 1.78 | 152.68 |
| killerofmaps optimized | 23.92 | 0 | 52.35 | 1.62 | 2.29 | 80.18 |
| killerofmaps optimized_128mb | 23.92 | 0 | 52.35 | 1.64 | 2.29 | 80.20 |

The 306-image `registration.npz` is 25,077,282 bytes. Coarse budget 128 MiB is runtime working storage, not a 128 MiB saved cache file. Earlier legacy SIYI output was 1,422.04 decimal MB before cleanup. A saved cleanup manifest records 4,130,376,735 bytes removed across regenerable intermediates/trial outputs; those are historical, not currently present cache bytes.

## Exact ODM comparison

- Saved baseline is ODM 3.6.2 in ARM64 Docker, 12 visible CPUs and 7.75 GiB guest RAM. Commands are `python3 /code/run.py --project-path /datasets <dataset> --orthophoto-png --optimize-disk-space`. Reconstruction quality remains default: DSPSIFT/high, 10,000 minimum feature target, FLANN, incremental SfM, camera-parameter optimization, medium point-cloud quality, default 5 cm requested orthophoto resolution. Densification logs explicitly use `--cuda-device -2` (CPU). ADM runs natively with six OpenCV workers and Metal/MPS.
- ODM stages (`ODM/stages/odm_app.py`): dataset -> split/merge -> OpenSfM features/matching/tracks -> 3D reconstruction/bundle adjustment and undistortion -> OpenMVS dense depth/fusion -> point filtering -> 3D and 2.5D meshing -> texturing -> georeferencing -> optional DEM products -> orthophoto -> reports.
- Saved options have skip_3dmodel=false and use_3dmesh=false: ODM builds/textures both 3D and 2.5D meshes; the orthophoto uses the 2.5D mesh. DSM/DTM exports are false; do not claim those raster products were enabled merely because internal surface reconstruction occurs.
- ODM estimates 3D camera/scene geometry and camera parameters; ADM aligns 2D homographies then fits a geographic similarity. ADM has no lens-distortion calibration/undistortion stage. ODM texture selection/seam leveling operates on reconstructed surfaces; ADM selects coarse connected photo regions in the flat map and applies bounded exposure bias.
- Saved feature-processing widths: ODM 1920 SIYI /640 Archive; ADM 1400/640. ODM output rasters: 877x678 at ~5 cm/pixel (SIYI), 1621x1384 at ~21.64 cm/pixel (Archive, GSD-limited). ADM: 6500x5458 at .7947 cm/pixel; 6500x5689 at 5.6664 cm/pixel. Finer raster sampling does not establish more resolved detail or accuracy.
- ODM wall times: 279.824 s SIYI and 352.776 s Archive; ratios to saved optimized ADM are 9.33x and 10.49x. They compare different workloads and environments. From consecutive ODM log stage starts, dense MVS + meshing + texturing take about 207.52 s SIYI and 303.04 s Archive; ADM has no corresponding stages. ADM's own legacy->optimized SIYI alignment improves 49.921 s to .459 s; that comparison isolates its alignment revision better than ODM totals.
- Actual retained ODM outputs with optimize-disk-space enabled, excluding input image folders: 144.94 MiB SIYI, 145.15 MiB Archive. Corresponding ADM optimized directories are 197.07 and 152.68 MiB. Do not claim ADM always saves less disk: these ADM outputs are larger and include three large PNGs at different sampling resolutions. Persistent match caches, diagnostic output, runtime memory, and total retained disk are different measures.
- Archive held-out nominal-center RMSE: ADM 2.987 m, ODM 2.982 m. These are supplied synthetic/crop centers, not surveyed checkpoints. SIYI GPS participates in fitting. Homography correspondence residuals and ODM 3D reprojection residuals have different definitions and are not directly rankable as accuracy.
- Neutral-map comparison resamples onto a common geographic grid/physical resolution. SIYI optimized SSIM against ODM .07410, Archive .60021. Agreement with ODM is not ground truth. The optimized compositor improves the saved goal-detail example versus legacy ADM while SIYI geographic SSIM falls from .10432; no blanket quality win.
- No corresponding saved ODM timing was located for the 306-image killerofmaps run. Do not extrapolate a measured ODM speedup for it. ODM's `--fast-orthophoto` exists but was not the measured baseline; it skips dense MVS and forces skip-3dmodel while retaining SfM and surface mapping.

## Evidence and validation

Quality attribution reviewed against saved tests/comparisons:

- No stage-by-stage percentage attribution exists. GPU numerical parity tests, synthetic failure cases, and multi-change real-run comparisons answer different questions. Do not present runtime shares or source-boundary counts as percentages of map quality.
- Latest 306-image 128 MiB run: preparation 14.462 s (12.16%), matching 67.104 s (56.43%), alignment/geographic fit 35.222 s (29.62%), composition/index/export 2.136 s (1.80%), lighting skipped. Shares use the 118.924 s internal timer; external process time is 119.488 s. Existing timers do not isolate SIFT, descriptor matching, RANSAC, cache writing, GPS fitting, or individual compositor stages.
- GPS-only versus optimized Archive comparisons: geographic ODM-agreement SSIM .28493 -> .60021; MAE 46.959 ->13.215 on a common physical grid. This compares multiple changed stages, not an isolated matching ablation; it is also agreement with ODM rather than independent accuracy.
- SIYI source-transition fraction drops .17888065 -> .00232177 (17.888% -> .232%, a 98.70% reduction). This quantifies label coherence, not overall image-quality improvement. Geographic SSIM against ODM simultaneously drops .10432 -> .07410 versus legacy visually aligned ADM.
- Synthetic displaced-object test asserts output interior equals one original photograph, preserving one object without duplicated/erased copies. Exposure test uses source values 80/120 and requires neighboring output jumps <8 levels in the tested interior. These validate specific cases, not a real-dataset average benefit.
- Corner delta threshold .02 -> .10 experiment: 226 evaluated patches, 0 ->119 passing delta alone, 0 ->2 passing all gates, zero pairs with four corners. All 190 edges/homographies/correspondence sets remain identical; no map-quality improvement demonstrated (`output/101SIYI_IMG_corner10/RESULTS.md`). This experiment predates the 187-edge optimized SIYI comparison.
- Lasso synthetic tests preserve surviving pixels exactly, remove protruding/black borders, retain >80% of covered pixels in the test fixture, preserve an interior dark region and canvas, and reject unsupported central geometry. It is off in the latest measured run.
- 32 ->128 MiB coarse budget changes 306-image grid 142x96 ->285x193; saved alignment residuals are identical. No independent compositor-quality comparison for this budget change was located. Larger budget does not have a measured quality percentage.
- Existing legacy Metal test confirms recomputed warps versus disk-cache mode are pixel-identical; optimized resume test confirms unchanged neutral output and unchanged match-cache timestamp. Cache persistence is for reuse, not additional map information.

- Implementation sources above; saved results: `output/comparison/{siyi,archive}/metal_optimized/`, sibling timing/comparison JSONs, `output/killerofmaps/metal_optimized*/`, and `output/comparison/{siyi,archive}/odm/{options.json,log.json,opensfm/config.yaml,timing.json}`.
- Historical `output/metal_validation.json` reports 38 tests, zero failures/errors, lighting max absolute error 1.19209e-7. Timestamp Sept 14 00:34 precedes the final configurable-budget edit at 00:40. This review read the tests/reports; it did not rerun the GPU suite or full benchmarks.
- Review ran a tiny temporary-file probe: parsed actual CLI default as vectors and invoked the common compositor export path to confirm neutral/coverage/source-ID PNGs are written and no project.json appears. Temporary probe data were removed automatically. No implementation changes or cache deletion were performed.

## ODM native performance update (2026-09-16)

See `ODM-Fedora/PERFORMANCE_UPDATE.md` for retained changes and validation limits. Installed candidate-score cache passes 1,000,000 bit-identical score checks; 18.3% lower focused score time is not an end-to-end gain. CPU cap and stage-specific RAM budget added. Rejected projection SIMD and batched mesh-read experiments are not shipped. Active 564-source/94-selected ultra job `output/test` was preserved and still texturing during installation; its elapsed time includes concurrent development work.

## Python parallel update and Metal validation (2026-09-16)

`acceleration/` contains the canonical shared Python runtime, downloaded pinned
AMD docs, and validation reports. Runtime copies ship in ODM-Fedora and MapRepair.
OpenSfM feature scheduling and FSR tiles are bounded; MapRepair photo preparation
and grading strips run in parallel while decisions remain ordered. Pixel math
uses optional PyTorch Metal/ROCm with exact parity and transfer-inclusive timing
gates. No NPU implementation exists. HIP FSR now requires exact pixels.

Four real Ultra DSPSIFT images: serial63.67s/newparallel22.25s/upstreamparallel22.28s,
identical points/descriptors/colors. No material gain over upstream parallel
extraction. MapRepair real6-photo scan/repair 4.84s->2.97s, identical repaired
pixels/ownership/report. Grading JPEG bytes match CPU and Metal. 34 MapRepair
tests pass. Metal100batches25,048,960pixels exact; liveallocation0->0, reserved
peak75,988,992bytes. HIP host equations36cases exact, NOT hardware-tested.
Scheduler1000job stress:24MiB estimated budget, injectedfailure handled.
Six-photo native Ultra smoke185.65s under old94-run contention completed. All
six feature arrays match old run; final orthophoto grids differ (historical8
threads versus4); full pipeline bitwise parity is NOT established.

User then requested all36 already-deduplicated photos directly, no deduplication
or stride for this test. Benchmark is `output/odm_macos_ultra36_parallel_20260916`,
8threads/Ultra/2.3cm, CLI `fedora/benchmark_selected.py`. It temporarily SIGSTOPs
old94 texrecon63558 and resumes it in finally. Check benchmark.json for current
status and resumption. Production app still selects every sixth unique photo.

36-photo benchmark completed:522.749665s, all36 cameras,83722points,.982px
reprojection(not surveyed),sampled peakRSS4716576768bytes. OpenSfM169.84s,
OpenMVS288.87s,filter8.25s,mesh15.46s,texture16.63s,geo8.23s,ortho2.06s.
Base2070x1499,FSR4140x2998,boundsand2xalphaexact. Old94texrecon63558resumed
automatically and confirmedrunning. User now targets<300s for~250deduplicated4K
photoswithoutqualityloss; unmet. Fastorthoskipsdense anduses8xcoarserDSMstep
inthisODM;not a quality-equivalent speedup. UserconfirmednolaptopSSH;include
on-laptoptests.


## 2026-09-16 custom AMD kernel / adaptive Ultra source bundle

Latest user authorizes custom HIP/XDNA2 code and hybrid flat-region Ultra,
but accuracy cannot be traded silently. Added canonical acceleration/parallel_runtime/
kernels/{hip.cpp,composite.h,flat.h,xdna_composite.cpp}, native.py, xdna.py,
validate_native.py, test_native.py, synced into both apps. Native HIP exact BGRA
composite uses two streams/pinned reusable staging (32MiB); classifier one block
per32px tile. XDNA2 16-lane uint32 source + IRON transform_parallel two DMA
channels/column; pinned published wheel1.4.4.dev3+gf50bef7 SHA in setup-amd.sh,
Peano22.0.0.2026090701+3e93bf7b. These do NOT port SfM/dense/meshing to AMD.
No HIP compiler/device or XDNA NPU here; host C++ math only. Actual device parity,
max-size classifier/composite stress and speed must run on laptop. Auto native
HIP selects only after byte parity and >=10% transfer-inclusive speed improvement;
NPU opt-in ADM_ENABLE_XDNA=1 after compatible Python3.12/XRT setup. Failed target
libraries renamed .failed. Native target test reports hardware_execution only
after certification executes on device. No under5min/250 result.

OpenMVS patch SuperBuild/cmake/openmvs-hybrid.patch + adm_hybrid.h/adm_flat.h:
ODM_HYBRID=1 enables strict inverse-depth-plane/NCC flat tiles, 32px tiles,
3x3 eligible-neighbor erosion; >=4 views; skips middle full-res photometric
iterations only. Full first/last/coarse/geometric passes retained. Off by default
because it changes algorithm/geometry; not a WebODM fast-ortho merge or proven
quality parity. Engine fingerprint includes hybrid flag. Mac native build passed.
Latest actual 36-depthmap 640px integration smoke in fedora/validation/
hybrid_native_audit/result.json: 26.563s, 36 classifier calls, ZERO skipped pixels
(8,294,400 retained). This is neither Ultra speed benchmark nor proof of shortcut
quality. Earlier console LOG was paused by OpenMVS; audit now prints stderr.
Six selected originals smoke output/odm_macos_hybrid6_smoke_20260916 initially
failed because copying build binary skipped install RPATH. cmake --install fixed
Mac library lookup; cached-SfM resume completed44.650s, quality_summary6/6.
Do not report resume as fresh benchmark. Initial failed output folders retained.

Pre-kernel full36 Ultra output/odm_macos_ultra36_terrain_20260916:
495.443862s vs522.749665 prior (5.22% single-trial total reduction).
Peak sampled process-treeRSS4,733,861,888bytes. SfM142.53s,dense297.08s
(dense slower than baseline288.87s). All36registered,1component84,873points;
2035x1509 GeoTIFF ->4070x3018 FSR, EPSG32610, bounds/nearest2xalphaexact;
preview inspected grass field. Fullpipelinepixelparity not claimed; native
reconstruction stochastic and raster extent differs frombaseline2070x1499.
This run preceded ready-first dispatch and new hybrid/kernels. Feature cache,
single-producer calibrated map and degraded4K DSPSIFT arrays exact. Four noisy/
blur/exposure/JPEG fixtures: ordered12.035s vsready12.030s (no speed claim).
Default skips PDF charts, includes tinyquality_summary; ODM_FULL_REPORT=1 restores.

Tests: 34MapRepair10.456s;16shared scheduler/pixel/native source tests;3feature
cache tests;Qtctest pass. No AMD/Fedora execution performed. Scripts setup-amd.sh,
optional setup-xdna-userspace.sh (Fedora44+, userspace -nokmod, source-pinned),
run-bounded.sh (systemduser scope70%high/80%max/1Gswap; may terminate job onOOM).
MainnativeRSS not proven boundedbyPython scheduler. Installed .so files excluded
from sourcebundles. AMD_KERNELS.md /UPDATE_LAPTOP.md describe actual limits.
Old94texrecon PID63558 exited naturally and output/test/flight/odm_orthophoto
andFSR exist; it is NOT suspended. No need to resume or kill it.


## 2026-09-16 MapRepair full-frame 1080p grading update

User requests instant slider-release preview, full-resolution download and
professional curves/control. Replaced center crop/manual preview with cached
full-image BGRA proxy bounded1920x1080, premultiplied-alpha area reduction,
2-entrycachefingerprinted by source size/mtime. Warm preview skips master/JPEG
loading. Full export always reads original lossless master (legacyJPEGfallback),
uses full-res strip grading with dynamic spatial halo. Separate preview/export
locks. LUT whitebalance/exposure avoids per-pixel powers. Appservedgrading.js
routes+51sliders/numericfields,8bandHSL,3zoneRGBbalance,gamma/hue/fade/clarity/
texture/sharpen. Master/RGBsmoothHermitecurves16points,S/mattepresets,drag/keyboard/
numberediting. PerjobbrowserlocalStorage,latest-revisionresponseguard,oneactive
preview/latestpending,exportclicksnapshotssettings,originalcomparison/reset.
Top original JPEG download explicitly labelled original finish.

38tests passed11.787s; after cachefastpath additional8gradingtests passed1.217s.
Browserreal3676x2228losslessmaster->1782x1080fullpreview (no crop), export3676x2228;
release-preview .1923s,51sliders,curvespreset/keyboard,queuedlatestchange,refresh
persistence,compare/reset,mobilenooverflow,zeroJSerrors. ArtifactsMapRepair/
validation/grading_1080p. Warmserver renderneutral .043s,exposure .165s,complex
curves/HSL/clarity/sharpen .26s beforefinalskipzerooptimizations. Not a guaranteed
latencyonFedora. Spatialdetailpreviewisapproximateafterdownsampling;fullsizeexport
is authoritative. No geometry/alpha/master changes. Updatedserver8768 PID62512
replaces58517afterconfirminglatestjobcomplete. Older8765/8766servers untouched;
restartthemifusingnewstaticUI,becausetheyneednewgrading.jsroute.

## 2026-09-17 MapRepair offline adaptive cinema

User requests genuinely crisper, cinematic mapping, offline dynamic grading and
ray tracing. New `MapRepair/adaptive_finish.py` is the default FINISH_MODE in
run.py: bounded scene proxy + native-patch Haar noise estimate; restrained neutral
balance, scene-dependent monotone tone, noise-gated multiscale luma detail,
local-extrema edge protection and gamut compression. No generated grass/detail,
bloom/grain/vignette. Existing real OpenRT CPU sun/sky tracer retained; adaptive
strength .35, FSR RCAS .10. No registered mesh supplied, so geometry is a plane:
artistic lighting, not newly recovered scene shadows. CLI --finish legacy.
Ray field now sampled per strip, avoiding full-size float RGB field allocation.

Interactive Auto cinema computes editable settings offline from the saved master
(half-strength because already finished). Added 52nd control halo protection,
lossless PNG default graded export (JPEG97/4:4:4 optional), and click-position
640x480 native detail inspector. Crop pixel output exactly matches the full
export including spatial halos. Auto response and crop response guarded against
stale edits; comparison also updates detail crop. Existing 1080p full-frame
release-driven preview preserved. PNG has alpha but no embedded georeferencing;
retain/rename world file (.pgw for PNG) + CRS. Old JPEG-only jobs cannot undo
existing JPEG loss. No new dependencies, cloud/model download or inference.

45 tests passed13.521s: serial/parallel strips exact, hidden-color invariance,
alpha, noise amplification gate, sinusoidal texture response, step edge no
overshoot, grayscale ramp monotonic/endpoints, full/detail crop parity and exact
PNG decode. Real maps enhancement-only:1838x1114 .254s,6500x4394 .2.508s,
6500x5978 1.987s (single trials, no IO/ray/FSR/export). Comparisons visually read;
improvement is restrained, blurry source areas still blurry. Saturated-channel
fraction can increase on real maps; do not generalize synthetic ramp test into
no clipping on every source. No ground-truth quality recovery or universal
blade-of-grass resolution claim. Artifacts MapRepair/validation/adaptive_cinema.
Browser52 controls,Auto->preview .486s,lossless3676x2228download,640x480crop,
original/grade,persistence,mobilezerooverflow,noJSerrors,noexternalrequests.

Actual complete web run36 photos (no dedup) job610c7acfb278477facc0df5cb8f77f2a:
16.194964s,4repairedtiles,sourcecrosschecks15.189s,adaptive.1967s,ray.1711s,
FSR.088s,export.1676s. New result3676x2228; full36-finished.png retained in
validation. This is MapRepair, NOT ODM reconstruction performance. Latest old
jobec5251fc7cd640998712a5c7b926e048 was complete before restarting server8768;
oldPID62512 replaced by80254. Older servers8765/8766 still untouched. Refresh
http://127.0.0.1:8768 for new routes/UI and completed36 result.
MapRepair-linux.tar.gz rebuilt202729bytes SHA256
94414ef2ac85f04d25835e6185d5f814b77ca17c3b3e3c7e31aa8e98b892ab0a.
Source bytes verified against tar. ODM bundle unchanged this turn.
