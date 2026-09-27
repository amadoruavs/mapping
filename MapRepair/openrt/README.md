# OpenRT integration

This adapter compiles the actual [Project-10/OpenRT](https://github.com/Project-10/OpenRT)
core from commit `ef6205f8ca51fd123da4a2f0b7b7a6ad27809b80`. The bundled
upstream sources and BSD-3-Clause license are in `../third_party/OpenRT`.
Three files carry a GCC portability fix: `BxDF.h` explicitly includes `<cmath>`
and uses the float overload of `std::sqrt`; `BRDFPhong.cpp` and `BRDFBlinn.cpp`
use the float overload of `std::pow`. GCC's standard library did not expose
the original `std::sqrtf`/`std::powf` spellings. Lighting algorithms are unchanged.
Its demos, datasets and GUI entry points are not needed by the adapter.

## Build

Requires CMake, a C++20 compiler, and **OpenCV 4 development libraries/headers**
(core, imgproc, imgcodecs, highgui, features2d, ml). Python's `opencv-python`
wheel alone does not supply the development package. On this Mac the existing
Homebrew `opencv@4` 4.14.0 installation was used; no system packages were installed.

```sh
# From the ADM directory. First use builds the worker automatically:
.venv/bin/python MapRepair/openrt_backend.py

# Or configure/build explicitly:
cmake -S MapRepair/openrt -B MapRepair/build/openrt -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/path/to/opencv4/cmake
cmake --build MapRepair/build/openrt --config Release --parallel 8
```

`OpenCV_DIR` can also be an environment variable for automatic builds. The
worker executable is cached by the adapter/upstream source hash. Build details
are in `MapRepair/build/openrt-<system>-<architecture>-<id>/build.log`. The id
includes the folder location, platform/C++ runtime and custom compiler/OpenCV
paths, preventing copied Mac binaries or relocated CMake caches from being used
on Linux. `bash run.sh` bootstraps the Linux dependencies and calls this builder.
The explicit CMake commands above remain available for a manual build.
macOS/Apple silicon and Ubuntu 24.04 x86-64 (GCC, OpenCV 4.6, container under
emulation) are tested. Windows CMake paths are supplied but not yet executed.

Set `RAYTRACE_BACKEND='openrt'` in `run.py` to use it. Set `'native'` to use the
existing smaller C++17 tracer if OpenRT's dependencies are unavailable. An
OpenRT build failure is explicit, with no silent substitution of another tracer.

## What it does

The worker uses OpenRT's `CScene`, `CPrimTriangle`, BSP acceleration,
`CShaderDiffuse`, point sun light, sampled sky light and `rayTrace` method.
It traces orthographic rays into a neutral diffuse scene using 16 deterministic
sky samples per point, and normalizes the illumination by a separately traced,
unoccluded ground plane. The lighting grid is interpolated over the original
map; map pixels are never resampled through the rendering camera.

The default input contains only the flat plane. With `OPENRT_LIGHTING='neutral'`,
both renders agree exactly and this step remains byte-exact. The launcher now
selects `'golden_hour_5pm'`: a warm sun (RGB 1.38/.98/.60) and cool sky
(.92/1/1.12), each normalized to unit Rec.709 luminance and weighted .65/.35.
The reference keeps neutral light, so the colored illumination survives the
ratio. `CLightSky` supplies its own pi factor; the adapter does not add it twice.
OpenRT's BGR buffers are converted to an explicit linear RGB lighting field.
Python applies this in linear light at .65 strength, fading the effect near
white to retain highlight detail. The 15° elevation / 255° azimuth is an
artistic low-sun setup, not a calculation of the actual sun at 5 pm.

This warms the existing image and preserves its photographed shadows. It does
not recover albedo or create real tree/building shadows from pixels. If a
registered `MESH_OBJ` is provided, its geometry can produce actual occlusion.
OBJ X/Y are original map column/row coordinates and Z is height in the same
pixel-unit scale. Materials are assumed neutral diffuse. This adapter evaluates
direct sun plus sky occlusion, **not** recursive global illumination.

The worker is a separate process to keep C++ OpenCV 4 independent of Python's
OpenCV version. Inputs/outputs are temporary float32 buffers; no render cache or
window is opened. Tests exercise neutral plane identity, a known wall shadow,
warm RGB ordering/luminance, highlight ramps, alpha preservation and exact
repeatability across thread counts. The launcher restores the previous
`siyi_low_light` + `pro_camera` finish and disables `cinematic_finish.py` by
default; the additional golden-hour color comes from OpenRT's colored lights.
