# AMD FidelityFX FSR1 source

Upstream: https://github.com/GPUOpen-Effects/FidelityFX-FSR

Pinned revision: `a21ffb8f6c13233ba336352bdff293894c706575`.
`ffx_fsr1.h` and `LICENSE` are unmodified upstream files. The FP32 EASU and
RCAS equations are adapted to Metal in `shaders/fsr.metal`. This is an ADM
port of the spatial algorithms, not AMD's FSR2/3/4 temporal runtime or frame
generation SDK. It uses buffer loads and full-precision division/square root
instead of texture gathers and AMD's approximate reciprocal intrinsics.
ADM adds transparent-boundary handling and a sharpening strength control.
