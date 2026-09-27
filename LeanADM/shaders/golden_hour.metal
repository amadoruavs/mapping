#include <metal_stdlib>
using namespace metal;

// Buffers contain float4 RGB and ENU normals to avoid float3 host alignment traps.
// Input/output radiance is linear. Encoding and tone mapping happen at export.
kernel void golden_hour(
    device const float4 *rgb [[buffer(0)]],
    device const float4 *normal [[buffer(1)]],
    device const float *visibility [[buffer(2)]],
    device float4 *output [[buffer(3)]],
    constant float4 &parameters [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float elevation = parameters.x * M_PI_F / 180.0f;
    float azimuth = parameters.y * M_PI_F / 180.0f;
    float3 sun = float3(cos(elevation)*sin(azimuth),cos(elevation)*cos(azimuth),sin(elevation));
    float direct = max(dot(normal[id].xyz,sun),0.0f) / max(sun.z,.15f) * visibility[id];
    float shade = 1.0f + parameters.z*(clamp(.65f+.35f*direct,.35f,1.45f)-1.0f);
    float luminance = dot(rgb[id].xyz,float3(.2126f,.7152f,.0722f));
    float3 lit = rgb[id].xyz * float3(1.065f,1.015f,.94f)*shade + float3(.008f,.004f,0.0f)*(1.0f-luminance);
    output[id] = float4(lit,rgb[id].w);
}
