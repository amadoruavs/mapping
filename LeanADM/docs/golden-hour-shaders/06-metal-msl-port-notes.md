# 06 — Porting the golden-hour shaders to Metal (MSL) on Apple silicon

How each construct used in files 03–05 maps to the Metal Shading Language, what changes on an
Apple GPU (M4 Pro: TBDR, 16-bit register file), a minimal MetalKit host, offline compilation,
and the automatic converters (SPIRV-Cross, naga, Tint). MSL facts are checked against the
**Metal Shading Language Specification, Version 4.1** (Apple, PDF dated 2026-06-04,
[link](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)).

---

## 1. Side-by-side: GLSL ES 3.00 vs MSL vs WGSL

| Construct (as used in file 04/05) | GLSL ES 3.00 (WebGL2) | MSL (Metal 3/4) | WGSL (WebGPU) |
|---|---|---|---|
| Entry point | `void main()` per stage, separate files | `vertex VSOut vs_main(...)`, `fragment float4 fs_main(...)`, `kernel void cs_main(...)` — many in one `.metal` file | `@vertex fn vs_main(...)`, `@fragment fn fs_main(...)`, `@compute @workgroup_size(8,8) fn cs_main(...)` |
| Vertex attributes | `layout(location=0) in vec3 aPosition;` | `struct VSIn { float3 position [[attribute(0)]]; float3 normal [[attribute(1)]]; float2 uv [[attribute(2)]]; }; vertex VSOut vs(VSIn in [[stage_in]], ...)` — layout described by `MTLVertexDescriptor` (spec §5.2.4, §5.5) | `struct VSIn { @location(0) position: vec3<f32>, ... }` |
| Varyings | `out vec3 vWorldPos;` / `in vec3 vWorldPos;` | Return struct from the vertex function; fields with `[[user(name)]]` optional; fragment takes it as `[[stage_in]]` | `struct VSOut { @builtin(position) clip: vec4<f32>, @location(0) worldPos: vec3<f32> }` |
| Clip position | `gl_Position` | `float4 position [[position]];` in the output struct | `@builtin(position)` |
| Front facing | `gl_FrontFacing` | `bool ff [[front_facing]]` argument | `@builtin(front_facing)` |
| Uniforms | loose `uniform mat4 uViewProj;` (or UBO `layout(std140)`) | `constant FrameUniforms& u [[buffer(1)]]` — a struct in the `constant` address space; **float3 is 16 bytes/16-aligned, use `packed_float3` (12 B, 4-aligned) or float4 for tight CPU structs** (spec §2.5, Table of packed types) | `@group(0) @binding(0) var<uniform> u: FrameUniforms;` (std140-like: vec3 aligned to 16) |
| Uniform arrays of matrices | `uniform mat4 uLightVP[3];` | `float4x4 lightVP[3];` inside the constant struct (64 B stride) | `array<mat4x4<f32>, 3>` (64 B stride) |
| 2D texture + sampler | `uniform sampler2D uAlbedo; texture(uAlbedo, uv)` (combined) | `texture2d<float> albedo [[texture(0)]], sampler samp [[sampler(0)]]` → `albedo.sample(samp, uv)`; or `constexpr sampler s(filter::linear, address::repeat)` in-shader (spec §2.10) | `var t: texture_2d<f32>; var s: sampler; textureSample(t, s, uv)` |
| Shadow/comparison sampling | `sampler2DShadow` / `sampler2DArrayShadow`, `texture(sm, vec4(uv, layer, ref))` | `depth2d<float> sm [[texture(1)]]`, `depth2d_array<float>`; `constexpr sampler shadowSamp(filter::linear, compare_func::less, address::clamp_to_edge)`; `sm.sample_compare(shadowSamp, uv, ref)` / `.sample_compare(samp, uv, layer, ref)` — "To use sample_compare with a depth format you need to declare depth2d/depth2d_array/depthcube" (spec §2.9) | `texture_depth_2d_array` + `sampler_comparison`, `textureSampleCompare(t, s, uv, layer, ref)` |
| Texel fetch (DSM) | `texelFetch(uDsm, ivec2, 0)` | `dsm.read(uint2(x, y))` | `textureLoad(t, vec2<i32>, 0)` |
| Float textures | `R32F` via `texImage2D(..., gl.R32F, ...)` | `MTLPixelFormat.r32Float`; `texture2d<float, access::read>` | `r32float` (not filterable by default) |
| Derivatives | `dFdx/dFdy` | `dfdx/dfdy` | `dpdx/dpdy` |
| Matrix × vector | `M * v` (column-major) | `M * v` (column-major, `float4x4`) | `M * v` (column-major) |
| Constants | `const vec3 X = vec3(...)` | `constant float3 X = float3(...)` at file scope, or `constexpr` | `const X = vec3<f32>(...)` |
| Half precision | `mediump` hints (WebGL ignores on desktop) | `half`, `half3` real 16-bit types (spec Table 2.2) | `f16` with `enable f16;` (optional feature) |
| Depth range | NDC z −1..1 | NDC z 0..1 | NDC z 0..1 |
| Compute | not in WebGL2 | `kernel void f(... uint2 gid [[thread_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]]); threadgroup float tile[16][16];` | `@compute @workgroup_size(16,16)`, `var<workgroup> tile: array<...>`, `@builtin(global_invocation_id)` |
| Framebuffer fetch / tile ops | `EXT_shader_framebuffer_fetch` (mobile) | `[[color(0)]]` input argument, imageblocks, tile shaders (spec §2.11, §5.6) | not available |
| sRGB decode | manual or `EXT_sRGB` | `MTLPixelFormat.rgba8Unorm_srgb`: sampler returns linear | `rgba8unorm-srgb` |

Packed/alignment facts from the MSL spec (Table "Size and alignment of packed vector data
types"): `packed_float3` = 12 bytes, 4-byte alignment; `packed_half3` = 6 bytes, 2-aligned;
ordinary `float3` is 16-byte aligned. The cross-API layout gist by teoxoy
([link](https://gist.github.com/teoxoy/936891c16c2a3d1c3c5e7204ac6cd76c)) recommends the same
portable rule: **never put a vec3 in a shared uniform struct — use vec4 or explicit padding.**

---

## 2. Mesh relighting pass in MSL (sketch)

```metal
#include <metal_stdlib>
using namespace metal;

struct FrameUniforms {                  // matches a Swift struct with the same layout; keep 16-byte alignment
    float4x4 viewProj;
    float4x4 view;
    float4x4 lightViewProj[3];
    float4   cascadeFar;                // xyz used
    float4   sunDir;                    // xyz, w unused
    float4   sunColor;
    float4   skyZenith, skyHorizon, skyGround;
    float4   cameraPos;
    float4   fogParams;                 // density, heightFalloff, groundZ, sunPower
    float4   fogColorSky, fogColorSun;
    float4   misc;                      // roughness, rimStrength, rimPower, shadowNormalOffset
    float2   shadowTexel;
};

struct VSIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};
struct VSOut {
    float4 clip     [[position]];
    float3 worldPos;
    half3  normal;                      // half is plenty for a normal
    float2 uv;
};

vertex VSOut vs_mesh(VSIn in [[stage_in]], constant FrameUniforms& u [[buffer(1)]]) {
    VSOut o;
    float4 wp   = float4(in.position, 1.0);        // uModel = identity for ODM
    o.worldPos  = wp.xyz;
    o.normal    = half3(normalize(in.normal));
    o.uv        = in.uv;
    o.clip      = u.viewProj * wp;
    return o;
}

constexpr sampler albedoSamp(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(8));
constexpr sampler shadowSamp(filter::linear, address::clamp_to_edge, compare_func::less);

static half shadowPCF(depth2d_array<float> sm, float2 uv, uint layer, float ref, float2 texel) {
    half sum = 0.0h;
    for (int y = -2; y <= 2; ++y)
        for (int x = -2; x <= 2; ++x)
            sum += half(sm.sample_compare(shadowSamp, uv + float2(x, y) * texel, layer, ref));
    return sum / 25.0h;
}

fragment half4 fs_mesh(VSOut in [[stage_in]],
                       constant FrameUniforms& u [[buffer(1)]],
                       texture2d<half>     albedoTex [[texture(0)]],     // rgba8Unorm_srgb -> linear on sample
                       depth2d_array<float> shadowMap [[texture(1)]],
                       bool frontFacing [[front_facing]])
{
    half3 albedo = albedoTex.sample(albedoSamp, in.uv).rgb;
    float3 n = normalize(float3(in.normal)); if (!frontFacing) n = -n;
    float3 v = normalize(u.cameraPos.xyz - in.worldPos);
    float3 l = u.sunDir.xyz;

    // cascade select
    float viewDepth = -(u.view * float4(in.worldPos, 1.0)).z;
    uint layer = 2; for (uint i = 0; i < 3; ++i) if (viewDepth < u.cascadeFar[i]) { layer = i; break; }
    float NdL = saturate(dot(n, l));
    float3 offPos = in.worldPos + n * (u.misc.w * sqrt(1.0 - NdL * NdL) * float(layer + 1));
    float4 lp = u.lightViewProj[layer] * float4(offPos, 1.0);
    float3 uvz = lp.xyz / lp.w;
    float2 suv = uvz.xy * float2(0.5, -0.5) + 0.5;            // Metal: y flips, z already 0..1
    float bias = max(0.05 * (1.0 - NdL), 0.005) / (u.cascadeFar[layer] * 0.5);
    half shadow = shadowPCF(shadowMap, suv, layer, uvz.z - bias, u.shadowTexel);

    half diffuse = half(NdL);                                  // or Oren-Nayar as in file 04
    half3 direct = albedo * half3(u.sunColor.rgb) * diffuse * shadow;
    float up = n.z * 0.5 + 0.5;
    half3 skyAmb = half3(mix(u.skyGround.rgb, mix(u.skyHorizon.rgb, u.skyZenith.rgb, smoothstep(0.5, 1.0, up)), smoothstep(0.0, 0.5, up)));
    half3 color = direct + albedo * skyAmb;

    // height fog (IQ analytic), float precision for the exp terms
    float a = u.fogParams.x, b = u.fogParams.y;
    float3 rd = -v; float t = length(in.worldPos - u.cameraPos.xyz);
    float rz = copysign(max(abs(rd.z), 1e-4), rd.z);
    float f = (a / b) * exp(-(u.cameraPos.z - u.fogParams.z) * b) * (1.0 - exp(-t * rz * b)) / rz;
    half fog = half(1.0 - exp(-f));
    half sunAmt = half(pow(max(dot(rd, l), 0.0), u.fogParams.w));
    half3 fogCol = mix(half3(u.fogColorSky.rgb), half3(u.fogColorSun.rgb), sunAmt);
    color = mix(color, fogCol, fog);
    return half4(color, 1.0h);                                 // into an rgba16Float HDR target
}
```

Notes: `saturate`, `copysign`, `smoothstep`, `mix` are in `metal_stdlib`; `half` literals take an
`h` suffix; mixing `half` and `float` requires explicit casts (no implicit widening in
expressions across types — the compiler errors rather than silently promoting in many cases).

---

## 3. DSM shadow pass as a compute kernel

The 2D orthophoto pass in file 04 is a natural **compute kernel**: one thread per DSM texel,
ray-march toward the sun, write a shadow/sky-visibility texture (`r8Unorm` or `rg16Float`) that
the final grading pass samples. On Apple GPUs use `threadgroup` memory to cache a tile of the DSM
plus the apron in the sun direction.

```metal
#include <metal_stdlib>
using namespace metal;

struct DsmParams {
    float2 texSize;          // pixels
    float  gsd;              // metres per pixel
    float  noData;
    float3 sunDirENU;        // unit, (E, N, U)
    float  sunAngularRadius; // 0.00465
    float  maxShadowDist;    // metres
    uint   steps;
};

kernel void dsm_shadow(texture2d<float, access::read>  dsm     [[texture(0)]],
                       texture2d<half,  access::write> shadowOut [[texture(1)]],
                       constant DsmParams& p [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(p.texSize.x) || gid.y >= uint(p.texSize.y)) return;
    float z0 = dsm.read(gid).r;
    if (z0 <= p.noData + 0.5) { shadowOut.write(half4(1.0h), gid); return; }

    float3 s = p.sunDirENU;
    float horiz = length(s.xy);
    if (s.z <= 0.0) { shadowOut.write(half4(0.0h), gid); return; }
    if (horiz < 1e-4) { shadowOut.write(half4(1.0h), gid); return; }
    float slope = s.z / horiz;
    float2 dirPx = float2(s.x, -s.y) / horiz / p.gsd;      // pixels per horizontal metre (row 0 = north)
    float bias = p.gsd * slope + 0.05;
    float dist = p.gsd, minClear = 1e9;
    float stepMul = pow(p.maxShadowDist / p.gsd, 1.0 / float(p.steps));
    for (uint i = 0; i < p.steps && dist <= p.maxShadowDist; ++i, dist *= stepMul) {
        float2 pxf = float2(gid) + 0.5 + dirPx * dist;
        if (any(pxf < 0.0) || any(pxf >= p.texSize)) break;
        float zs = dsm.read(uint2(pxf)).r;                 // nearest; use a bilinear read via sampler for smoother penumbrae
        if (zs <= p.noData + 0.5) continue;
        float clearance = (z0 + bias + slope * dist) - zs;
        if (clearance < 0.0) { shadowOut.write(half4(0.0h), gid); return; }
        minClear = min(minClear, clearance / dist);
    }
    float k = saturate(minClear / p.sunAngularRadius);
    shadowOut.write(half4(half(k * k * (3.0 - 2.0 * k))), gid);
}
```

Dispatch with `threadsPerThreadgroup = MTLSize(16, 16, 1)` and `dispatchThreads(MTLSize(w, h, 1), ...)`
(non-uniform threadgroups are supported on Apple GPUs, so no manual edge handling is required
beyond the bounds check). For a fixed sun azimuth, the O(N) line-sweep horizon algorithm (file 04
§4) is better done as one thread per *sun-direction scanline* with a running max — a
`threadgroup` prefix-max over a row.

Kernel-side declarations checked against the spec: `kernel void f(device T* p [[buffer(0)]], uint gid [[thread_position_in_grid]])`
(spec §5.2, examples in §2.7), `threadgroup` address space (spec §4), `access::read/write`
texture template parameters.

---

## 4. Apple-GPU specifics

### 4.1 Half precision

- Apple's GPU register file is made of **16-bit registers**; a `float` consumes two. "That's one
  reason to use 16-bit types, like half, when appropriate, since they use half the register space"
  — WWDC21 *Optimize high-end games for Apple GPUs* ([session 10148](https://developer.apple.com/videos/play/wwdc2021/10148/)),
  which shows a shader going from F16 = 0 % to a mixed F32/F16 build and gaining ~30 % execution
  time and almost double occupancy.
- Use `half` for: albedo, sky/fog colours, the lighting accumulation, the PCF sum, normals,
  UV-derived values, the final tone-mapped colour. Keep `float` for: world positions, depth,
  shadow-map comparisons, fog's `exp()` integrals, the DSM ray march (`clearance/dist` needs range),
  matrix transforms.
- MSL `half` is IEEE binary16 (spec Table 2.2; scalar types section); fast-math is on by default —
  pass `-fno-fast-math` if the CCT/xy conversions in file 05 show banding.

### 4.2 Tile-based deferred rendering

Apple's guide *Tailor your apps for Apple GPUs and tile-based deferred rendering*
([link](https://developer.apple.com/documentation/metal/tailor-your-apps-for-apple-gpus-and-tile-based-deferred-rendering))
and the Kodeco chapter ([link](https://www.kodeco.com/books/metal-by-tutorials/v3.0/chapters/15-tile-based-deferred-rendering)):

- Render targets that only live within one render pass (the HDR colour before tone mapping, if
  you tone map in the same pass via a tile shader or `[[color(0)]]` fetch) can be
  `storageMode = .memoryless` — they stay in tile memory and never hit DRAM.
- Set `loadAction = .clear/.dontCare` and `storeAction = .dontCare` for the depth buffer and any
  memoryless attachment; storing is the expensive part on a TBDR.
- **Avoid `discard_fragment()`** in the mesh pass (breaks hidden-surface removal); ODM meshes are
  opaque, so alpha test is unnecessary. The orthophoto's cutline alpha is fine in the 2D pass
  (single quad).
- Programmable blending / `[[color(0)]]` fragment inputs let the tone-map + grade run as a second
  draw *in the same pass* reading the HDR value without a texture round-trip — bloom, however,
  needs the mip chain, so it stays a separate pass (or use `MPSImageGaussianPyramid`).
- Two-frame overlap: TBDR GPUs overlap the vertex stage of frame N+1 with the fragment stage of
  frame N when there are no dependencies (WWDC21) — keep the shadow pass on its own command buffer.
- Texture compression: ASTC for the albedo atlas (`MTLPixelFormat.astc_6x6_srgb`) halves bandwidth
  vs. RGBA8; ODM's atlases are 8192² PNG/JPEG, so compress at load time (or offline).

### 4.3 Compute vs fragment for the DSM pass

Both work; compute is simpler for arbitrary-length ray marches and lets you write two outputs
(shadow, sky visibility) without MRT. For a 20k × 20k orthophoto, tile the dispatch (4k × 4k
chunks) and keep the DSM as an `r32Float` texture with `.read` access; `.sample` with a
`constexpr sampler(filter::linear)` is allowed on `r32Float` on Apple GPUs (32-bit float
filtering is supported on Apple silicon per the Metal feature set tables — verify with
`device.supports32BitFloatFiltering`).

---

## 5. Minimal MetalKit host (Swift)

Pattern from Metal by Example's *Modern Metal* part 1 ([link](https://metalbyexample.com/modern-metal-1/))
and Apple's `MTKViewDelegate` docs ([link](https://developer.apple.com/documentation/metalkit/mtkviewdelegate)):

```swift
import MetalKit

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let meshPipeline: MTLRenderPipelineState
    let depthState: MTLDepthStencilState
    var frame = FrameUniforms()                 // Swift struct mirroring the MSL struct (use SIMD4<Float>, simd_float4x4)

    init(view: MTKView) throws {
        device = view.device!
        queue = device.makeCommandQueue()!
        view.colorPixelFormat = .bgra8Unorm_srgb  // final output; HDR intermediate is a separate .rgba16Float texture
        view.depthStencilPixelFormat = .depth32Float

        let lib = try device.makeLibrary(URL: Bundle.main.url(forResource: "golden", withExtension: "metallib")!)   // offline-compiled (see §6)
        let vd = MTLVertexDescriptor()
        vd.attributes[0].format = .float3; vd.attributes[0].offset = 0;  vd.attributes[0].bufferIndex = 0
        vd.attributes[1].format = .float3; vd.attributes[1].offset = 12; vd.attributes[1].bufferIndex = 0
        vd.attributes[2].format = .float2; vd.attributes[2].offset = 24; vd.attributes[2].bufferIndex = 0
        vd.layouts[0].stride = 32

        let pd = MTLRenderPipelineDescriptor()
        pd.vertexFunction   = lib.makeFunction(name: "vs_mesh")
        pd.fragmentFunction = lib.makeFunction(name: "fs_mesh")
        pd.vertexDescriptor = vd
        pd.colorAttachments[0].pixelFormat = .rgba16Float          // HDR target
        pd.depthAttachmentPixelFormat = .depth32Float
        meshPipeline = try device.makeRenderPipelineState(descriptor: pd)

        let dsd = MTLDepthStencilDescriptor(); dsd.depthCompareFunction = .less; dsd.isDepthWriteEnabled = true
        depthState = device.makeDepthStencilState(descriptor: dsd)!
        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { /* re-create HDR + depth textures */ }

    func draw(in view: MTKView) {
        guard let cb = queue.makeCommandBuffer(),
              let rpd = view.currentRenderPassDescriptor,        // for the final tone-mapped pass
              let drawable = view.currentDrawable else { return }
        // 1) shadow cascades (render pass into depth2DArray), 2) mesh pass into HDR texture,
        // 3) sky pass, 4) bloom compute, 5) tone map + grade into `rpd`
        let enc = cb.makeRenderCommandEncoder(descriptor: rpd)!
        enc.setRenderPipelineState(meshPipeline)                  // (illustrative; real code uses the HDR pass descriptor)
        enc.setDepthStencilState(depthState)
        enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        enc.setVertexBytes(&frame, length: MemoryLayout<FrameUniforms>.stride, index: 1)
        enc.setFragmentBytes(&frame, length: MemoryLayout<FrameUniforms>.stride, index: 1)
        enc.setFragmentTexture(albedoTexture, index: 0)
        enc.setFragmentTexture(shadowArray, index: 1)
        enc.drawIndexedPrimitives(type: .triangle, indexCount: indexCount, indexType: .uint32, indexBuffer: indexBuffer, indexBufferOffset: 0)
        enc.endEncoding()
        cb.present(drawable)
        cb.commit()
    }
}
```

`MTKTextureLoader` loads the PNG/JPEG atlas (`options: [.SRGB: true]`), and `MDLAsset` /
ModelIO can import the OBJ directly (`MDLAsset(url:vertexDescriptor:bufferAllocator:)`) —
or read ODM's GLB with a glTF loader; remember the GLB is Z-up (file 02).

---

## 6. Offline compilation

From the Metal Programming Guide's *Metal Tools* chapter (archive docs,
[link](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Dev-Technique/Dev-Technique.html), Listing 8-1/8-2)
and the current *Metal libraries* page ([link](https://developer.apple.com/documentation/metal/metal-libraries)):

```sh
# compile each .metal to AIR (Apple IR); -c is optional for a single file, -o names the output
xcrun -sdk macosx metal -c shaders/golden.metal -o build/golden.air
# optional: archive several .air files
xcrun -sdk macosx metal-ar r build/golden.metalar build/golden.air build/dsm.air
# link into a library
xcrun -sdk macosx metallib build/golden.air -o build/golden.metallib
```

Useful `metal` front-end flags (Clang-derived): `-std=metal3.1` (language version;
[forum](https://developer.apple.com/forums/thread/105719)), `-Os`, `-ffast-math` (default) /
`-fno-fast-math`, `-gline-tables-only -frecord-sources` for GPU debugging in Xcode (from the
[Metal libraries](https://developer.apple.com/documentation/metal/metal-libraries) page; flag
names not re-verified in this session — check `xcrun metal --help`). Load at runtime with
`device.makeLibrary(URL:)` (Swift) / `newLibraryWithFile:error:` (ObjC).

**Observed on this machine (2026-09-13):** `xcrun -sdk macosx metal --version` fails with
`unable to find utility "metal"` — the Metal toolchain is not installed. With Xcode 15+/26 the
shader compiler is a separate download: install full Xcode (not just Command Line Tools) and run
`xcodebuild -downloadComponent MetalToolchain` (Xcode 26) or install the "Metal Toolchain" from
Xcode → Settings → Components. Until then, runtime compilation via `device.makeLibrary(source:options:)`
still works (it uses the system's built-in compiler service) — good for iteration, but ship a
`.metallib`.

---

## 7. Automatic translation

| Tool | Input → Output | Invocation | Notes | Licence |
|---|---|---|---|---|
| **SPIRV-Cross** (Khronos) | SPIR-V → GLSL / HLSL / **MSL** | `glslangValidator -V mesh.frag -o mesh.spv` then `spirv-cross --msl --msl-version 20300 mesh.spv --output mesh.metal` | Descriptor sets → argument buffers with `--msl-argument-buffers` (bindings become `[[id(N)]]`); otherwise resources get `[[buffer(n)]]/[[texture(n)]]/[[sampler(n)]]`; combined samplers are split into texture + sampler; `CompilerMSL::add_msl_resource_binding` for custom mapping. From the README ([link](https://github.com/KhronosGroup/SPIRV-Cross/blob/main/README.md)): "Metal supports indirect argument buffers (--msl-argument-buffers)"; the `--msl` flag is documented in the test section. | Apache 2.0 |
| **naga** (wgpu) | WGSL / SPIR-V / GLSL 440+ → SPIR-V / **MSL** / HLSL / GLSL / WGSL | `cargo install naga-cli; naga mesh.wgsl mesh.metal` | Same translator WebGPU-on-Metal (wgpu, Firefox) uses; output uses `[[buffer(n)]]` bindings you map from `@group/@binding` via a binding map. README ([link](https://github.com/gfx-rs/wgpu/blob/trunk/naga/README.md)). | MIT/Apache-2.0 (wgpu) |
| **Tint** (Dawn / Chrome) | WGSL / SPIR-V → SPIR-V / HLSL / **MSL** / GLSL / WGSL | `tint --format msl mesh.wgsl -o mesh.metal` (Dawn build) | What Chrome uses for WebGPU on macOS, so its MSL is the "reference" behaviour for your WGSL. The README fetch failed in this session (404 on both mirror paths) — command syntax from memory, verify in `dawn/src/tint/`. | BSD-3 (Dawn) |
| **Slang** | Slang/HLSL → MSL, WGSL, SPIR-V | `slangc mesh.slang -target metal` | Single-source path if you want one shader language for all three backends ([Metal-specific docs](http://shader-slang.org/slang/user-guide/metal-target-specific)). | Apache 2.0 w/ LLVM exception |

kvark's *Shader translation benchmark* ([link](http://kvark.github.io/naga/shader/2022/02/17/shader-translation-benchmark.html))
compares naga, Tint and SPIRV-Cross speed and coverage. Practical advice: hand-write the MSL
for the two hot shaders (mesh fragment, DSM kernel) using `half` where useful — translators emit
all-`float` code — and use naga/Tint only to *verify* semantics (translate the WGSL and diff
against the hand port).

---

## 8. Porting checklist

1. Flip shadow-map `y` (`uv.y = 1 − uv.y` or the `float2(0.5, −0.5)` trick) and drop the `z*0.5+0.5`
   remap — Metal NDC depth is 0..1.
2. Replace `sampler2DArrayShadow` with `depth2d_array<float>` + `compare_func::less` sampler;
   `sample_compare` returns 0/1 with hardware 2×2 PCF when `filter::linear`.
3. Convert uniforms to one `constant` struct with 16-byte-aligned members; use `setVertexBytes`
   / `setFragmentBytes` for < 4 KB, otherwise a triple-buffered `MTLBuffer`.
4. `texture()` → `.sample()`, `texelFetch()` → `.read()`, `dFdx` → `dfdx`.
5. Introduce `half` for colour math; keep `float` for positions and the fog/shadow math.
6. Tone map + grade in the final pass writing to `bgra8Unorm_srgb` (Metal encodes sRGB on write;
   do *not* encode manually).
7. Compile offline with `xcrun metal` / `metallib`; enable `-gline-tables-only -frecord-sources`
   for the Xcode GPU debugger while iterating.

---

## Sources

- Metal Shading Language Specification v4.1 (PDF): https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- Apple — Tailor your apps for Apple GPUs and tile-based deferred rendering: https://developer.apple.com/documentation/metal/tailor-your-apps-for-apple-gpus-and-tile-based-deferred-rendering
- Apple — Metal libraries: https://developer.apple.com/documentation/metal/metal-libraries
- Apple — Metal Programming Guide, Metal Tools (archive): https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Dev-Technique/Dev-Technique.html
- Apple — MTKViewDelegate: https://developer.apple.com/documentation/metalkit/mtkviewdelegate
- WWDC21 — Optimize high-end games for Apple GPUs (16-bit registers, occupancy): https://developer.apple.com/videos/play/wwdc2021/10148/
- WWDC20 — Harness Apple GPUs with Metal: https://developer.apple.com/videos/play/wwdc2020/10602/
- WWDC22 — Target and optimize GPU binaries with Metal 3: https://developer.apple.com/videos/play/wwdc2022/10102/
- Apple forum — Set Metal language version (-std): https://developer.apple.com/forums/thread/105719
- Apple forum — using xcrun to compile .metal shader: https://developer.apple.com/forums/thread/726165
- Kodeco — Metal by Tutorials ch. 15, Tile-Based Deferred Rendering: https://www.kodeco.com/books/metal-by-tutorials/v3.0/chapters/15-tile-based-deferred-rendering
- Metal by Example — Writing a Modern Metal App from Scratch, part 1: https://metalbyexample.com/modern-metal-1/
- carolight/Metal-Shadow-Map (depth2d + compare_func::less example): https://github.com/carolight/Metal-Shadow-Map/blob/master/Shadows/Shader.metal
- teoxoy — Shader buffer memory layouts across GLSL/WGSL/MSL: https://gist.github.com/teoxoy/936891c16c2a3d1c3c5e7204ac6cd76c
- WGSL specification: https://www.w3.org/TR/WGSL/
- SPIRV-Cross README: https://github.com/KhronosGroup/SPIRV-Cross/blob/main/README.md
- naga README (wgpu): https://github.com/gfx-rs/wgpu/blob/trunk/naga/README.md
- kvark — Shader translation benchmark: http://kvark.github.io/naga/shader/2022/02/17/shader-translation-benchmark.html
- Slang — Metal target specifics: http://shader-slang.org/slang/user-guide/metal-target-specific
- alain.xyz — A review of shader languages: https://alain.xyz/blog/a-review-of-shader-languages
- DarthShader (fuzzing WGSL translators; lists Tint/naga backends): https://arxiv.org/pdf/2409.01824
