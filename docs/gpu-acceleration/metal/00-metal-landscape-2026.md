# Metal for compute in 2026 — landscape for the ODM/Apple-Silicon port

Research date: 2026-09-13. Target machine: MacBook Pro, Apple M4 Pro (12 CPU cores, 16-core GPU,
24 GB unified memory), macOS 26.6, Command Line Tools only (no Xcode.app installed).
Everything marked **[verified locally]** was measured on that machine with a small
Objective-C++ probe compiled with the CLT `clang++` (see §9). Everything else is cited.
Claims marked **[uncertain]** could not be confirmed against a primary source.

---

## 1. What "Metal 4" is (macOS 26 / WWDC 2025)

Metal 4 is the API revision that ships with macOS 26 ("Tahoe") and is supported on every
Apple-silicon Mac ("Mac with Apple silicon" running "macOS Tahoe 26 or later") and on
A14/M1 and later in general ([Apple Support 102894](https://support.apple.com/en-us/102894),
[WWDC25 "Discover Metal 4"](https://developer.apple.com/videos/play/wwdc2025/205/)).
The Metal 3 API is not removed; Metal 4 adds a parallel set of `MTL4*` types that can be
adopted incrementally ("porting [is] approachable and modular ... shader compilation,
command encoding, resource management" — WWDC25 205).

The pieces that matter for a compute-only workload such as PatchMatch stereo or feature
extraction:

| Metal 4 feature | Why it matters for ODM | Source |
|---|---|---|
| `MTL4CommandQueue` / `MTL4CommandBuffer` / `MTL4CommandAllocator` | Command buffers are decoupled from the queue and can be encoded from several CPU threads in parallel; command memory is explicit. Useful if several DensifyPointCloud worker threads each drive the GPU. | WWDC25 205 |
| Unified compute encoder | One `MTL4ComputeCommandEncoder` also does blit and acceleration-structure work; fewer encoder switches. | WWDC25 205 |
| `MTL4ArgumentTable` + residency sets | "Bindless" resource binding; you populate a residency set once at start-up and attach it to the queue, then bind by table slot. Removes per-dispatch `setBuffer` calls. | WWDC25 205 |
| Barrier API | Explicit stage-to-stage barriers "that map well to barriers in other APIs" (i.e. closer to Vulkan/CUDA streams than Metal 3's implicit hazard tracking). | WWDC25 205 |
| `MTLTensor` + `MTL4MachineLearningCommandEncoder` | Run a whole CoreML-format network on the GPU timeline inside a Metal command buffer; tensors are a first-class resource type "supported across all contexts". | WWDC25 205, [WWDC25 262](https://developer.apple.com/videos/play/wwdc2025/262/) |
| Shader ML / Metal Performance Primitives | `matmul2d` and `convolution2d` on `MTLTensor` directly inside your own kernels; the OS compiler inlines a device-tuned implementation. | WWDC25 262 |
| `MTL4Compiler` | Compilation is a separate object with explicit CPU control; pipelines can be built from unspecialised Metal IR and specialised later. | WWDC25 205 |
| Placement sparse resources | Buffers/textures allocated without pages; pages come from a placement heap. Relevant for very large depth-map working sets. | WWDC25 205 |

Metal 4 also requires C++17-based MSL (Metal 1–3 MSL was C++14-based; "C++17-based in Metal 4",
[Wikipedia: Metal (API)](https://en.wikipedia.org/wiki/Metal_(API))).

**Practical note for this project.** Nothing in the ODM pipeline needs a Metal-4-only feature.
The openMVS Metal backend that landed upstream in June 2026 (see `01-openmvs-on-metal.md`) uses
plain Metal 3 objects (`id<MTLDevice>`, `newLibraryWithSource:`, `MTLComputeCommandEncoder`,
`MTLResourceStorageModeShared`). Metal 4 is an optimisation avenue (argument tables, explicit
barriers, ML encoder), not a prerequisite.

---

## 2. Metal Shading Language (MSL) versions

* The current spec is **Metal Shading Language Specification Version 4.1**
  ([PDF](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)); a
  "Version 4" PDF dated 2025-10-23 is also mirrored at developer-rno.apple.com. The spec is
  ~10 MB and could not be fetched in full for this document, so the OS→MSL-version table is not
  quoted here **[uncertain: exact mapping of 4.1 to a macOS 26.x point release]**.
* **[verified locally]** On macOS 26.6, `[MTLCompileOptions new].languageVersion` defaults to
  `0x40000` = `MTLLanguageVersion4_0`, and a kernel using `bfloat` arithmetic and `simd_sum`
  compiled and ran at runtime through `newLibraryWithSource:` without the Metal toolchain
  installed.
* `bfloat` (16-bit brain float, 8-bit exponent / 7-bit stored mantissa) is part of MSL since
  3.1 (macOS 14) and is a native type in MSL 4; the spec describes it as "a truncated version of
  float for machine learning applications" (MSL spec search excerpt). Whether M4 executes
  bfloat *arithmetic* natively or via conversion to float is not stated in public docs
  **[uncertain]**; the Apple9 row in the Feature Set Tables lists bfloat support as available
  ([Feature Set Tables PDF](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)).
  For ODM's purposes (`float` NCC costs, `float` depths) this is irrelevant; `half` is the
  precision worth exploiting for image samples.
* `double` is **not supported** on Apple GPUs at all ("Apple Silicon GPUs do not have hardware
  support for double-precision floating point" — [AdaptiveCpp Metal backend doc](https://adaptivecpp.github.io/AdaptiveCpp/install-metal/);
  same limitation reported by [PyTorch MPS](https://docs.pytorch.org/docs/2.13/notes/mps.html)).
  **[verified locally]** OpenCL reports `CL_DEVICE_DOUBLE_FP_CONFIG = 0` on the M4 Pro.
  Any CUDA code in OpenMVS that touches `double` on device must be rewritten in `float`.
* 64-bit atomics: Feature Set Tables list "the full set of 64-bit atomic operations" from
  Apple9; AdaptiveCpp's doc says atomic64 is "not supported" on Metal — these disagree, most
  likely because AdaptiveCpp targets the lowest common Apple family **[uncertain]**. OpenMVS'
  PatchMatch kernels use no atomics, so it does not matter for the densify port.

---

## 3. Compute kernel model (what a CUDA developer must relearn)

Terminology mapping (from the MSL spec and Feature Set Tables; numbers **[verified locally]**
on M4 Pro unless cited):

| CUDA | Metal | M4 Pro value |
|---|---|---|
| grid / block / thread | grid / **threadgroup** / thread | `maxThreadsPerThreadgroup` = 1024×1024×1024, `maxTotalThreadsPerThreadgroup` = 1024 per pipeline |
| warp (32) | **SIMD-group** | `threadExecutionWidth` = 32 |
| `__shared__` | `threadgroup` address space | `maxThreadgroupMemoryLength` = 32768 bytes (32 KB) |
| `__shfl_*`, `__ballot` | `simd_shuffle*`, `simd_ballot`, `simd_sum/min/max`, `quad_*` | available; `simd_sum` tested |
| `__constant__` | `constant` address space (bound as buffer) | — |
| `cudaTextureObject_t` + bilinear | `texture2d<float>` + `constexpr sampler(filter::linear)` | `supports32BitFloatFiltering` = 1 (bilinear on R32Float works) |
| `curand` | none — bring your own hash/PCG (upstream openMVS uses a PCG in MSL) | — |
| kernel launch on stream | `dispatchThreads:threadsPerThreadgroup:` on a compute encoder | — |
| `cudaMalloc` | `newBufferWithLength:options:` (`Private`/`Shared`) | `maxBufferLength` = 14.30 GB |
| argument passing | `[[buffer(n)]]`, `[[texture(n)]]`; or argument buffers (Metal 3) / argument tables (Metal 4) | argument-buffer tier 2 (`argumentBuffersSupport` = 1 → Tier2) |

Non-uniform threadgroups (`dispatchThreads`) are supported on all Apple families, so a
W×H image can be dispatched exactly without edge guards. Dynamic indexing of arrays of textures
(`array<texture2d<float>, N>`) works and is what upstream openMVS uses for the multi-view image
set.

Reference on microarchitecture (register file, occupancy, ALU rates per core) is community
work: [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks).

---

## 4. Unified memory: zero-copy and storage modes

On Apple silicon the CPU and GPU share one physical memory; `MTLDevice.hasUnifiedMemory` is
`YES` **[verified locally]**. Storage modes that matter:

* `MTLResourceStorageModeShared` — one allocation visible to CPU and GPU; no blit needed. On
  Apple silicon this is the default and correct choice for inputs/outputs. Upstream openMVS'
  Metal backend uses it for cameras, planes, costs, RNG state, selected-view masks.
* `MTLResourceStorageModePrivate` — GPU-only; only marginally faster on Apple GPUs and forces a
  blit; use for scratch that the CPU never reads.
* `MTLStorageModeManaged` — Intel/discrete-GPU concept (two copies + `didModifyRange:`); not
  needed on Apple silicon.
* `MTLStorageModeMemoryless` — tile memory for render passes; irrelevant to compute.

Zero-copy wrapping of existing host memory: `newBufferWithBytesNoCopy:length:options:deallocator:`
requires page-aligned pointer and length (16 KB pages on Apple silicon). **[verified locally]**
wrapping a 16 KB-aligned `aligned_alloc` block as a Shared buffer succeeded and a kernel wrote
results straight into the host array. This means a `cv::Mat` whose data pointer is 16 KB-aligned
(allocate via `posix_memalign`/`aligned_alloc` and wrap with `cv::Mat(rows, cols, type, ptr)`)
can be handed to the GPU without a copy. Synchronisation is by command-buffer completion
(`waitUntilCompleted` or a completion handler); there is no CPU cache flush API on Apple silicon.

Apple's storage-mode guide is at
`developer.apple.com/documentation/metal/resource-fundamentals/choosing-a-resource-storage-mode-for-apple-gpus`
(JS-rendered; it could not be fetched as text for quotation **[uncertain: exact wording]**).

Memory budget: **[verified locally]** `recommendedMaxWorkingSetSize` = 19.07 GB on the 24 GB
machine (≈ 79 %), `maxBufferLength` = 14.30 GB for a single buffer. Community reports put the
working-set figure at 65–75 % of RAM on other configurations
([devnote on Metal VRAM cap](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap),
[MTLDevice.recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)).
The important consequence for ODM: GPU memory *is* system memory. Running OpenMVS with
`--max-threads 12` CPU workers plus GPU depth maps competes for the same 24 GB; upstream's
Metal backend drains an `@autoreleasepool` per resolution level precisely to keep resident GPU
memory bounded (`PatchMatchMetal.mm`, see `01-openmvs-on-metal.md`).

---

## 5. GPU family and limits for the M4 Pro

* GPU family: `MTLGPUFamilyApple9` ("Apple A17, M3, and M4 GPUs" —
  [MTLGPUFamily.apple9](https://developer.apple.com/documentation/metal/mtlgpufamily/apple9)).
  **[verified locally]** `supportsFamily:` returns true for Apple7/8/9, Metal3 and Metal4.
  (Note: an older revision of the Feature Set Tables PDF labels families by A-series chips
  only; trust the `MTLGPUFamily` doc page and the runtime query.)
* `system_profiler SPDisplaysDataType` reports `Chipset Model: Apple M4 Pro`,
  `Total Number of Cores: 16`, `Metal Support: Metal 4` **[verified locally]**. This string is
  the cheapest Python-side detection (`05-build-recipe-macos-metal.md`).
* Ray tracing, function pointers, dynamic libraries, mesh shaders: all available from Apple9
  (Feature Set Tables; `supportsRaytracing`/`supportsFunctionPointers` = 1 verified).
* Limits that shape kernel design: 32 KB threadgroup memory per threadgroup, 1024 threads per
  threadgroup, SIMD width 32, max buffer 14.3 GB on this machine, 2 GB per-buffer minimum
  guaranteed by the family table.

---

## 6. Higher-level Apple libraries

* **Metal Performance Shaders (MPS)** — Objective-C framework of pre-built kernels:
  `MPSImageConvolution`, `MPSImageGaussianBlur`, `MPSImagePyramid`, `MPSMatrixMultiplication`,
  `MPSNNGraph`, plus `MPSGraph` (a graph-compiled tensor library used by PyTorch-MPS). MPS is
  not deprecated but Apple's 2025 direction for new ML work is `MTLTensor` + Metal Performance
  Primitives ([WWDC25 262](https://developer.apple.com/videos/play/wwdc2025/262/)). Known
  `MPSMatrixMultiplication` caveats: no double precision, and a community-reported failure when
  a dimension exceeds 65535 on Apple silicon
  ([neurolabusc/Metal notes](https://github.com/neurolabusc/Metal),
  [Apple forum 105534](https://developer.apple.com/forums/thread/105534)) **[uncertain: whether
  the 65535 limit still applies in macOS 26]**. For ODM, MPS is relevant to (a) Gaussian
  pyramids for SIFT, (b) brute-force descriptor matching as a GEMM, (c) MPSGraph as the runtime
  behind PyTorch-MPS if learned features are used.
* **Accelerate** (CPU): BLAS/LAPACK (new interface since macOS 13.3), vImage (image
  convolution/geometry/histogram), vDSP, BNNS. Accelerate's BLAS drives the on-die matrix unit
  (AMX on M1–M3, SME on M4) and "will appear to use one thread" while beating multi-threaded
  OpenBLAS at small/medium sizes ([dev.to comparison](https://dev.to/frosnerd/comparing-openblas-and-accelerate-on-apple-silicon-for-blas-routines-2pb9),
  [Eclectic Light on M4 SME](https://eclecticlight.co/2024/11/27/inside-m4-chips-matrix-processing-and-power-modes/),
  [Apple-vs-Oranges HPC paper](https://arxiv.org/pdf/2502.05317)). See `02-opensfm-features-on-metal.md` §6 for the conda-forge switch.
* **Core ML / ANE**: reachable from ODM only through onnxruntime's CoreML EP or coremltools;
  see `03-onnxruntime-coreml-mps.md`.
* **RealityKit `PhotogrammetrySession`** (Object Capture) — Apple's own closed photogrammetry
  engine, Metal-accelerated, command-line usable; see `01-openmvs-on-metal.md` §7.

---

## 7. Toolchain: compiling MSL and linking Metal without Xcode.app

Two distinct things are needed and they have different requirements:

1. **Host-side code that calls Metal** (`.mm` Objective-C++ or C++ via metal-cpp) only needs the
   macOS SDK. **[verified locally]** with Command Line Tools only:
   `clang++ -std=c++17 -fobjc-arc probe.mm -framework Metal -framework Foundation` builds and
   runs. The SDK is at `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`.
   Caveat found while probing: the CLT `swiftc` (6.2.0.19.9) refused to import `Metal` because
   the installed SDK was built with a slightly older Swift (6.2.0.16.14) — a CLT/SDK version skew.
   Clang was unaffected. Keep the port in Objective-C++/C++, not Swift.
2. **Ahead-of-time MSL compilation** (`xcrun metal` → `.air`, `xcrun metallib` → `.metallib`)
   requires the **Metal Toolchain**, which since Xcode 26 is an optional ~700 MB component
   installed with `xcodebuild -downloadComponent MetalToolchain` (or
   `-downloadComponent metalToolchain -exportPath …` + `-importComponent`), and which needs
   Xcode.app — Command Line Tools alone are insufficient
   ([polpiella.dev](https://www.polpiella.dev/metal-toolchain-ci-cd),
   [actions/runner-images #13080](https://github.com/actions/runner-images/issues/13080),
   [Apple forum 787294](https://developer.apple.com/forums/thread/787294)).
   **[verified locally]** `xcrun --find metal` → `unable to find utility "metal"` and
   `xcodebuild` → `requires Xcode` on this machine. Known quirks: the toolchain is only visible
   to the user who installed it ([FB20389216](https://openradar.appspot.com/FB20389216)) and may
   need manual mounting ([Tietze, Jan 2026](https://christiantietze.de/posts/2026/01/manually-mount-metal-toolchain/)).
3. **Runtime MSL compilation** via `-[MTLDevice newLibraryWithSource:options:error:]` needs
   *no* toolchain — the compiler lives in the OS. **[verified locally]**. Cost is a one-off
   ~100 ms-class compile per library versus ~20 ms to load a prebuilt `.metallib`
   ([Apple forum 696543](https://developer.apple.com/forums/thread/696543)); Apple's guidance is
   to precompile when possible ([Metal Programming Guide: Functions and Libraries](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Prog-Func/Prog-Func.html)).
   Upstream openMVS chose runtime compilation of an embedded MSL string exactly to avoid the
   Xcode dependency (see `01-openmvs-on-metal.md` §3).

**metal-cpp** — Apple's header-only C++ wrapper (`MTL::Device`, `MTL::ComputePipelineState`,
…), Apache-2.0, C++17, one TU defines `NS_PRIVATE_IMPLEMENTATION`/`MTL_PRIVATE_IMPLEMENTATION`;
releases track the OS (macOS 26 release added Metal 4 `MTL4` classes; a macOS 27 drop exists)
([apple/metal-cpp](https://github.com/apple/metal-cpp), [developer.apple.com/metal/cpp](https://developer.apple.com/metal/cpp/),
[WWDC22 10160](https://developer.apple.com/videos/play/wwdc2022/10160/)). It lets a pure-C++
project (CMake, no `OBJCXX` language) call Metal; you still link `-framework Metal -framework
Foundation -framework QuartzCore`.

**conda-forge clang and Metal.** conda-forge's `clangxx_osx-arm64` (19.1.7 in ODM's
`pixi.lock`) is a normal LLVM clang that compiles Objective-C/C++ and links frameworks provided
the Apple SDK is on the sysroot; conda's activation passes `-isysroot ${CONDA_BUILD_SYSROOT}`
and expects `CONDA_BUILD_SYSROOT`/`SDKROOT` to point at an installed SDK
([clang-compiler-activation #94](https://github.com/conda-forge/clang-compiler-activation-feedstock/issues/94),
[conda-forge knowledge base](https://conda-forge.org/docs/maintainer/knowledge_base/),
[PKGW how-to](https://newton.cx/~peter/howto/fix-conda-compilers-macos/)). The CLT SDK path
above satisfies that. Upstream openMVS' CMake does `ENABLE_LANGUAGE(OBJCXX)` +
`FIND_LIBRARY(METAL_FRAMEWORK Metal)`, which works with any clang that sees the SDK
**[uncertain: not yet exercised with the conda clang in ODM's SuperBuild — see open questions]**.

---

## 8. OpenCL on macOS 26 — deprecated, still alive

* Apple deprecated OpenCL (and OpenGL) in macOS 10.14 and says "apps that use OpenCL for
  computational tasks should now adopt Metal and Metal Performance Shaders"
  ([developer.apple.com/opencl](https://developer.apple.com/opencl)). An Apple engineer on the
  forums: "Yes, OpenGL and OpenCL still run on Apple Silicon for now"
  ([forum 694866](https://developer.apple.com/forums/thread/694866)).
* **[verified locally on macOS 26.6]** `clGetPlatformIDs` → 1 platform;
  `clGetDeviceIDs(CL_DEVICE_TYPE_ALL)` → 1 device: `Apple M4 Pro`, `CL_DEVICE_VERSION = "1.2"`,
  type GPU, 16 compute units, no fp64. There is **no CPU OpenCL device** on Apple silicon (also
  noted in forum 694866). The framework binary is in the dyld shared cache (no file at
  `OpenCL.framework/Versions/A/OpenCL`, but `dlopen` of the framework path works).
* Apple's OpenCL runs on top of the Metal/AIR compiler stack; philipturner's
  [opencl-metal-stdlib](https://github.com/philipturner/opencl-metal-stdlib) exposes Metal's
  SIMD-group functions from OpenCL C via inline `__asm` and documents the limits: 256 threads
  per work-group (vs 1024 in Metal), no fp64, work-group sizes must be multiples of 32.
* Conclusion: OpenCL 1.2 is a *possible* stop-gap for code that already has OpenCL kernels
  (e.g. OpenCV's T-API — but see `04-*` for why conda-forge OpenCV has it compiled out), not a
  foundation for new work. No removal date is announced, but every Apple statement says it can
  disappear.

---

## 9. Portability layers

| Layer | What it is | Status on Apple silicon 2026 | Fit for ODM |
|---|---|---|---|
| **MoltenVK** | Vulkan 1.4 over Metal; SPIR-V→MSL via SPIRV-Cross at runtime; ships in the LunarG Vulkan SDK | v1.4.x (Aug–Nov 2025) passes most CTS for 1.0–1.4 in portability mode; subgroup size 32 on Apple GPUs ([MoltenVK README](https://github.com/KhronosGroup/MoltenVK), [Phoronix](https://www.phoronix.com/news/MoltenVK-1.4)); Metal 4 adoption only tracked ([#2560](https://github.com/KhronosGroup/MoltenVK/issues/2560)) | Only useful if one writes Vulkan compute (GLSL/SPIR-V). No existing Vulkan MVS to reuse (see `01`). Adds a translation layer and a Vulkan SDK dependency; no fp64 either. |
| **Kompute** | C++/Python Vulkan compute framework (LF AI & Data), GLSL→SPIR-V | v0.7.0; no explicit macOS statements in README, works through MoltenVK ([KomputeProject/kompute](https://github.com/KomputeProject/kompute)) | Same caveats as MoltenVK plus a thin abstraction; low value over raw Metal. |
| **wgpu (Rust) / Dawn (C++)** | WebGPU-native; Metal backend on macOS, WGSL shaders | wgpu selects Metal automatically on M-series ([docs.rs/wgpu](https://docs.rs/wgpu)); Dawn is Chromium's implementation | Portable (would also cover the Radeon/Linux track via Vulkan), but WGSL lacks fp64, has no texture arrays of dynamic size in all backends, and adds a Rust/Chromium build dependency to the SuperBuild. Reasonable only if a *single* shader source for Metal + Vulkan is a hard goal. |
| **AdaptiveCpp (SYCL) Metal backend** | SYCL kernels → LLVM IR → MSL at runtime | Experimental; "only been tested on macOS 26"; no double, no printf, SYCL only ([install-metal.md](https://adaptivecpp.github.io/AdaptiveCpp/install-metal/)) | Interesting for a CUDA-like single-source C++ port, but experimental and would drag in a custom LLVM. Not recommended for production ODM. |
| **OpenCL 1.2** | Apple's deprecated runtime | Works (see §8) | Stop-gap only. |
| **metal-cpp** | Apple's C++ headers | Current | Recommended host API if you want to avoid `.mm` files. |
| **Plain Objective-C++ + MSL** | Apple's first-party path | Current; what upstream openMVS uses | **Recommended** for ODM: zero extra dependencies, matches upstream. |

---

## 10. Local probe (reproducible)

Compiled and run on the target machine with Command Line Tools only:

```
clang++ -std=c++17 -fobjc-arc -O2 -o probe_mm probe.mm -framework Metal -framework Foundation
name: Apple M4 Pro
apple7 1 apple8 1 apple9 1 metal3 1
metal4 1
maxBufferLength 14.30 GB
recommendedMaxWorkingSetSize 19.07 GB
maxThreadgroupMemoryLength 32768
maxThreadsPerThreadgroup 1024 x 1024 x 1024
hasUnifiedMemory 1, supportsRaytracing 1, supportsFunctionPointers 1, supports32BitFloatFiltering 1, supportsBCTextureCompression 1
argumentBuffersSupport tier 1 (enum value 1 = MTLArgumentBuffersTier2), currentAllocatedSize 0.1 MB
MTLCompileOptions.languageVersion default = 0x40000
runtime MSL compile OK (bfloat arithmetic + simd_sum); maxTotalThreadsPerThreadgroup 1024, threadExecutionWidth 32
newBufferWithBytesNoCopy shared: OK
dispatch status 4 (Completed), host[0]=34.0  (1 + 1 + simd_sum over 32 lanes)
```

The probe source lives only in the session scratchpad; it is ~40 lines and trivially
re-creatable from the table in §3.

---

## 11. Take-aways for the port

1. Target Metal 3-level API in Objective-C++ (or metal-cpp); adopt Metal 4 pieces later.
2. Ship MSL as an embedded string compiled at runtime (no Xcode requirement), optionally with a
   build-time `.metallib` fast path when the Metal Toolchain is present.
3. Everything on the GPU is `float`/`half`; audit CUDA code for `double`.
4. Use `Shared` buffers and 16 KB-aligned host allocations for zero-copy; budget GPU use against
   the 19 GB working-set ceiling shared with the CPU workers.
5. OpenCL still enumerates the GPU on macOS 26.6 but must be treated as legacy.

## Sources

- Apple, "Discover Metal 4" (WWDC25 205): https://developer.apple.com/videos/play/wwdc2025/205/
- Apple, "Combine Metal 4 machine learning and graphics" (WWDC25 262): https://developer.apple.com/videos/play/wwdc2025/262/
- Apple, Metal Shading Language Specification 4.1: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- Apple, Metal Feature Set Tables: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
- Apple, MTLGPUFamily.apple9: https://developer.apple.com/documentation/metal/mtlgpufamily/apple9
- Apple Support, "Support for Metal on Apple devices": https://support.apple.com/en-us/102894
- Apple, MTLDevice.recommendedMaxWorkingSetSize: https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize
- Apple, MTLDevice.maxBufferLength: https://developer.apple.com/documentation/metal/mtldevice/2966563-maxbufferlength
- Apple, metal-cpp: https://github.com/apple/metal-cpp and https://developer.apple.com/metal/cpp/
- Apple, "Program Metal in C++ with metal-cpp" (WWDC22 10160): https://developer.apple.com/videos/play/wwdc2022/10160/
- Apple, Metal Programming Guide – Functions and Libraries: https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Prog-Func/Prog-Func.html
- Apple, OpenCL "Transition to Metal": https://developer.apple.com/opencl
- Apple Developer Forums 694866 (OpenGL/OpenCL on Apple silicon): https://developer.apple.com/forums/thread/694866
- Apple Developer Forums 696543 (newLibraryWithSource vs metallib timings): https://developer.apple.com/forums/thread/696543
- Apple Developer Forums 787294 (Xcode 26 Metal toolchain): https://developer.apple.com/forums/thread/787294
- Metal by Example, "Getting Started with Metal 4": https://metalbyexample.com/metal-4/
- Wikipedia, Metal (API): https://en.wikipedia.org/wiki/Metal_(API)
- polpiella.dev, Metal toolchain on CI: https://www.polpiella.dev/metal-toolchain-ci-cd
- actions/runner-images #13080: https://github.com/actions/runner-images/issues/13080
- OpenRadar FB20389216: https://openradar.appspot.com/FB20389216
- C. Tietze, "Manually Mount the Metal Toolchain": https://christiantietze.de/posts/2026/01/manually-mount-metal-toolchain/
- llama.cpp #6946 (xcrun metal missing): https://github.com/ggml-org/llama.cpp/issues/6946
- philipturner/opencl-metal-stdlib: https://github.com/philipturner/opencl-metal-stdlib
- philipturner/metal-benchmarks: https://github.com/philipturner/metal-benchmarks
- KhronosGroup/MoltenVK: https://github.com/KhronosGroup/MoltenVK ; issue #2560: https://github.com/KhronosGroup/MoltenVK/issues/2560 ; Phoronix on 1.4: https://www.phoronix.com/news/MoltenVK-1.4
- KomputeProject/kompute: https://github.com/KomputeProject/kompute
- wgpu: https://docs.rs/wgpu ; Dawn: https://dawn.googlesource.com/dawn
- AdaptiveCpp Metal backend: https://adaptivecpp.github.io/AdaptiveCpp/install-metal/
- PyTorch MPS notes: https://docs.pytorch.org/docs/2.13/notes/mps.html
- conda-forge clang activation #94: https://github.com/conda-forge/clang-compiler-activation-feedstock/issues/94
- conda-forge knowledge base: https://conda-forge.org/docs/maintainer/knowledge_base/
- PKGW, fixing conda compilers on macOS: https://newton.cx/~peter/howto/fix-conda-compilers-macos/
- ivanopcode, Metal VRAM cap note: https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap
- dev.to, OpenBLAS vs Accelerate: https://dev.to/frosnerd/comparing-openblas-and-accelerate-on-apple-silicon-for-blas-routines-2pb9
- Eclectic Light, M4 matrix processing: https://eclecticlight.co/2024/11/27/inside-m4-chips-matrix-processing-and-power-modes/
- arXiv 2502.05317, Apple M-series for HPC: https://arxiv.org/pdf/2502.05317
- neurolabusc/Metal (MPS matmul limits): https://github.com/neurolabusc/Metal
