# AMD integrated Radeon compute landscape for ODM (September 2026)

Scope: what actually runs on a Ryzen laptop iGPU today, and which of it ODM could use.
Target parts: Radeon 780M/760M (`gfx1103`, Phoenix/Hawk Point, RDNA3), Radeon 890M/880M
(`gfx1150`, Strix Point, RDNA3.5), Radeon 8060S (`gfx1151`, Strix Halo, mentioned for
context), and Vega 7/8 (`gfx90c`, Renoir/Cezanne/Lucienne, GCN5). Both Linux and Windows 11.

Status labels used below: **official** (in an AMD support matrix), **unofficial-works**
(reproducible community reports), **unofficial-risky** (reports of crashes/hangs), **no**.

---

## 1. ROCm / HIP on Linux

### 1.1 Official support matrices (two of them)

AMD publishes two different matrices and it matters which one you read:

* **ROCm core (data-center oriented).** The Linux system-requirements page for ROCm 7.14.0
  lists Instinct parts plus RDNA3/RDNA4 discrete cards (`gfx1100`, `gfx1101`, `gfx1200`,
  `gfx1201`). It lists **no** `gfx1103`, `gfx1150`, `gfx1151` or `gfx90c`, and redirects
  APU users to the Radeon/Ryzen docs. Ubuntu 24.04.4 is supported with kernel 6.8 (GA)
  and 6.17 (HWE).
  [ROCm Linux system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* **"Use ROCm on Radeon and Ryzen" (consumer).** For ROCm 7.2.1 on Ubuntu 24.04.4 the Ryzen
  matrix lists Ryzen AI Max+ 395 / Max 390 / Max 385 (`gfx1151`) and Ryzen AI 9 HX 475 /
  470 / 375 / 370, AI 9 365 / 465 (`gfx1150`), with PyTorch 2.9.1 on Python 3.12 as the
  validated framework and "only FP16 has been officially validated".
  [Ryzen Linux compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)

Net: **`gfx1150`/`gfx1151` are official on Linux (ROCm 7.2.x line). `gfx1103` (780M) and
`gfx90c` (Vega) are not official anywhere.**

Version note: AMD renumbered in late 2025/2026. The Radeon/Ryzen docs track a "7.2.1"
line while the core Linux docs now say 7.14.0 and mention "unified documentation beginning
at Core SDK 7.13.0". Treat 7.2.x and 7.1x as the same generation of stack; check which one
your distro packages are from.

### 1.2 What "unsupported" actually means for gfx1103

The compiler and runtime know `gfx1103`; what is missing is **pre-built library kernels**:

* rocBLAS/Tensile ships no `gfx1103` kernels: `Cannot read .../rocblas/library/TensileLibrary.dat:
  Illegal seek for GPU arch: gfx1103` (available variants: gfx1100/1101/1102/1150/1151...).
  [kvark/inferena#61](https://github.com/kvark/inferena/issues/61)
* MIOpen has no precompiled convolution databases for `gfx1103`, so PyTorch convolutions fail
  at runtime even though "PyTorch runtime support for gfx1103 has been merged, including
  hipBLASLt and aotriton". Filed April 2026, still open.
  [ROCm/rocm-libraries#6335](https://github.com/ROCm/rocm-libraries/issues/6335)
* TheRock (AMD's new open build system) lists `gfx1103 "AMD Radeon 780M Laptop iGPU"` in
  families `igpu-all gfx110X-all gfx110X-igpu`, and `gfx1150 "AMD Strix Point iGPU"`,
  `gfx1151`, `gfx1152`, `gfx1153` in `gfx115X-all`; `gfx90c` is also registered
  (`gfx90c-igpu`). Nightly indexes exist at `https://rocm.nightlies.amd.com/v2/gfx110X-all/`
  (I could only confirm `win_amd64` wheels for `rocm-sdk-core` there; Linux wheels for that
  family were not visible in my sampling — verify before relying on it).
  [therock_amdgpu_targets.cmake](https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake),
  [TheRock#1939](https://github.com/ROCm/TheRock/issues/1939)

**Why this matters for ODM:** OpenMVS PatchMatch and PopSift need only the HIP runtime,
`hipcc`/clang, and hipRAND device headers — no rocBLAS, no MIOpen. Code compiled with
`--offload-arch=gfx1103` should run natively without any override (this is exactly how
llama.cpp is built for RDNA3 iGPUs). Mark as *expected*, not verified on ODM.
[llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

### 1.3 `HSA_OVERRIDE_GFX_VERSION` — what it does and the risks

The variable makes the HSA runtime report a different ISA so that pre-built code objects
for that ISA are loaded.

| iGPU | Override | Evidence | Label |
|---|---|---|---|
| 780M `gfx1103` | `11.0.2` (pretend gfx1102) | Ollama: 6.2 → 18.7 tok/s vs CPU, "no issues so far" ([ollama#3189](https://github.com/ollama/ollama/issues/3189)); ROCm 7.2 + Ollama 0.30.6 on kernel 7.0 ([k8s.it, June 2026](https://www.k8s.it/posts/running-a-local-llm-on-amd-radeon-780m-gfx1103-rocm-and-the-gpu-that-wasnt-supposed-to-work/)); inferena adds `HSA_ENABLE_SDMA=0` because the async copy engine is unstable on this iGPU ([inferena#61](https://github.com/kvark/inferena/issues/61)) | unofficial-works for llama.cpp-class kernels |
| 780M `gfx1103` | `11.0.0` / `11.0.2` in a PyTorch+MIOpen context | "11.0.0 causes GPU page faults and hard hangs; 11.0.2 causes full system hard locks" ([rocm-libraries#6335](https://github.com/ROCm/rocm-libraries/issues/6335)) | unofficial-risky |
| Vega `gfx90c` | `9.0.0` (pretend gfx900) + `PYTORCH_ROCM_ARCH=gfx900` | [catbraincell gist](https://gist.github.com/catbraincell/391721ee1b196773ed259baaab10f18b), [ROCm#1743](https://github.com/ROCm/ROCm/issues/1743) | unofficial-works, old stack (ROCm 5.x era) |
| 890M `gfx1150` | none needed | official in 7.2.1 | official |

Rules of thumb: the override is only needed for libraries lacking your ISA; it is
"telling the hardware to lie about what it is"; and it is **not supported on Windows**
(llama.cpp docs). Never run it on production data without a CPU fallback.

### 1.4 Kernel driver and memory on an APU (GTT vs UMA carve-out)

* `amdgpu` is in-tree; Ubuntu 24.04 6.8/6.17 kernels are fine for RDNA3. For `gfx1151`
  a kernel ≥ 6.16.9 was required before ROCm saw the full allocation (15.5 GB vs 96 GB).
  [ROCm#5444](https://github.com/ROCm/ROCm/issues/5444)
* iGPUs have no VRAM. Two pools exist: the BIOS "UMA frame buffer"/dedicated carve-out
  (permanently stolen from the OS) and **GTT** (system pages mapped on demand, default
  ~50% of RAM). AMD's Strix Halo optimisation guide recommends keeping the dedicated VRAM
  reservation small (e.g. 0.5 GB) and raising the GTT/TTM limit instead; the tunable is
  `/sys/module/ttm/parameters/pages_limit` (in pages) and AMD ships an `amd-ttm` helper.
  [Strix Halo system optimization](https://rocmdocs.amd.com/en/develop/how-to/system-optimization/strixhalo.html),
  [ROCm#5562 (amd-ttm tool)](https://github.com/ROCm/ROCm/issues/5562)
* Community recipe on a 780M box: `amdgpu.gttsize=24576` (MiB) in GRUB to expose 24 GB;
  `amdgpu.gttsize` is considered deprecated in favour of `ttm.pages_limit` (pages = MiB×256,
  and `ttm.pages_limit` must be ≥ the gttsize).
  [k8s.it](https://www.k8s.it/posts/running-a-local-llm-on-amd-radeon-780m-gfx1103-rocm-and-the-gpu-that-wasnt-supposed-to-work/),
  [llama.cpp#18159](https://github.com/ggml-org/llama.cpp/issues/18159)
* GTT allocations were not charged to the memory cgroup, which confuses OOM accounting in
  containers on RDNA3.5 APUs. [ROCm#6370](https://github.com/ROCm/legacy-rocm-build/issues/6370)
* Bandwidth is the real ceiling: a single-channel DDR5-5600 laptop measured ~31 GB/s and the
  CPU beat the iGPU on memory-bound work; dual-channel roughly doubles it. [k8s.it]
* How much can the iGPU address? In practice: dedicated carve-out + GTT limit, i.e. most of
  system RAM if you raise `ttm.pages_limit`. This is shared with ODM's own CPU stages, so a
  16 GB laptop has very little headroom (see `05-cpu-first-tuning-amd-laptop.md`).

Also set the GPU DPM level for compute runs: the 780M idled at 800 MHz under power
management; `echo high > /sys/class/drm/card0/device/power_dpm_force_performance_level`
raised prefill throughput 37 → 54 tok/s. [k8s.it]

## 2. ROCm on Windows 11

### 2.1 HIP SDK for Windows 7.2.0

* Supported table includes **"AMD Ryzen AI 300 Series | RDNA3.5 | gfx1150 | ✅ | ✅"** and
  **"AMD Ryzen AI Max+ Pro 395 | RDNA3.5 | gfx1151 | ✅ | ✅"**, plus RX 7900/9070-class
  dGPUs. `gfx1103` and `gfx90c` are absent. Windows 11 22H2+ only.
  [HIP SDK Windows system requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html)
* Components: HIP runtime (closed source), hipcc/clang++, HIP math and primitive libraries,
  HIPIFY, a Visual Studio plugin. **Not available:** MIOpen, MIGraphX, PyTorch/TF via the SDK,
  communication libs; **"CMake HIP Language: Unsupported"**; rocm-smi/rocminfo replaced by
  `hipInfo`. [Component support](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html)
* Unofficial 780M on Windows: the community `ROCmLibs-for-gfx1103-AMD780M-APU` project ships
  rocBLAS/Tensile files for `gfx1103`, `gfx90c`, `gfx1150` (experimental) etc. for HIP SDK
  5.7 → 7.1.1; used by Ollama, llama.cpp, ZLUDA users. This tells us the Windows HIP runtime
  *does* launch kernels on gfx1103 — only the BLAS libs are missing.
  [likelovewant/ROCmLibs](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU)

### 2.2 PyTorch / ONNX on Windows via ROCm

* PyTorch 2.9.1 + ROCm 7.2.1 on Windows officially covers the same nine Ryzen AI parts
  (`gfx1150`/`gfx1151`) and RX 7000/9000; "the entire ROCm stack is not yet supported on
  Windows". Limitations: inference only, Python 3.12 only, disable WDAG and Smart App
  Control. [Windows Ryzen compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html),
  [Ryzen limitations](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.1.1/docs/limitations/limitationsryz.html)
* There is **no ONNX Runtime ROCm/MIGraphX build for Windows**. For ONNX on Windows the
  options are DirectML or the WebGPU EP (section 5).

### 2.3 WSL2

* ROCm on WSL2 used to be dGPU-only (7900 XTX validated). In May 2026 AMD's `librocdxg`/ROCDXG
  1.2 added **GFX11.0.3 (Radeon 780M)** and GFX11.5.2 (Krackan) iGPUs, on top of Strix/Strix
  Halo. Requirements: Windows 11, Adrenalin 26.2.2+, Ubuntu 22.04/24.04 in WSL2.
  Limitations: no `rocm-smi`, **no MIGraphX**, no multi-GPU, JAX issues, "performance may be
  limited compared to native Linux".
  [Phoronix on librocdxg 1.2](https://www.phoronix.com/news/librocdxg-1.2-More-Ryzen-ROCm),
  [WSL how-to](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html)

## 3. Vulkan compute

* Linux: RADV (Mesa) is the driver AMD itself now invests in ("In summer of 2025 AMD began to
  refocus from AMDVLK to RADV"). Mesa 26.x adds `VK_KHR_performance_query`,
  `VK_EXT_descriptor_heap`, etc. Works on every part in scope including `gfx90c`.
  [RADV docs](https://docs.mesa3d.org/drivers/radv.html), [Phoronix Mesa 26.1 APU note](https://www.phoronix.com/news/Mesa-26.1-RadeonSI-Rusticl-APUs)
* Windows: the Adrenalin driver's Vulkan implementation.
* Vulkan is the only GPU API that is **official on all four iGPUs on both OSes**. Its cost is
  that nothing in ODM speaks Vulkan today. llama.cpp's Vulkan backend "works with integrated
  and discrete GPUs across Windows, Linux, macOS" and is the usual fallback when ROCm refuses
  an iGPU. [llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

## 4. OpenCL

| Stack | Where | Notes |
|---|---|---|
| ROCm OpenCL (`rocm-opencl`) | Linux, official-matrix GPUs | conda-forge packages `rocm-opencl`, `rocm-opencl-runtime` exist (linux-64). [hip-feedstock](https://github.com/conda-forge/hip-feedstock) |
| Mesa **rusticl** on radeonsi | Linux, any Radeon | `RUSTICL_ENABLE=radeonsi` (not advertised by default). Mesa 26.1 fixed unified-memory reporting on APUs; Mesa 26.2 reports OpenCL 3.1. [rusticl docs](https://docs.mesa3d.org/rusticl.html), [Phoronix](https://www.phoronix.com/news/Mesa-26.1-RadeonSI-Rusticl-APUs) |
| Mesa clover | Linux | legacy, avoid |
| AMD Windows OpenCL | Windows, Adrenalin | works on all in-scope iGPUs; not something ODM uses |

Important ODM detail: conda-forge's OpenCV is built with `-DWITH_OPENCL=0`
([opencv-feedstock build.sh](https://github.com/conda-forge/opencv-feedstock/blob/main/recipe/build.sh)),
so OpenCV's Transparent API/UMat is a no-op in the pixi env. OpenCL is therefore only useful
if ODM gained a purpose-written OpenCL kernel (nobody has written one for PatchMatch or SIFT
that is maintained).

## 5. ML inference stacks for ODM's ONNX models

| Execution provider | Linux | Windows | iGPU status | Source |
|---|---|---|---|---|
| ROCm EP | last wheels `onnxruntime_rocm 1.22.1` for ROCm 7.0; **removed since ORT 1.23** | no | needs override on gfx1103 | [ROCm EP](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html) |
| MIGraphX EP | `onnxruntime_migraphx 1.23.2` for ROCm 7.2.1 (cp310/cp312) | no | gfx1150 in MIGraphX target list; gfx1103 SIGSEGVs at compile ([immich#27387](https://github.com/immich-app/immich/issues/27387)) | [MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html) |
| DirectML EP | no | `onnxruntime-directml 1.24.4` (Mar 2026) | any DX12 GPU incl. Vega | [DML EP](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html), [PyPI](https://pypi.org/project/onnxruntime-directml/) |
| WebGPU EP (native, Dawn) | Vulkan | D3D12/Vulkan | any Vulkan GPU; maturity unproven for ODM | [WebGPU EP](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html) |
| OpenVINO EP | Intel only | Intel only | no | — |
| "Vulkan EP" | does not exist; WebGPU EP is the Vulkan route | | | |

DirectML is "in sustained engineering"; "new feature development has moved to WinML", it was
declared deprecated at Build 2025 and the GitHub README says maintenance mode, yet
`onnxruntime-directml` still ships (1.24.4, 2026-03-17). Fine for ODM's two small models for
the next few years; not a platform to build new features on.
[microsoft/DirectML](https://github.com/microsoft/DirectML), [ORT#23783](https://github.com/microsoft/onnxruntime/issues/23783)

Details in `03-onnxruntime-on-amd-igpu.md`.

## 6. CUDA-on-AMD translation: ZLUDA

* History: Intel-funded, then AMD-funded (2022–24); AMD withdrew, the code "has been rolled back
  to the pre-AMD state" (Oct 2024) and rewritten under a new commercial sponsor targeting
  RDNA1+ consumer GPUs, Linux mainline, Windows secondary.
  [ZLUDA's third life](https://vosen.github.io/ZLUDA/blog/zludas-third-life/)
* Q4 2025 (published Jan 2026): ROCm 7 support, big Windows loader refactor, llama.cpp
  "complete", PyTorch WIP, bundled patched LLVM.
  [Q4 2025 update](https://vosen.github.io/ZLUDA/blog/zluda-update-q4-2025/),
  [Phoronix](https://www.phoronix.com/news/ZLUDA-ROCm-7)
* Q1/Q2 2026 (June 2026): "no longer commercially funded, so it's back to being my weekend
  project"; texture support is "basic" (PhysX/Blender cases); cuRAND/cuFFT not mentioned.
  ZLUDA 6 stable (June 2026), 7 previews.
  [Q1&Q2 2026 update](https://vosen.github.io/ZLUDA/blog/zluda-update-q1q2-2026/)
* It sits on top of ROCm/HIP, so every iGPU caveat above applies (override on gfx1103,
  community rocBLAS on Windows). On Windows the `CUDA-for-AMD-Windows` guide packages that.
  [Speedstu/CUDA-for-AMD-Windows](https://github.com/Speedstu/CUDA-for-AMD-Windows)
* Legal caveat: the CUDA EULA §1.2 limitation 8 says "You may not reverse engineer, decompile
  or disassemble any portion of the output generated using SDK elements for the purpose of
  translating such output artifacts to target a non-NVIDIA platform." Whether running an
  unmodified binary through ZLUDA at runtime falls under this is untested; not legal advice.
  [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html)

## 7. Other portability layers

* **chipStar 1.3**: compiles HIP/CUDA to SPIR-V and runs on OpenCL (rusticl qualifies) or
  Level Zero; "mature enough for wider testing and even production use" per its 2026 paper.
  Texture support and RDNA iGPU coverage are not documented; treat as experimental.
  [chipStar](https://github.com/CHIP-SPV/chipStar), [Phoronix 1.3](https://www.phoronix.com/news/chipStar-1.3-Released), [paper](https://doi.org/10.1177/10943420261423001)
* **SYCL**: AdaptiveCpp's generic SSCP flow JITs to amdgcn at runtime for ROCm GPUs; Codeplay
  ships "oneAPI for AMD GPUs" for DPC++ (supported gfx list not fetched — verify).
  PopSift's maintainer, who has a SYCL branch, reports "Intel's SYCL compiler doesn't do very
  well with the PopSift port, while U.Stuttgart's compiler [AdaptiveCpp] doesn't support
  hardware interpolation via sycl::image. CUDA and HIP still win."
  [AdaptiveCpp SSCP](https://adaptivecpp.github.io/hipsycl/sscp/compiler/generic-sscp/),
  [Codeplay plugin](https://developer.codeplay.com/products/oneapi/amd/home/),
  [popsift#186 comments](https://github.com/alicevision/popsift/pull/186)
* **HIPCL / SYCLomatic** were the options the ODM forum floated in 2023; neither produced a
  port. [forum 16485](https://community.opendronemap.org/t/amd-radeon-support/16485)

## 8. Decision table (OS × workload)

| Workload | Linux, gfx1150/1151 | Linux, gfx1103 | Linux, gfx90c | Windows 11, gfx1150/1151 | Windows 11, gfx1103 | Windows 11, gfx90c |
|---|---|---|---|---|---|---|
| OpenMVS densify (needs a HIP port, see 01) | HIP, official | HIP, build for gfx1103, no override expected | HIP with `HSA_OVERRIDE_GFX_VERSION=9.0.0`, risky | HIP SDK 7.2, official; MSVC integration is the hard part | HIP SDK runtime works per community; unofficial | doubtful; CPU |
| SIFT features (PopSift HIP PR #186) | HIP (PR tested on gfx1151 Windows/TheRock, gfx1100 Linux) | HIP, unverified | probably not (wave64 path was tested only on gfx90a) | HIP | unofficial | CPU |
| ONNX sky/bg models | MIGraphX EP | CPU (MIGraphX crashes) or WebGPU EP | CPU/WebGPU | DirectML (or WebGPU) | DirectML | DirectML |
| Everything else (BA, meshing, texturing, DEM, ortho) | CPU | CPU | CPU | CPU | CPU | CPU |
| Zero-code-change option | ZLUDA on CUDA binaries (fragile) | ZLUDA + override | no | ZLUDA + HIP SDK | ZLUDA + community rocBLAS | no |

Ordering of practical value for a laptop owner today: (1) CPU tuning (doc 05), (2) DirectML /
MIGraphX for the ONNX models (doc 03), (3) a HIP OpenMVS build (doc 01), (4) PopSift HIP
(doc 02). Vulkan/OpenCL are portable but require writing kernels nobody has written.

## Sources

- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.1.1/docs/limitations/limitationsryz.html
- https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html
- https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html
- https://www.phoronix.com/news/librocdxg-1.2-More-Ryzen-ROCm
- https://rocmdocs.amd.com/en/develop/how-to/system-optimization/strixhalo.html
- https://github.com/ROCm/ROCm/issues/5444 , https://github.com/ROCm/ROCm/issues/5562 , https://github.com/ROCm/legacy-rocm-build/issues/6370
- https://github.com/ROCm/rocm-libraries/issues/6335
- https://github.com/kvark/inferena/issues/61
- https://github.com/ollama/ollama/issues/3189
- https://www.k8s.it/posts/running-a-local-llm-on-amd-radeon-780m-gfx1103-rocm-and-the-gpu-that-wasnt-supposed-to-work/
- https://gist.github.com/catbraincell/391721ee1b196773ed259baaab10f18b , https://github.com/ROCm/ROCm/issues/1743
- https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake , https://github.com/ROCm/TheRock/issues/1939
- https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md , https://github.com/ggml-org/llama.cpp/issues/18159
- https://docs.mesa3d.org/drivers/radv.html , https://docs.mesa3d.org/rusticl.html , https://www.phoronix.com/news/Mesa-26.1-RadeonSI-Rusticl-APUs
- https://github.com/conda-forge/hip-feedstock , https://github.com/conda-forge/opencv-feedstock/blob/main/recipe/build.sh
- https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html , https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html , https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html , https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html
- https://github.com/microsoft/DirectML , https://github.com/microsoft/onnxruntime/issues/23783 , https://pypi.org/project/onnxruntime-directml/
- https://github.com/immich-app/immich/issues/27387
- https://vosen.github.io/ZLUDA/blog/zludas-third-life/ , https://vosen.github.io/ZLUDA/blog/zluda-update-q4-2025/ , https://vosen.github.io/ZLUDA/blog/zluda-update-q1q2-2026/ , https://www.phoronix.com/news/ZLUDA-ROCm-7 , https://github.com/Speedstu/CUDA-for-AMD-Windows
- https://docs.nvidia.com/cuda/eula/index.html
- https://github.com/CHIP-SPV/chipStar , https://www.phoronix.com/news/chipStar-1.3-Released , https://doi.org/10.1177/10943420261423001
- https://adaptivecpp.github.io/hipsycl/sscp/compiler/generic-sscp/ , https://developer.codeplay.com/products/oneapi/amd/home/
- https://github.com/alicevision/popsift/pull/186
- https://community.opendronemap.org/t/amd-radeon-support/16485
