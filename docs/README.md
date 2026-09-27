# ADM research docs

Research-only documentation written 2026-09-13 for the ADM project (an OpenDroneMap 3.6.2
clone under `../ODM`). **No code was changed.** Start with `research-log.md` for what was
verified vs. estimated, then the two overview/roadmap files.

## 00-context — what we are working with
| File | Summary |
|---|---|
| `00-context/01-project-context.md` | Repo state, ODM stage graph, binaries, build system, the three asks |
| `00-context/02-hardware-profile.md` | M4 Pro / 24 GB / Metal 4 machine facts (verified) and what they imply |
| `00-context/03-gpu-touchpoints-today.md` | The four CUDA-only GPU touchpoints with file:line references |
| `00-context/04-toolchain-osx-arm64.md` | Compiler/library versions pinned by `pixi.lock` for osx-arm64 |

## golden-hour-shaders — custom golden hour look for ODM outputs
| File | Summary |
|---|---|
| `00-overview.md` | **Start here.** De-light + relight + grade pipeline, what ODM outputs, reading order |
| `01-golden-hour-physics.md` | Elevation bands, air mass, Rayleigh/Mie, CCT and direct/diffuse vs elevation tables, shadow length |
| `02-sun-position-from-metadata.md` | pysolar/NREL SPA, EXIF → sun vector in ENU/NED/engine frames, golden-hour time solver, ODM UTC caveat |
| `03-sky-and-sun-models.md` | Preetham vs Hosek–Wilkie vs Bruneton vs Hillaire vs three.js Sky; compact GLSL sun+sky function |
| `04-glsl-relighting-shaders.md` | Full GLSL ES 3.00 mesh relight (CSM PCF, fog, rim) and ortho+DSM ray-marched shadow shaders; WGSL notes |
| `05-color-grading-and-tonemapping.md` | Kelvin→RGB fits, ACES/AgX/PBR Neutral, split toning, bloom, `.cube` LUTs, golden-hour defaults |
| `06-metal-msl-port-notes.md` | GLSL↔MSL↔WGSL mapping, Metal compute kernel for DSM shadows, MTKView host, toolchain, converters |
| `07-delighting-photogrammetry-textures.md` | Removing baked lighting: tools, papers, ONNX models, cheap DSM+sun heuristic pipeline |
| `08-orthophoto-dsm-relight.md` | 2.5D relight: gdaldem, horizon/SVF tools, GPU heightfield ray-march, COG delivery, unit traps |
| `09-threejs-webgpu-viewer-integration.md` | three.js r186 `SunLight`/Sky/CSM, 3DTilesRendererJS, CesiumJS, texture-size limits, KTX2 |
| `10-blender-cycles-golden-hour.md` | Nishita sky, Sun Position add-on, Cycles-Metal, AgX, script outline replacing 2.7x-era contrib |
| `11-input-side-golden-hour-considerations.md` | Shooting at golden hour: failure modes, ODM flags, colour normalisation, synthetic-data angle |
| `12-golden-hour-references.md` | 76-entry annotated bibliography with licences |

## gpu-acceleration — making the pipeline fast on laptops
| File | Summary |
|---|---|
| `gpu-acceleration/00-roadmap.md` | **Start here.** Headline findings, phased Mac plan, AMD plan, decision table |
| `anatomy/01-openmvs-cuda-anatomy.md` | Every CUDA file/kernel in ODM's openMVS fork, `--cuda-device` semantics, memory model, portability matrix |
| `anatomy/02-opensfm-features-anatomy.md` | `feature_type` dispatch, array contracts, matching, config keys ODM writes; contract for a new backend |
| `anatomy/03-pypopsift-anatomy.md` | pypopsift API/build, PopSift structure, texture-size limit, Metal/HIP mapping |
| `anatomy/04-prior-art-non-cuda-odm.md` | 2017→2026 timeline of AMD/OpenCL/Apple GPU attempts in ODM; what was and wasn't tried |
| `cpu-baseline/00-pipeline-stage-map.md` | Stage-by-stage binaries, parallelism, cost drivers with exact numeric mappings from the code |
| `cpu-baseline/01-macos-arm64-build-notes.md` | Every APPLE/arm64 branch in the SuperBuild, toolchain, commands, CI build times, known issues |
| `cpu-baseline/02-apple-silicon-cpu-tuning.md` | `-mcpu=apple-m4`, NEON, Accelerate BLAS, OpenMP on macOS, memory sizing, thermals |
| `cpu-baseline/03-fast-settings-playbook.md` | Three flag presets (fast ortho / balanced / overnight) with trade-offs |
| `cpu-baseline/04-benchmark-and-profiling-plan.md` | Datasets, timing sources, samplers, profilers, results template |
| `metal/00-metal-landscape-2026.md` | Metal 4, MSL, compute model, unified memory, limits measured on this M4 Pro, OpenCL status, portability layers |
| `metal/01-openmvs-on-metal.md` | Upstream Metal PatchMatch backend (PR #1279) and how to bring it into ODM; alternatives with effort |
| `metal/02-opensfm-features-on-metal.md` | SIFT/feature options on Apple GPU; OpenCV has no OpenCL here; learned features; BA on Accelerate |
| `metal/03-onnxruntime-coreml-mps.md` | CoreML EP for the sky/bg models; version pin gap; described provider change |
| `metal/04-opencv-gdal-pdal-accel-on-apple.md` | What else could use Apple GPU (little); `odm_orthophoto` as a Metal render pass; OpenMP pitfalls |
| `metal/05-build-recipe-macos-metal.md` | pixi build on this Mac, proposed `osx-metal` feature, CMake/runtime shader strategy, `has_metal()` |
| `radeon/00-amd-igpu-compute-landscape-2026.md` | ROCm/HIP/Vulkan/OpenCL/DirectML/ZLUDA status for gfx1103/gfx1150/gfx90c; decision table |
| `radeon/01-openmvs-hip-port.md` | hipify plan for PatchMatchCUDA, CMake/ODM deltas, ZLUDA, effort and honest speed-up |
| `radeon/02-popsift-and-features-on-amd.md` | PopSift HIP PR #186, alternatives, Ceres/BLAS on CPU |
| `radeon/03-onnxruntime-on-amd-igpu.md` | MIGraphX / DirectML / WebGPU EPs; ROCm EP removed in ORT 1.23 |
| `radeon/04-build-recipe-linux-windows-amd.md` | Ubuntu ROCm install, `amd` pixi feature, Windows HIP SDK, WSL2, Docker |
| `radeon/05-cpu-first-tuning-amd-laptop.md` | Sure CPU wins on a Ryzen laptop before any GPU work |

## Conventions
* Claims verified on this machine are tagged "verified locally"; unverified ones are marked
  `[uncertain]` or "verify". Every file ends with a Sources section.
* File paths are relative to `/Users/avneh/Code/ADM/ODM` unless stated.
