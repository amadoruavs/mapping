# Research log — 2026-09-13

Owner request: (1) custom golden hour shaders, (2) docs on compiling the pipeline for Metal and
for an integrated Radeon GPU so it is fast on a laptop, (3) deep research saved as many Markdown
files, **no code changes** (another agent, "Codex", is expected to land code separately).

## What was done

* Read the ODM 3.6.2 clone: `run.py`, `stages/*`, `opendm/gpu.py`, `opendm/osfm.py`,
  `opendm/config.py`, `SuperBuild/CMakeLists.txt` and every `External-*.cmake`, `pixi.toml`,
  `pixi.lock` (osx-arm64 versions), Dockerfiles, `contrib/blender`.
* Verified host facts on this MacBook Pro (M4 Pro, 8P+4E, 16-core GPU, 24 GB, macOS 26.6,
  Metal 4, CLT only, no `xcrun metal`, OpenCL.framework present, pixi absent).
* Ran six parallel research tracks (~350 web searches/fetches, scratch clones of
  OpenDroneMap/openMVS@355, OpenSfM@c532843, pypopsift, popsift, all deleted afterwards):
  golden-hour physics/GLSL; golden-hour applied to ODM outputs; Metal; AMD Radeon;
  CPU baseline on macOS; OpenMVS/OpenSfM/PopSift GPU anatomy and prior art.
* Wrote 41 Markdown files (~10k lines) under `docs/`. `ODM/` git status is clean.

## Verified vs. estimated

* **Verified locally**: hardware table; absence of the Metal toolchain; runtime MSL compile
  works via `clang++ -framework Metal` + `newLibraryWithSource`; Metal device limits
  (`maxBufferLength` 14.3 GB, working set 19 GB, 32 KB threadgroup memory, SIMD 32); OpenCL 1.2
  enumerates the GPU; system-python onnxruntime 1.22.1 exposes `CoreMLExecutionProvider`.
* **Verified from source/primary docs**: all ODM line references; openMVS PR #1279; PopSift
  PR #186; ODM's 2021 silx OpenCL SIFT history; `KHR_materials_unlit` in `gltf.py`;
  EXIF-as-UTC in `photo.py`; conda-forge OpenCV `WITH_OPENCL=0`; ORT ROCm EP removal at 1.23.
* **Estimated / uncertain (flagged in-doc)**: all speed-up multipliers; person-week efforts;
  Metal backend accuracy on aerial data; gfx1103 behaviour under HIP; CCT-vs-elevation table
  (computed, not from a peer-reviewed source); Hillaire LUT sizes; Tint CLI syntax.

## Open questions (merged)

Golden hour
1. Code availability for Song & Qin 2024 (arXiv 2409.03032) inverse rendering with known sun.
2. Aerial shadow detector with permissive licence + ONNX export (S-EO U-Net is the lead).
3. Does Obj2Tiles keep `KHR_materials_unlit`?
4. ODM stamps EXIF local time as UTC (`opendm/photo.py`) — decide fix vs. workaround.
5. Blender 4.2+ Sun Position extension property names; MetalRT default on M4.

Metal
6. Does pixi's conda-clang activation expose the CLT SDK so `find_library(Metal)` and
   `enable_language(OBJCXX)` work inside the SuperBuild? (untested — nothing built yet)
7. Rebase cost of the ODM openMVS fork onto upstream `develop` vs cherry-picking PR #1279.
8. Metal PatchMatch accuracy/memory on real aerial datasets.
9. Correct pixi build-string wildcard for the Accelerate BLAS variant.

CPU / build
10. PoissonRecon `libomp.dylib` load failure on macOS under SIP (PR #2037 symptom) — verify
    with `otool -l` after first build.
11. Does conda clang 19.1.7 map `-mcpu=native` to `apple-m4`?
12. Does conda-forge OpenCV keep AArch64 runtime dispatch (NEON_FP16/BF16/DOTPROD)?
13. Obj2Tiles macOS binary: x64 under Rosetta or arm64?
14. No native Apple-silicon ODM timings exist anywhere; Phase 0 establishes them.

AMD
15. HIP + `cudaFilterModeLinear` float textures on ROCm 7.2 (PatchMatchCUDA uses exactly that).
16. HIP-compiled OpenMVS on gfx1103 with no override — untried by anyone.
17. conda-forge `hip-*` package coverage (no hiprand feedstock found).
18. Windows: HIP SDK clang vs the repo's `cl`+Ninja SuperBuild; HIP runtime DLL name.

Anatomy
19. Effective fatbin arch set when `CUDA_ARCHITECTURES "50;72;75"` and
    `MAX_CUDA_COMPATIBILITY` coexist.
20. Angle units of OpenSfM `points[:,3]` (degrees vs radians) across backends.

## Suggested next actions (after Codex lands)

1. Install pixi, free ~10 GB, build, run the Phase 0 benchmarks (`gpu-acceleration/cpu-baseline/04`).
2. Prototype the ortho+DSM golden-hour shader in a standalone three.js page against a real
   ODM output (`golden-hour-shaders/08`, `04`) — no ODM code needed.
3. Spike the openMVS rebase onto upstream with the Metal backend (`gpu-acceleration/metal/01` §5).
