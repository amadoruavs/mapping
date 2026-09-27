# Prior art: non-CUDA / non-x86 GPU acceleration in the OpenDroneMap ecosystem

Scope: every forum thread, GitHub issue/PR, commit and doc page found that
concerns running ODM's GPU paths on AMD, Apple Silicon, Intel, ARM/Jetson, or
via OpenCL/ROCm/HIP/Metal/Vulkan/SYCL. Sources are community.opendronemap.org
(fetched), GitHub (API + fetch), the ODM tree at `/Users/avneh/Code/ADM/ODM`,
and upstream cdcseacave/openMVS. Web search quota was exhausted part-way, so
coverage of the forum is from targeted queries plus GitHub API; it is
unlikely to be exhaustive for forum threads after mid-2024.

---

## 1. Timeline

| Date | Where | What happened |
|---|---|---|
| 2017-02-17 | ODM #489 "Is possible implement the program on GPU with CUDA?" | Early request; closed |
| 2017-04-20 | openMVS #206 (PolarNick239) | Asked for CUDA kernel sources (they were PTX strings in `SceneRefineCUDA.cpp`) to write `SceneRefineOpenCL.cpp`; no maintainer reply visible; closed |
| 2018-05-24 → 06-11 | Forum `rocm-docker/652` (Fabio_Raimondo) | Asked about ROCm-docker for AMD APUs. pierotofy: "We currently don't leverage the GPU, but this has been brought up multiple times". User "switched to Nvidia GPU instead" |
| 2021-02-09 | ODM commit `3b2a60737` "Add gpu dockerfile"; OpenSfM fork `ed7c8fb5f` "Rewrote changes from cojosef96/gpu" | Start of GPU work |
| 2021-02-24 / 03-05 | OpenSfM fork `eb4b8ff84` "PoC GPU support", `aac21e23f` "Merged gpu"; **ODM PR #1239 "GPU Support"** (pierotofy, merged 2021-03-05) | "Adds support for GPU SIFT (**OpenCL based**)" using `silx.image.sift` (`opensfm/sift_gpu.py`). GPU SIFT *matching* was also implemented but left disabled: "in my benchmarks it seems to underperform the CPU based FLANN". ODM commit `5e3737a3f` "GitHub GPU workflow, use SIFT_GPU when GPUs are available". This is the origin of the README log line `Found GPU device: Intel(R) OpenCL HD Graphics` (`README.md:110`) still present in 3.6.2 |
| 2021-03-08 | ODM #1240 "Add support for Mac Silicon (M1) Chips" (shawndearmond) | Docker image failed under QEMU. pierotofy: "ODM does not support M1 chips at the moment. It's something we'd like to add at some point in the future." (2021-03-08); "Still not supported." (2021-04-14). charliep reported building an image on M1 (2021-05-23) but hit a missing Python module; unanswered |
| 2021-05-14/15 | Forum `odm-gpu-on-radeon/7622` (machinefly, RX 570; hblanken, Mac Radeon Pro 570X) | Tried amdgpu-pro OpenCL and ROCm; `clinfo` inside the container showed `CL_DEVICE_TYPE_GPU ... Available: Yes` but `--gpu` needs the NVIDIA container hook. No resolution; the OpenCL SIFT never ran on AMD in Docker because the runtime wiring was NVIDIA-only |
| 2021-07-17 | Forum `building-odm-on-the-apple-m1-chipset/8334` (charliep) | `docker build --platform linux/arm64/v8`; Saijin_Naib: "things will get much better/easier for building on Apple M1 chips with the funding campaign". No GPU/Metal mention |
| 2021-10-22 → 12-17 | openMVS fork commits `8e24a58c8` "Expose --cuda-device option", `4b85f74e0` "Target sm_35", `7bd32db29` "Add OpenMVS_MAX_CUDA_COMPATIBILITY", `9de2e1d86`; ODM commit `15275d9aa` "Update start-dev-env.sh, OpenMVS" | OpenMVS CUDA densification enabled in ODM |
| 2021-10-29 | ODM #1365 "SIFT_GPU fails with very large images (>11000 pixels wide)" (pierotofy) | Workaround: HAHOG. pierotofy 2021-11-22: "Plan is to replace the current GPU code with https://github.com/uav4geo/pypopsift" |
| 2021-12-16/17 | OpenSfM fork `c90b0344a` "GPU code cleanup, **pypopsift in, silx out**"; ODM `0ca7d1cef` "Pypopsift support", `239f05194` "Popsift check fix"; pypopsift `838b7aecc` "Add fitsTexture method" | **OpenCL support removed**; GPU SIFT becomes CUDA-only |
| 2022-06-10 → 22 | Forum `gpu-acceleration-w-amd/11947` (NGE_Drone_Services, Radeon Pro 5500M; geoma, RX 5600) | Saijin_Naib: "Unless AMD gains support for CUDA, I would think it would be unlikely, unfortunately"; "You will not be able to as the Radeon hardware/software stack does not support CUDA"; "I'm waiting for more news about Intel's SYCLomatic to see if it might be an option" |
| 2022-09-07 → 27 | Forum `using-specific-gpu-with-nodeodm/12983` (ChrisDAT, Johnny5) | Multi-GPU NVIDIA only: `NVIDIA_VISIBLE_DEVICES` in `docker-compose.nodeodm.gpu.nvidia.yml`. "Success!!! Two separate tasks using separate GPU cores" |
| 2022-12-15/16 | Forum `why-webodm-not-use-my-gpu/14070` (TOPOGRAFIA_SC, RTX 3060 Ti) | GPU SIFT refused 20 MP images; Saijin_Naib: "it isn't primarily the memory usage, but rather there are finite image dimensions that can be properly accelerated within CUDA" → use Resize Images |
| 2023-06-21 | Forum `amd-radeon-support/16485` (doletas) | Why do AMD/Intel options exist in WebODM's `.sh`? Saijin_Naib: "**We had OpenCL support before CUDA, hence the AMD and Intel targets**"; "There might be a time where Intel's SYCLomatic can be used to run the CUDA code over Intel/AMD OpenCL/Vulkan" |
| 2023-08-09 | ODM #1685 "Can't build GPU docker for Nvidia Jetson devices" (Dennis-Johnson, Tegra/arm64) | pierotofy 2023-08-25 suggested `start-dev-env.sh ... GPU=YES` then noted "ah, never mind, I see that you're running on ARM". grunlab 2025-03-12: "I was finally able to build the odm-gpu image on ARM64" (Turing Pi RK1; image `registry.gitlab.com/grunlab/opendronemap/odm-gpu:3.5.5-arm64`, later reported missing by piaoyaoi 2026-02-03). pierotofy: "Yes you need quite a bit of memory to build ODM". Note: RK1 (Rockchip) has no NVIDIA GPU, so this was a CUDA-*linked* arm64 build, not a working Tegra GPU path |
| 2023-09-09/10 | Forum `help-requested-navigate-source-code-to-hipify-for-amd-gpu-support/17500` (brewfalcon) | Wanted to run HIPify/ROCm; could not locate the CUDA code. Saijin_Naib pointed at SuperBuild deps and suggested SYCLomatic instead ("code should run equally well on any device (Intel/AMD/NVIDIA/other)"). brewfalcon: "I'll poke around, thanks!" — no follow-up |
| 2023-10-25 / 11-28 | openMVS #1076 "opencl support?" (zoey9628: "I want to run it not in nvidia drivers"), #1085 "Can we use GPU other then CUDA" | No maintainer replies; both closed |
| 2023-12-21 → 26 | Forum `is-there-an-odm-gpu-docker-image-for-ubuntu-arm64/18686` (oyzw0720, Jetson AGX Orin) | `no matching manifest for linux/arm64/v8`. pierotofy: "**We currently have no GPU support for ARM (but we'd welcome contributions).**" oyzw0720: "I'll try to work on this." — nothing landed |
| 2024-03-06 | OpenSfM fork #29 "GPU Accelerated BRUTEFORCE feature matching" (Elliot-Construct, cv2.cuda) | pierotofy: duplicate of forum `gpu-accelerated-feature-mapping/19571`; closed |
| 2024-03-26/27 | Forum `gpu-enabled/19805` (apdim, RTX 3090, Windows) | `--gpu` "not supported on Windows" message; root cause: "If I use dspsift-it is not using GPU / If I use sift-it is using GPU". Saijin_Naib: "In short, only SIFT is currently GPU-accelerated. Everything else will need to be re-written." (imprecise — OpenMVS densification is also CUDA, see doc 01) |
| 2024-04-18/19 | Forum `odm-on-macos/20064` (andrewhbradley9) | Native Apple Silicon install errors. pierotofy: "It's been a while since I've tested the native macOS build, so minor issues might be present." alanterra moved to Asahi Linux + Docker. No GPU/Metal mention |
| 2024-05-09 | Forum `cuda-or-opencl/20330` (wmolevelt, Radeon) | pierotofy: "**You can still use WebODM; you just will not get a boost in processing during certain parts of the computation.**" |
| 2024-07 | ODM commits `709afd16f` "working gpu docker build", `8ce00e01a`, `51a7470eb` | GPU image maintenance (CUDA) |
| 2024-11-19 | ODM PR #1813 / commit `36897fc0e` "Cannot initialize CUDA even if nvidia-smi detected" | Hardened `gpu.py` fallback |
| 2025-07-24 | ODM PR #1895 "Minor updates for GPU" | CUDA |
| 2025-08/09 | pypopsift PR #15, OpenSfM fork PR #43 "24.04", ODM PR #1904 (NathanMOlson) | CUDA 13 compile fixes; pypopsift moved to upstream alicevision/popsift |
| 2026-01-20 | upstream openMVS v2.4.0 | "CUDA support: New GPU architectures (A100, etc.)" — still CUDA-only |
| 2026-02-05 | ODM #1993 "ARM64 gpu support" (brandonwheat, DGX Spark) | Bot redirected to the forum; closed |
| 2026-03-12 | ODM PR #2003 "Fix GPU pipeline failure in Kubernetes/GKE (libcuda.so.1 and nvidia-smi PATH)" | Open; CUDA |
| 2026-05-11 → 06-24 | ODM PR #2037 "Moving build system to pixi" | `pixi.toml` adds `osx-arm64` (macOS 14) as a **CPU** platform; the `gpu` feature is limited to `linux-64-cuda-12-0` / `win-64-cuda-12-0` (`pixi.toml:4-14`) |
| 2026-06-08 → 06-26 | **upstream openMVS PR #1279 "Add Metal compute backend for PatchMatch densification (Apple Silicon)"** (leNeo) | Merged into `develop` 2026-06-26. MSL port of the CUDA PatchMatch; `--gpu-device` replaces `--cuda-device`; validated on M4 Max; maintainer cross-checked against CUDA ("F1 score is 1-2% higher", then fixed). **Not in any ODM release**; ODM's openMVS fork tag `355` (2026-05-16) is based on a 2023-08-20 upstream state |
| 2026-08-19/24 | ODM commits `95db3c54d` "Mock the CUDA virtual package during the GPU docker build", `cb4ed84be`, `c18c27571` | GPU image (CUDA 12.9 runtime, `gpu.Dockerfile:34`) |
| 2026-08-31 | ODM PR #2074 "Openmvs 240" | Open; targets OpenMVS 2.4.0 (pre-Metal) |

---

## 2. Official documentation state

- `README.md:95-115` (ODM 3.6.2): "ODM has support for doing SIFT feature
  extraction on a GPU, which is about 2x faster than the CPU on a typical
  consumer laptop"; use `opendronemap/odm:gpu` with `--gpus all` and
  `--feature-type sift`; "The SIFT GPU implementation is CUDA-based, so should
  work with most NVIDIA graphics cards of the GTX 9xx Generation or newer."
  The sample log still shows `Found GPU device: Intel(R) OpenCL HD Graphics`,
  a leftover from the 2021 OpenCL era. OpenMVS GPU densification is not
  mentioned in the README at all.
- `docs.opendronemap.org`: no `/gpu/` page (HTTP 404). The installation page
  (fetched, "OpenDroneMap 3.6.0" docs) says under Hardware Recommendations:
  "A graphics card (GPU) currently has no impact on performance." — stale
  relative to the code since 2021.
- `pixi.toml:13-14` / `SuperBuild/CMakeLists.txt:62-67`: CUDA is enabled only
  through the pixi `gpu` feature (`ODM_ENABLE_CUDA=ON`); the feature exists for
  linux-64 and win-64 only.
- `gpu.Dockerfile` (3.6.2): pixi build with `CONDA_OVERRIDE_CUDA=12.0` (line 14),
  runtime `nvidia/cuda:12.9.1-runtime-ubuntu24.04` (34), smoke test uses
  `/usr/local/cuda/compat` for `libcuda.so.1` (55-60). Commit history
  (GitHub, `path=gpu.Dockerfile`): 36 commits from `3b2a60737` (2021-02-09) to
  `c18c27571` (2026-08-24); none touch non-NVIDIA runtimes.
- WebODM: `webodm.sh` still carries `--gpu` handling with AMD/Intel branches
  (per forum 16485, "We had OpenCL support before CUDA, hence the AMD and Intel
  targets"); those branches have had nothing to drive since Dec 2021.
- NodeODM: `opendronemap/nodeodm:gpu` image exists for NVIDIA (`--gpus all`);
  arm64 NodeODM images are CPU-only.

---

## 3. What was actually tried (and its outcome)

| Attempt | Who / when | Outcome |
|---|---|---|
| OpenCL GPU SIFT (`silx.image.sift`, pyopencl) in OpenSfM fork + ODM `gpu.Dockerfile` | pierotofy, Feb-Mar 2021 (ODM PR #1239) | Shipped; portable in principle to AMD/Intel, but the Docker wiring only handled NVIDIA (forum 7622). Failed on >11000 px images (#1365). **Replaced by CUDA pypopsift in Dec 2021** (`c90b0344a`) |
| OpenCL GPU SIFT *matching* | same PR | Implemented, disabled: slower than FLANN |
| CUDA PatchMatch in OpenMVS via ODM fork (`--cuda-device`, `MAX_CUDA_COMPATIBILITY`) | pierotofy, Oct 2021 | Shipped; still the production GPU densifier (doc 01) |
| ROCm-docker | Fabio_Raimondo, 2018 | Abandoned (bought NVIDIA) |
| AMD OpenCL/ROCm drivers + `--gpu` | machinefly, hblanken, 2021 | Failed at the NVIDIA container hook; never reached the OpenCL code |
| HIPify the CUDA code | brewfalcon, 2023 | Could not find the code; no result |
| SYCLomatic (suggested repeatedly by Saijin_Naib 2022-2023) | — | Never attempted in public |
| Jetson / Tegra GPU build | Dennis-Johnson 2023, oyzw0720 2023 | Build failures; pierotofy: no ARM GPU support. grunlab (2025) produced an arm64 CUDA-linked image on non-NVIDIA hardware |
| DGX Spark (arm64 + Blackwell) | brandonwheat, 2026 | Redirected to forum; no public build |
| `cv2.cuda` brute-force matching | Elliot-Construct, 2024 | Redirected; not pursued |
| Native Apple Silicon build (CPU) | pierotofy (native macOS build), community 2021-2024; pixi `osx-arm64` 2026 | Works/maintained as CPU-only; no Metal anywhere in ODM |
| **Metal PatchMatch in upstream OpenMVS** | leNeo + cdcseacave, Jun 2026 | Merged upstream; not consumed by ODM |

---

## 4. What has *not* been attempted (as far as the public record shows)

1. Any Metal code path inside ODM, WebODM, NodeODM, the ODM OpenMVS fork,
   pypopsift or the OpenSfM fork. The only Metal work is upstream OpenMVS
   PR #1279 (densification only, June 2026).
2. A HIP/ROCm build of OpenMVS or PopSift (no fork, PR or gist found; upstream
   openMVS issues #206/#1076/#1085 asking for OpenCL went unanswered).
3. A Vulkan/WGSL backend for either PatchMatch or SIFT in this ecosystem
   (VulkanSift exists as an independent project tested on Apple M2, but no
   one has wired it into OpenSfM).
4. SYCL/SYCLomatic translation, despite being the maintainers' suggested route
   in 2022-2023.
5. GPU feature *matching* in production (rejected on benchmarks in 2021).
6. GPU acceleration of any stage other than SIFT extraction and OpenMVS
   depth-map estimation (mesh reconstruction, texturing, DEM, orthophoto are
   CPU everywhere; upstream OpenMVS has CUDA mesh refinement but ODM does not
   call `RefineMesh`).
7. An ARM64 + NVIDIA (Jetson/DGX Spark) **published** GPU image from the
   project.
8. Detecting non-NVIDIA GPUs in `opendm/gpu.py` (`has_gpu` checks only
   `nvidia-smi` / `nvcuda.dll`).

---

## 5. Implications for a Metal / AMD effort (observations only)

- Precedent for a non-CUDA backend *exists* upstream (Metal PatchMatch,
  1358 additions) and the maintainer was receptive ("great contribution, this
  PR is very valuable"), conditioned on quantitative comparison against CUDA.
  Adopting it in ODM requires moving the ODM fork forward ~3 years of upstream
  history and switching `stages/openmvs.py` from `--cuda-device` to
  `--gpu-device`.
- ODM's own gate is purely NVIDIA-shaped (`gpu.py`, pixi `gpu` feature,
  `External-PyPopsift.cmake` on `SB_ENABLE_CUDA`); a Metal/AMD path needs
  build-system and detection changes independent of any kernel work.
- The OpenCL precedent shows the failure mode to avoid: kernels that were
  portable but a runtime/detection layer that was not.
- Maintainer statements consistently frame non-NVIDIA support as "welcome
  contributions" rather than roadmap.

## 6. Supporting detail

### 6.1 The 2021 OpenCL era (the only non-CUDA GPU code ODM ever shipped)

ODM PR #1239 "GPU Support" (pierotofy, opened and merged 2021-03-05, 10 files,
+163/-4) touched: `.github/workflows/publish-docker-gpu.yaml`, `README.md`,
`SuperBuild/cmake/External-OpenSfM.cmake`, `VERSION`, `configure.sh`,
`gpu.Dockerfile`, `opendm/gpu.py`, `opendm/osfm.py`, `requirements.gpu.txt`,
`start-dev-env.sh`. PR text: "Adds support for GPU SIFT (OpenCL based). There's
also the option to enable GPU SIFT matching, however in my benchmarks it seems
to underperform the CPU based FLANN, so I'm not going to enable it. It can be
enabled by adding `matcher_type: SIFT_GPU` to the `config.yaml`."

The OpenSfM side lived in `opensfm/sift_gpu.py` (created by `eb4b8ff84`
"PoC GPU support", 2021-02-24; built on `ed7c8fb5f` "Rewrote changes from
cojosef96/gpu", 2021-02-09, whose files included `requirements.txt` — the
silx/pyopencl dependency). `aac21e23f` "Merged gpu" (2021-03-05) landed it.
`c90b0344a` (2021-12-16, "GPU code cleanup, pypopsift in, silx out") deleted
`sift_gpu.py` and changed `opensfm/src/CMakeLists.txt`, `features.py`,
`feature_loading.py`, `dataset.py`, `config.py`, `requirements.txt`.

Why it was dropped: ODM #1365 (2021-10-29) — `ROOT_SIFT_GPU` crashed on images
>11000 px wide (and, per rumenmitrev 2021-12-11, on 4000 px images with low
VRAM: "I had to resize-to 3000"). pierotofy 2021-11-19: "The workaround at the
moment is to use HAHOG"; 2021-11-22: "Plan is to replace the current GPU code
with https://github.com/uav4geo/pypopsift". The replacement traded portability
(OpenCL ran on Intel/AMD in principle) for speed and robustness, and nobody has
revisited that trade since.

### 6.2 Thread specifics worth keeping verbatim

- `rocm-docker/652` (2018): Fabio_Raimondo, 2018-05-24: "does anyone know
  rocm-docker container? Rocm is a hardware accelerated AMD Radeon (APU) API
  for linux." pierotofy, same day: "We currently don't leverage the GPU, but
  this has been brought up multiple times". Fabio_Raimondo, 2018-06-11: moved
  to NVIDIA.
- `odm-gpu-on-radeon/7622` (2021): machinefly installed both amdgpu-pro
  (OpenCL) and ROCm; Saijin_Naib asked "Can you do me a favor and install
  clinfo inside the ODM docker container and paste the output?"; the output
  showed a usable `CL_DEVICE_TYPE_GPU`; hblanken (Mac, Radeon Pro 570X):
  "Looks like it is searching for a nvidia-container hook". The OpenCL
  extractor was present in the image at the time; the `--gpu` launcher was the
  blocker.
- `gpu-acceleration-w-amd/11947` (2022-06): Saijin_Naib, 06-11: "Unless AMD
  gains support for CUDA, I would think it would be unlikely, unfortunately";
  06-22: "You will not be able to as the Radeon hardware/software stack does
  not support CUDA"; "I'm waiting for more news about Intel's SYCLomatic to
  see if it might be an option". Auto-closed after 30 days.
- `amd-radeon-support/16485` (2023-06-21): doletas: "I know that webodm only
  supports Nvidia cards, but why do these AMD and INTEL options appear in the
  .sh file?" Saijin_Naib: "We had OpenCL support before CUDA, hence the AMD
  and Intel targets." and "There might be a time where Intel's SYCLomatic can
  be used to run the CUDA code over Intel/AMD OpenCL/Vulkan".
- `help-requested-navigate-source-code-to-hipify-for-amd-gpu-support/17500`
  (2023-09-09): brewfalcon, "somewhat proficient in Python and C++", proposed
  using ROCm HIP utilities; Saijin_Naib (09-10) suggested SYCLomatic because
  "code should run equally well on any device (Intel/AMD/NVIDIA/other)" and
  pointed at SuperBuild dependencies (OpenSfM, OpenCV); brewfalcon: "I'll poke
  around, thanks!"; Saijin_Naib deferred to "Piero and others". Nothing
  further. (The CUDA code brewfalcon was looking for is in the OpenMVS fork
  and pypopsift/PopSift, neither of which is in the ODM repo itself — see
  docs 01 and 03.)
- `is-there-an-odm-gpu-docker-image-for-ubuntu-arm64/18686` (2023-12):
  oyzw0720 (Jetson AGX Orin): "I get 'no matching manifest for linux/arm64/v8
  in the manifest list entries'"; pierotofy: "We currently have no GPU support
  for ARM (but we'd welcome contributions)."; oyzw0720 (12-26): "I'll try to
  work on this."
- ODM #1685 (2023-08 → 2026-02): Dennis-Johnson (Tegra) could not build
  `gpu.Dockerfile` (libavcodec/OpenCV issues); zfb132 (2023-10-12) saw the same
  import error on amd64; grunlab (2025-03-08/12) built an arm64 GPU image on a
  Turing Pi RK1 and published CI steps at
  `gitlab.com/grunlab/opendronemap/-/blob/main/build/gitlab-ci.yml`;
  pierotofy: "Yes you need quite a bit of memory to build ODM 👍 thanks for
  posting your steps."; piaoyaoi (2026-02-03) reported the image gone.
- `gpu-enabled/19805` (2024-03): apdim (RTX 3090, 64 GB, Ryzen 7600X): "If I
  use dspsift-it is not using GPU / If I use sift-it is using GPU";
  Saijin_Naib: "In short, only SIFT is currently GPU-accelerated. Everything
  else will need to be re-written." This is the most common user confusion:
  ODM's default `--feature-type dspsift` (`opendm/config.py:233`) never uses
  the GPU.
- `cuda-or-opencl/20330` (2024-05-09): wmolevelt: "for me being on AMD -
  Radeon - OpenCL it is not possible to use WebODM. Is that correct?";
  pierotofy (19:48): "You can still use WebODM; you just will not get a boost
  in processing during certain parts of the computation."
- `why-webodm-not-use-my-gpu/14070` (2022-12): TOPOGRAFIA_SC (RTX 3060 Ti,
  20 MP P4P images): "I find it strange because in pix4D, 3Dsurvey and
  Metashape the memory of my GPU is perfectly adequate"; Saijin_Naib: "it
  isn't primarily the memory usage, but rather there are finite image
  dimensions that can be properly accelerated within CUDA, and if the input
  images don't fit, CUDA can't accelerate them." (See doc 03 §4 for what the
  limit actually is.)
- `using-specific-gpu-with-nodeodm/12983` (2022-09): NVIDIA multi-GPU via
  `NVIDIA_VISIBLE_DEVICES` and `nvidia-docker2`; Johnny5 (09-27): "Success!!!
  Two separate tasks using separate GPU cores..."
- ODM #1240 (2021-03-08): pierotofy: "ODM does not support M1 chips at the
  moment. It's something we'd like to add at some point in the future."
  (2021-04-14) "Still not supported." — the thread is about running the x86
  Docker image under QEMU, not about the GPU.
- `odm-on-macos/20064` (2024-04-19): pierotofy: "It's been a while since I've
  tested the native macOS build, so minor issues might be present." No
  GPU/Metal mention anywhere in the Apple-Silicon threads found.

### 6.3 `gpu.Dockerfile` commit history (GitHub, `path=gpu.Dockerfile`)

```
3b2a60737 2021-02-09 Add gpu dockerfile
1f2d93fd0 2021-02-10 Dockerfile gpu changes
5e3737a3f 2021-03-05 GitHub GPU workflow, use SIFT_GPU when GPUs are available
409c70f03 2021-03-15 Fix GPU dockerfile
539871bac 2021-09-02 Support for Ubuntu 21.04
15275d9aa 2021-10-22 Update start-dev-env.sh, OpenMVS
7fd8fb4c8 2021-11-15 matcher_type freedom
4eba4cf31 2021-11-17 Smoke test in docker build
0ca7d1cef 2021-12-16 Pypopsift support
c8033cc26 2022-01-12 Better smoke test
d9cf169cb 2022-01-12 Fix dockerfile
d3ae0adda 2022-02-17 Update old opensfm path
8410186b6 2023-01-03 Don't let PDAL resolve canonical paths while searching for drivers
be1fec2bd 2023-07-15 update cuda ver
d04e30109 2024-07-03 test fix
709afd16f 2024-07-04 working gpu docker build
8ce00e01a 2024-07-05 fixed gpu docker build (missing ffmeg)
51a7470eb 2024-07-16 Add libtbb2 to gpu.Dockerfile
36897fc0e 2024-11-19 Cannot initialize CUDA even if nvidia-smi detected
58aba0241 2025-07-24 Minor updates for GPU (#1895)
ff0004793 2025-10-28 24.04 (#1904) (#1942)
44e3ff6e3 2026-03-04 Fixing CI - Compiling GDAL for non-windows (#1995)
ddbd64202 2026-05-11 Clean up dockerfiles and devcontainers
811cb53df 2026-05-11 Working Docker and devcontainer
132ff7dbb 2026-05-19 feat: Remove legacy build
d94d05ebe 2026-06-15 Cleanup docker images to reduce size
41017632e 2026-06-17 Switch to python scripts
64eaaa331 2026-06-17 minor cleanup
321184a5f 2026-06-23 Fix gpu image deps
0803c9bf1 2026-06-23 Installer and PR cleanup
291a0ce96 2026-07-15 Add OCI image labels carrying git commit SHA to Docker images
8125b4b6e 2026-07-15 Add static OCI descriptive labels to Docker images
e7b553abc 2026-08-13 Ship contrib tools in the runtime images
95db3c54d 2026-08-19 Mock the CUDA virtual package during the GPU docker build
cb4ed84be 2026-08-19 Give the GPU smoke test a driver library to load
c18c27571 2026-08-24 Simplify all the docker build workflows
```

Every entry is NVIDIA/CUDA housekeeping; the base image has moved
`nvidia/cuda:10.x` → `11.x` (2023-07-15) → `12.9.1-runtime-ubuntu24.04`
(current `gpu.Dockerfile:34`).

### 6.4 Upstream OpenMVS requests for non-CUDA GPUs

- #206 (2017-04-20, PolarNick239): asked why kernels were PTX-only and
  offered to add nvcc/CMake compilation so an OpenCL `SceneRefineOpenCL.cpp`
  could be written. Closed without a visible maintainer reply.
- #1076 (2023-10-25, zoey9628): "I want to run it not in nvidia drivers, but
  in other drivers and I wanted to speed things up." Offered to contribute.
  No reply visible; closed.
- #1085 (2023-11-28, datasciritwik): "Can we use GPU other then CUDA of
  processing?" for densify and mesh. No reply visible; closed.
- PR #1279 (2026-06, leNeo): the first accepted non-CUDA backend. Maintainer
  review (2026-06-22): "great contribution, this PR is very valuable; pls help
  with the last part in order to be able to merge it: full testing on various
  datasets and configurations and compare to the existing CUDA
  implementation". The author added a Tanks-and-Temples harness
  (`eval/tnt/` in `leNeo/openMVS-metal`), the maintainer ran the CUDA column
  himself (2026-06-24), a geometric-consistency discrepancy was fixed, a
  latent worker-shutdown race in the shared `DenseReconstructionEstimate` was
  found and fixed (`00ace26`), and on 2026-06-26 the maintainer unified backend
  selection under `--gpu-device` (`75ffb42`) and merged. The PR body notes it
  was "Generated with Claude Code".

### 6.5 Adjacent projects (not ODM, but relevant precedent)

- **VulkanSift** (maelaubert, MIT, C): SIFT detection and matching in Vulkan
  compute; README lists tests on NVIDIA RTX 4070 / 2060 / GTX 960M, Qualcomm
  Adreno 615 (Android) and Apple M2 (macOS, i.e. via MoltenVK). No pybind
  wrapper; would need a `pypopsift`-shaped shim (doc 02 §7a).
- **Upstream OpenMVS `ports/siftgpu`** (CUDA SiftGPU, behind
  `OpenMVS_USE_SIFTGPU`): CUDA-only, irrelevant to portability.
- No HIP/OpenCL/Metal fork of PopSift was found (doc 03 §5).

## Sources

Forum (community.opendronemap.org): `/t/rocm-docker/652`,
`/t/odm-gpu-on-radeon/7622`, `/t/building-odm-on-the-apple-m1-chipset/8334`,
`/t/gpu-acceleration-w-amd/11947`, `/t/using-specific-gpu-with-nodeodm/12983`,
`/t/why-webodm-not-use-my-gpu/14070`, `/t/amd-radeon-support/16485`,
`/t/help-requested-navigate-source-code-to-hipify-for-amd-gpu-support/17500`,
`/t/is-there-an-odm-gpu-docker-image-for-ubuntu-arm64/18686`,
`/t/gpu-enabled/19805`, `/t/odm-on-macos/20064`, `/t/cuda-or-opencl/20330`
(also surfaced but not fetched: `/t/running-webodm-on-mac-m1/6799`,
`/t/apple-m1-performance-prediction/8745`, `/t/mac-m1-update-for-2022/10017`,
`/t/newby-question-m1-macbook-pro-16gb-1tb-best-setup-settings/14736`,
`/t/gpu-setup-assistance/10452`, `/t/gpu-accelerated-feature-mapping/19571`).

GitHub: OpenDroneMap/ODM issues #489, #1240, #1365, #1685, #1993; PRs #1239,
#1813, #1895, #2003, #2037, #2074; commits on `gpu.Dockerfile`
(`repos/OpenDroneMap/ODM/commits?path=gpu.Dockerfile`); OpenDroneMap/OpenSfM
issue #29 and commits `ed7c8fb5f`, `eb4b8ff84`, `aac21e23f`, `c90b0344a`;
OpenDroneMap/pypopsift PR #15, issue #7; cdcseacave/openMVS issues #206,
#1076, #1085, PR #1279, releases v2.1.0-v2.4.0; maelaubert/VulkanSift README.

Local: `/Users/avneh/Code/ADM/ODM/README.md:95-135`, `gpu.Dockerfile`,
`pixi.toml`, `SuperBuild/CMakeLists.txt:62-67`, `opendm/gpu.py`,
`SuperBuild/cmake/External-PyPopsift.cmake`.

Docs: https://docs.opendronemap.org/installation/ (GPU sentence quoted above);
https://docs.opendronemap.org/gpu/ → 404.
