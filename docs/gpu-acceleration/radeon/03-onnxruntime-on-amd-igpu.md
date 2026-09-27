# ODM's ONNX models (sky / background removal) on an AMD iGPU

ODM runs two small segmentation models through ONNX Runtime, only when the user asks:
`--sky-removal` (`opendm/skyremoval/skyfilter.py`, used in `stages/dataset.py`) and
`--bg-removal` (`opendm/bgfilter.py`). Both files pick the provider identically:

```python
provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
...
self.session = ort.InferenceSession(self.model, providers=[provider])
```

(`skyfilter.py` L15/L30, `bgfilter.py` L16/L29). ODM pins conda-forge `onnxruntime==1.22.2`
(CPU build) and, in the `gpu` feature, the `*cuda*` build. This document lists what AMD iGPU
providers exist in 2026, what they need, and what the two files would have to do differently.
Nothing here has been run.

---

## 1. Provider inventory

### 1.1 ROCm Execution Provider (Linux) — end of life

* "ROCm Execution Provider has been removed since 1.23 release" and "ROCm 7.0 is the last
  officially AMD supported distribution of this provider"; AMD recommends MIGraphX.
  Last wheels: `onnxruntime_rocm-1.22.1` for cp310/cp312 at
  `https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/` (`manylinux_2_28_x86_64`).
  Provider string `ROCMExecutionProvider`; options `device_id`, `tunable_op_enable`,
  `gpu_mem_limit`. [ROCm EP](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html)
* iGPU: relies on rocBLAS/MIOpen kernels → `gfx1103` needs `HSA_OVERRIDE_GFX_VERSION=11.0.2`
  and still lacks MIOpen convolution DBs (doc 00 §1.2). Do not build on this.

### 1.2 MIGraphX Execution Provider (Linux) — AMD's current answer

* Wheels: `onnxruntime_migraphx-1.23.2` for cp310/cp312 at
  `https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/`; install with
  `pip3 install onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/`.
  Provider string `MIGraphXExecutionProvider`; options `device_id`, `migraphx_fp16_enable`,
  `migraphx_int8_enable`, `migraphx_exhaustive_tune` (save/load compiled model deprecated
  in 6.4). Falls back to CPU for unsupported nodes.
  [MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html),
  [AMD install guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-onnx.html)
* iGPU status:
  * `gfx1150`/`gfx1151`: listed among MIGraphX-supported targets (gfx1030, gfx1100–1102,
    gfx1150, gfx1151, gfx1200, gfx1201) — expected to work natively.
    [llm-tracker AMD GPUs](https://llm-tracker.info/howto/AMD-GPUs)
  * `gfx1103`: MIGraphX compilation **SIGSEGVs** on a Phoenix iGPU with ROCm 7.2 / ORT
    1.24.2 (immich's ROCm image); `HSA_OVERRIDE_GFX_VERSION` did not help; users were told to
    use the CPU image. Root cause is the missing gfx1103 rocBLAS/CK kernels.
    [immich#27387](https://github.com/immich-app/immich/issues/27387)
  * `gfx90c`: no reports; assume no.
* Not available on Windows, not available in WSL2 ("MIGraphX ... aren't currently supported").
  [WSL how-to](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html)
* Version coupling: the wheel is built against a specific ROCm (1.23.2 ↔ ROCm 7.2/7.2.1);
  the system `/opt/rocm` must match. This is a different Python distribution from conda-forge's
  `onnxruntime`; both install the `onnxruntime` package, so a pixi env must carry exactly one.

### 1.3 DirectML Execution Provider (Windows) — works on every AMD iGPU

* `pip install onnxruntime-directml` — latest 1.24.4 (2026-03-17), `win_amd64`, Python
  3.11–3.14. Provider string `DmlExecutionProvider`. Requirements: Windows 10 1903+, any
  DirectX 12 GPU ("AMD GCN 1st Gen and above"), so Vega 8, 780M and 890M all qualify with the
  stock Adrenalin driver — no ROCm, no SDK.
  [PyPI](https://pypi.org/project/onnxruntime-directml/), [DML EP](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
* Mandatory session settings: `sess_options.enable_mem_pattern = False` and
  `sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL`; `device_id` selects the
  adapter (0 = default, which on a laptop with only an iGPU is the iGPU). Only one thread may
  call `Run()` per session. Best performance with static shapes (ODM's models run fixed-size
  inputs after resize, so fine).
* Status: "in sustained engineering ... new feature development has moved to WinML"; the
  DirectML repo banner says maintenance mode; a Feb 2025 issue asking "Is DML being
  deprecated?" got no official reply, yet wheels keep shipping. Adequate for ODM's two models
  through at least the ORT 1.2x line; not a base for new features.
  [microsoft/DirectML](https://github.com/microsoft/DirectML), [ORT#23783](https://github.com/microsoft/onnxruntime/issues/23783)

### 1.4 WebGPU Execution Provider (Linux + Windows) — cross-vendor, young

* Native WebGPU EP built on Dawn: Windows D3D12/Vulkan, Linux Vulkan, macOS Metal.
  `pip install onnxruntime onnxruntime-ep-webgpu` (plugin EP), provider string
  `WebGpuExecutionProvider`, options `preferredLayout`, `deviceId`, `powerPreference`,
  `forceCpuNodeNames`, `enableGraphCapture`.
  [WebGPU EP](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html)
* Runs on RADV (Linux) and Adrenalin (Windows) for all four iGPUs. Operator coverage and
  fp32 accuracy for ODM's specific models are unverified; treat as the portable fallback and
  test the output masks against the CPU provider before enabling by default.

### 1.5 Not applicable

OpenVINO EP (Intel only), Vitis AI EP (Xilinx/Ryzen AI **NPU**, not the iGPU — a separate
topic: the XDNA NPU in Phoenix/Strix could run these models via Ryzen AI Software/Vitis EP,
Windows-first; out of scope), CUDA EP, TensorRT EP.

## 2. Package availability inside pixi

* conda-forge `onnxruntime-feedstock` builds **CPU and CUDA 12.x variants only**; no ROCm,
  MIGraphX, DirectML or WebGPU variants.
  [onnxruntime-feedstock](https://github.com/conda-forge/onnxruntime-feedstock)
* Therefore an `amd` feature must pull the AMD/Microsoft wheels through
  `[feature.amd.target.<platform>.pypi-dependencies]` (doc 04 §A.3) and exclude the
  conda-forge `onnxruntime` from that solve group. The AMD MIGraphX wheel needs the
  `find-links`/index URL above; DirectML is on PyPI.
* The ODM pin (`onnxruntime==1.22.2`) does not match any AMD wheel (MIGraphX is 1.23.2 for
  ROCm 7.2.1, ROCm EP 1.22.1 for 7.0). ODM's models are opset-conservative; the version drift
  is unlikely to matter, but it means the `amd` env runs a different ORT than `cpu`/`gpu`.

## 3. What the ODM code would need (description only, no patch)

Both `skyfilter.py` and `bgfilter.py` would replace the two-way choice with a preference
list filtered by availability and by `ODM_NO_GPU`/`--no-gpu`, e.g.

```
preferred = ["CUDAExecutionProvider",
             "MIGraphXExecutionProvider",   # Linux, ROCm 7.2 wheels
             "ROCMExecutionProvider",       # legacy wheels only
             "DmlExecutionProvider",        # Windows
             "WebGpuExecutionProvider",     # plugin EP, both OSes
             "CPUExecutionProvider"]
providers = [p for p in preferred if p in ort.get_available_providers()]
```

with two provider-specific details:

1. **DirectML** requires `SessionOptions` with `enable_mem_pattern = False` and
   `execution_mode = ORT_SEQUENTIAL`; pass `providers=[("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]`.
2. **MIGraphX / WebGPU** as plugin EPs may need registration (`ort.register_execution_provider_library`
   for plugin EPs in ORT ≥ 1.23) before they show up in `get_available_providers()`; the
   WebGPU package documents its own registration call.

Also: today the module-level `provider` variable is evaluated at import time, before
`args.no_gpu` is known; `has_gpu()` in `opendm/gpu.py` is not consulted here at all. Moving the
choice into `__init__` and passing the ODM args (or checking `ODM_NO_GPU`) would make the
`--no-gpu` flag cover the ONNX path too. The fallback-to-CPU-on-failure pattern that
`stages/openmvs.py` uses for OpenMVS would be worth copying: wrap `InferenceSession` creation
and the first `run()` in a try/except that rebuilds the session on `CPUExecutionProvider`.

## 4. Expectations

* These models run once per image at reduced resolution, so even on CPU the cost is seconds
  per image; on a 100-image set that is a few minutes out of ~30 (doc 05 §6). An iGPU
  provider might make it 2–5× faster (unmeasured; DirectML on small U-Nets typically lands
  there), i.e. a minute or two saved. **It is the cheapest AMD acceleration to ship because it
  needs no C++ build, but it is also the smallest win.**
* Memory: MIGraphX/DirectML allocate GPU (shared) memory per session; on a 16 GB laptop with
  ODM's own workers this is another few hundred MB. Harmless.
* Accuracy: fp32 everywhere by default; do not enable `migraphx_fp16_enable` without checking
  mask quality.
* Community precedent that DirectML on an AMD iGPU is real: the gfx1103 rocBLAS community
  project reports ROCm "often 2–3 times faster than DirectML" for LLMs — DirectML is the
  slower but safer baseline. [ROCmLibs](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU)

## 5. Recommendation

| OS / iGPU | Provider to document | Confidence |
|---|---|---|
| Windows 11, any AMD iGPU | `onnxruntime-directml` → `DmlExecutionProvider` | high (stock driver, on PyPI) |
| Linux, gfx1150 / gfx1151 | `onnxruntime-migraphx` (ROCm 7.2.1) → `MIGraphXExecutionProvider` | medium (official targets; needs system ROCm) |
| Linux, gfx1103 | CPU, or `onnxruntime-ep-webgpu` on RADV | low-medium (MIGraphX crashes; WebGPU untested on these models) |
| Linux, gfx90c | CPU | — |
| WSL2 | CPU (MIGraphX unsupported in WSL) or WebGPU | low |

## 6. The models themselves (from the ODM sources)

* **Sky filter** (`opendm/skyremoval/skyfilter.py`): `SkyFilter(model, width=384, height=384)`;
  input is the RGB image resized to 384×384 with `INTER_AREA`, scaled to `[0,1]` float32,
  fed as `session.get_inputs()[0]`; the mask is resized back with `INTER_LANCZOS4` and
  written as uint8. Applied in `stages/dataset.py` only to images without an existing mask
  that are not nadir (or lack orientation / are fisheye) and have no spaces in the filename.
* **Background filter** (`opendm/bgfilter.py`, "based on https://github.com/danielgatis/rembg"):
  input resized to 320×320, ImageNet-normalised (mean 0.485/0.456/0.406, std 0.229/0.224/0.225),
  float32; output thresholded to uint8 and resized with `INTER_LANCZOS4`. Same file-name rules.
* Both are single-input, single-output, fixed-shape fp32 CNNs — the friendliest possible case
  for DirectML (static shapes) and MIGraphX (compile once per session, no dynamic dims).
  Session creation happens once per ODM run, so MIGraphX's compile-time cost is paid once.
* The model weights are downloaded on first use into the ODM tree; they are ordinary ONNX
  files, so `onnxruntime`'s `check_model`/`onnx.checker` and a CPU-vs-GPU mask diff are enough
  to validate a new provider.

## 7. Verification procedure for a new provider (describe-only)

1. In the `amd` env: `python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"`
   — expect `DmlExecutionProvider` (Windows) or `MIGraphXExecutionProvider` (Linux gfx1150/1151).
2. Run `SkyFilter` on ~10 oblique images once with `providers=["CPUExecutionProvider"]` and
   once with the GPU provider; compare masks (`np.abs(a-b).mean()` well under 1/255, IoU > 0.99).
3. Time both runs; if the GPU path is not at least ~2× on the per-image loop, keep CPU as the
   default and offer the provider as opt-in via an environment variable — the stage is a
   minor fraction of the pipeline (§4).
4. Kill switch: honour `ODM_NO_GPU`/`--no-gpu` and fall back to CPU on any
   `InferenceSession` or first-`run()` exception, mirroring `stages/openmvs.py`.

## Sources

- ODM sources: `opendm/skyremoval/skyfilter.py`, `opendm/bgfilter.py`, `stages/dataset.py`, `opendm/gpu.py`, `pixi.toml`
- https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html
- https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-onnx.html
- https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ , https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html , https://pypi.org/project/onnxruntime-directml/
- https://github.com/microsoft/DirectML , https://github.com/microsoft/onnxruntime/issues/23783
- https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html , https://github.com/microsoft/onnxruntime/issues/22077
- https://github.com/immich-app/immich/issues/27387
- https://llm-tracker.info/howto/AMD-GPUs
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html
- https://github.com/conda-forge/onnxruntime-feedstock
- https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU
