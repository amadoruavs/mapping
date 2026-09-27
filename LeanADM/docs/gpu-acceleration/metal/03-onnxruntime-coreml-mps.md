# Running ODM's ONNX models (sky removal, background removal) on Apple GPU / Neural Engine

Research date: 2026-09-13. Scope: `opendm/skyremoval/skyfilter.py`, `opendm/bgfilter.py`,
`stages/dataset.py`, and the `onnxruntime==1.22.2` pin in `pixi.toml`.

---

## 1. What ODM does today

Both filters (read locally) choose a provider at import time:

```python
provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"
...
self.session = ort.InferenceSession(self.model, providers=[provider])
```

* **SkyFilter** (`--sky-removal`): model from
  `https://github.com/OpenDroneMap/SkyRemoval/releases/download/v1.0.5/model.zip`
  ([OpenDroneMap/SkyRemoval](https://github.com/OpenDroneMap/SkyRemoval), AGPL-3.0); input
  resized to **384×384** (`INTER_AREA`), `float32` NCHW without normalisation, single output
  mask, resized back with `INTER_LANCZOS4`, then a **guided filter** (`guidedfilter.py`,
  radius 20, eps 0.01) on the CPU. One session shared under a `Lock` (`mutex`) across the
  image-mask thread pool. The network architecture is not documented in the repo README
  **[uncertain: exact architecture; a small encoder–decoder segmentation net is implied by
  the fixed 384×384 input]**.
* **BgFilter** (`--bg-removal`): U²-Net (`u2net.zip` from ODM's v2.9.0 release; code "based on
  rembg"), input **320×320**, ImageNet mean/std normalisation, output `pred[:,0,:,:]`.
* On the Mac today `get_available_providers()` from the conda-forge build in `pixi.lock`
  (`onnxruntime-1.22.2-py312h4d4766a_1_cpu`) is expected to be `['AzureExecutionProvider',
  'CPUExecutionProvider']` — **[uncertain: not executed inside the pixi env; see §3]** — so
  both models run on 12 CPU cores through onnxruntime's MLAS/NEON kernels.

Per-image cost is small (a 384² or 320² CNN is tens of ms on the CPU), but ODM runs it for
every input photo before feature extraction, serialised by the mutex, so on a 2 000-image
job the sky step is minutes, not hours. The GPU/ANE win is real but bounded; it also frees
the CPU for the parallel undistort/resize work.

---

## 2. onnxruntime CoreML Execution Provider

Docs: [onnxruntime.ai CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
(and [gh-pages source](https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/execution-providers/CoreML-ExecutionProvider.md)).

* Availability: macOS ≥ 10.15; "Official Python wheels include CoreML EP" on macOS
  (`pip install onnxruntime`), as do the official macOS arm64 C/C++/Node packages. Local
  builds need `./build.sh --use_coreml`. Check with `'CoreMLExecutionProvider' in
  ort.get_available_providers()`. **[verified locally]** the system `python3` (3.13, pip
  onnxruntime 1.22.1) on the target Mac reports
  `['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']`.
* Provider options (new-style API, `providers=[("CoreMLExecutionProvider", {...})]`):

| Option | Default | Values / meaning |
|---|---|---|
| `ModelFormat` | `NeuralNetwork` | `MLProgram` (Core ML 5+, macOS 12+) — **use this**; explicit tensor types, no silent fp16 |
| `MLComputeUnits` | `ALL` | `CPUOnly`, `CPUAndNeuralEngine`, `CPUAndGPU`, `ALL` |
| `RequireStaticInputShapes` | `0` | `1` = only take subgraphs with static shapes (dynamic shapes hurt perf) |
| `EnableOnSubgraphs` | `0` | allow CoreML inside Loop/Scan/If bodies |
| `SpecializationStrategy` | `Default` | `FastPrediction` |
| `ProfileComputePlan` | `0` | `1` = log which unit (CPU/GPU/ANE) ran each op |
| `AllowLowPrecisionAccumulationOnGPU` | `0` | `1` = fp16 accumulation on GPU |
| `ModelCacheDirectory` | off | path; caches the compiled CoreML model (hash of graph/metadata); user must invalidate stale caches |

* Op coverage (MLProgram list in the docs): `Conv` (1D/2D only), `ConvTranspose`,
  `BatchNormalization`, `InstanceNorm`, `GroupNorm`, `LayerNorm`, `MaxPool` (2D),
  `GlobalAveragePool/GlobalMaxPool`, `Resize`, `Pad` (limited modes), `Concat`, `Slice`,
  `Split`, `Gather`, `Gemm`, `MatMul`, `Softmax`, `Sigmoid`, `Relu/LeakyRelu/PRelu/Gelu`,
  `Tanh`, `Erf`, `GridSample`, `Reshape/Transpose/Squeeze/Unsqueeze/Flatten`, `Cast`, `Clip`,
  arithmetic and reductions. Unsupported nodes fall back to CPU per partition; many
  partitions = many CPU↔CoreML copies = slower than pure CPU.
  A U²-Net (conv/BN/ReLU/maxpool/bilinear upsample/concat/sigmoid) and a small segmentation
  encoder–decoder are squarely inside this list; the only likely hazards are `Resize` mode
  attributes (`align_corners`/`half_pixel`) and any 5-D or `Pad` reflect modes — run with
  `ProfileComputePlan=1` or `onnxruntime` verbose logging to count partitions.
* Precision pitfall: with the default `NeuralNetwork` format Core ML runs fp16 on the GPU and
  can flip borderline predictions ("values between ~0.4998779 and 0.5 round up ... causing some
  negative predictions to become positive") — fixed by `ModelFormat=MLProgram`
  ([ym2132 write-up](https://ym2132.github.io/ONNX_MLProgram_NN_exploration)). For a sky
  *mask* that then goes through a guided filter, fp16 is almost certainly fine, but MLProgram
  should still be the default for reproducibility against the Linux/CPU results.
* Performance shape (M2 MacBook Pro numbers, onnxruntime + CoreML, [Xybrid](https://www.xybrid.ai/blog/onnx-runtime-6x-faster-apple-silicon-coreml)):
  MobileNet v2 12.4 ms CPU → 1.8 ms CoreML (6.8×); MNIST 0.15 ms → 0.8 ms (dispatch overhead
  dominates tiny models); a dynamic-shape transformer ~1×. "Cold-start compilation penalty" of
  1–5 s on first session creation; recommendation: warm up once, use static shapes.
  Fixed 384×384 / 320×320 inputs are the best case.
* Expectation for ODM on M4 Pro **[estimate, not measured]**: U²-Net-full at 320² is ~40 GFLOP
  per image; the 16-core M4 Pro GPU (≈ 8–9 TFLOPS fp32 class **[uncertain]**) or the ANE
  should bring it to the ~10–30 ms range vs ~100–300 ms on CPU; the 384² sky model is smaller
  and will land in single-digit ms, at which point the Python pre/post-processing
  (`cv2.resize`, Lanczos upscale, guided filter at full resolution) dominates. So: expect the
  *sky stage* to speed up 2–4×, not 10×, unless the guided filter is also moved (vImage or a
  Metal box-filter — trivial kernels) or the mutex is dropped in favour of per-thread sessions.

---

## 3. Packaging: which onnxruntime build has the CoreML EP?

* PyPI wheels: yes (docs; verified on the system Python).
* conda-forge `onnxruntime` osx-arm64: CoreML was enabled by
  [PR #173](https://github.com/conda-forge/onnxruntime-feedstock/pulls) "Enable CoreML provider
  for osx-arm64 builds" (opened 2026-03-05, closed 2026-05-17) and shipped in
  [PR #182](https://github.com/conda-forge/onnxruntime-feedstock/pull/182) "1.26.0 with CoreML
  execution provider for osx" (merged 2026-05-16). The current recipe (1.30.0) tests
  `assert 'CoreMLExecutionProvider' in ort.get_available_providers()  # [osx and arm64]`
  and sets `onnxruntime_USE_COREML=ON` for osx-arm64 only
  ([feedstock meta.yaml](https://github.com/conda-forge/onnxruntime-feedstock/blob/main/recipe/meta.yaml)).
  **ODM's pin `onnxruntime==1.22.2` predates this**; the locked build string `_1_cpu` (build
  number 1) is therefore expected to lack CoreML **[uncertain: confirm with
  `pixi run python -c "import onnxruntime as o; print(o.get_available_providers())"`]**.
* Options for ODM: (a) bump the pin to `>=1.26` conda-forge on osx-arm64 (a `[target.osx-arm64.dependencies]`
  override; the Linux/Windows `gpu` feature already overrides the build string with `*cuda*`),
  (b) take the PyPI wheel on macOS via `[target.osx-arm64.pypi-dependencies]`, or
  (c) keep 1.22.2 on CPU. Note the "cuda" feature's `onnxruntime = { version = "==1.22.2",
  build = "*cuda*" }` pattern in `pixi.toml` is the template for a future `metal` feature.
  Third-party wheels such as [xaviviro/onnxruntime-coreml](https://github.com/xaviviro/onnxruntime-coreml)
  exist but are unnecessary now that official wheels include the EP
  ([microsoft/onnxruntime #10367](https://github.com/microsoft/onnxruntime/issues/10367)).

---

## 4. Code change ODM would need (description only — not implemented)

In both `skyfilter.py` and `bgfilter.py`, replace the two-way provider choice with an ordered
provider list, e.g.

```
available = ort.get_available_providers()
providers = []
if "CUDAExecutionProvider" in available:  providers.append("CUDAExecutionProvider")
if "CoreMLExecutionProvider" in available:
    providers.append(("CoreMLExecutionProvider", {
        "ModelFormat": "MLProgram", "MLComputeUnits": "ALL",
        "RequireStaticInputShapes": "1", "ModelCacheDirectory": <ODM cache dir>}))
providers.append("CPUExecutionProvider")
```

and honour `--no-gpu` / `ODM_NO_GPU` (today the filters ignore `opendm.gpu` entirely — they
only look at onnxruntime's provider list). Other points to design:

* The `mutex` serialises inference; with CoreML the session is thread-safe for `run()` and the
  ANE/GPU are shared anyway, so keeping the lock is acceptable, but per-worker sessions would
  overlap CPU pre/post-processing with device inference.
* Warm-up: run one dummy inference at `load_model()` so the 1–5 s CoreML compile is not
  attributed to the first image (and is cached by `ModelCacheDirectory`).
* Fallback: wrap session creation in `try/except` and retry with CPU (CoreML EP can fail on
  op-set issues at session creation, not at `run()`), mirroring `stages/openmvs.py`'s
  GPU→CPU retry.
* Logging: `log.ODM_INFO(' ?> Using provider %s' % provider)` should print the resolved list
  (`session.get_providers()`), which also shows whether CoreML actually took the graph.
* `MLComputeUnits`: `ALL` lets Core ML pick the ANE for the CNN parts; `CPUAndGPU` is the
  safer choice if ANE-specific numeric differences show up in mask comparisons
  **[uncertain: no measurement]**.

---

## 5. Alternatives

| Route | Pros | Cons |
|---|---|---|
| onnxruntime CoreML EP (§2) | no new dependency if onnxruntime ≥ 1.26 on osx-arm64; same `.onnx` files; runtime fallback | partition boundaries; first-run compile; conda pin bump |
| coremltools conversion (`ct.convert(onnx_model, convert_to="mlprogram")`) + `coremltools`/PyObjC to run `.mlpackage` | full Core ML control (fp16/fp32, compute units, ANE profiling in Xcode) | second model artefact to ship per platform; new runtime path in Python; ONNX→Core ML conversion in coremltools is via the ONNX frontend which has been deprecated in favour of PyTorch/TF frontends **[uncertain: current coremltools ONNX support status]** |
| PyTorch MPS | trivial if the models exist in PyTorch (rembg's U²-Net does) | +torch dependency; MPS fp32 only, no fp64; op fallbacks; ODM ships ONNX not `.pt` |
| Metal 4 `MTL4MachineLearningCommandEncoder` | run the CoreML package on the GPU timeline next to Metal kernels (e.g. fuse mask + guided filter) | Objective-C++ code path; macOS 26+ only; overkill for this stage ([WWDC25 262](https://developer.apple.com/videos/play/wwdc2025/262/)) |
| Keep CPU | zero work | leaves 2–4× on the table for a stage that is already short |

**Recommendation:** enable the CoreML EP behind the existing `has_gpu()`-style switch once the
onnxruntime pin is bumped for osx-arm64; measure mask parity (IoU vs CPU) on a sample of ODM
test images; move the guided filter to vImage/Metal only if profiling shows it dominating.

---

## Sources

- ODM: `/Users/avneh/Code/ADM/ODM/opendm/skyremoval/skyfilter.py`, `/Users/avneh/Code/ADM/ODM/opendm/bgfilter.py`, `/Users/avneh/Code/ADM/ODM/stages/dataset.py`, `/Users/avneh/Code/ADM/ODM/pixi.toml`, `/Users/avneh/Code/ADM/ODM/pixi.lock` (read locally)
- OpenDroneMap/SkyRemoval: https://github.com/OpenDroneMap/SkyRemoval
- onnxruntime CoreML EP docs: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html ; source: https://github.com/microsoft/onnxruntime/blob/gh-pages/docs/execution-providers/CoreML-ExecutionProvider.md ; provider factory header: https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h
- microsoft/onnxruntime #10367 (CoreML EP in macOS builds): https://github.com/microsoft/onnxruntime/issues/10367
- conda-forge onnxruntime-feedstock: https://github.com/conda-forge/onnxruntime-feedstock ; meta.yaml: https://github.com/conda-forge/onnxruntime-feedstock/blob/main/recipe/meta.yaml ; PR #182: https://github.com/conda-forge/onnxruntime-feedstock/pull/182 ; PR #206 (megabuilds): https://github.com/conda-forge/onnxruntime-feedstock/pull/206 ; issue/PR search: https://api.github.com/search/issues?q=repo:conda-forge/onnxruntime-feedstock+coreml
- xaviviro/onnxruntime-coreml wheels: https://github.com/xaviviro/onnxruntime-coreml
- Xybrid, "How We Made ONNX Runtime 6.8x Faster on Apple Silicon with CoreML": https://www.xybrid.ai/blog/onnx-runtime-6x-faster-apple-silicon-coreml
- ym2132, "ONNX Runtime & CoreML May Silently Convert Your Model to FP16": https://ym2132.github.io/ONNX_MLProgram_NN_exploration
- ort (pyke) EP overview: https://ort.pyke.io/perf/execution-providers
- U²-Net to Core ML (MLBoy): https://rockyshikoku.medium.com/u2net-to-coreml-machine-learning-segmentation-on-iphone-eac0c721d67b
- PyTorch MPS notes: https://docs.pytorch.org/docs/2.13/notes/mps.html
- Apple WWDC25 262 (ML command encoder): https://developer.apple.com/videos/play/wwdc2025/262/
