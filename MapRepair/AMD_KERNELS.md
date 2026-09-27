# AMD kernel and experimental hybrid update — September 16, 2026

This bundle contains custom HIP and XDNA2 kernel sources, CPU fallbacks, target
build scripts and device tests. It is **not a completed GPU/NPU port of ODM**.
Feature matching, bundle adjustment, dense PatchMatch and meshing still run on
the CPU. The under-five-minute target for 250 4K images is **not achieved**.
Fedora installation and actual HIP/XDNA2 execution cannot be verified on this Mac.

## Implemented paths

| Path | Work | Default |
|---|---|---|
| Native HIP | Exact packed BGRA compositing, 256 threads/block, two asynchronous streams with bounded reusable pinned staging buffers | Linux auto selection after exact parity and a transfer-inclusive speed check |
| HIP FSR | Existing tiled FP32 FSR1 EASU/RCAS display upscale, gfx1150 build | CPU fallback when device/parity checks fail |
| HIP flat classifier | One workgroup per 32×32 depth tile, parallel confidence/plane checks | Experimental hybrid only; every mask checked against CPU |
| XDNA2 | New 16-lane integer vector compositing, IRON parallel tile workers, two DMA channels per column | Experimental, explicitly enabled after installation and device checks |
| OpenMVS hybrid | Skip middle full-resolution photometric refinement inside exceptionally flat, confident tiles | **Off**, enable with `ODM_HYBRID=1` |

The NPU kernel operates on map export pixels, not SfM/depth inference. It cannot
provide a large reconstruction speedup. GPU/NPU selection is based on actual
device execution; installing ROCm or detecting Linux alone is insufficient.
On short transfers the CPU may be faster and auto mode then keeps CPU execution.
No reduced-precision model, quantization, fake detail or approximate integer math.

## Hybrid behavior and quality limits

The hybrid is a conservative adaptive Ultra prototype, **not WebODM's sparse
fast-orthophoto surface pasted into buildings**. At full depth resolution it
tests every sample of a 32×32 tile against an inverse-depth plane and NCC cost
≤0.03, with relative inverse-depth residual ≤0.00005. It requires at least four
available views and a complete 3×3 neighborhood of accepted tiles. Border,
rough-surface, hole, nonfinite and uncertain tiles retain full processing.
The first and last photometric passes, all coarse levels, both geometric passes,
filtering, meshing and original-image texture/seam processing remain intact.

These gates are heuristics. A wrong smooth depth plane can still pass; small
raised objects or bad alignment can defeat them. Skipping a refinement pass can
change both geometry and final pixels. Neither surveyed accuracy nor end-to-end
pixel parity has been established. Default Ultra therefore retains all passes.
Google Maps screenshots are visual references, not 4K multiview ground truth.
FSR changes display sampling/sharpness, not measured ground detail.

## Install and verify on Fedora

Stop the apps, extract the new archives over their existing parent directory,
then use the app's normal installer. Reuse existing paths to retain dependencies.
The native ODM update needs a rebuild; first installation requires downloads.
Select a **new output project** after updating or changing hybrid mode.

```bash
# From ODM-Fedora
./install-fedora.sh
./verify-acceleration.sh
./setup-amd.sh hip
./run-bounded.sh

# Experimental hybrid trial, separate output folder
ODM_HYBRID=1 ./run-bounded.sh

# From MapRepair
./run.sh --setup-only
./verify-acceleration.sh
./setup-amd.sh hip
./run-bounded.sh --port 8766
```

`setup-amd.sh hip` uses Fedora's `rocm-hip-devel`, builds gfx1150 sources locally,
then runs real-device parity and repeated allocation checks. It disables the
new library on failure. It does not install replacement kernel/display drivers.
`ADM_PIXEL_BACKEND=cpu` forces CPU compositing; `ODM_FSR_BACKEND=cpu` forces CPU
FSR. `ADM_PIXEL_BACKEND=native-hip` requires the custom HIP kernel and fails if
unavailable; `hip` alone is the older optional PyTorch ROCm path.

`run-bounded.sh` places the app and child processes in a systemd user scope:
MemoryHigh=70%, MemoryMax=80%, MemorySwapMax=1G, Python budget 18000 MiB.
On 32 GiB this reserves roughly 6.4 GiB outside the scope. Exceeding the limit
can terminate a job; this protects system RAM, not guaranteed job completion.
Driver allocations and other apps can still consume memory. Do not process in
both apps simultaneously. Plain launchers remain available without the OS cap.

## Optional XDNA2 setup

The published IRON wheel is pinned by URL and SHA256 to
`1.4.4.dev3+gf50bef7`, paired with Peano `22.0.0.2026090701+3e93bf7b`.
The matching upstream API and AMD driver docs were inspected; provenance is in
`AMD_DOCS_MANIFEST.json`. This requires Python **3.12**, working Linux amdxdna
device/firmware, XRT userspace and an accessible `/dev/accel/accel*` device.

```bash
# From ODM-Fedora, whose locked Python is already 3.12
# Only if XRT userspace is missing; builds the pinned AMD source, no kernel module.
./setup-xdna-userspace.sh
./setup-amd.sh xdna
```

The userspace helper targets Fedora 44+, builds AMD's pinned XRT/SHIM with two
compiler jobs and `-nokmod`, and leaves the existing kernel/firmware in place.
This optional source build can take substantial time. It has **not been run on
Fedora here**. If device permissions, firmware or memlock configuration prevent
execution, the target check fails with an error; the regular CPU/HIP app remains
usable. The official AMD instructions below cover those machine-specific cases.

MapRepair's normal runtime may use another Python version. For its NPU path,
install Python 3.12 (`sudo dnf install python3.12`), rerun
`MAPREPAIR_PYTHON=python3.12 ./run.sh --setup-only`, and set `ADM_PYTHON` to that
new `.runtime/.../bin/python` when calling the AMD scripts. XRT must use the same
Python ABI. After successful checks, source `/opt/xilinx/xrt/setup.sh` and launch
MapRepair with `ADM_ENABLE_XDNA=1 MAPREPAIR_PYTHON=python3.12 ./run.sh`.
Auto mode still rejects an NPU path slower than CPU. `ADM_PIXEL_BACKEND=xdna`
forces it for diagnostics and never silently counts a CPU fallback as NPU.

## Validation and scope

- Actual portable C++ scalar and 16-lane vector sources compile and match the
  NumPy CPU reference. Every foreground/alpha byte combination is covered;
  division-by-255 is exhaustive over the entire intermediate range.
- Plane, roof edge, rough canopy, zero-depth hole, high cost, NaN, infinity and
  boundary/halo rejection fixtures pass. Device classifier tests compare HIP
  masks with these compiled CPU decisions.
- Native target stress uses repeated small, awkward-size and 3840×2160 inputs,
  exact pixels, live process RSS growth checks after warmup. The HIP composite
  staging allocation is bounded to 32 MiB per process; flat-map GPU buffers are
  limited to two 4096×4096 float maps plus a small mask. These are bounded tests,
  **not proof that every workload can never run out of memory**.
- Native hybrid integration completed on 36 depth maps at 640px, exercising
  all 36 classifier calls. **Zero pixels qualified for skipping** on this grass
  dataset; it demonstrates integration, not a speedup or Ultra quality result.
  A six-photo full pipeline also completed after resuming its cached SfM stage;
  that resume time is not a fresh-run benchmark.
- Mac application regressions: 34 MapRepair tests, 16 shared runtime/kernel
  tests, three feature-cache tests, and Qt integration passed.
- Completed 36-photo Ultra run: **495.44 s**, all 36 registered, one component,
  sampled peak process-tree RSS **4.41 GiB**. Previous run was 522.75 s. This is
  about 5.2% faster in one comparison, **before the new hybrid/kernel work**.
  Original 2035×1509 GeoTIFF and 4070×3018 FSR export retain identical geographic
  bounds and exact 2× alpha. Full reconstruction pixel parity is not claimed.

HIP compilation, NPU compilation/execution, target memory stress and Ryzen speed
remain **on-laptop checks**. Passing Mac host arithmetic is not an AMD hardware
pass. Save `validation/target/*.json` and the application logs for review.

Primary references: [HIP streams and synchronization](https://github.com/ROCm/HIP/blob/f7929cc4022333400ca95b34ab2e8dd56a352636/docs/how-to/hip_runtime_api/asynchronous.rst),
[AMD Linux XDNA driver](https://github.com/amd/xdna-driver/tree/d715319ca8c4db291542abbce7cd850b9c2a1b57),
[IRON parallel transform](https://github.com/Xilinx/mlir-aie/blob/f50bef7/python/iron/algorithms/_transform.py),
[IRON non-Ubuntu setup](https://github.com/Xilinx/mlir-aie/blob/20a9c2fb2dca34d9d0f6f0c91793107d737423c3/docs/buildHostLinNonUbuntu.md).
