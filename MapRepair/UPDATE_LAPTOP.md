# Install or update the Fedora bundles

Stop both apps. Copy both archives and their `.sha256` files to `~/Downloads`.
These commands assume `~/Applications`; reuse your existing install parent if
different. Extraction preserves environments and results. Choose a new ODM
project after updating. The first native build takes time and needs internet.

```bash
cd ~/Downloads
sha256sum -c ODM-Fedora.tar.gz.sha256
sha256sum -c MapRepair-linux.tar.gz.sha256
mkdir -p ~/Applications
tar -xzf ODM-Fedora.tar.gz -C ~/Applications
tar -xzf MapRepair-linux.tar.gz -C ~/Applications

cd ~/Applications/ODM-Fedora
./install-fedora.sh
./verify-acceleration.sh
./setup-amd.sh hip

cd ~/Applications/MapRepair
./run.sh --setup-only
./verify-acceleration.sh
./setup-amd.sh hip
```

The HIP setup builds locally, checks real-device parity and repeated allocations,
and disables its new library on failure. CPU remains usable. No PyTorch install
is needed for custom HIP. Main reconstruction still runs on CPU.

Start ODM at http://127.0.0.1:8765:

```bash
cd ~/Applications/ODM-Fedora
./run-bounded.sh
```

For an experimental hybrid trial, choose a new project:

```bash
ODM_HYBRID=1 ./run-bounded.sh
```

Start MapRepair at http://127.0.0.1:8766:

```bash
cd ~/Applications/MapRepair
./run-bounded.sh --port 8766
```

The bounded launchers use systemd's 70% soft / 80% hard RAM limits and 1 GiB swap.
A job may be terminated when it exceeds the cap. Do not process in both apps
simultaneously. Ordinary `./run-fedora.sh` / `./run.sh` remain available.

Defaults retain 2.3 cm/pixel, every-sixth unique photo selection and full Ultra
refinement. FSR is a separate display upscale. Hybrid shortcuts can change
geometry, so they are off by default. MapRepair now has a full-image 1080p preview, 51 grading/detail controls and
editable master/RGB curves. Release a control to update; download grades the
original resolution. `ODM_FULL_REPORT=1` restores the optional PDF/diagnostic charts.

Custom XDNA2 vector sources are included with a pinned toolchain installer;
setup is optional and requires compatible XRT/driver/Python 3.12. From ODM-Fedora:

```bash
./setup-xdna-userspace.sh   # only if XRT userspace is missing; Fedora 44+
./setup-amd.sh xdna
```

See [AMD_KERNELS.md](AMD_KERNELS.md) for precise scope, XDNA2 setup for MapRepair,
fallback controls, memory limits and primary AMD documentation.

Local result: 36-photo Ultra 495.44 s (previous 522.75 s), all 36 registered,
4.41 GiB sampled peak process-tree RSS. This predates the new hybrid/kernels.
Mac host math/application tests passed. Actual Fedora installation, GPU/NPU
compilation/execution and Ryzen performance remain target checks. Full-pipeline
pixel parity and the 250-image under-five-minute target are not established.

MapRepair September 17 finish update: new runs use offline adaptive cinema with
noise-gated detail and OpenRT lighting. Completed maps have **Auto cinema ·
offline**, a full-resolution detail inspector, and lossless PNG downloads.
No additional dependencies are required. Restart the running MapRepair server
after extracting this update; a browser refresh alone cannot load new routes.
For a MapRepair-only update:

```bash
cd ~/Downloads
sha256sum -c MapRepair-linux.tar.gz.sha256
tar -xzf MapRepair-linux.tar.gz -C ~/Applications
cd ~/Applications/MapRepair
./run.sh --setup-only
./run-bounded.sh --port 8766
```
