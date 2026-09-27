> Updated 2026-09-16: see [QUALITY_UPDATE.md](QUALITY_UPDATE.md) for the current
> ultra-quality defaults, native CPU tuning, FSR export and optional HIP setup.
> The every-sixth selection remains unchanged. Older validation notes below
> describe the preceding standard-quality release.

# ODM Desktop for Fedora

A C++ localhost server and offline browser interface for the existing OpenDroneMap engine. It builds and runs natively on Fedora, with Python orchestration and the original C++ reconstruction tools. No Docker or other containers are involved.

## Install

Target: **Fedora Workstation x86_64**, including AMD Ryzen AI 9 365 and Intel processors. An internet connection and permission to install Fedora packages with `sudo` are required for the initial build. Use a regular user account.

Extract the supplied archive into a permanent location such as `~/Applications/ODM-Fedora`. Keep the installation path free of spaces and special characters because the upstream native build has path restrictions.

```bash
cd ~/Applications/ODM-Fedora
./install-fedora.sh
./run-fedora.sh
```

Open **http://127.0.0.1:8765** in your browser and leave the terminal running. **ODM Flight Line** also appears in the applications menu. The initial installation downloads the native reconstruction dependencies and compiles the engine; it is not a prebuilt binary install. Reserve at least 20 GB for the installation, plus working space for your photos and reconstruction. Large flights need considerably more space and memory. Keep the installation at the same path after building.

The installer only installs missing Fedora packages, disables optional RPM recommendations, builds the orthophoto tools without the unused classification/3D export tools, and stops refreshing existing native source checkouts. Once successful, repeating setup reuses the built engine without reinstalling its dependencies.

The OpenSfM build now honors the configured job limit; previously it overrode that limit with every CPU core, which could exhaust memory during compilation.

The installer uses Fedora Qt, a checksum-verified Pixi 0.77.1 download, the repository's `pixi.lock`, and its original SuperBuild. Python packages stay in the local `.pixi` environment. The default build concurrency is two jobs per nested build to limit compiler memory use. Use `ODM_BUILD_JOBS=1 ./install-fedora.sh` if the first build exhausts memory.

```bash
./install-fedora.sh --check   # System, memory, free space; makes no changes
./run-fedora.sh --doctor     # Checks native libraries, tools and engine imports
./install-fedora.sh --offline # Recheck a completed install without downloads
```

Setup failures are recorded in `fedora-install.log`. This installer is for conventional Fedora Workstation; atomic variants such as Silverblue require different system package handling.

## Reconstruction mode

The app defaults to **Fast orthophoto** (ODM `--fast-orthophoto`): the same ultra
DSPSIFT feature extraction and calibration, but the map is rendered from the
sparse reconstruction and the dense OpenMVS pass is skipped. Choose **Ultra
dense** in the Reconstruction selector (or `--mode ultra` on the command line)
for the full dense terrain reconstruction, which takes far longer. Changing the
mode on a saved project reruns from the calibration stage.

**Input exposure** (`--exposure <stops>`, or the field next to the mode selector)
brightens dark flights before feature extraction. It is applied to the staged
JPEG copies with GPS/camera metadata preserved; originals are untouched.

## Make an orthophoto

1. Enter the full path of a folder of overlapping, geotagged RGB flight photographs (JPEG, PNG, or TIFF), or use **Choose a full folder** in the browser. Subfolders are included. A direct path avoids the browser import copy. Wait for the camera transfer to finish first.
2. Choose a new, empty project folder outside your photo folder.
3. Keep the default **2.3 cm/pixel** resolution, or set the same resolution used in your WebODM task. The engine respects the detail available in the photos.
4. Click **Build orthophoto**. The app shows stage progress and engine logs; **Stop** preserves completed work for resuming.

Photos are sorted by their full relative filename with numeric ordering (2 before 10). Exact byte duplicates are removed using SHA-256, retaining the first copy. The app then keeps unique photos **1, 7, 13, 19, …** across the entire folder. At least 13 unique photos are required to retain three. Only selected photos are copied into the processing project; originals are never deleted. `input-manifest.json` records the selection and counts. Similar-looking images are not treated as duplicates.

Keeping every sixth image reduces overlap and can reduce reconstruction quality or coverage compared with processing the full flight. Compare with WebODM using the same selected images when checking engine parity.

The thread default considers available RAM and logical CPU count; lower it if other applications need memory. This is a starting estimate, not a memory limit.

## Offline flight-line operation

After the first online installation, disconnect the network and run `./run-fedora.sh`. The C++ server listens only on 127.0.0.1. All HTML, CSS, JavaScript, photos, logs, preview images, and map downloads are served locally. No CDN, cloud processing, telemetry, or online basemap is used. The app disables ODM's model downloader and PROJ network access for processing; the standard pipeline needs no downloaded AI masks.

`--offline` does not install missing dependencies onto a fresh machine. Complete setup on the flight-line laptop before leaving connectivity. Browser imports are stored under the user's local ODM application-data directory; incomplete imports cannot be started. Direct folder paths avoid that extra copy. Large GeoTIFF downloads stream from disk.

The map is `PROJECT/flight/odm_orthophoto/odm_orthophoto.tif`. Open it in QGIS or another GeoTIFF viewer. A small preview uses ODM's report image when available. The engine retains reconstruction intermediates for resume, and can produce additional files needed by its report. Changing only map resolution reruns orthophoto export and subsequent reporting. If the source photos or engine change, choose a new project to avoid mixing incompatible results.

The desktop currently covers RGB flights with embedded GPS. It does not expose GCP import, multispectral controls, custom camera parameters, or every WebODM option. Files named `*_mask` are excluded; masks and other sidecar inputs require the original ODM CLI.

## Quality and performance

The processing engine is the **ODM 3.6.2 source in this checkout**. Its camera calibration, dense reconstruction, terrain surface, georeferencing, texturing, and orthophoto renderer are retained. The desktop fixes these standard quality settings:

| Setting | Value |
| --- | --- |
| Feature type / quality | DSPSIFT / high |
| Minimum features | 10,000 |
| Matcher / SfM | FLANN / incremental |
| Point cloud quality | medium |
| Map resolution | 2.3 cm/pixel by default; adjustable |
| Separate full 3D model | skipped; the standard 2.5D map surface is retained |

Quality parity requires the same ODM version, input photos, camera/GCP settings, and processing options as the WebODM task. **Pixel-identical output and a measured speedup have not been established.** This checkout also contains its existing native-build compatibility patches. The previous experimental direct OpenMVS renderer is excluded from the Fedora package.

Speed improvements preserve the standard map pipeline: omit the separate full 3D model, reuse completed stages, reuse the small report preview instead of exporting a second full-size PNG, and limit nested CPU thread pools. Dense reconstruction still takes time.

### Ryzen AI 9 365 and Radeon graphics

This build uses parallel CPU processing on AMD or Intel. ODM's existing GPU feature extraction uses **NVIDIA CUDA**; it has no AMD GPU compute backend. Installing ROCm does not make that pipeline run on the Radeon GPU. Experimental custom HIP/XDNA2 export kernels are now available; see [AMD_KERNELS.md](AMD_KERNELS.md). The main reconstruction remains CPU. No Metal, CUDA, or ROCm runtime is required by this CPU build. See [ODM's GPU support information](https://opendronemap.org/nodeodm/).

## Command line

```bash
./run-fedora.sh --cli --input /home/user/flight-photos \
  --output /home/user/maps/flight-01 --threads 6 --resolution 2.3
```

Use Ctrl+C to stop and the same command to resume. The project contains `project.json`, `input-manifest.json`, `processing.log`, and ODM's own `flight/log.json` for diagnostics. One app instance may process a given project at a time. Close processing jobs before rebuilding the engine. Do not modify files inside a managed project's `flight` folder.

## Validation status

The C++ application compiles locally with Qt 6 on macOS, and its automated process-runner and HTTP tests use a mock engine. Tests cover deduplication and sixth-image selection, recursive folder upload, incomplete-upload protection, localhost request checks, map download, quality arguments, outputs, rerendering, failed-stage resume, cancellation, project locks, changed inputs/engine, and unsafe project paths. They **do not validate photogrammetric quality**.

An installed Chrome browser also passed full-folder import (14 files → one exact duplicate removed → three selected), mocked processing and GeoTIFF download, page refresh, and mobile-width layout checks. No JavaScript errors or external web requests occurred.

The Fedora installer, Linux process cleanup, and Ryzen hardware performance still need testing on Fedora. No Fedora executable or end-to-end mapping benchmark was produced on the development Mac. During installation the script builds on the target machine, runs the desktop tests, and checks the actual native engine imports/executables before creating the application shortcut.

For a meaningful comparison, process the same flight in WebODM and this app using matching ODM versions and settings; compare seam placement, building/terrain edges, coverage, georeferencing, and processing time. The app's `project.json` records its arguments and elapsed processing time.

## Development

```bash
cmake -S fedora -B fedora/build -DCMAKE_BUILD_TYPE=Release
cmake --build fedora/build --parallel 2
ctest --test-dir fedora/build --output-on-failure
```

The app requires Qt 6 Core, Gui, Widgets, Concurrent, Network, and Test. The inherited ODM license and third-party licenses apply; see `LICENSE` and the dependency source trees downloaded under `SuperBuild/src`.
