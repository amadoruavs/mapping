# Native Apple Silicon Mac port

The same ODM-Fedora source now runs natively on Apple Silicon macOS using locked
Pixi/conda native libraries and a Homebrew Qt interface. It preserves exact
photo deduplication, every-sixth selection, ultra DSPSIFT feature extraction,
24,000 feature target, mesh 500,000/depth 12 and 2.3 cm/pixel defaults.
FSR runs on CPU with SIMD; no ROCm, CUDA or Metal is used by this Mac port.

## Fast orthophoto (default) vs Ultra dense

The **Reconstruction** selector (web page, Qt window, and `--mode fast|ultra`
on the command line) chooses the pipeline after camera calibration:

- **Fast orthophoto** (default) passes ODM's `--fast-orthophoto`. The map is
  rendered from the sparse OpenSfM reconstruction with Poisson meshing; the dense
  OpenMVS pass is skipped entirely. Feature extraction, matching and calibration
  are unchanged, so georeferencing accuracy is the same. Expect a fraction of the
  Ultra runtime; fine terrain relief is coarser, which mostly matters on steep or
  built-up scenes rather than flat fields.
- **Ultra dense** keeps the previous full dense reconstruction and texturing.

**Input exposure** (web page, Qt window, `--exposure <stops>`) brightens or darkens
the staged JPEG copies before feature extraction: +1 doubles the light in linear
space, highlights roll off softly above 85% instead of clipping, and 0 is exact.
Exif/XMP/ICC segments are copied byte-for-byte into the re-encoded photos, so GPS,
camera model and orientation are unchanged; originals are never modified. PNG/TIFF
inputs are copied as-is. Changing the exposure on a saved project re-stages the
photos and reruns from calibration.

Both modes still write the original GeoTIFF plus the separate 2× FSR GeoTIFF.
Switching mode on a saved project reruns from the calibration stage so sparse
and dense products are never mixed; `project.json` records `"fast"`.

```bash
./install-macos.sh
./run-macos.sh
```

Open http://127.0.0.1:8767. Existing dependency downloads/builds are reused.
First setup needs internet, Homebrew and Xcode command line tools. Build on the
Mac where you will run it; don't move its compiled environment to Fedora.

On the development Mac, MapRepair is separately available at
http://127.0.0.1:8768, launched with:

```bash
cd ../MapRepair
./run.sh --port 8768 --runs-dir web_runs_mac
```

To test the complete workflow, download **Download GeoTIFF** from ODM and load
that original map into MapRepair with its source photos. MapRepair already
performs its own 2× FSR, so feeding it ODM's FSR 2× export would upscale twice.
The extra **Download FSR 2× GeoTIFF** in ODM is available for direct comparison.

The web page displays elapsed time. Use a fresh output project for a cold
reconstruction benchmark; restarting a completed project reuses cached stages.
`project.json` records the input manifest, quality arguments and elapsed time.
`flight/log.json` records stage timings. Native Mac runtime checks and Qt
integration tests pass. This is not a Ryzen performance measurement.
