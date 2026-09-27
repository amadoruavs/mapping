# OpenSfM feature extraction & matching anatomy (ODM fork @ `c5328439`)

Scope: the code path ODM 3.6.2 runs for `detect_features` and `match_features`,
read from a clone of `https://github.com/OpenDroneMap/OpenSfM` at commit
`c5328439465e6ace011f39077d1077d7b1cdd65d` (2025-08-18, "Merge pull request #43
from NathanMOlson/24.04"). Compared with `mapillary/OpenSfM` `main` the fork is
296 ahead / 296 behind with a merge-base of 2022-06-15 (GitHub compare API).
Line numbers are for that commit. ODM config generation is read from
`/Users/avneh/Code/ADM/ODM/opendm/osfm.py` and `/Users/avneh/Code/ADM/ODM/opendm/gpu.py`.

---

## 1. File map

| File | LOC | Role |
|---|---|---|
| `opensfm/features.py` | 701 | `FeaturesData` container + npz I/O; resize, RootSIFT, coordinate normalisation; one `extract_features_<type>` per backend; the `feature_type` dispatch; `build_flann_index` |
| `opensfm/features_processing.py` | 310 | `run_features_processing` (multi-process driver), `detect()` (calls `extract_features`, masking, sorting, save) |
| `opensfm/feature_loader.py` | 4 | `instance = feature_loading.FeatureLoader()` — a module-level singleton |
| `opensfm/feature_loading.py` | — | `FeatureLoader`: LRU-cached `load_all_data`, `load_features_index` (→ `build_flann_index`), `load_words`, masks |
| `opensfm/matching.py` | 1038 | pair matching drivers, FLANN / BRUTEFORCE / WORDS matchers, robust (F / E) filtering |
| `opensfm/pairs_selection.py` | 815 | candidate pair selection (GPS distance/neighbours, graph, time, order, BoW, VLAD) |
| `opensfm/config.py` | 397 | defaults |
| `opensfm/actions/detect_features.py` | 30 | command entry |
| `opensfm/src/features/` | C++ | `hahog.cc` (204), `dspsift.cc` (249), `akaze_bind.cc` (57), `matching.cc` (150) → pybind module `pyfeatures` (`python/pybind.cc`) |
| `opensfm/src/dense/` | C++ | `depthmap.cc` (637): OpenSfM's own CPU PatchMatch (unused by ODM) + `OpenMVSExporter` (used by ODM's `export_openmvs`) → module `pydense` |

---

## 2. `extract_features` and the `feature_type` dispatch

`opensfm/features.py:592-667`:

```python
def extract_features(image, config, is_panorama) -> Tuple[points, desc, colors]:
    extraction_size = config["feature_process_size_panorama" if is_panorama else "feature_process_size"]
    features_count  = config["feature_min_frames_panorama"   if is_panorama else "feature_min_frames"]
    image = resized_image(image, extraction_size)                     # 627
    ... (h,w)->(h,w,1); RGB->gray via cv2.COLOR_RGB2GRAY               # 628-634
    feature_type = config["feature_type"].upper()                     # 637
    if   feature_type == "SIFT":     points, desc = extract_features_sift(image_gray, config, features_count)
    elif feature_type == "SURF":     ... extract_features_surf
    elif feature_type == "AKAZE":    ... extract_features_akaze
    elif feature_type == "HAHOG":    ... extract_features_hahog
    elif feature_type == "ORB":      ... extract_features_orb
    elif feature_type == 'SIFT_GPU': points, desc = extract_features_popsift(image_gray, config, features_count)   # 648-649
    elif feature_type == 'DSPSIFT':  ... extract_features_dspsift
    else: raise ValueError("Unknown feature type (must be SURF, SIFT, AKAZE, HAHOG, SIFT_GPU or ORB)")          # 652-655
    xs = points[:, 0].round().astype(int); ys = points[:, 1].round().astype(int)
    colors = image[ys, xs]                                            # 657-661  <-- points must be in-bounds pixel coords
    return normalize_features(points, desc, colors, image.shape[1], image.shape[0])   # 667
```

Helpers:

- `resized_image` (271-279): if `0 < max_size < max(w,h)` resize so the **longer
  side == max_size** with `cv2.INTER_AREA`; integer floor for the other side.
- `root_feature` (282-288): L1-normalise each row, then `sqrt` (RootSIFT). Used
  by SIFT/SURF when `config["feature_root"]` (406-407, 476-477).
- `normalized_image_coordinates` (314-321): `x' = (x + 0.5 − w/2)/max(w,h)`,
  same for `y`; `normalize_features` (334-340) also divides `points[:,2]`
  (size) by `max(w,h)`. Angle (`points[:,3]`) is left untouched.

Backend functions and what they return:

| Backend | Lines | Extractor | Input | Returns |
|---|---|---|---|---|
| `extract_features_sift` | 350-412 | `cv2.SIFT_create(nfeatures=int(features_count*3/2), edgeThreshold, contrastThreshold)` (359-361); loop lowering `sift_peak_threshold *= 2/3` until `len(points) >= features_count` or threshold `< 0.0001` (381-402) | uint8 gray | `points = np.array([(pt.x, pt.y, size, angle) ...])` float64 (408); `desc` float32 `[N,128]` from OpenCV, RootSIFT'd (406-407) |
| `extract_features_popsift` | 414-427 | `from opensfm import pypopsift` (417); `pypopsift.popsift(image, peak_threshold, edge_threshold, target_num_features=features_count, use_root=bool(config["feature_root"]))` (422-425) — note: `downsampling` not passed (defaults to −1 → 2× upsampling inside popsift) | uint8 gray `(H,W)` | `[points float32 [N,4], desc float32 [N,128]]`; RootSIFT applied **inside** popsift (`NormMode RootSift`), so `root_feature` is *not* called here |
| `extract_features_surf` | 429-482 | `cv2.xfeatures2d.SURF_create` (contrib) | uint8 gray | `[N,4]`, `[N,64]` float32 |
| `extract_features_akaze` | 494-520 | `pyfeatures.akaze(image, AKAZEOptions)` | uint8 gray | `[N,4]` (cast to float, 519), desc float32 or uint8 (MLDB) |
| `extract_features_hahog` | 523-545 | `pyfeatures.hahog(image.astype(float32)/255, peak, edge, target)` (528-533) | float32 gray in `[0,1]` | `[N,4]`, `[N,128]`; `sqrt` if root (536); scaled to uchar `*362` (root) or `*512` and rounded when `hahog_normalize_to_uchar` (541-542) |
| `extract_features_dspsift` | 548-563 | `pyfeatures.dspsift(image.astype(float32)/255, peak_threshold=0.00667 (hardcoded), edge_threshold=10, target_num_features, feature_root, estimate_affine_shape=False)` (553-560); `domain_size_pooling` default True (pybind default) | float32 gray `[0,1]` | `[N,4]`, `[N,128]` float32 (`dspsift.cc:244-246`) |
| `extract_features_orb` | 566-589 | `cv2.ORB_create(nfeatures)` | uint8 gray | `[N,4]`, `[N,32]` uint8 |

The C++ extractors return `py::tuple(points [N,4], desc [N,dim])`
(`dspsift.cc:128-135, 244-246`; `hahog.cc:165-201`), with `points[i] =
(x, y, size, angle)` in **pixel coordinates of the resized image**.

Empty-result edge case: OpenCV backends return `np.zeros((0,3))` for both arrays
(410-411, 480-481, 585-586) — 3 columns, not 4; downstream code tolerates it
only because `colors = image[ys, xs]` on empty arrays is empty.

---

## 3. Persisted format (`FeaturesData`, `features.py:48-268`)

`FeaturesData.save` (95-135) writes `np.savez_compressed` with keys
`points` (float32 `[N,4]`, normalised coordinates), `descriptors`
(dtype: **uint8** for `AKAZE` with `MLDB*`, `HAHOG` with
`hahog_normalize_to_uchar`, and `ORB`; **float32** otherwise — lines 98-108),
`colors`, `segmentations`, `instances`, `segmentation_labels`,
`OPENSFM_FEATURES_VERSION = 3`. Loaded via `from_file` (138-142) and
`_from_file_v3` (239-268). The uint8-vs-float32 choice is also what selects
the FLANN index type (`build_flann_index`, 670-701: float32 → `KDTREE`/`KMEANS`
per `flann_algorithm`, uint8 → `LSH`) and the brute-force norm
(`match_brute_force`, `matching.py:719-724`: uint8 → `BruteForce-Hamming`).

`features_processing.detect` (225-294): `extract_features` (256) →
optional segmentation bake (264) → mask (272-275) → sort by size, keep best →
`data.save_features` (294). `run_features_processing` (18-80) uses a
reader thread + `processes` workers (config `processes`, set by ODM to
`--max-concurrency`).

`FeatureLoader.load_features_index` (`feature_loading.py:169`) lazily builds
the per-image FLANN index once and caches it.

---

## 4. Matching stage (`opensfm/matching.py`)

Entry: `match_images(data, config_override, ref_images, cand_images)` (26-58)
→ `pairs_selection.match_candidates_from_metadata` (46-52) →
`match_images_with_pairs` (61-94): `context.parallel_map(match_unwrap_args, args,
processes, jobs_per_process=2)` with `processes` reduced to fit 512 MB each
(75-80).

`_match_descriptors_impl` (337-457) dispatches on `config["matcher_type"].upper()`:

| `matcher_type` | Code | Mechanism |
|---|---|---|
| `FLANN` | 396-430 | `index.knnSearch(f2, 2, params={checks: flann_checks})` on a `cv2.flann_Index` (`context.py:34-40`), Lowe ratio on **squared** distances (`match_flann`, 669-683); `symmetric_matching` intersects both directions (686-702) |
| `BRUTEFORCE` | 431-435 | `cv2.DescriptorMatcher_create("BruteForce"|"BruteForce-Hamming").knnMatch(f1, k=2)` + Lowe ratio (705-738) |
| `WORDS` (ODM `bow`) | 373-395 | `pyfeatures.match_using_words(f1, words1, f2, words2[:,0], lowes_ratio, bow_num_checks)` (624-642); requires the HAHOG-uchar vocabulary `bow_hahog_root_uchar_10000.npz` (`config.py:107`) — which is why ODM forces `feature_type=HAHOG` under `bow` (`osfm.py:290-294`) |

Then `robust_match` (885): `robust_match_fundamental` (767-789,
`cv2.findFundamentalMat(FM_RANSAC, threshold, 0.9999)`) or, for calibrated
cameras, bearing-based epipolar filtering via `pygeometry`
(828-848). Guided matching (`_match_descriptors_guided_impl`, 256-334) only
supports BRUTEFORCE.

Pair selection knobs read at `pairs_selection.py:587-599`:
`matching_gps_distance`, `matching_gps_neighbors`, `matching_graph_rounds`,
`matching_time_neighbors`, `matching_order_neighbors`, `matching_bow_*`,
`matching_vlad_*`. Without GPS on all images, `matching_gps_neighbors` is
disabled with a log line (607).

Nothing in matching is GPU-aware. The one GPU-matching attempt in ODM history
(2021 PR #1239: "option to enable GPU SIFT matching ... underperform[s] the CPU
based FLANN, so I'm not going to enable it") was removed with the silx backend;
a 2024 request to use `cv2.cuda` brute-force (`OpenDroneMap/OpenSfM#29`) was
redirected to the forum.

---

## 5. C++ modules

`opensfm/src/features/CMakeLists.txt` builds `features` (links OpenCV, `akaze`,
`foundation`, `vl` = VLFeat) and `pybind11_add_module(pyfeatures python/pybind.cc)`;
`pybind.cc` exports `akaze`, `hahog(image, peak_threshold=0.003,
edge_threshold=10, target_num_features=0)`, `dspsift(image, peak_threshold,
edge_threshold, target_num_features, feature_root=True,
domain_size_pooling=True, estimate_affine_shape=True)`, `match_using_words`,
`compute_vlad_descriptor`, `compute_vlad_distances`. All CPU (VLFeat SIFT / DSP-SIFT).

`opensfm/src/dense/src/depthmap.cc` (637 LOC) is OpenSfM's original CPU MVS:
`DepthmapEstimator::ComputePatchMatch` (200) = `RandomInitialization` (235) +
alternating `PatchMatchForwardPass`/`BackwardPass` (281-298) with an
`NCCEstimator` (47-75); plus `DepthmapCleaner`/`DepthmapPruner`. It is bound
as `pydense` (`dense/python/pybind.cc`) and driven by `opensfm/dense.py`. ODM
does not run `compute_depthmaps`; it exports to OpenMVS via
`pydense.OpenMVSExporter` (`pybind.cc:11-15`) and densifies there.

`pypopsift` is **not** part of this tree: it is a separate CMake project whose
`.so` is dropped into `${SB_INSTALL_DIR}/bin/opensfm/opensfm/` by
`/Users/avneh/Code/ADM/ODM/SuperBuild/cmake/External-PyPopsift.cmake`
(`-DOUTPUT_DIR=...`), which makes `from opensfm import pypopsift`
(`features.py:417`) resolve. The fork's history: `c90b0344a` (2021-12-16,
"GPU code cleanup, pypopsift in, silx out") removed the earlier OpenCL
`silx.image.sift` backend (`opensfm/sift_gpu.py`, added by `eb4b8ff84`
2021-02-24 "PoC GPU support").

---

## 6. Config keys ODM sets (`opendm/osfm.py`)

`osfm.py:249-266` (base list), 268-331 (conditionals):

| Key | Value | Source |
|---|---|---|
| `use_exif_size` | `no` | 250 |
| `flann_algorithm` | `KDTREE` ("more stable, faster than KMEANS"; OpenSfM default is `KMEANS`, `config.py:94`) | 251 |
| `feature_process_size` | computed (see below) | 252 |
| `feature_min_frames` | `--min-num-features` (default 10000, `opendm/config.py:220-228`) | 253 |
| `processes` | `--max-concurrency` | 254 |
| `matching_gps_neighbors` | `--matcher-neighbors` (default 0) or 3 when ≤3 photos | 255, 237-247 |
| `matching_gps_distance` | `0` | 256 |
| `matching_graph_rounds` | `50` unless `--matcher-neighbors > 0` | 257, 237-242 |
| `sift_peak_threshold` | `0.066` (OpenSfM default 0.2) | 262 |
| `matching_order_neighbors` | `--matcher-order` (non-georeferenced only) | 268-272 |
| `matcher_type` | `bow→WORDS`, `flann→FLANN`, `bruteforce→BRUTEFORCE`; `bow` auto-selected when no GPS and `--matcher-type` unset | 277-296 |
| `feature_type` | `--feature-type` upper-cased; forced `HAHOG` under `bow`; **`SIFT_GPU` only if `SIFT` was requested AND `has_gpu(args)` AND `has_popsift_and_can_handle_texsize(w,h)`** | 278, 290-317 |
| `use_altitude_tag`, `align_method`, hybrid BA, camera projection, etc. | — | 274-275, 319-331 |

`feature_process_size` (202-232): start 2048; `factor = min(1, scale[quality] ×
multiplier)` with `scale = {ultra 1, high .5, medium .25, low .125, lowest .0675}`,
`multiplier = 2` if `< 2 MP`, `0.5` if `> 42 MP`; result clamped to
`[320, 4480]` and multiplied against the **largest photo's longer side**.
Default `--feature-quality high` on a 20 MP (5472 px) photo → 2736 px.

GPU gating (`osfm.py:299-317`, `gpu.py`):

```python
if feature_type == "SIFT":
    if has_gpu(args) and max_dims is not None:
        w, h = scaled so the longer side == feature_process_size          # 302-308
        if has_popsift_and_can_handle_texsize(w, h):                      # gpu.py:12-35
            feature_type = "SIFT_GPU"; self.gpu_sift_feature_extraction = True
```

- `has_gpu` (`gpu.py:73-95`): `ODM_NO_GPU` env or `--no-gpu` → False; Windows:
  `%SYSTEMROOT%\system32\nvcuda.dll` exists; else: `shutil.which('nvidia-smi')`.
  **NVIDIA-only by construction** — a Metal/AMD backend must change this function.
- `has_popsift_and_can_handle_texsize` (`gpu.py:12-35`): loads `libcuda.so` /
  `nvcuda.dll` via ctypes, requires CC ≥ 3.5, then
  `pypopsift.fits_texture(int(w*1.02), int(h*1.02))`. Any exception → False
  with a warning (so a missing `pypopsift` silently means CPU SIFT).
- Failure fallback (`osfm.py:418-439`): if `detect_features` raises and
  `gpu_sift_feature_extraction` was set, rewrite `feature_type: SIFT`, delete
  `features/`, re-run.

ODM's default `--feature-type` is **`dspsift`** (`opendm/config.py:230-234`),
so the GPU SIFT path is only taken when the user passes `--feature-type sift`
(as the README's GPU example does, `README.md:100`). The pixi `gpu` feature also
pulls `onnxruntime ... *cuda*` (`pixi.toml:91-92`) for the sky/background
segmentation models, unrelated to features.

---

## 7. Contract for a hypothetical non-CUDA feature backend (documentation only)

Two integration shapes are possible; both are described, neither is implemented.

### 7a. Drop-in `pypopsift` replacement (zero OpenSfM changes)

Provide a Python extension importable as `opensfm.pypopsift` exposing:

```python
popsift(image: np.ndarray[uint8, (H, W)], peak_threshold: float = 0.1, edge_threshold: float = 10,
        target_num_features: int = 4000, use_root: bool = True, downsampling: float = -1) -> list
fits_texture(width: int, height: int, downsampling: float = -1) -> bool
```

(`pypopsift/src/main.cpp:19-31`). Return value is a 2-element list:
`points` float32 `[N,4]` = `(x, y, sigma, orientation)` in pixel coordinates of
the input image, `x,y` must satisfy `0 ≤ round(x) ≤ W−1`, `0 ≤ round(y) ≤ H−1`
(`features.py:657-659` indexes the image with them; pypopsift rounds and clamps,
`popsift.cpp:107-108`); `desc` float32 `[N,128]`, already RootSIFT-normalised
when `use_root` (OpenSfM will **not** post-normalise). It must honour the
"keep lowering the peak threshold until `N >= target_num_features`" semantic
(`popsift.cpp:82-122`), otherwise sparse images will yield too few features.
Ordering: popsift sorts `LargestScaleFirst`; OpenSfM re-sorts by size later,
so order is not load-bearing. **Unit caveat**: OpenCV SIFT reports `angle` in
degrees, popsift in radians (moderately confident; `s_orientation.cu:43` computes
`th` in `[-π, π)`); `points[:,3]` appears unused downstream — verify before
relying on it.

Also required: make `opendm/gpu.py:has_gpu` and
`has_popsift_and_can_handle_texsize` detect the new runtime (today both are
hard-wired to NVIDIA driver files / `nvidia-smi` / `libcuda.so`), and add a
`SuperBuild` external project for the new module (`External-PyPopsift.cmake`
is gated on `SB_ENABLE_CUDA`).

### 7b. New `feature_type` (e.g. `SIFT_METAL`, `ONNX_XXX`)

- Add a branch at `features.py:648-651` calling a new
  `extract_features_<name>(image_gray, config, features_count)` returning
  `(points [N,4], desc [N,D])` with the same coordinate contract. `D` need not
  be 128: FLANN (`build_flann_index`, 670-692) and brute-force
  (`matching.py:719-724`) are dimension-agnostic; **dtype must be float32 or
  uint8** (`FeaturesData.save`, 98-108 decides storage dtype by
  `feature_type` — extend that `if` or the new type will be saved float32,
  which is correct for float descriptors).
- Learned descriptors (SuperPoint/ALIKED/DISK via ONNX/CoreML) are typically
  256-D float32 with L2 semantics; Lowe's ratio at `lowes_ratio 0.8`
  (`config.py:84`) is tuned for SIFT and may need retuning; `WORDS`/BoW cannot
  be used (vocabulary is HAHOG-specific). Scale (`points[:,2]`) feeds
  `reprojection_error_sd`-style logic and must be a plausible pixel radius, not 0.
- Add the name to the `ValueError` at 652-655 and to `opendm/config.py:234`
  choices; map it in `osfm.py:298-317`; extend the CPU fallback (`osfm.py:428-433`)
  to know the new GPU type.
- Threshold-loop semantics (`features_count` target) should be reproduced for
  parity with SIFT/HAHOG/DSPSIFT.

### 7c. Where a GPU could also help, for completeness

- `resized_image` + grayscale conversion are trivial.
- Matching: `match_flann` per pair is CPU FLANN; a GPU brute-force k=2 matcher
  would slot into `match_brute_force` (705-738) if it returned
  `List[Tuple[int,int]]` of Lowe-ratio-filtered pairs; the 2021 benchmark
  found GPU matching slower than FLANN for SIFT-sized sets.
- `robust_match_fundamental` uses OpenCV RANSAC on CPU.

---

## 8. Open questions

1. Angle units of `points[:,3]` across backends and whether anything reads it
   (grep suggests only save/load).
2. Whether `np.str` at `features.py:122` (removed in NumPy 1.24) is reachable
   in ODM (only with semantic segmentation data; ODM's sky masks go through
   `mask_list.txt`, not `SemanticData`) — probably dead code for ODM.
3. Exact `feature_process_size` used for the `fits_texture` probe vs the size
   OpenSfM actually resizes to (both use the longer-side rule, but `osfm.py`
   uses the *largest* photo's aspect while OpenSfM resizes each photo).

## Sources

- Clone: `https://github.com/OpenDroneMap/OpenSfM` @ `c5328439465e6ace011f39077d1077d7b1cdd65d`
- `opensfm/features.py`, `opensfm/features_processing.py`, `opensfm/feature_loader.py`, `opensfm/feature_loading.py`, `opensfm/matching.py`, `opensfm/pairs_selection.py`, `opensfm/config.py`, `opensfm/context.py`, `opensfm/actions/detect_features.py`, `opensfm/src/features/{CMakeLists.txt,python/pybind.cc,src/dspsift.cc,src/hahog.cc}`, `opensfm/src/dense/{python/pybind.cc,src/depthmap.cc}`
- ODM: `/Users/avneh/Code/ADM/ODM/opendm/osfm.py:195-330, 418-439`; `/Users/avneh/Code/ADM/ODM/opendm/gpu.py`; `/Users/avneh/Code/ADM/ODM/opendm/config.py:220-262, 388-392`; `/Users/avneh/Code/ADM/ODM/SuperBuild/cmake/External-OpenSfM.cmake`, `External-PyPopsift.cmake`; `/Users/avneh/Code/ADM/ODM/README.md:95-115`
- pypopsift: `https://github.com/OpenDroneMap/pypopsift` @ `fe2d1ccc` `src/main.cpp`, `src/popsift.cpp`
- GitHub API: `repos/mapillary/OpenSfM/compare/main...OpenDroneMap:OpenSfM:c5328439...`; commits `eb4b8ff84`, `ed7c8fb5f`, `aac21e23f`, `c90b0344a`
- https://github.com/OpenDroneMap/ODM/pull/1239 (2021-03-05 "GPU Support")
- https://github.com/OpenDroneMap/OpenSfM/issues/29
