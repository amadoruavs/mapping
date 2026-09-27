# Known failures

These happened on this project. If you see the same symptom, start here instead of rewriting the solver.

## GPS median 30.09 m exceeds 5.00 m

```
ValueError: GPS camera positions contradict the visual mosaic:
median 30.09 m exceeds 5.00 m; see .../geographic_alignment.json
```

**Meaning:** the visual mosaic and the GPS camera centers disagree. On `mapv2` (94 photos) this was the **scale-collapse** mode: mosaic-space photo area fell to ~0.008 of the registration frame. Reported visual residual can still look small.

**What fixed it (measured):** `alignment_symmetric_transfer=True`. Same matches (508 edges, 60875 correspondences): GPS median 2.38 m, p95 6.83 m.

**What is not a fix:** raising `gps_tolerance_m` to 31; `allow_gps_mismatch=True` (bypass only); more iterations; blaming the 5 m constant.

**Still seen:** 2026-09-26 a run of `ISM/run.py` on `/Users/avneh/Downloads/mapv2` printed the 30.09 m error again and dumped an optimization trace that walks objectives down toward ~7457. If that reproduces on this tree, inspect whether symmetric transfer actually ran (`registration_report.json` / encoded kernels `align_transfer_*`) and whether footprint areas collapsed.

## 600 iterations, tiny residual, no map

564 photos, 1094 s wall, 600 iterations, objective still improving, median 0.0028 px. `transforms.npy`: 523/564 footprints < 1% of registration area, median area ratio 2.9e-6. Rejected steps included NaNs.

**Meaning:** collapsing scale. Stop. Do not celebrate the residual.

## Disconnected match graph

LeanADM + `mapv2`: 512 verified pairs, still disconnected, 9.46 s, no map.

ISM added `match_graph.py`: only if the graph is disconnected, retry cross-group GPS/sequence candidates at 1280 with the same ratio/inlier/coverage gates, convert matches back to the 640 grid. Cap 256 attempts. Report filenames. Never GPS-only edges.

564-photo set needed recovery contrast 0.01 for those retry detections (11 bridges). Then it hit the alignment RAM guard.

## Projection intersects horizon

A pair homography that sends the image across the vanishing line is rejected. Seen during 564-photo work after 86 s, before the horizon guard was tightened. Do not project those pairs.

## Alignment workspace guard

`alignment_budget_mb` (128–2048, ISM 1024). 564 photos at 1280 planned more than 512 MiB scratch. Raise the budget or drop registration width. Do not remove the guard.

## FSR duplicated roads

Recursive frame generation + compositor using generated views produced large warped pavement pieces. Gate: rotation ≤ 15°, motion ≤ ¼ image, finite midpoint, parent consistency ≤ 1 native px on a 9×9 grid; generated pixels must be seen in **both** parents. Or set `interpolation_levels=0`.

## Sharp rectangular MapRepair patches

Early repairs looked stamped-on. `patch_blend.py` added slow RGB offsets and wide feathers. Fixture RGB error 28.4 → 4.2 (was 1.85 with a harsher blend). Gentler edges, less "correction."

## OpenRT / GCC build

`std::sqrtf` / `std::powf` missing in GCC's `std`. Patched to `std::sqrt` / `std::pow` in three upstream OpenRT files. Already in the tree. Lighting math equivalent.

## OpenRT first compile missing `Ray.h`

Adapter include path. Fixed. If you clean-build and see it, check `MapRepair/openrt/`.

## Direct `opensfm_planar` init from metadata

Failed on `mapv2` (40.96 s, no map) because of `NO GPS connected` + bad yaw. Must visual-init first.

## Copied Mac OpenRT binaries on Linux

CMake cache has OS/arch/C++ runtime/install path. `run.sh` builds into `build/openrt-<platform>-<arch>-<id>`. Delete copied `build/` on the destination.

## RPATH after OpenMVS hybrid install (ODM port)

Copying a built binary skipped `cmake --install` RPATH. `cmake --install` fixed Mac library lookup. Relevant only if you rebuild the ODM port.

## Web UI CSP / Playwright

First browser test used a CSP-incompatible wait expression. Replaced with locator assertions. If you add browser tests, do not `eval` inline.

## User-canceled work — do not resume

Motion-blur handling, extra RAM schemes, and inner-loop GPS/yaw priors were **explicitly canceled** on 2026-09-14. Do not continue them unless a human asks again.

## MapRepair "0 patches" is not a crash

A consistent report with flagged/unresolved tiles and zero patches means the evidence gates held. The SIYI `odm.png` + 40 photos run did this in 14.2 s.
