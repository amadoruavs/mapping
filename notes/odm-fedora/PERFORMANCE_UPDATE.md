# Performance update — 2026-09-16

Installed on the development Mac; included in the Fedora source package.

* CPU requests are capped at the hardware thread count (20 requested becomes 12 on this M4 Pro). The CLI wrapper also enforces this for servers already running when updated.
* Meshing separately budgets about 2 GiB per 2000px tile worker, using at most 60% of currently available RAM. Tile size, mesh vertex target, depth, and quality are unchanged. This addresses concurrent full-resolution tiles exhausting memory; the Mac had about 12.5 GiB in swap during the 20-worker run. Swap use alone does not measure how much time swapping cost.
* OpenMVS caches smoothness and depth-prior terms once per candidate instead of recomputing them for each target image. It retains the original score multiplication order and FP32 operations. The patched DensifyPointCloud is installed locally; the SuperBuild applies the patch on future installs.
* Every sixth unique photo, DSPSIFT ultra/24k features, ultra dense reconstruction, 500k mesh target, depth 12, 2.3 cm/pixel and FSR remain unchanged.

Validation: 1,000,000 candidate/view scores are bit-identical to the original function, covering randomized depths, sloped normals and 0–4 smoothness neighbors. Focused score timing in the saved run was 2.74252 s original versus 2.24151 s cached (18.3% less time). This is a kernel measurement under concurrent system load, NOT an end-to-end speedup. Qt integration: 14 passed, 1 Linux-only skip. Memory budgeting: 2 tests passed.

Real six-photo dense runs completed with original and cached binaries, but concurrent 94-photo processing made their timings unsuitable for a reliable overall speedup claim. No new full-pipeline speedup or Ryzen result has been established. The existing 94-selected-photo job was preserved and was still texturing when this update was installed; it had started with the old engine. Its duration includes development/benchmark contention and must not be used as an isolated baseline.

A projection/bilinear SIMD experiment was rejected after noisy/slower real-photo timings. A batched mesher read experiment was also rejected because its mesh differed in validation and no speed benefit was measured. Neither is in the installed patch. The earlier FSR SIMD code is retained.

Local evidence: `fedora/validation/performance_20260916/score_final.txt`, `dense_benchmark.json`, `repeat/`, `cache_repeat/`, and `mesh/`. These bulky development artifacts are excluded from release archives. The score parity source is distributed as `fedora/tests/openmvs_score_check.cpp`; configure the patched OpenMVS with `-DODM_SCORE_CHECK_SOURCE=/absolute/path/to/that/file`, build `ODMScoreCheck`, then run it with the dependency environment's libraries available.

The active interfaces remain ODM at http://127.0.0.1:8767 and MapRepair at http://127.0.0.1:8768. New jobs pick up the updated native matcher and Python memory limits; restarting ODM also loads the updated desktop worker reporting. Do not interrupt an in-progress reconstruction solely to update its interface.
