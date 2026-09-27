# Benchmark and profiling plan: measuring where ODM's time goes on the M4 Pro

Status: documented, **not executed**. Nothing is built in this checkout yet. This plan is the
CPU baseline that the Metal work must later beat, so it fixes datasets, flags, environment
capture and output format up front.

## 1. Datasets (pick three sizes)

Image counts were read from the GitHub tree API (`images/` folder); dimensions and camera
model from the first image of each repo (`sips -g pixelWidth -g pixelHeight -g model`):

| Repo | Images | Resolution (MP) | Camera | GPS | Size | Use |
|---|---|---|---|---|---|---|
| `OpenDroneMap/odm_data_benchmark` | 50 | 3648x2736 (10) | Sony DSC-HX5V | yes | small | smoke-level timing, quick A/B of flags |
| `pierotofy/drone_dataset_brighton_beach` | 18 | - | - | yes | 62 MB | the odm-benchmarks "small" reference (most rows) |
| `OpenDroneMap/odm_data_toledo` | 87 | 4000x3000 (12) | DJI FC300X (Phantom 3) | yes | 449 MB | **primary**: most-benchmarked mid-size set (Ryzen 9: 14m05s; M1 Pro Docker: 22m12s) |
| `OpenDroneMap/odm_data_aukerman` | 77 | 4896x3672 (18) | Sony DSC-WX220 | yes | 543 MB | **primary**: 18 MP stresses feature/depth resolution tables; many community numbers |
| `OpenDroneMap/odm_data_seneca` | 167 | 3240x2430 (8) | Canon ELPH 300 HS | yes | 358 MB | medium count, small images |
| `OpenDroneMap/odm_data_lewis` | 145 | 4000x3000 (12) | Canon S100 | yes | 610 MB | medium |
| `OpenDroneMap/odm_data_helenenschacht` | 176 | 5472x3648 (20) | Autel XT705 (Evo II Pro RTK) | RTK + GCP | 2.5 GB | **memory stress** on 24 GB at defaults |
| `OpenDroneMap/odm_data_waterbury` | 248 | 4896x3672 (18) | Sony DSC-WX220 | RTK | 1.9 GB | large-for-laptop; expect `--pc-quality low` or `--split` |
| `OpenDroneMap/odm_data_zoo` | 524 | 4896x3672 (18) | Sony DSC-WX220 | yes | 3.5 GB | overnight / split-merge only |
| `OpenDroneMap/odm_data_langley` | 200 | 4608x3456 (16) | Canon ELPH 130 IS | **no GPS** | 706 MB | BOW-matcher path, all-pairs cost |
| `OpenDroneMap/odm_data_copr` | 41 | - | - | GCP only | 150 MB | GCP path |
| `OpenDroneMap/odm_data_apt` | 76 | 5232x3488 (18) | Nikon 1 J4 | yes | - | alternative 18 MP |
| `pierotofy/drone_dataset_sheffield_park_1` (78), `_2` (54), `_3` (32), `drone_dataset_sheffield_cross` (172, GCP) | | | | | | odm-benchmarks references |

Notes: `odm_data_lafayette` and `odm_data_rectangular_grid` do not exist under the
OpenDroneMap org (404); Brighton Beach and Sheffield live under `pierotofy`. Index of all
datasets: https://github.com/OpenDroneMap/ODMData . Other rows in odm-benchmarks (Wietrznia
225, Shitan 493, Tuniu 271/297) are hosted off-GitHub.

Recommended trio: **Toledo (87, 12 MP)**, **Aukerman (77, 18 MP)**, **Helenenschacht (176,
20 MP)**. Toledo and Aukerman have the densest community comparison data; Helenenschacht
probes the 24 GB ceiling. Clone with `git clone --depth 1`; each repo has `images/` at the
root, which is what `--project-path <parent> <name>` expects.

## 2. How ODM records time (no flags needed)

* `--time` is in `DEPRECATED` (`opendm/config.py:919`): accepted, warned about, ignored.
  Timing is always on.
* `<project>/benchmark.txt`: one line per stage, `"<stage> runtime: <seconds> seconds"`
  (`opendm/system.py::benchmark`, `opendm/types.py:451`). Recreated by the `dataset` stage
  if it exists (`stages/dataset.py:53-60`); on `--rerun-from` the earlier stages' lines are
  therefore lost - copy the file after each run.
* `<project>/log.json` (`opendm/log.py`): `odmVersion, memory, cpus, images, options,
  startTime, stages[{name, startTime, messages[]}], processes[{command, exitCode, output
  tail}], success, endTime, totalTime`. Only the **last** stage gets `totalTime`; per-stage
  durations must be derived from consecutive `startTime`s. `processes[]` holds every
  `system.run` command line (with the exact `DensifyPointCloud`/`PoissonRecon`/`texrecon`
  arguments) in order, which is the easiest way to attribute sub-steps.
* Sub-step timings inside `opensfm`: OpenSfM writes `opensfm/reports/*.json`
  (`features.json`, `matches.json`, `tracks.json`, `reconstruction.json`) with wall times
  per action, and `opensfm/profile.log` when profiling is enabled in its config
  (UNVERIFIED for this fork; check `opensfm/reports/` after a run). `--optimize-disk-space`
  deletes `reports/`, so do not use it during benchmarking.
* `odm_report/stats.json` -> `odm_processing_statistics.total_time` (+ `_human`) and
  `average_gsd`; `report.pdf` shows total time only. `opendm/dem/commands.py` and
  `opendm/mesh.py` also log `Completed ... in <delta>` lines for DEM steps to stdout.
* Console: ODM prints `[INFO] running <cmd>` before each subprocess with a timestamp-free
  line; wrap the run in `ts` (moreutils) or `awk '{print strftime("%H:%M:%S"), $0}'` to get
  timestamps in the log.

## 3. Environment capture (once per session, scripted)

```bash
mkdir -p ~/odm-bench/$(date +%Y%m%d)/env && cd $_
sysctl hw.model machdep.cpu.brand_string hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.memsize > sysctl.txt
sw_vers > sw_vers.txt; pmset -g batt > power.txt; pmset -g | grep -i "lowpowermode\|highpowermode" >> power.txt
(cd /Users/avneh/Code/ADM/ODM && git rev-parse HEAD && cat VERSION && pixi info && pixi list --json > ~/odm-bench/pixi-list.json)
otool -L /Users/avneh/Code/ADM/ODM/SuperBuild/install/bin/{DensifyPointCloud,PoissonRecon,texrecon,odm_orthophoto,FPCFilter} > otool.txt
pixi run -- python -c "import cv2;print(cv2.getBuildInformation())" > opencv-build.txt
pixi run -- python -c "import numpy;numpy.show_config()" > numpy-config.txt
```

Record: plugged in, lid open, Energy Mode (High Power), no other heavy apps, free disk
before/after.

## 4. Run matrix

Each cell = one full run into a fresh copy of the dataset folder (ODM writes next to
`images/`), with `benchmark.txt`, `log.json`, `opensfm/reports/`, `odm_report/stats.json`
and the system samplers archived. Repeat the primary cells 2x to see variance (thermal state
matters; run the second repeat after a 10-minute idle).

| ID | Dataset | Flags | Purpose |
|---|---|---|---|
| T-def | Toledo | defaults + `--dsm` | baseline vs odm-benchmarks (14m05s on 5900X) |
| A-def | Aukerman | defaults + `--dsm` | baseline vs 15m52s (5900X), 9 min (i7-10700K) |
| A-fast | Aukerman | preset A from 03 | fast-ortho ratio |
| A-high | Aukerman | `--feature-quality ultra --pc-quality high --dsm --orthophoto-resolution 2 --dem-resolution 2` | "High Resolution" preset (36m15s on i7-10700K) |
| A-c8 / A-c12 | Aukerman | defaults, `--max-concurrency 8` vs `12` | P-only vs P+E |
| A-geo | Aukerman | defaults + `--pc-skip-geometric` | openmvs-only saving |
| A-blas | Aukerman | defaults with `libblas=*=*_newaccelerate` env | BLAS A/B (opensfm stage only matters) |
| A-mcpu | Aukerman | defaults, SuperBuild rebuilt with `-mcpu=apple-m4` | compiler A/B |
| H-def | Helenenschacht | defaults + `--dsm` | memory ceiling; watch for `--fusion-mode 1` fallback in log |
| H-low | Helenenschacht | `--pc-quality low --max-concurrency 8` | laptop-safe large set |
| L-def | Langley | defaults | no-GPS BOW path |

For the later Metal comparison the cells that matter are **A-def, A-high, H-def** (dense
reconstruction dominates) and **A-def with `--feature-type sift`** (the only feature path that
has a GPU implementation upstream).

## 5. System samplers during each run

Start these in a second terminal before `pixi run odm`, stop after `odm_postprocess`:

```bash
# power, frequency, thermal pressure, per-process P/E split; 5 s cadence, CSV-ish text
sudo powermetrics --samplers cpu_power,gpu_power,thermal --show-process-amp --show-plimits -i 5000 -o powermetrics.txt &
# memory: page-outs and compressor growth, 5 s cadence
vm_stat 5 > vm_stat.txt &
# which binary is running when (1 s cadence, cheap)
while sleep 1; do date +%T; ps -Ao pid,pcpu,rss,comm | egrep 'opensfm|DensifyPointCloud|ReconstructMesh|PoissonRecon|dem2mesh|texrecon|FPCFilter|renderdem|odm_orthophoto|pdal|gdal' ; done > ps.txt &
```

`powermetrics` needs root; `vm_stat`/`ps` do not. Note the local `powermetrics -h` on macOS
26.6 lists `--show-process-amp`, `--show-plimits`, `--show-cpu-qos`, `--show-process-energy`.
Per-stage aggregation: join `ps.txt` timestamps with `log.json` stage `startTime`s.

## 6. Profiling the hot binaries (after the baseline exists)

* **Sampling a running tool (no Xcode needed)**: `sample <pid|name> 30 1 -file out.txt`
  (30 s at 1 ms; `sample` is in `/usr/bin`, verified). Attach to `DensifyPointCloud`,
  `PoissonRecon`, `texrecon`, `odm_orthophoto` while ODM runs. The call graph shows whether
  time is in Eigen/OpenMVS patch-match, `libomp` barriers (`__kmp_*` = idle spin) or I/O.
* **Instruments** (`xctrace`): requires full Xcode (`xctrace` currently errors: "requires
  Xcode"; only CLT installed). With Xcode: `xctrace record --template 'Time Profiler'
  --time-limit 120s --output densify.trace --launch -- SuperBuild/install/bin/DensifyPointCloud
  <args from log.json>` or `--attach <pid>`; `xctrace export --toc`. Also the "CPU Counters"
  template for IPC on P vs E clusters (counter names for M4 UNVERIFIED).
* **Python (OpenSfM stage)**: `sudo py-spy record -o opensfm.svg --subprocesses --pid <pid
  of opensfm>` (py-spy needs root on macOS; not installed - `pip install py-spy` inside the
  pixi env or `brew install py-spy`). `sample` on the same pid shows the C++ side
  (`pybundle`, `pyfeatures`, `cv2`). For `reconstruct`, run it standalone:
  `pixi run -- SuperBuild/install/bin/opensfm/bin/opensfm reconstruct <project>/opensfm`
  with `python -X importtime` / `cProfile` if a pure-Python hotspot is suspected.
* **Single-tool micro-benchmarks**: replay the exact command lines from `log.json`
  `processes[]` with `hyperfine --warmup 1 --runs 3 '<cmd>'` (`brew install hyperfine`;
  not installed). This isolates `DensifyPointCloud` (vary `--max-threads 4/8/12`,
  `--resolution-level`), `PoissonRecon` (`--threads`, `--depth 10/11/12`), `texrecon`
  (`--num_threads`) from the Python glue. Use `--prepare 'rm -rf <outputs>'` so each run
  starts clean. Because inputs are on disk after one ODM run, this is cheap.
* **OpenMP scaling curve**: `OMP_NUM_THREADS=1,2,4,8,12` for `odm_orthophoto`/`renderdem`
  (they read the env var directly, `stages/odm_orthophoto.py:110`, `opendm/dem/commands.py:95`)
  and `--max-threads` for OpenMVS. Speedup vs threads tells whether E-cores help or hurt.
* **Memory high-water**: `/usr/bin/time -l <cmd>` prints "maximum resident set size" on
  macOS; `footprint <pid>` (present in `/usr/bin`) gives compressed/dirty breakdown of a
  live process. Record both for `DensifyPointCloud` and `PoissonRecon`.
* **I/O**: `fs_usage -w -f filesys <pid>` (root) if `texrecon` or the undistort step look
  IO-bound; the undistorted TIFs for 176 images at 5472 px are ~10 GB.

## 7. Results table template

One row per run; keep the raw files next to it.

```
| run id | date | dataset | images | flags | max-conc | total (s) | dataset | opensfm | openmvs | filterpoints | meshing | mvs_texturing | georef | dem | orthophoto | report | peak RSS (GB) | pageouts (delta) | thermal (max level) | avg CPU pkg W | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-def-1 | | aukerman | 77 | --dsm | 12 | | | | | | | | | | | | | | | | |
```

Derive stage columns from `benchmark.txt`; `total` from `log.json.totalTime`; `peak RSS`
from `ps.txt` (max RSS per binary) or `/usr/bin/time -l` replays; `pageouts` from the
first/last `vm_stat` lines; thermal and watts from `powermetrics.txt`.

Secondary quality columns (so speedups are not bought with failures): reconstructed shots /
total (`odm_report/stats.json` -> `reconstruction_statistics`), mean reprojection error,
dense point count (`point_cloud_statistics`), orthophoto pixel count.

## 8. Comparing against community CPU numbers

* odm-benchmarks (https://github.com/OpenDroneMap/odm-benchmarks, `data/benchmarks.csv`)
  records only total time with columns DATASET, PROCESSING_TIME, RAM, CPU, CORES, OS,
  VM_TYPE, ODM_VERSION, CONFIG_NAME, CONFIG_RESIZE, TEST_DATE. Match on dataset + "default"
  config; the closest hardware rows are Ryzen 9 5900X (Toledo 14m05s, Aukerman 15m52s,
  ODM 3.5.5) and Ryzen 7 8845HS laptop / 16 GB (Toledo 27 min, Aukerman 34 min, ODM 3.5.4).
  Apple rows are Docker-only and old (M1 Pro/ODM 2.9.2 Toledo 22m12s). A native M4 Pro row
  would be new information for the community; the repo accepts PRs to the CSV.
* ODM version drift: those rows are 3.5.x (OpenMVS 355 branch, same as now) but different
  OpenSfM/Ceres builds; expect +/-20 % noise from version alone. Use the *ratios*
  (fast/default, high/default, per-stage share) rather than absolute seconds when comparing
  across machines.
* The per-stage split has no community reference; the numbers produced here become the
  reference. Publish `benchmark.txt` alongside any claim.

## 9. What the Metal track needs from this baseline

* `A-def`, `A-high`, `H-def` stage times for `openmvs` with the exact `DensifyPointCloud`
  command line (from `log.json`), plus peak RSS - the GPU port replaces this call and must
  report the same metric.
* `opensfm` sub-step split (features vs matching vs reconstruct) from `opensfm/reports/`,
  because only feature extraction has a GPU precedent (PopSift) and matching/BA will remain
  CPU; the ceiling on the whole-pipeline speedup follows from Amdahl's law on these numbers.
* `powermetrics` GPU power at idle during CPU runs (it should be ~0 W; a nonzero baseline
  means something else is using the GPU) so later GPU runs can attribute watts.
* Thermal steady state: if the CPU already runs thermally limited, moving work to the GPU
  will also raise the CPU clocks; account for that in the comparison.

## Sources

* ODM code: `opendm/config.py` (`DEPRECATED` list), `opendm/log.py`, `opendm/system.py`,
  `opendm/types.py`, `stages/dataset.py`, `stages/odm_report.py`, `stages/run_opensfm.py`,
  `stages/openmvs.py`, `stages/odm_orthophoto.py`, `opendm/dem/commands.py` (commit 1a71341)
* Dataset index: https://github.com/OpenDroneMap/ODMData ; repos under
  https://github.com/OpenDroneMap (image counts via `gh api repos/<org>/<repo>/git/trees/HEAD?recursive=1`)
* odm-benchmarks: https://github.com/OpenDroneMap/odm-benchmarks ,
  https://raw.githubusercontent.com/OpenDroneMap/odm-benchmarks/master/data/benchmarks.csv ,
  https://raw.githubusercontent.com/OpenDroneMap/odm-benchmarks/master/data-parsed/by-dataset.md
* Community benchmark/profiling threads: https://community.opendronemap.org/t/odm-benchmarks-update/10367 ,
  https://community.opendronemap.org/t/profiling-opensfm/9179 ,
  https://community.opendronemap.org/t/processing-time-differences/15255
* macOS tools: local `sample`, `powermetrics -h`, `xctrace` (needs Xcode), `vm_stat`,
  `memory_pressure`, `footprint`, `fs_usage` (2026-09-13); man pages:
  https://keith.github.io/xcode-man-pages/sample.1.html ,
  https://keith.github.io/xcode-man-pages/xctrace.1.html ,
  https://ss64.com/mac/powermetrics.html , https://fleetdm.com/tables/macos_thermal_pressure
* py-spy on macOS (root required): https://raw.githubusercontent.com/benfred/py-spy/master/README.md
* hyperfine: https://github.com/sharkdp/hyperfine
* Instruments from the CLI: https://www.jviotti.com/2024/01/29/using-xcode-instruments-for-cpp-cpu-profiling.html
