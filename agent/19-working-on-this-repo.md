# Working on this repo

## Before you edit

1. Read [00-start-here.md](00-start-here.md) and [09-do-not-claim.md](09-do-not-claim.md).
2. Decide which lineage you are in: `MapRepair/`, `ISM/`, `LeanADM/`, or root `adm/`.
3. Do not "share a fix" across lineages unless you copy it on purpose. They do not import each other.

## Preferential edit sites

| Task | Edit |
|---|---|
| ISM flight defaults | `ISM/run.py` SETTINGS only, unless the user asked to change the engine |
| ISM solver / compositor | `ISM/runtime/adm/` + `ISM.metal` |
| LeanADM | only if the user wants the frozen backup changed |
| MapRepair geometry | `map_geo_checker.py`, `patch_blend.py`, `map_io.py` |
| MapRepair look | `adaptive_finish.py` / `run.py` FINISH_MODE |
| MapRepair UI | `web.py`, `web_ui/` |
| Docs for the next agent | `agent/` and a dated note at the **top** of `MEMORY.md` |

## After you change shaders

ISM: `python ISM/run.py --compile-only` then `--self-test`. The dylib is hashed from source; an already-running worker will **not** pick up a new shader until it restarts.

MapRepair: `python run.py --compile-only` builds `native.cpp` and OpenRT.

## Output

New empty directory. Keep the user's input JPEGs and any existing `final_test` / `web_runs` they still have.

## Claims in PRs / chat

Name the dataset, the photo count, the AGL, the solver, and whether interpolation was on. If you did not run it, say so.

## If you are tempted to add dependencies

MapRepair's web UI is stdlib HTTP + static files on purpose. Do not add Node. Do not add a cloud model. Adaptive cinema is offline.

ISM is macOS / Metal on purpose. A Radeon port is a new backend, not a compile flag.

## MEMORY.md protocol

The workspace rule still applies: read MEMORY.md, then recheck source. When you finish a real change, prepend a dated section with what ran, what did not, and hashes if you have them. Do not rewrite history in the old sections.
