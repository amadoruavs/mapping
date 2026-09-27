#!/usr/bin/env bash
# Whole-process-tree limit, in addition to the Python scheduler's work budget.
set -euo pipefail
adm_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ "$(uname -s)" == Linux ]] || { echo 'This memory-limited launcher uses Linux systemd.' >&2; exit 1; }
command -v systemd-run >/dev/null || { echo 'systemd-run is required; use the normal launcher without a hard memory limit.' >&2; exit 1; }
adm_launch="$adm_root/run-fedora.sh"
[[ -f "$adm_launch" ]] || adm_launch="$adm_root/run.sh"
exec systemd-run --user --scope -p MemoryHigh=70% -p MemoryMax=80% -p MemorySwapMax=1G \
  env ADM_MEMORY_MB="${ADM_MEMORY_MB:-18000}" "$adm_launch" "$@"
