#!/usr/bin/env bash
set -euo pipefail
repair_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$repair_root"
repair_python="${MAPREPAIR_PYTHON:-}"
if [[ -z "$repair_python" ]]; then
 for candidate in "$repair_root"/.runtime/python-"$(uname -s)"-*/bin/python; do
  if [[ -x "$candidate" ]]; then repair_python="$candidate"; break; fi
 done
fi
[[ -x "$repair_python" ]] || { echo 'Run ./run.sh --setup-only first, or set MAPREPAIR_PYTHON.' >&2; exit 1; }
mkdir -p validation/target
"$repair_python" -m parallel_runtime.doctor --output validation/target/hardware.json
"$repair_python" -m unittest parallel_runtime.test_scheduler parallel_runtime.test_pixels parallel_runtime.test_native
"$repair_python" -m unittest discover -s tests
case "${1:-cpu}" in
 cpu) ;;
 native-hip|xdna)
  [[ "$1" != native-hip ]] || bash parallel_runtime/kernels/build-hip.sh
  "$repair_python" -m parallel_runtime.validate_native --backend "$1" --output "validation/target/$1.json"
  ;;
 metal|hip) "$repair_python" -m parallel_runtime.validate --backend "$1" --output "validation/target/pixels-$1.json" ;;
 *) echo 'Usage: ./verify-acceleration.sh [cpu|metal|hip|native-hip|xdna]' >&2; exit 2 ;;
esac
