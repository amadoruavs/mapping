#!/usr/bin/env bash
set -euo pipefail
adm_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
adm_hipcc="${HIPCC:-$(command -v hipcc || true)}"
[[ -n "$adm_hipcc" ]] || adm_hipcc=/opt/rocm/bin/hipcc
[[ -x "$adm_hipcc" ]] || { echo 'Install rocm-hip-devel first.' >&2; exit 1; }
# Strix Point / Ryzen AI 9 365. Override explicitly for a different AMD GPU.
"$adm_hipcc" -x hip -std=c++17 -O3 -ffp-contract=off -fno-fast-math \
  --offload-arch="${ADM_HIP_ARCH:-gfx1150}" -fPIC -shared -pthread \
  "$adm_root/hip.cpp" -o "$adm_root/libadm_hip.so.tmp"
mv "$adm_root/libadm_hip.so.tmp" "$adm_root/libadm_hip.so"
