#!/usr/bin/env bash
# Shared by the two source bundles. Uses Fedora's ROCm packages, not a replacement driver.
set -euo pipefail
adm_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$adm_root"
[[ "$(uname -s)" == Linux ]] || { echo 'Run AMD setup on the Fedora laptop.' >&2; exit 1; }
adm_mode="${1:-hip}"
case "$adm_mode" in hip|xdna) ;; *) echo 'Usage: ./setup-amd.sh [hip|xdna]' >&2; exit 2;; esac
adm_python="${ADM_PYTHON:-$adm_root/.pixi/envs/default/bin/python}"
if [[ ! -x "$adm_python" ]]; then
  for adm_candidate in "$adm_root"/.runtime/python-Linux-*/bin/python; do
    if [[ -x "$adm_candidate" ]]; then adm_python="$adm_candidate"; break; fi
  done
fi
[[ -x "$adm_python" ]] || { echo 'Run the application installer first.' >&2; exit 1; }
mkdir -p validation/target
if [[ "$adm_mode" == hip ]]; then
  if ! command -v hipcc >/dev/null && [[ ! -x /opt/rocm/bin/hipcc ]]; then
    sudo dnf install -y --setopt=install_weak_deps=False rocm-hip-devel
  fi
  bash parallel_runtime/kernels/build-hip.sh
  if ! timeout 300 "$adm_python" -m parallel_runtime.validate_native --backend native-hip --output validation/target/native-hip.json; then
    mv parallel_runtime/kernels/libadm_hip.so parallel_runtime/kernels/libadm_hip.so.failed
    echo 'HIP target checks failed. CPU fallback remains available.' >&2; exit 1
  fi
  if [[ -f fedora/finish/build-hip.sh ]]; then
    bash fedora/finish/build-hip.sh
    if ! timeout 300 "$adm_python" fedora/finish/validate.py --hip --output validation/target/fsr-hip.json; then
      mv fedora/finish/libfsr_hip.so fedora/finish/libfsr_hip.so.failed
      echo 'FSR HIP parity failed. CPU FSR remains available.' >&2; exit 1
    fi
  fi
  echo 'HIP compilation and target parity/stress checks passed. Auto mode also checks transfer-inclusive speed.'
else
  [[ "$("$adm_python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == 3.12 ]] || {
    echo 'The pinned IRON wheel needs Python 3.12. For MapRepair: sudo dnf install python3.12; MAPREPAIR_PYTHON=python3.12 ./run.sh --setup-only; then select that runtime with ADM_PYTHON=/absolute/path/to/its/bin/python ./setup-amd.sh xdna.' >&2; exit 1;
  }
  compgen -G '/dev/accel/accel*' >/dev/null || { echo 'No Linux NPU device. Install current Fedora kernel/firmware and reboot; see AMD_KERNELS.md.' >&2; exit 1; }
  [[ -f /opt/xilinx/xrt/setup.sh ]] || { echo 'XRT userspace is missing. Run ./setup-xdna-userspace.sh first (see AMD_KERNELS.md).' >&2; exit 1; }
  sudo dnf install -y --setopt=install_weak_deps=False llvm
  # Wheels are pinned to a published build whose IRON API was inspected.
  "$adm_python" -m pip install 'https://github.com/Xilinx/mlir-aie/releases/download/latest-wheels-4/mlir_aie-1.4.4.dev3%2Bgf50bef7-cp312-cp312-manylinux_2_35_x86_64.whl#sha256=02c5229aa6229cca9c70d694cf69a96a50b2a2c11f5fcecb855ea72926208366' \
    'llvm-aie==22.0.0.2026090701+3e93bf7b' -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
  set +u
  source /opt/xilinx/xrt/setup.sh
  set -u
  export PATH="$(dirname "$adm_python"):$PATH"
  timeout 60 xrt-smi examine
  timeout 900 "$adm_python" -m parallel_runtime.validate_native --backend xdna --iterations 10 --output validation/target/xdna.json
  echo 'XDNA2 device checks passed. Enable automatic performance-gated selection with ADM_ENABLE_XDNA=1; force only for testing with ADM_PIXEL_BACKEND=xdna.'
fi
