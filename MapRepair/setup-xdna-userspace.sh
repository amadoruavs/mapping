#!/usr/bin/env bash
# Build only XRT userspace and its SHIM against the existing Fedora amdxdna driver.
set -euo pipefail
adm_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source /etc/os-release
[[ "${ID:-}" == fedora && "${VERSION_ID:-0}" -ge 44 ]] || { echo 'This pinned XRT build requires Fedora 44 or newer.' >&2; exit 1; }
compgen -G '/dev/accel/accel*' >/dev/null || { echo 'No NPU device; update Fedora kernel/firmware and reboot first.' >&2; exit 1; }
adm_python="${ADM_PYTHON:-$adm_root/.pixi/envs/default/bin/python}"
[[ -x "$adm_python" ]] || { echo 'Set ADM_PYTHON to the application Python 3.12 executable.' >&2; exit 1; }
[[ "$("$adm_python" -c 'import sys;print(sys.version_info[:2]==(3,12))')" == True ]] || { echo 'Python 3.12 is required.' >&2; exit 1; }
adm_jobs="${ODM_BUILD_JOBS:-2}"
[[ "$adm_jobs" =~ ^[1-8]$ ]] || { echo 'ODM_BUILD_JOBS must be 1..8.' >&2; exit 1; }
sudo dnf install -y --setopt=install_weak_deps=False git git-lfs cmake gcc-c++ make rpm-build
adm_source="$adm_root/.local-tools/xdna-driver"
mkdir -p "$adm_root/.local-tools"
if [[ ! -d "$adm_source/.git" ]]; then
  git clone --no-checkout https://github.com/amd/xdna-driver.git "$adm_source"
fi
git -C "$adm_source" checkout --detach d715319ca8c4db291542abbce7cd850b9c2a1b57
git -C "$adm_source" lfs install --local
git -C "$adm_source" submodule update --init --recursive
sudo bash "$adm_source/tools/amdxdna_deps.sh"
adm_prefix="$("$adm_python" -c 'import sys;print(sys.prefix)')"
cd "$adm_source/xrt/build"
./build.sh -npu -opt -disable-werror -j "$adm_jobs" \
  -cmake-flags "-DPython3_EXECUTABLE=$adm_python -DPython3_ROOT_DIR=$adm_prefix"
sudo cmake --install Release
set +u
source /opt/xilinx/xrt/setup.sh
set -u
cd "$adm_source/build"
./build.sh -release -nokmod -j "$adm_jobs"
sudo cmake --install Release
sudo ldconfig
timeout 60 /opt/xilinx/xrt/bin/xrt-smi examine
echo 'XRT userspace installed. Now run ./setup-amd.sh xdna from the application folder.'
