#!/usr/bin/env bash
# First run installs dependencies, builds the processors, then opens the local UI.
set -Eeuo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$APP_DIR"

fail() { printf '\nMapRepair: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nMapRepair setup stopped at line %s. Fix the error above, then rerun ./run.sh.\n" "$LINENO" >&2' ERR

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Usage: ./run.sh [--port 8765] [--no-browser] [--runs-dir /path/to/results]
       ./run.sh --setup-only

Sets up a private Python environment, compiles MapRepair and opens its local UI.
Linux: installs missing system dependencies with apt-get, dnf or pacman.
It may ask for your sudo password. Do not run the whole script with sudo.
First setup needs internet access. Later launches reuse the environment/builds.

Optional environment variables:
  MAPREPAIR_PYTHON=/path/to/python3   Select Python 3.10 or newer.
  MAPREPAIR_INSTALL_SYSTEM=0         Check dependencies without installing them.
  OpenCV_DIR=/path/to/opencv4/cmake   Use a custom OpenCV development install.

Drag in the map and source photos folder, click Run, then Download full JPEG.
HELP
  exit 0
fi

[[ -f web.py && -f web_worker.py && -f requirements.txt && -f third_party/OpenRT/modules/core/Scene.cpp ]] ||
  fail 'Copy the whole MapRepair folder, including third_party/OpenRT, not just run.sh.'

system="$(uname -s)"
machine="$(uname -m)"
# Homebrew OpenCV is keg-only; discover an existing install before invoking brew.
if [[ "$system" == Darwin && -z "${OpenCV_DIR:-}" ]]; then
  for opencv_prefix in /opt/homebrew/opt/opencv@4 /usr/local/opt/opencv@4 /opt/homebrew/opt/opencv /usr/local/opt/opencv; do
    if [[ -f "$opencv_prefix/lib/cmake/opencv4/OpenCVConfig.cmake" ]]; then
      export OpenCV_DIR="$opencv_prefix/lib/cmake/opencv4"
      break
    fi
  done
fi
PYTHON_BIN=""
pick_python() {
  local candidate fallback=""
  if [[ -n "${MAPREPAIR_PYTHON:-}" ]]; then
    "$MAPREPAIR_PYTHON" -c 'import sys; assert sys.version_info >= (3,10)' >/dev/null 2>&1 ||
      fail 'MAPREPAIR_PYTHON must point to a working Python 3.10+ interpreter.'
    PYTHON_BIN="$MAPREPAIR_PYTHON"
    return
  fi
  for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; assert sys.version_info >= (3,10)' >/dev/null 2>&1; then
      [[ -n "$fallback" ]] || fallback="$(command -v "$candidate")"
      if "$candidate" -c 'import venv, ensurepip' >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v "$candidate")"
        return
      fi
    fi
  done
  PYTHON_BIN="$fallback"
}

has_opencv() {
  if [[ -n "${OpenCV_DIR:-}" && -f "${OpenCV_DIR}/OpenCVConfig.cmake" ]]; then return 0; fi
  command -v pkg-config >/dev/null 2>&1 && pkg-config --atleast-version=4 opencv4
}

has_build_tools() {
  command -v cmake >/dev/null 2>&1 &&
    { command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1; }
}

as_admin() {
  if [[ "$(id -u)" == 0 ]]; then "$@";
  elif command -v sudo >/dev/null 2>&1; then sudo "$@";
  else fail "Administrator access is needed to install dependencies. Ask an admin to run: $*";
  fi
}

pick_python
needs_system=0
[[ -n "$PYTHON_BIN" ]] || needs_system=1
if [[ -n "$PYTHON_BIN" ]] && ! "$PYTHON_BIN" -c 'import venv, ensurepip' >/dev/null 2>&1; then needs_system=1; fi
has_build_tools || needs_system=1
has_opencv || needs_system=1

if [[ "$needs_system" == 1 ]]; then
  [[ "${MAPREPAIR_INSTALL_SYSTEM:-1}" != 0 ]] ||
    fail 'Missing dependencies: Python 3.10+ with venv/ensurepip, C++20 compiler, CMake, pkg-config, OpenCV 4 development libraries. Automatic installation is disabled.'
  printf '\nInstalling missing MapRepair system dependencies…\n'
  case "$system" in
    Linux)
      if command -v apt-get >/dev/null 2>&1; then
        as_admin apt-get update
        as_admin apt-get install -y --no-install-recommends python3 python3-venv python3-pip build-essential cmake pkg-config libopencv-dev xdg-utils
      elif command -v dnf >/dev/null 2>&1; then
        as_admin dnf install -y python3 python3-pip gcc-c++ cmake pkgconf-pkg-config opencv-devel xdg-utils
      elif command -v pacman >/dev/null 2>&1; then
        # Refuse a partial system upgrade; pacman's normal full-upgrade prompt
        # remains visible rather than silently upgrading an Arch installation.
        as_admin pacman -Syu --needed python python-pip base-devel cmake pkgconf opencv xdg-utils
      else
        fail 'Install Python 3.10+ with venv/pip, a C++20 compiler, CMake, pkg-config and OpenCV 4 development libraries, then rerun. Automatic setup supports apt-get, dnf and pacman.'
      fi
      ;;
    Darwin)
      command -v brew >/dev/null 2>&1 || fail 'On macOS, install Homebrew and Xcode Command Line Tools, then rerun.'
      xcrun --find clang++ >/dev/null 2>&1 || fail 'Run xcode-select --install, finish installation, then rerun.'
      brew install python@3.12 cmake pkg-config opencv@4
      PYTHON_BIN="$(brew --prefix python@3.12)/bin/python3.12"
      export OpenCV_DIR="$(brew --prefix opencv@4)/lib/cmake/opencv4"
      ;;
    *) fail 'run.sh supports Linux and macOS. On Windows, use python web.py with the dependencies installed.' ;;
  esac
  [[ "$system" != Linux ]] || pick_python
fi

[[ -n "$PYTHON_BIN" ]] || fail 'Install Python 3.10 or newer, or set MAPREPAIR_PYTHON.'
"$PYTHON_BIN" -c 'import venv, ensurepip' >/dev/null 2>&1 ||
  fail "This Python lacks venv/ensurepip: $PYTHON_BIN. Install its matching venv package, or set MAPREPAIR_PYTHON=/usr/bin/python3."
has_build_tools || fail 'A C++20 compiler and CMake must be available on PATH.'

# A copied virtualenv contains absolute paths and cannot be shared across OSes.
# Key it to this install location, base interpreter and platform; keep old
# environments intact instead of deleting anything from a copied folder.
env_key="$("$PYTHON_BIN" - "$APP_DIR" <<'PY'
import hashlib, platform, sys
print(hashlib.sha256(repr((sys.argv[1],sys.executable,sys.version,platform.platform())).encode()).hexdigest()[:12])
PY
)"
ENV_DIR="$APP_DIR/.runtime/python-${system}-${machine}-${env_key}"
mkdir -p "$APP_DIR/.runtime"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  printf '\nCreating MapRepair Python environment…\n'
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi
APP_PYTHON="$ENV_DIR/bin/python"
requirements_hash="$("$APP_PYTHON" -c 'import hashlib; print(hashlib.sha256(open("requirements.txt","rb").read()).hexdigest())')"
installed_hash=""
[[ ! -f "$ENV_DIR/.requirements.sha256" ]] || installed_hash="$(cat "$ENV_DIR/.requirements.sha256")"
if [[ "$installed_hash" != "$requirements_hash" ]] ||
   ! "$APP_PYTHON" -c 'import cv2,numpy,PIL,pyproj,rasterio' >/dev/null 2>&1; then
  printf '\nInstalling MapRepair Python packages…\n'
  "$APP_PYTHON" -m pip install --disable-pip-version-check --upgrade pip
  "$APP_PYTHON" -m pip install --disable-pip-version-check --prefer-binary -r requirements.txt
  "$APP_PYTHON" -c 'import cv2,numpy,PIL,pyproj,rasterio'
  printf '%s' "$requirements_hash" > "$ENV_DIR/.requirements.sha256"
fi

printf '\nChecking and compiling image processors…\n'
"$APP_PYTHON" -u run.py --compile-only
if [[ "${1:-}" == "--setup-only" ]]; then
  printf '\nSetup complete. Start the UI with ./run.sh\n'
  exit 0
fi
printf '\nStarting MapRepair…\n'
exec "$APP_PYTHON" -u web.py "$@"
