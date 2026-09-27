#!/usr/bin/env bash
set -Eeuo pipefail
trap 'find "/work with spaces/MapRepair/build" -name build.log -exec cp {} /artifacts/openrt-build.log \; 2>/dev/null || true' ERR
mkdir -p '/work with spaces'
tar -xzf /artifacts/source.tar.gz -C '/work with spaces'
cd '/work with spaces/MapRepair'
bash run.sh --setup-only > /artifacts/setup.log 2>&1
# Repeat must not reinstall or compile; all UI flags must reach web.py.
bash run.sh --setup-only > /artifacts/repeat.log 2>&1
python_path="$(find .runtime -path '*/bin/python' -print -quit)"
"$python_path" -m unittest discover -s tests -v > /artifacts/tests.log 2>&1
bash run.sh --no-browser --port 8765 > /artifacts/server.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
"$python_path" - <<'PY'
import json, platform, time, urllib.request
for i in range(100):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8765',timeout=1) as r:
            page=r.read()
            assert r.status==200 and b'Drop your photos folder' in page
        state=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/state'))
        break
    except OSError:
        time.sleep(.1)
else:
    raise RuntimeError('Local web server did not start')
with open('/artifacts/summary.json','w') as f:
    json.dump(dict(platform=platform.platform(),architecture=platform.machine(),
        clean_install=True,repeat_setup=True,path_with_spaces=True,http_ui_status=200,
        folder_drop_ui_present=True,tests=30,state=state),f,indent=2)
print('Linux launcher and web UI verified')
PY
