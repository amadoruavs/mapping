"""Build and run the actual OpenRT library in an isolated C++ worker process."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent
COMMIT='ef6205f8ca51fd123da4a2f0b7b7a6ad27809b80'


class OpenRT:
    def __init__(self,threads=8):
        self.threads=int(threads)
        if not 1<=self.threads<=256:raise ValueError('OpenRT threads must be 1..256')
        build=ROOT/'build'/'openrt';suffix='.exe' if os.name=='nt' else ''
        self.binary=build/('maprepair_openrt'+suffix)
        if not self.binary.exists() and (build/'Release'/self.binary.name).exists():self.binary=build/'Release'/self.binary.name
        upstream=ROOT/'third_party'/'OpenRT'
        sources=[ROOT/'openrt'/'CMakeLists.txt',ROOT/'openrt'/'trace.cpp',upstream/'cmake'/'types.h.in']
        sources+=sorted(p for p in (upstream/'modules'/'core').glob('*') if p.suffix in ('.cpp','.h'))
        sources+=sorted((upstream/'include').glob('*.h'))
        digest=hashlib.sha256(COMMIT.encode())
        for p in sources:
            if p.is_file():digest.update(p.read_bytes())
        self.source_sha256=digest.hexdigest();stamp=build/'source.sha256'
        if self.binary.exists() and stamp.exists() and stamp.read_text()==self.source_sha256:return
        if not shutil.which('cmake'):raise RuntimeError('OpenRT needs CMake, a C++20 compiler and OpenCV 4 development libraries. See MapRepair/openrt/README.md.')
        build.mkdir(parents=True,exist_ok=True)
        configure=['cmake','-S',str(ROOT/'openrt'),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release']
        opencv=os.environ.get('OpenCV_DIR')
        if not opencv:
            for prefix in (Path('/opt/homebrew'),Path('/usr/local')):
                candidates=sorted((prefix/'Cellar'/'opencv@4').glob('*/lib/cmake/opencv4'))
                if candidates:opencv=str(candidates[-1]);break
        if opencv:configure.append('-DOpenCV_DIR='+opencv)
        print('[OpenRT] Compiling upstream C++ library and map adapter',flush=True)
        with (build/'build.log').open('w') as log:
            for command in (configure,['cmake','--build',str(build),'--config','Release','--parallel',str(min(self.threads,8))]):
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:raise RuntimeError(f'OpenRT build failed; see {build/"build.log"}')
        if not self.binary.exists() and (build/'Release'/self.binary.name).exists():self.binary=build/'Release'/self.binary.name
        stamp.write_text(self.source_sha256)

    def trace(self,vertices,map_shape,grid=128,elevation=8.,azimuth=255.,**unused):
        h,w=map_shape[:2];gh=max(1,round(grid*h/max(h,w)));gw=max(1,round(grid*w/max(h,w)))
        data=np.ascontiguousarray(vertices,dtype=np.float32)
        if data.ndim!=3 or data.shape[1:]!=(3,3) or len(data)<2 or not np.isfinite(data).all():raise ValueError('OpenRT requires ground-plane triangles first, then registered mesh triangles')
        if not 1<=grid<=2048 or not np.isfinite([elevation,azimuth]).all() or not 1<=elevation<=90:raise ValueError('Invalid OpenRT grid or sun angles')
        with tempfile.TemporaryDirectory(prefix='maprepair-openrt-') as td:
            src=Path(td)/'triangles.bin';dst=Path(td)/'light.bin';data.tofile(src)
            command=[str(self.binary),str(src),str(dst),str(w),str(h),str(gw),str(gh),str(elevation),str(azimuth),str(self.threads),str(len(data))]
            result=subprocess.run(command,text=True,capture_output=True)
            if result.returncode:raise RuntimeError('OpenRT trace failed: '+result.stderr[-2000:])
            field=np.fromfile(dst,dtype=np.float32).reshape(gh,gw,3)
            if not np.isfinite(field).all():raise RuntimeError('OpenRT returned invalid lighting')
            return field

    def description(self):
        return dict(backend='Project-10/OpenRT C++ Scene, BSP, diffuse shader, sun and sky ray tracing',
            upstream='https://github.com/Project-10/OpenRT',commit=COMMIT,source_sha256=self.source_sha256,
            sky_samples=16,normalization='Identical unoccluded ground plane',
            lighting='Direct diffuse sun plus sampled sky occlusion; no inferred scene geometry')


if __name__=='__main__':print(OpenRT().binary)
