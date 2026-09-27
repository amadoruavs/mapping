"""Fresh old/new 94-photo benchmarks; no match caches or 306-photo inputs."""
import datetime
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ISM/benchmarks/odm_geo94'
INPUT=Path('/Users/avneh/Downloads/mapv2')


def main():
    photos=sorted(p for p in INPUT.iterdir() if p.suffix.lower() in ('.jpg','.jpeg'))
    if len(photos)!=94:raise ValueError('This benchmark is explicitly the 94-photo mapv2 batch')
    (OUT/'input_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in photos},indent=2)+'\n')
    cfg=json.loads((OUT/'old_config.json').read_text())
    # Reverse the first benchmark order to expose cache/order variation.
    for name,script,extra in [('new_repeat',ROOT/'ISM/run.py',{'geographic_solver':'opensfm_planar',
         'geographic_gps_sigma_m':10.,'geographic_altitude_sigma_m':10.,'geographic_orientation_sigma_deg':10.}),
        ('old_repeat',OUT/'source_old/run.py',{})]:
        folder=OUT/name
        folder.mkdir(parents=True,exist_ok=False)
        config=folder/'input_config.json';config.write_text(json.dumps({**cfg,**extra},indent=2)+'\n')
        command=[sys.executable,str(script),'--worker','--input',str(INPUT),'--output',str(folder),'--config',str(config)]
        print(f'Starting {name}; log {folder / "run.log"}',flush=True)
        with (folder/'run.log').open('w') as log:
            # Separate measurement process gives each child its own peak RSS.
            completed=subprocess.run([sys.executable,str(ROOT/'scripts/benchmark_process.py'),str(folder/'timing.json'),*command],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        report=json.loads((folder/'timing.json').read_text())
        print(f'{name}: {report["wall_seconds"]:.6f}s, exit {completed.returncode}',flush=True)
        if completed.returncode:raise RuntimeError((folder/'run.log').read_text()[-5000:])


if __name__=='__main__':main()
