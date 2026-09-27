"""Compare unchanged Ultra DSPSIFT arrays under serial and bounded scheduling."""
from pathlib import Path
import json, os, shutil, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ODM=ROOT/'ODM-Fedora'
sys.path[:0]=[str(ROOT/'acceleration'),str(ODM/'SuperBuild/install/bin/opensfm')]
from opensfm import dataset, features_processing as fp
from parallel_runtime.opensfm import extract
BASE=ROOT/'output/odm_macos_ultra94_20260916/flight/opensfm'
OUT=ROOT/'acceleration/validation/features';OUT.mkdir(parents=True,exist_ok=True)
paths=sorted(BASE.joinpath('image_list.txt').read_text().splitlines())[:4]
results={}
for variant in ('serial','parallel'):
 d=OUT/variant;d.mkdir(exist_ok=True)
 for name in ['camera_models.json','config.yaml']:shutil.copy2(BASE/name,d/name)
 shutil.copytree(BASE/'exif',d/'exif',dirs_exist_ok=True)
 (d/'image_list.txt').write_text('\n'.join(paths)+'\n')
 data=dataset.DataSet(str(d));data.config['processes']=1 if variant=='serial' else 4
 start=time.perf_counter()
 (fp.run_features_processing if variant=='serial' else extract)(data,data.images(),True)
 results[variant]={'seconds':time.perf_counter()-start,'images':len(data.images()),'feature_process_size':data.config['feature_process_size'],'minimum_features':data.config['feature_min_frames']}
 (OUT/'timing.json').write_text(json.dumps(results,indent=2))
a=dataset.DataSet(str(OUT/'serial'));b=dataset.DataSet(str(OUT/'parallel'));counts={}
for name in a.images():
 x=a.load_features(name);y=b.load_features(name)
 for attr in ('points','descriptors','colors'):
  if not np.array_equal(getattr(x,attr),getattr(y,attr)):raise AssertionError(f'{name}: {attr} differs')
 counts[name]=len(x.points)
results.update(passed=True,bit_identical_arrays=['points','descriptors','colors'],counts=counts)
(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
