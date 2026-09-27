"""Read-only hardware/runtime diagnostic; a detected device is not a passed test."""
import argparse,glob,importlib.util,json,os,platform,shutil,sys
from pathlib import Path
from .scheduler import available_bytes,memory_budget

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
 report={'platform':platform.platform(),'machine':platform.machine(),'cpu_threads':os.cpu_count(),'available_bytes':available_bytes(),'working_budget_bytes':memory_budget(),'hip_compiler':shutil.which('hipcc'),'rocm_root_present':Path('/opt/rocm').exists(),'npu_device_nodes':glob.glob('/dev/accel/accel*'),'xdna_driver_present':Path('/sys/module/amdxdna').exists(),'npu_kernels_implemented':False,'torch_installed':importlib.util.find_spec('torch') is not None}
 if report['torch_installed']:
  try:
   import torch
   report.update(torch_version=torch.__version__,hip_version=torch.version.hip,metal_available=torch.backends.mps.is_available(),hip_available=bool(torch.version.hip and torch.cuda.is_available()))
  except Exception as exc:report['torch_error']=str(exc)
 text=json.dumps(report,indent=2)+'\n'
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
 print(text)
if __name__=='__main__':main()
