"""Run exactly the same Python integer math on Metal or ROCm; never fake a GPU pass."""
from pathlib import Path
import argparse, gc, json, sys, time
import numpy as np
from parallel_runtime.pixels import cpu_composite, torch_composite, torch_repeat2x, certified_device

def main():
 p=argparse.ArgumentParser();p.add_argument('--backend',choices=['metal','hip'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=100);a=p.parse_args()
 report={'requested_backend':a.backend,'passed':False};start=time.perf_counter()
 try:
  import torch
  device=certified_device(a.backend);report.update(torch_version=torch.__version__,hip_version=torch.version.hip,device=device)
  if device=='mps':torch.mps.synchronize();baseline=torch.mps.current_allocated_memory()
  else:torch.cuda.synchronize();baseline=torch.cuda.memory_allocated()
  reserved_before=torch.mps.driver_allocated_memory() if device=='mps' else torch.cuda.memory_reserved()
  reserved_peak=reserved_before
  rng=np.random.default_rng(365);pixels=0
  for i in range(a.iterations):
   h,w=[(1,1),(7,13),(255,257),(513,517),(720,1280)][i%5]
   image=rng.integers(0,256,(h,w,4),dtype=np.uint8);bg=rng.integers(0,256,3).tolist()
   alpha=image[:,:,3:4].astype(np.float32)/255
   legacy=(image[:,:,:3]*alpha+np.array(bg,np.float32)*(1-alpha)+.5).astype(np.uint8)
   reference=cpu_composite(image,bg);result=torch_composite(image,bg,device)
   if not np.array_equal(reference,legacy):raise AssertionError('Legacy compositing pixel mismatch')
   if not np.array_equal(reference,result):raise AssertionError('GPU compositing pixel mismatch')
   alpha2=torch_repeat2x(image[:,:,3],device)
   if not np.array_equal(alpha2,np.repeat(np.repeat(image[:,:,3],2,0),2,1)):raise AssertionError('GPU alpha resampling mismatch')
   pixels+=h*w
   reserved_peak=max(reserved_peak,torch.mps.driver_allocated_memory() if device=='mps' else torch.cuda.memory_reserved())
   if reserved_peak>reserved_before+256*1024**2:raise AssertionError('GPU cached/driver allocation growth exceeded 256 MiB stress allowance')
  gc.collect()
  if device=='mps':torch.mps.synchronize();live=torch.mps.current_allocated_memory()
  else:torch.cuda.synchronize();live=torch.cuda.memory_allocated()
  if live>baseline+16*1024**2:raise AssertionError('Live GPU allocations grew beyond 16 MiB')
  report.update(passed=True,iterations=a.iterations,source_pixels=pixels,max_pixel_error=0,device_live_bytes_before=baseline,device_live_bytes_after=live,device_reserved_bytes_before=reserved_before,device_reserved_peak_bytes=reserved_peak)
 except Exception as exc:report['error']=str(exc)
 report['seconds']=time.perf_counter()-start;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
