"""Fail-closed target checks. Host source math, HIP and XDNA2 are distinct tests."""
import argparse, gc, json, os, resource, sys, time, unittest
from pathlib import Path
import numpy as np
from . import native
from .pixels import cpu_composite
from .test_native import NativeMathTests, flat_cases

def rss():
    # Live resident memory, rather than the monotonically increasing high water.
    if sys.platform.startswith('linux'):
        return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform=='darwin' else peak*1024

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['host','native-hip','xdna'],default='host')
    p.add_argument('--iterations',type=int,default=40);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=dict(backend=a.backend,passed=False,hardware_execution=False)
    start=time.perf_counter()
    try:
        if a.iterations<1:raise ValueError('Iterations must be positive')
        if a.backend=='host':
            result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeMathTests))
            if not result.wasSuccessful():raise RuntimeError('Host source math tests failed')
            report['tests']=result.testsRun
        else:
            native.certify(a.backend)
            report["hardware_execution"]=True
            if a.backend=='native-hip':
                # Compile the actual CPU classifier and compare every decision.
                NativeMathTests.setUpClass()
                try:
                    lib=native.hip_library()
                    for name,d,c in flat_cases():
                        h,w=d.shape;cpu=np.empty(((h+31)//32,(w+31)//32),np.uint8);gpu=np.empty_like(cpu)
                        NativeMathTests.lib.flat(d.ctypes.data,c.ctypes.data,w,h,cpu.ctypes.data)
                        if lib.adm_flat_tiles(d.ctypes.data,c.ctypes.data,w,h,gpu.ctypes.data):raise RuntimeError(lib.adm_hip_error().decode())
                        np.testing.assert_array_equal(cpu,gpu,err_msg=name)
                finally:NativeMathTests.tearDownClass()
            rng=np.random.default_rng(365);times=[];pixels=0;after_warmup=None;peak=0
            # Fixed backgrounds bound the XDNA JIT cache during stress.
            for i in range(a.iterations+5):
                h,w=[(1,1),(7,13),(255,257),(513,517),(2160,3840)][i%5]
                src=rng.integers(0,256,(h,w,4),dtype=np.uint8);bg=[17,128,241]
                expected=cpu_composite(src,bg)
                t=time.perf_counter();actual=native.run(src,bg,a.backend);times.append(time.perf_counter()-t)
                np.testing.assert_array_equal(actual,expected);pixels+=h*w
                del src,expected,actual
                if a.backend=='native-hip' and i%5==4:
                    # Stress maximum classifier allocation and irregular right/bottom tiles.
                    dh,dw=(4096,4096) if i%10==4 else (2173,4093)
                    d=np.full((dh,dw),100,np.float32);c=np.full_like(d,.01)
                    cpu=np.empty(((dh+31)//32,(dw+31)//32),np.uint8);gpu=np.empty_like(cpu)
                    NativeMathTests.lib.flat(d.ctypes.data,c.ctypes.data,dw,dh,cpu.ctypes.data)
                    if lib.adm_flat_tiles(d.ctypes.data,c.ctypes.data,dw,dh,gpu.ctypes.data):raise RuntimeError(lib.adm_hip_error().decode())
                    np.testing.assert_array_equal(cpu,gpu)
                    del d,c,cpu,gpu
                gc.collect()
                current=rss();peak=max(peak,current)
                if i==4:after_warmup=current
                if i>4 and current>after_warmup+256*1024**2:raise MemoryError('Resident growth exceeds 256 MiB after warmup')
            report.update(iterations=a.iterations,pixels=pixels,max_pixel_error=0,
                          rss_after_warmup=after_warmup,rss_peak=peak,rss_final=rss(),
                          median_call_seconds=float(np.median(times[5:])))
        report['passed']=True
    except Exception as exc:report['error']=str(exc)
    report['seconds']=time.perf_counter()-start
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
