"""Bounded-allocation and injected-failure stress; does not deliberately OOM the OS."""
import gc,json,resource,time
from pathlib import Path
import numpy as np
from parallel_runtime.scheduler import bounded_map,MIB
start=time.perf_counter();stats={};checksum=0
before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
for i,value in enumerate(bounded_map(lambda i:np.full((1024,1024),i%251,np.float32),range(1000),workers=8,bytes_per_item=8*MIB,budget=24*MIB,stats=stats)):
 if not np.all(value==i%251):raise AssertionError('Parallel data corruption')
 checksum+=int(value[0,0])
del value;gc.collect()
assert stats['peak_reserved_bytes']<=24*MIB and stats['peak_tasks']<=3
assert stats['submitted']==stats['released']==1000
injected=False
def fail(i):
 if i==5:raise MemoryError('Injected native allocation failure')
 return np.zeros((256,256),np.uint8)
try:list(bounded_map(fail,range(100),workers=4,bytes_per_item=MIB,budget=4*MIB))
except MemoryError:injected=True
assert injected
report=dict(passed=True,iterations=1000,working_budget_bytes=24*MIB,stats=stats,injected_allocation_failure_propagated=True,peak_rss_native_units_before=before,peak_rss_native_units_after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.perf_counter()-start,scope='Python scheduler allocations; not a proof of native ODM global memory bounds')
p=Path(__file__).parent/'validation/memory_stress.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
