"""Ordered, bounded task dispatch; no unbounded executor input/result queue."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
from pathlib import Path
import subprocess
import sys

MIB=1024**2
GIB=1024**3

def available_bytes():
    try:
        import psutil
        available=int(psutil.virtual_memory().available)
    except ImportError:
        if sys.platform.startswith('linux'):
            values={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
            available=values.get('MemAvailable',values.get('MemFree',256*MIB))
        elif sys.platform=='darwin':
            import re
            text=subprocess.check_output(['vm_stat'],text=True)
            page=int(re.search(r'page size of (\d+)',text).group(1))
            counts={k:int(v) for k,v in re.findall(r'([^\n:]+):\s+(\d+)\.',text)}
            available=page*sum(counts.get(k,0) for k in ['Pages free','Pages inactive','Pages speculative'])
        else:
            available=512*MIB
    # Account for a cgroup-v2 limit where visible at the current process path.
    if sys.platform.startswith('linux'):
        try:
            relative=next(line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
            base=Path('/sys/fs/cgroup')/relative.lstrip('/')
            while True:
                limit=(base/'memory.max').read_text().strip()
                if limit!='max':available=min(available,max(0,int(limit)-int((base/'memory.current').read_text())))
                if base==Path('/sys/fs/cgroup'):break
                base=base.parent
        except (OSError,ValueError,StopIteration):pass
    return max(0,available)

def memory_budget():
    ceiling=int(os.environ.get('ADM_MEMORY_MB','20480'))*MIB
    if ceiling<=0:raise ValueError('ADM_MEMORY_MB must be positive')
    return min(ceiling,int(available_bytes()*.60))

def worker_count(requested=None):
    if os.environ.get('ADM_PARALLEL','1')=='0':return 1
    hardware=os.cpu_count() or 1
    limit=hardware if requested is None else max(1,int(requested))
    default=min(4,hardware) if requested is None else limit
    configured=int(os.environ.get('ADM_PARALLEL_WORKERS',str(default)))
    return max(1,min(configured,limit,hardware))

def bounded_map(function, items, *, workers=None, bytes_per_item=1, budget=None, stats=None, ordered=True):
    """Yield in input order; reserve estimated scratch AND returned result memory.

    At most `workers` futures are retained. An oversized task is rejected before
    submission. On failure/early close, cancel queued jobs and join active jobs.
    Estimates must include native scratch; this does not intercept native malloc.
    Set ordered=False only when tasks/results are independent: completed work
    then frees a slot immediately instead of waiting behind the oldest task.
    """
    workers=worker_count(workers);budget=memory_budget() if budget is None else int(budget)
    if budget<1:raise MemoryError('No working-memory budget available')
    if stats is None:stats={}
    stats.update(peak_reserved_bytes=0,peak_tasks=0,submitted=0,released=0)
    pending=deque();source=iter(items);reserved=0;next_item=None;has_next=False;exhausted=False
    pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='adm')
    try:
        while pending or not exhausted or has_next:
            while len(pending)<workers and not exhausted:
                if not has_next:
                    try:next_item=next(source);has_next=True
                    except StopIteration:exhausted=True;break
                cost=bytes_per_item(next_item) if callable(bytes_per_item) else bytes_per_item
                cost=int(cost)
                if cost<1:raise ValueError('Task memory estimate must be positive')
                if cost>budget:raise MemoryError(f'Task needs estimated {cost/MIB:.1f} MiB; budget is {budget/MIB:.1f} MiB')
                if reserved+cost>budget:break
                future=pool.submit(function,next_item);pending.append((future,cost));reserved+=cost
                next_item=None;has_next=False;stats['submitted']+=1
                stats['peak_reserved_bytes']=max(stats['peak_reserved_bytes'],reserved)
                stats['peak_tasks']=max(stats['peak_tasks'],len(pending))
            if pending:
                if ordered:
                    future,cost=pending.popleft()
                else:
                    done,_=wait([f for f,_ in pending],return_when=FIRST_COMPLETED)
                    future,cost=next(pair for pair in pending if pair[0] in done)
                    pending.remove((future,cost))
                try:yield future.result()
                finally:reserved-=cost;stats['released']+=1;del future
    finally:
        for future,_ in pending:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        stats['reserved_at_close']=0
