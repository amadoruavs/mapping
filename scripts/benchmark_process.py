"""Time an actual pipeline command; works inside Linux ODM and on macOS."""
import datetime
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


def main():
    output = Path(sys.argv[1]); command = sys.argv[2:]
    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.perf_counter()
    process = subprocess.run(command)
    elapsed = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = dict(command=command, start_utc=start_utc, wall_seconds=elapsed,
                  exit_code=process.returncode, cpu_user_seconds=usage.ru_utime,
                  cpu_system_seconds=usage.ru_stime,
                  peak_child_rss_bytes=usage.ru_maxrss*(1024 if sys.platform.startswith('linux') else 1),
                  minor_page_faults=usage.ru_minflt, major_page_faults=usage.ru_majflt,
                  voluntary_context_switches=usage.ru_nvcsw, involuntary_context_switches=usage.ru_nivcsw,
                  input_blocks=usage.ru_inblock, output_blocks=usage.ru_oublock,
                  rss_note='Maximum child process RSS; not the simultaneous sum of all worker processes')
    if sys.platform.startswith('linux'):
        for name in ('memory.peak', 'memory.max', 'memory.events', 'cpu.stat', 'io.stat'):
            p=Path('/sys/fs/cgroup')/name
            if p.exists(): report['cgroup_'+name]=p.read_text().strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    raise SystemExit(process.returncode)


if __name__ == '__main__': main()
