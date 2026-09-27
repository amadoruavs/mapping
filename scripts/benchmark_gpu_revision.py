"""Fresh end-to-end run of the GPU alignment / coherent composite revision."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['siyi','archive','killerofmaps'])
    p.add_argument('--output',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--export-profile',choices=['png','vectors','diagnostic'],default='png')
    p.add_argument('--max-output-mb',type=int)
    args=p.parse_args()
    base=ROOT/'output'/('killerofmaps' if args.dataset=='killerofmaps' else 'comparison/'+args.dataset)
    output=(args.output or base/'metal_lean').resolve()
    if output.exists() and any(output.iterdir()):raise RuntimeError('Output is not empty; preserve it and choose a new benchmark directory')
    source=Path('/Users/avneh/Downloads/killerofmaps') if args.dataset=='killerofmaps' else base/'odm/images'
    config=args.config or base/('adm_config_128mb.json' if args.dataset=='killerofmaps' else 'adm_config.json')
    timing=output.with_name(output.name+'_timing.json');log_path=output.with_name(output.name+'.log')
    output.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(ROOT/'scripts/benchmark_process.py'),str(timing),sys.executable,'-m','adm.pipeline',str(source),
         '--output',str(output),'--config',str(config),'--gps-centers','--backend','metal','--gpu-optimized','--export-profile',args.export_profile]
    if args.export_profile=='diagnostic':cmd.extend(['--stylized-shadows','--max-output-mb','512'])
    elif args.max_output_mb:cmd.extend(['--max-output-mb',str(args.max_output_mb)])
    with log_path.open('w') as log:
        result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    print(timing.read_text())
    if result.returncode:print(log_path.read_text()[-6000:]);raise SystemExit(result.returncode)
    report=json.loads((output/'report.json').read_text());print(json.dumps(report,indent=2))
    assert not (output/'warps').exists() and not (output/'tiles').exists()
    assert not (output/'map_golden_linear.npy').exists()
    if args.export_profile!='diagnostic':
        assert not (output/'registration.npz').exists() and not (output/'coverage.png').exists()
        assert not (output/'source_ids.png').exists() and not (output/'vocabulary').exists()
    print('Saved bytes:',sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))


if __name__=='__main__':main()
