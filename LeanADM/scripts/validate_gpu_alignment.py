"""Compare GPU alignment with saved successful correspondence sets."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from adm.gpu_alignment import solve
from adm.metal import get_backend
from adm.geometry import project


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='siyi');p.add_argument('--iterations',type=int,default=60);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];base=root/'output/comparison'/args.dataset
    data=np.load(base/'metal/registration.npz',allow_pickle=True);names=data['names'].tolist();edges=data['edges'].tolist()
    out=base/'gpu_alignment_validation';gpu=get_backend();t=time.perf_counter()
    width=json.loads((base/'adm_config.json').read_text())['registration_width']
    hs=solve(names,edges,out,width,gpu,args.iterations);elapsed=time.perf_counter()-t
    reference=np.load(base/'metal/transforms.npy')
    def residual(transforms):return np.concatenate([np.linalg.norm(project(transforms[e['i']],e['a'])-project(transforms[e['j']],e['b']),axis=1) for e in edges])
    a=residual(reference);b=residual(hs)
    report=dict(wall_seconds=elapsed,cpu_reference_median=float(np.median(a)),gpu_median=float(np.median(b)),cpu_reference_p95=float(np.percentile(a,95)),gpu_p95=float(np.percentile(b,95)),
                nonregression=bool(np.median(b)<=np.median(a)*1.01+.01 and np.percentile(b,95)<=np.percentile(a,95)*1.01+.01))
    (out/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['nonregression']:raise SystemExit('GPU alignment exceeded the accuracy tolerance')


if __name__=='__main__':main()
