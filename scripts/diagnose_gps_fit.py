"""Reproduce registration only; save matches/transforms for GPS-fit diagnosis."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ISM/runtime'))
import cv2
import numpy as np
from adm.config import Config
from adm.geometry import gps_center_neighbors
from adm.metadata import read_metadata
from adm.registration import collect_matches, solve
from adm.selection import select_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse', type=Path, help='Use saved verified matches; skip photo processing')
    parser.add_argument('--symmetric', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new diagnostic directory')
    args.output.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location('ism_settings', ROOT / 'ISM/run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = Config(**runner.SETTINGS)
    cfg.export_profile = 'png'
    cfg.gpu_alignment = cfg.coherent_composite = cfg.compact_outputs = True
    cfg.alignment_symmetric_transfer = args.symmetric
    cfg.metal_recompute_warps = True
    cfg.validate()
    cv2.setNumThreads(6)
    cv2.setRNGSeed(cfg.seed)
    from adm.metal import get_backend
    gpu = get_backend()
    start = time.perf_counter()
    if args.reuse:
        from adm.metadata import FrameMetadata
        rows=json.loads((args.reuse/'metadata.json').read_text())
        metadata=[FrameMetadata(**{k:v for k,v in r.items() if k in FrameMetadata.__dataclass_fields__}) for r in rows]
        names=[m.name for m in metadata]
    else:
        names, selection = select_frames(args.input, cfg, {})
        metadata = [read_metadata(args.input / name, cfg)[0] for name in names]
    (args.output / 'metadata.json').write_text(json.dumps([m.to_dict() for m in metadata], indent=2))
    (args.output / 'config.json').write_text(json.dumps(cfg.to_dict(), indent=2))
    pairs, earth = gps_center_neighbors(metadata, cfg)
    print(f'Diagnostic registration only: {len(names)} photos, {len(pairs)} candidate pairs', flush=True)
    if args.reuse:
        with np.load(args.reuse/'registration.npz',allow_pickle=True) as cache:edges=cache['edges'].tolist()
    else:edges = collect_matches(args.input, names, metadata, pairs, cfg, args.output, gpu=gpu, save_cache=True)
    from adm.gpu_alignment import solve as solve_gpu
    fitted=solve_gpu(names, edges, args.output, cfg.registration_width, gpu=gpu,
          iterations=cfg.alignment_max_iterations, workspace_budget_mb=cfg.alignment_budget_mb,symmetric=args.symmetric)
    native=[fitted[i]@np.diag([cfg.registration_width/m.width,round(m.height*cfg.registration_width/m.width)/m.height,1.]) for i,m in enumerate(metadata)]
    from adm.pipeline import align_centers_to_geography
    align_centers_to_geography(native,metadata,earth,cfg,args.output)
    print(f'Diagnostic registration completed in {time.perf_counter()-start:.3f}s', flush=True)


if __name__ == '__main__':
    main()
