"""Rerun the corner threshold experiment on an existing GPS-center dataset.

Re-extract features and rematch the original photographs with only the delta-RMSE
threshold changed. Report separately whether corner observations can actually
estimate homographies. GPS-center mode uses SIFT to predict corner search regions;
accepted corners in this mode are consistency checks, not independent discovery.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from adm.config import Config
from adm.geometry import gps_center_neighbors
from adm.metadata import read_metadata
from adm.registration import collect_matches


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threshold',type=float,default=.10)
    args=parser.parse_args(); started=time.perf_counter()
    if not 0<args.threshold<=1: raise ValueError('Threshold must be in (0,1]')
    baseline=args.baseline.resolve(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    report=json.loads((baseline/'report.json').read_text())
    if report['geographic_alignment_mode']!='GPS camera centers': raise ValueError('This replay is specifically for the GPS-center matching mode')
    previous=json.loads((baseline/'resolved_config.json').read_text())
    previous.pop('camera_spec',None); old_threshold=previous['delta_rmse_threshold']
    cfg=Config(**{**previous,'delta_rmse_threshold':args.threshold}).validate()
    names=json.loads((baseline/'map_geometry.json').read_text())['input_names']
    inventory={r['name']:r for r in json.loads((baseline/'selection.json').read_text())}
    for name in names:
        actual=hashlib.sha256((args.source/name).read_bytes()).hexdigest()
        if actual!=inventory[name]['sha256']: raise ValueError(f'Input changed since baseline: {name}')
    metadata=[read_metadata(args.source/name,cfg)[0] for name in names]
    (out/'run_config.json').write_text(json.dumps(cfg.to_dict(),indent=2))
    cv2.setNumThreads(6); cv2.setRNGSeed(cfg.seed)
    pairs,_=gps_center_neighbors(metadata,cfg)
    edges=collect_matches(args.source,names,metadata,pairs,cfg,out)
    results=json.loads((out/'corner_matches.json').read_text())
    all_patches=[patch for pair in results for patch in pair['corners']]
    evaluated=[patch for patch in all_patches if 'delta_rmse' in patch]
    counts=[sum(p['accepted'] for p in pair['corners']) for pair in results]
    accepted_pairs=[pair for pair in results if any(p['accepted'] for p in pair['corners'])]
    hist=Counter(counts)
    # Distinguish the threshold's effect from the unchanged other quality gates.
    def gates(threshold):
        delta=[p for p in evaluated if p['delta_rmse']<=threshold]
        correlation=[p for p in delta if p['ncc']>=cfg.corner_min_ncc]
        unique=[p for p in correlation if p['peak_margin']>.02]
        return {'delta_rmse_threshold':threshold,'evaluated':len(evaluated),'delta_pass':len(delta),'delta_and_correlation_pass':len(correlation),'all_quality_gates_pass':len(unique)}
    base=np.load(baseline/'registration.npz',allow_pickle=True)['edges'].tolist()
    by_pair={(e['i'],e['j']):e for e in base}
    unchanged=len(base)==len(edges) and all((e['i'],e['j']) in by_pair and np.array_equal(e['H'],by_pair[(e['i'],e['j'])]['H']) and np.array_equal(e['a'],by_pair[(e['i'],e['j'])]['a']) and np.array_equal(e['b'],by_pair[(e['i'],e['j'])]['b']) for e in edges)
    summary={'threshold_only_change':True,'baseline_threshold':old_threshold,'new_threshold':args.threshold,'input_images_used':len(names),'candidate_pairs':len(pairs),'registered_pairs':len(edges),'corner_observations_total':len(all_patches),'baseline_gates_on_current_matches':gates(old_threshold),'new_gates':gates(args.threshold),'accepted_corner_count_histogram_by_pair':{str(k):v for k,v in sorted(hist.items())},'pairs_with_four_accepted_corners':sum(n>=4 for n in counts),'images_with_any_accepted_corner':len(set(name for pair in accepted_pairs for name in [pair['source'],pair['target']])),'registration_arrays_identical_to_baseline':unchanged,'map_rerendered':False,'unchanged_baseline_map':str(baseline/'map_golden.png') if unchanged else None,'mode':'SIFT-predicted corner consistency checks; no independent geographic validation','seconds':time.perf_counter()-started}
    (out/'comparison.json').write_text(json.dumps(summary,indent=2))
    (out/'accepted_pairs.json').write_text(json.dumps(accepted_pairs,indent=2))
    # Visual review of each accepted match at full-resolution patch detail.
    cards=[]
    for pair in accepted_pairs:
        a=cv2.imread(str(args.source/pair['source'])); b=cv2.imread(str(args.source/pair['target']))
        for patch in pair['corners']:
            if not patch['accepted']: continue
            pa=cv2.getRectSubPix(a,(50,50),tuple(map(float,patch['source'])))
            pb=cv2.getRectSubPix(b,(50,50),tuple(map(float,patch['target'])))
            card=np.full((340,660,3),245,np.uint8)
            for x,im in [(20,pa),(350,pb)]: card[60:310,x:x+250]=cv2.resize(im,(250,250),interpolation=cv2.INTER_NEAREST)
            cv2.putText(card,f"{pair['source']} -> {pair['target']} | corner {patch['corner']}",(20,22),0,.55,(0,0,0),1)
            cv2.putText(card,f"delta RMSE {patch['delta_rmse']:.4f}, NCC {patch['ncc']:.4f}",(20,46),0,.5,(0,0,0),1)
            cards.append(card)
    if cards: cv2.imwrite(str(out/'accepted_patch_review.jpg'),np.concatenate(cards))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
