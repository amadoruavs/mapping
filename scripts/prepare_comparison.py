"""Freeze identical inputs for the ODM/default versus ADM/Metal comparison."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adm.config import Config
from adm.metadata import read_metadata
from adm.selection import select_frames

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'/'comparison'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    siyi=json.loads((ROOT/'output/101SIYI_IMG/map_geometry.json').read_text())['input_names']
    for dataset, names, source, altitude in (
        ('siyi', siyi, Path('/Users/avneh/Downloads/101SIYI_IMG'), 15.),
        ('archive', None, Path('/Users/avneh/Downloads/Archive'), 60.),
    ):
        if names is None:
            names, selection=select_frames(source, Config(skip_first=0))
        else: selection=None
        dest=OUT/dataset; images=dest/'odm'/'images'; images.mkdir(parents=True, exist_ok=True)
        manifest=[]
        for name in names:
            original=source/name; linked=images/name
            if not linked.exists(): os.link(original, linked)
            meta,_=read_metadata(original, Config(altitude_agl_m=altitude))
            manifest.append(dict(name=name, sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                                 bytes=original.stat().st_size, metadata=meta.to_dict()))
        cfg=Config(skip_first=0, altitude_agl_m=altitude, metal_recompute_warps=True)
        if dataset=='archive': cfg.registration_width=640
        (dest/'adm_config.json').write_text(json.dumps(cfg.to_dict(), indent=2))
        (dest/'inputs.json').write_text(json.dumps(dict(source=str(source), images=manifest,
            count=len(names), total_bytes=sum(r['bytes'] for r in manifest),
            total_pixels=sum(r['metadata']['width']*r['metadata']['height'] for r in manifest),
            selection=selection, truth_used_for_reconstruction=False,
            altitude_note='Nominal 15 m SIYI setting from existing run' if dataset=='siyi' else '60 m nominal flight altitude; true coordinates and GSD are held out'), indent=2))
        if dataset=='archive':
            # Evaluation-only path is never mounted as image data or passed to reconstruction.
            rows=list(csv.DictReader((source/'flight_log.csv').open()))
            (dest/'evaluation_truth.json').write_text(json.dumps(rows, indent=2))
    commands={
        'docker': ['docker','info','--format','{{json .}}'],
        'odm_image': ['docker','image','inspect','opendronemap/odm:3.6.2'],
        'background_containers': ['docker','ps','--format','{{json .}}'],
        'machine': ['system_profiler','SPHardwareDataType','SPDisplaysDataType','-json'],
    }
    for name,command in commands.items():
        result=subprocess.run(command,capture_output=True,text=True)
        (OUT/f'{name}.json').write_text(result.stdout)
    print(OUT)


if __name__=='__main__': main()
