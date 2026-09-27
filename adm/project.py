"""Small replayable mapping projects: original-image references and transforms."""
import hashlib
import json
from pathlib import Path

from .io import write_json


def source_digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):
            digest.update(block)
    return digest.hexdigest()


def write_recipe(source, names, transforms, geometry, cfg, out):
    selection_path=out/'selection.json'
    rows=json.loads(selection_path.read_text()) if selection_path.exists() else []
    hashes={row['name']:row['sha256'] for row in rows}
    value=dict(version=1,kind='ADM vector rendering project',source=str(source.resolve()),
               images=[dict(name=name,sha256=hashes.get(name) or source_digest(source/name)) for name in names],
               image_to_map=[h.tolist() for h in transforms],geometry=geometry,config=cfg.to_dict(),
               pixels_copied=False,rendering='Recompute bounded source-selection fields and stream original photographs on demand')
    write_json(out/'project.json',value)
    return value
