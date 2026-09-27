"""Report completed default-ODM versus full Metal/MPS comparisons."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/comparison'


def main():
    reports={name:json.loads((OUT/name/'comparison.json').read_text())
             for name in ('siyi','archive') if (OUT/name/'comparison.json').exists()}
    if not reports:raise RuntimeError('No completed full-pipeline comparisons')
    lines=['# Full Metal/MPS pipeline versus ODM','',
           'The full pipeline is restored: visual feature matching, global alignment, compositing, indexing and lighting. GPS-only and corner-only experiments are archived separately.','',
           '| Dataset | ODM seconds | Full Metal seconds | ODM / Metal | Metal dimensions |',
           '| --- | ---: | ---: | ---: | --- |']
    rows=[]
    for name,r in reports.items():
        t=r['timing'];g=r['metal_geometry']
        lines.append(f"| {name} | {t['odm']['wall_seconds']:.3f} | {t['metal']['wall_seconds']:.3f} | {t['odm_over_metal_wall_ratio']:.2f}× | {g['width']}×{g['height']} |")
        for backend in ('odm','metal'):
            rows.append(dict(dataset=name,backend=backend,**{key:t[backend][key] for key in ('wall_seconds','cpu_user_seconds','cpu_system_seconds','peak_child_rss_bytes')}))
        lines += ['',f"[ODM PNG]({name}/odm.png), [full Metal PNG]({name}/metal.png), [metrics]({name}/comparison.json).",'']
        lines += [f'## {name} full Metal stages','','| Stage | Seconds |','| --- | ---: |']
        for stage,seconds in sorted(r['metal_report']['timings_seconds'].items(),key=lambda item:-item[1]):
            lines.append(f'| {stage} | {seconds:.3f} |')
    missing=[name for name in ('siyi','archive') if name not in reports]
    if missing:lines += ['',f"No completed full Metal comparison for: {', '.join(missing)}. Experimental runs are not substituted."]
    lines += ['',
        'Timings are single completed runs, including process startup and image exports. ODM performs 3D reconstruction; ADM makes a planar mosaic, so these are different workloads and output resolutions. ODM ran ARM64 in Docker; Metal ran natively on Apple M4 Pro. CPU limits, background applications and filesystem caches were not controlled. Docker was unavailable during the successful full SIYI Metal run after an earlier disk-full failure. Failed runs are excluded. Pixel agreement and GPS-prior residuals are not independent surveyed accuracy.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (OUT/'timing.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('adm','shaders','scripts','tests') for p in (ROOT/folder).rglob('*')
            if p.is_file() and p.suffix in ('.py','.m','.metal')}
    (OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(OUT/'REPORT.md')


if __name__=='__main__':main()
