"""Translate pinned upstream FSR3 frame-generation GLSL to Metal, without rewriting it."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SDK=ROOT/'third_party/fidelityfx_sdk/sdk'
OUT=ROOT/'build/fsr3'
REVISION='c6efa6bf7f2027b3ec94f28578bb5965eabb9e55'


def build():
    OUT.mkdir(parents=True,exist_ok=True);passes={}
    for source in sorted((SDK/'src/backends/vk/shaders/frameinterpolation').glob('*.glsl')):
        if 'debug_view' in source.name:continue
        name=source.stem.removeprefix('ffx_frameinterpolation_').removesuffix('_pass')
        if name=='pass':name='interpolation'
        spv=OUT/(name+'.spv');msl=OUT/(name+'.metal');reflection=OUT/(name+'.json')
        subprocess.run(['glslangValidator','-V','--target-env','vulkan1.1','-S','comp',
            '-DFFX_GPU=1','-DFFX_GLSL=1','-DFFX_HALF=0','-I'+str(SDK/'include/FidelityFX/gpu'),str(source),'-o',str(spv)],check=True,capture_output=True)
        subprocess.run(['spirv-opt','-O',str(spv),'-o',str(spv)],check=True,capture_output=True)
        subprocess.run(['spirv-cross',str(spv),'--msl','--msl-version','40000','--output',str(msl)],check=True,capture_output=True)
        subprocess.run(['spirv-cross',str(spv),'--reflect','--output',str(reflection)],check=True,capture_output=True)
        text=msl.read_text();entry=next(line for line in text.splitlines() if line.startswith('kernel void'))
        bindings=[dict(name=n,kind=k,index=int(i)) for n,k,i in re.findall(r'(\w+)\s*\[\[(buffer|texture|sampler)\((\d+)\)\]\]',entry)]
        info=json.loads(reflection.read_text())
        passes[name]=dict(bindings=bindings,threads=info['entryPoints'][0]['workgroup_size'],
            source=str(source.relative_to(ROOT)),metal_sha256=hashlib.sha256(msl.read_bytes()).hexdigest())
        print(name,flush=True)
    manifest=dict(upstream_revision=REVISION,algorithm='AMD FidelityFX Frame Interpolation 1.1.3 (SDK 1.1.4)',
        translation='Unmodified upstream GLSL -> SPIR-V -> MSL 4.0, FP32',passes=passes)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':build()
