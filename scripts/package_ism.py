"""Create the standalone ISM package from the current validated ADM sources.

The generated ISM.metal is the package's sole shader source. Generated FSR
helpers are namespaced and their entry points have stable exported host names.
This packager is a development tool; running ISM never needs this repository.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ISM'


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError(f'Expected one packaging anchor: {old[:90]}')
    return text.replace(old,new,1)


def package():
    runtime=OUT/'runtime';(runtime/'adm').mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'build/fsr3/manifest.json').read_text())
    parts=['// ISM: consolidated ADM + AMD FSR shader source.\n// Host/file operations are in runtime/ and run.py. See GUIDE.md.\n',
           '#include <metal_stdlib>\n#include <metal_atomic>\n#include <metal_raytracing>\n#include <simd/simd.h>\n']
    sources={};kernel_names=[]
    for name in ('pipeline','alignment','geographic','fsr','coherent','lasso','presentation','raytracing'):
        path=ROOT/'shaders'/f'{name}.metal';source=path.read_text()
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        source=re.sub(r'^#include .*\n','',source,flags=re.M)
        parts.append(f'\n// ===== ADM {name} =====\n'+source)
        kernel_names.extend(re.findall(r'kernel void (\w+)\(',source))
    # Include the upstream notices in the actual consolidated source as well.
    for license_path in ('third_party/fidelityfx_sdk/LICENSE.txt','third_party/fidelityfx_fsr1/LICENSE'):
        parts.append('\n/*\n'+(ROOT/license_path).read_text().replace('*/','* /')+'\n*/\n')
    for name,info in manifest['passes'].items():
        path=ROOT/'build/fsr3'/f'{name}.metal';source=path.read_text()
        if hashlib.sha256(path.read_bytes()).hexdigest()!=info['metal_sha256']:raise ValueError(f'Stale FSR shader: {name}')
        sources[str(path.relative_to(ROOT))]=info['metal_sha256']
        source=re.sub(r'^#include .*\n','',source,flags=re.M)
        entry='ism_fsr_'+name
        source=replace_once(source,'kernel void main0(',f'[[host_name("{entry}")]] kernel void {entry}(')
        parts.append(f'\n// ===== AMD FSR {name} =====\nnamespace ism_{name} {{\n{source}\n}}\n')
        info['entrypoint']=entry
        info['reflection']=json.loads((ROOT/'build/fsr3'/f'{name}.json').read_text())
        kernel_names.append(entry)
    shader=''.join(parts);(OUT/'ISM.metal').write_text(shader)
    (runtime/'fsr_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for path in (ROOT/'adm').glob('*.py'):shutil.copy2(path,runtime/'adm'/path.name)

    path=runtime/'adm/metal.py';text=path.read_text()
    text=replace_once(text,'ROOT = Path(__file__).resolve().parents[1]','ROOT = Path(__file__).resolve().parents[2]')
    text=text.replace("ROOT / 'shaders' / 'ADMMetal.m'","ROOT / 'runtime' / 'ADMMetal.m'")
    text=text.replace("ROOT/'shaders/ADMMetal.m'","ROOT/'runtime/ADMMetal.m'")
    text=replace_once(text,"ROOT / 'shaders' / 'pipeline.metal'","ROOT / 'ISM.metal'")
    old=next(line for line in text.splitlines() if 'shader_bytes=b' in line)
    text=replace_once(text,old,"        shader_bytes=(ROOT/'ISM.metal').read_bytes()")
    text=replace_once(text,"            'adm_error':", "            'adm_shader_library': (C.c_void_p, [C.c_void_p]),\n            'adm_function_names': (C.c_char_p, [C.c_void_p]),\n            'adm_error':")
    path.write_text(text)

    path=runtime/'ADMMetal.m';text=(ROOT/'shaders/ADMMetal.m').read_text()
    start=text.index('        for (NSString *extra in ');end=text.index('        MTLCompileOptions *options=',start)
    text=text[:start]+text[end:]
    text=replace_once(text,'        // Preserve cancellation-sensitive',
        '        if (@available(macOS 26.0, *)) options.languageVersion=MTLLanguageVersion4_0;\n'
        '        else { creationError=@"ISM requires macOS 26 / Metal 4"; return NULL; }\n'
        '        // Preserve cancellation-sensitive')
    text+='\nvoid *adm_shader_library(void *p) { return (__bridge void *)ctx(p).library; }\n'
    text+='''const char *adm_function_names(void *p) {
    static _Thread_local char *copy;
    @autoreleasepool {
        NSData *json=[NSJSONSerialization dataWithJSONObject:ctx(p).library.functionNames options:0 error:nil];
        NSString *string=[[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding];
        free(copy);copy=strdup(string.UTF8String);
    }
    return copy;
}
'''
    path.write_text(text)

    path=runtime/'FSRMetal.m';text=(ROOT/'shaders/FSRMetal.m').read_text()
    text=replace_once(text,'@property NSArray *samplers;','@property NSArray *samplers;\n@property id<MTLLibrary> library;')
    start=text.index('int fsr_compile(');end=text.index('void *fsr_texture(',start)
    text=text[:start]+'''int fsr_set_library(void *p,void *library){@autoreleasepool{
    if(!library)return fail(context(p),@"ISM shared shader library is missing");
    context(p).library=(__bridge id<MTLLibrary>)library;return 0;
}}
int fsr_compile(void *p,const char *name,const char *unused){@autoreleasepool{
    FSRContext *c=context(p);NSError *e=nil;
    NSString *entry=[@"ism_fsr_" stringByAppendingString:@(name)];
    id<MTLFunction> function=[c.library newFunctionWithName:entry];
    if(!function)return fail(c,[@"Missing consolidated FSR kernel: " stringByAppendingString:entry]);
    id<MTLComputePipelineState> pipeline=[c.device newComputePipelineStateWithFunction:function error:&e];
    if(!pipeline)return fail(c,e.description);c.pipelines[@(name)]=pipeline;return 0;
}}
''' + text[end:]
    path.write_text(text)

    path=runtime/'adm/fsr_framegen.py';text=path.read_text()
    text=replace_once(text,'ROOT=Path(__file__).resolve().parents[1]','ROOT=Path(__file__).resolve().parents[2]')
    start=text.index("        if not (BUILD/'manifest.json').exists():");end=text.index("        source=ROOT/'shaders/FSRMetal.m'",start)
    text=text[:start]+"        BUILD.mkdir(parents=True,exist_ok=True)\n        self.manifest=json.loads((ROOT/'runtime/fsr_manifest.json').read_text())\n"+text[end:]
    text=replace_once(text,"ROOT/'shaders/FSRMetal.m'","ROOT/'runtime/FSRMetal.m'")
    text=replace_once(text,"            'fsr_compile':", "            'fsr_set_library':(C.c_int,[ptr,ptr]),\n            'fsr_compile':")
    start=text.index('        for name in self.passes:');end=text.index('        self.shape=None;',start)
    text=text[:start]+'''        from .metal import get_backend
        gpu=get_backend()
        self.check(self.lib.fsr_set_library(self.handle,gpu.lib.adm_shader_library(gpu.handle)))
        for name in self.passes:self.check(self.lib.fsr_compile(self.handle,name.encode(),b''))
''' + text[end:]
    text=replace_once(text,"info=json.loads((BUILD/(pass_name+'.json')).read_text())","info=self.passes[pass_name]['reflection']")
    path.write_text(text)

    shutil.copy2(ROOT/'requirements.txt',OUT/'requirements.txt')
    # Numerical tests ship with the portable package; no tests load parent ADM.
    (OUT/'tests').mkdir(exist_ok=True)
    for name in ('test_presentation_metal.py','test_cinematic_grade.py','test_fsr3.py','test_raytracing.py','test_lasso.py'):
        shutil.copy2(ROOT/'tests'/name,OUT/'tests'/name)
    notices='\n\n'.join((ROOT/path).read_text() for path in ('third_party/fidelityfx_sdk/LICENSE.txt','third_party/fidelityfx_fsr1/LICENSE','third_party/PRESENTATION_NOTICES.txt','third_party/opensfm_reference/LICENSE'))
    shutil.copy2(ROOT/'docs/OPENSFM_PORT.md',runtime/'OPENSFM_PORT.md')
    (runtime/'LICENSES.txt').write_text(notices)
    (runtime/'bundle_manifest.json').write_text(json.dumps(dict(shader='ISM.metal',kernel_count=len(kernel_names),
        entrypoints=kernel_names,source_hashes=sources,shader_sha256=hashlib.sha256(shader.encode()).hexdigest(),
        upstream_revision=manifest['upstream_revision'],
        opensfm_revision='c5328439465e6ace011f39077d1077d7b1cdd65d',
        geographic_port='DataPriorError, UpVectorError/Cauchy, WGS84; fixed-plane camera adaptation'),indent=2)+'\n')
    print(json.dumps({'folder':str(OUT),'shader_bytes':len(shader.encode()),'kernel_count':len(kernel_names)},indent=2))


if __name__=='__main__':package()
