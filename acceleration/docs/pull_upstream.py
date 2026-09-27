"""Fetch bounded, commit-pinned AMD documentation snapshots; never runs installers."""
import concurrent.futures, hashlib, json, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'upstream'
SOURCES={'HIP':('ROCm/HIP','develop',['README.md','docs/how-to/hip_runtime_api/asynchronous.rst','docs/how-to/hip_runtime_api/memory_management/device_memory.rst']), 'ROCm':('ROCm/ROCm','develop',['README.md','docs/index.md']), 'xdna-driver':('amd/xdna-driver','main',['README.md']), 'mlir-aie':('Xilinx/mlir-aie','main',['README.md','programming_guide/README.md','docs/buildHostLinNonUbuntu.md'])}
def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'ADM-docs-validation'})
 with urllib.request.urlopen(req,timeout=45) as f:return f.read()
def pull(item):
 name,(repo,branch,paths)=item
 commit=json.loads(get(f'https://api.github.com/repos/{repo}/commits/{branch}'))['sha']
 result={'repo':repo,'commit':commit,'files':[],'missing':[]}
 for path in paths:
  url=f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
  try:content=get(url)
  except Exception as e:result['missing'].append({'path':path,'error':str(e)});continue
  output=ROOT/name/path;output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(content)
  result['files'].append({'path':path,'url':url,'sha256':hashlib.sha256(content).hexdigest()})
 return name,result
if __name__=='__main__':
 ROOT.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=dict(pool.map(pull,SOURCES.items()))
 (ROOT/'manifest.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
