from pathlib import Path
import importlib.util,json,os,sys,tempfile,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'MapRepair'),str(ROOT/'MapRepair/tests')]
import map_geo_checker as new
import advanced_grade as grade_new
from test_pipeline import fixture
from map_io import read_map

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
old=load('checker_original',ROOT/'acceleration/validation/source_before/map_geo_checker.py')
grade_old=load('grade_original',ROOT/'acceleration/validation/source_before/advanced_grade.py')
OUT=ROOT/'acceleration/validation';results={}
def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k!='seconds'}
 if isinstance(value,list):return [clean(x) for x in value]
 return value

def compare(name,image,paths,kwargs):
 outputs=[];times=[]
 for module in [old,new]:
  cv2.setNumThreads(4);cfg=module.Settings(**kwargs);cv2.setRNGSeed(cfg.seed)
  start=time.perf_counter();checker=module.Checker(image,None,cfg,lambda s:None);checker.scan(paths);repaired,report,owners=checker.repair()
  outputs.append((repaired,clean(report),owners));times.append(time.perf_counter()-start)
 np.testing.assert_array_equal(outputs[0][0],outputs[1][0]);np.testing.assert_array_equal(outputs[0][2],outputs[1][2])
 if outputs[0][1]!=outputs[1][1]:raise AssertionError(f'{name}: geometry decisions differ')
 return {'passed':True,'serial_seconds':times[0],'parallel_seconds':times[1],'photos':len(paths),'identical':['repaired_pixels','ownership','geometry_report']}
with tempfile.TemporaryDirectory() as td:
 _,image,photos=fixture(td)
 results['synthetic_repair']=compare('synthetic',image,sorted(photos.glob('*.png')),dict(map_width=960,photo_width=800))
 directory=Path(td);rng=np.random.default_rng(365);image=rng.integers(0,256,(777,513,4),dtype=np.uint8)
 image.tofile(directory/'grade_master.bgra');(directory/'final_geometry.json').write_text(json.dumps({'width':513,'height':777,'jpeg_background_rgb':[241,128,17]}))
 for backend in ['cpu','metal']:
  os.environ['ADM_PIXEL_BACKEND']=backend
  for i,settings in enumerate([{},dict(exposure=.3,temperature=15,sharpen=100,radius=3,shadows=11)]):
   a=grade_old.render(directory,settings);b=grade_new.render(directory,settings)
   if a!=b:raise AssertionError(f'{backend}: JPEG export differs')
 results['grading']={'passed':True,'cpu_and_metal_jpeg_byte_identical':True}
base=ROOT/'output/odm_macos_ultra94_20260916/flight'
image,_=read_map(base/'odm_orthophoto/odm_orthophoto.tif',None)
results['real_photos']=compare('real',image,sorted((base/'images').glob('*.jpg')),{})
(OUT/'maprepair_parity.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
