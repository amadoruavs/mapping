"""Optional 640px proposal experiment; never changes ODM's selected pairs/settings.

The original-resolution verification is always executed, even if coarse matching
fails. This measures a proposal/refinement strategy; it is not a proven speedup.
"""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np

def features(path,width):
 image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
 if image is None:raise ValueError(f'Cannot read {path}')
 h,w=image.shape;scale=min(1.,width/max(h,w)) if width else 1.
 if scale<1:image=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
 points,desc=cv2.SIFT_create(nfeatures=24000,contrastThreshold=.025).detectAndCompute(image,None)
 xy=np.array([p.pt for p in points],np.float32).reshape(-1,2)
 xy*=np.array([w/image.shape[1],h/image.shape[0]],np.float32)
 return xy,desc,(h,w)

def match(a,b):
 pa,da,_=a;pb,db,_=b
 if da is None or db is None:return {'accepted':False,'matches':0}
 pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(da,db,k=2)
 good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.72*n.distance]
 if len(good)<12:return {'accepted':False,'matches':len(good)}
 x=pa[[m.queryIdx for m in good]];y=pb[[m.trainIdx for m in good]];held=np.arange(len(x))%5==0
 cv2.setRNGSeed(365);H,mask=cv2.findHomography(x[~held],y[~held],cv2.RANSAC,4.,maxIters=4000,confidence=.999)
 if H is None:return {'accepted':False,'matches':len(good)}
 errors=np.linalg.norm(cv2.perspectiveTransform(x[held,None],H)[:,0]-y[held],axis=1)
 return {'accepted':bool(np.median(errors)<4.),'matches':len(good),'held_out_median_px':float(np.median(errors)),'H_original_coordinates':H.tolist()}

def main():
 p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report={'production_enabled':False,'drops_baseline_pairs':False,'full_resolution_verification_always_runs':True,'coarse_long_edge_px':640}
 for name,width in [('coarse',640),('original',0)]:
  start=time.perf_counter();left=features(a.left,width);right=features(a.right,width);report[name]=match(left,right);report[name].update(seconds=time.perf_counter()-start,feature_counts=[len(left[0]),len(right[0])])
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
