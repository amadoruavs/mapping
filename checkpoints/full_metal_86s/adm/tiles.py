"""Disk-backed spatial multimap: one cell can hold observations from many images."""
from pathlib import Path
import json, sqlite3
import numpy as np
import cv2
from .geometry import project

class TileIndex:
    def __init__(self,path):
        self.path=Path(path); self.db=sqlite3.connect(self.path)
        self.info=json.loads(self.db.execute('select value from metadata where key=?',('info',)).fetchone()[0])
    def close(self): self.db.close()
    def lookup(self,x,y,radius=0):
        cell=self.info['cell_size']; q=np.floor(np.array([x-radius,y-radius,x+radius,y+radius])/cell).astype(int)
        rows=self.db.execute('select image_id,tile_id,x,y,lat,lon,valid_w,valid_h from tiles where cell_x between ? and ? and cell_y between ? and ?', (int(q[0]),int(q[2]),int(q[1]),int(q[3]))).fetchall()
        return rows if radius==0 else [r for r in rows if (r[2]-x)**2+(r[3]-y)**2<=radius**2]
    def pixels(self,image_id,tile_id):
        path=self.path.parent/'tiles'/f'{image_id:04}.npy'
        return np.load(path,mmap_mode='r')[tile_id]

def build_index(source,names,transforms,cfg,out,earth=None,independent_prior=True,gpu=None):
    out=Path(out); directory=out/'tiles'; directory.mkdir(exist_ok=True)
    db=sqlite3.connect(out/'tiles.sqlite')
    db.executescript('DROP TABLE IF EXISTS tiles; DROP TABLE IF EXISTS metadata; CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE tiles(image_id INTEGER,tile_id INTEGER,cell_x INTEGER,cell_y INTEGER,x REAL,y REAL,lat REAL,lon REAL,valid_w INTEGER,valid_h INTEGER,PRIMARY KEY(image_id,tile_id));')
    cell=cfg.geo_hash_cell_m if earth is not None else cfg.local_hash_cell_px
    info={'coordinate_system':'WGS84 local east/north metres' if earth else 'local map pixels','cell_size':cell,'tile_size':cfg.tile_size,'names':names,'origin_latlon':None if earth is None else [earth.lat,earth.lon],'independent_geographic_prior':earth is not None and independent_prior,'edge_policy':'zero padded; valid_w and valid_h exclude padding'}
    db.execute('insert into metadata values (?,?)',('info',json.dumps(info))); total=0; s=cfg.tile_size
    for i,name in enumerate(names):
        bgr=cv2.imread(str(source/name)); h,w=bgr.shape[:2]; rows,cols=(h+s-1)//s,(w+s-1)//s
        if gpu is None:
            padded=np.pad(bgr[:,:,::-1],((0,rows*s-h),(0,cols*s-w),(0,0)))
            tiles=padded.reshape(rows,s,cols,s,3).transpose(0,2,1,3,4).reshape(-1,s,s,3)
        else: tiles=gpu.tiles(bgr,s)
        np.save(directory/f'{i:04}.npy',tiles)
        yy,xx=np.mgrid[0:rows,0:cols]; xs=xx.ravel()*s; ys=yy.ravel()*s
        vw=np.minimum(s,w-xs); vh=np.minimum(s,h-ys)
        centers=np.c_[xs+(vw-1)/2,ys+(vh-1)/2]; xy=project(transforms[i],centers); keys=np.floor(xy/cell).astype(np.int64)
        ll=earth.latlon(xy) if earth else None
        batch=[(i,k,int(keys[k,0]),int(keys[k,1]),float(p[0]),float(p[1]),None if ll is None else float(ll[k,0]),None if ll is None else float(ll[k,1]),int(vw[k]),int(vh[k])) for k,p in enumerate(xy)]
        db.executemany('insert into tiles values (?,?,?,?,?,?,?,?,?,?)',batch); db.commit(); total+=len(batch)
        print('indexed',name,len(batch),'tiles',flush=True)
    db.execute('CREATE INDEX cell_lookup ON tiles(cell_x,cell_y)'); db.commit(); db.close()
    (out/'tile_report.json').write_text(json.dumps({**info,'observations':total},indent=2))
    return TileIndex(out/'tiles.sqlite')
