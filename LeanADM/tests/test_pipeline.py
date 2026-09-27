import json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import IFDRational
from adm.config import Config
from adm.metadata import FrameMetadata,read_metadata
from adm.geometry import LocalEarth,ground_homography,project,intrinsics,neighbor_pairs
from adm.tiles import build_index
from adm.vocabulary import match_corners,delta_map
from adm.lighting import normals_from_height,visibility_from_height,shade_linear,srgb_to_linear,linear_to_srgb,render_lighting
from adm.selection import select_frames
from adm.pipeline import run,align_centers_to_geography

def write_exif(path,rgb,lat=35.,lon=-97.,yaw=0):
    def dms(value):
        v=abs(value); degrees=int(v); minutes=int((v-degrees)*60)
        return (IFDRational(degrees),IFDRational(minutes),IFDRational((v-degrees-minutes/60)*3600))
    exif=Image.Exif(); exif[34853]={1:'N' if lat>=0 else 'S',2:dms(lat),3:'E' if lon>=0 else 'W',4:dms(lon),5:0,6:IFDRational(150),16:'T',17:IFDRational(yaw)}
    Image.fromarray(rgb).save(path,quality=97,exif=exif)

class PipelineTests(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory(prefix='adm-test-'); self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def test_exif_gps_heading_and_altitude_policy(self):
        path=self.root/'gps.jpg'; write_exif(path,np.full((64,96,3),100,np.uint8),-35,97,92)
        meta,raw=read_metadata(path,Config(altitude_agl_m=42))
        self.assertTrue(meta.geographic); self.assertAlmostEqual(meta.latitude,-35); self.assertAlmostEqual(meta.longitude,97); self.assertEqual(meta.yaw_deg,92)
        self.assertEqual(meta.altitude_agl_m,42); self.assertEqual(meta.exif_altitude_m,150); self.assertGreater(raw['gps_tag_count'],0)

    def test_empty_exif_and_xmp_adapter(self):
        path=self.root/'xmp.jpg'; Image.new('RGB',(96,64)).save(path)
        empty,_=read_metadata(path,Config()); self.assertFalse(empty.geographic)
        xml=b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" drone-dji:GpsLatitude="35" drone-dji:GpsLongitude="-97" drone-dji:GimbalYawDegree="123" drone-dji:GimbalPitchDegree="-85"/></rdf:RDF></x:xmpmeta>'
        payload=b'http://ns.adobe.com/xap/1.0/\0'+xml; data=path.read_bytes(); path.write_bytes(data[:2]+b'\xff\xe1'+(len(payload)+2).to_bytes(2,'big')+payload+data[2:])
        meta,_=read_metadata(path,Config()); self.assertTrue(meta.geographic); self.assertEqual(meta.yaw_deg,123); self.assertEqual(meta.pitch_deg,-85)
        meta,_=read_metadata(path,Config(),{'yaw_deg':44,'gps_reference':'image_center'}); self.assertEqual(meta.yaw_deg,44)

    def test_nadir_yaw_and_geographic_roundtrip(self):
        cfg=Config(); earth=LocalEarth(35,-97); m=FrameMetadata('a',3840,2160,35,-97,0)
        centre=np.array([[(m.width-1)/2,(m.height-1)/2]])
        H=ground_homography(m,cfg,earth); np.testing.assert_allclose(project(H,centre),0,atol=1e-8)
        right=project(H,centre+[[100,0]])[0]; up=project(H,centre+[[0,-100]])[0]
        self.assertGreater(right[0],0); self.assertGreater(up[1],0)
        m.yaw_deg=90; H=ground_homography(m,cfg,earth); self.assertLess(project(H,centre+[[100,0]])[0,1],0)
        xy=np.array([[20.,-12.],[-50,80]]); ll=earth.latlon(xy); np.testing.assert_allclose(earth.xy(ll[:,0],ll[:,1]),xy,atol=1e-7)
        m.pitch_deg=-1
        with self.assertRaises(ValueError): ground_homography(m,cfg,earth)

    def test_image_center_gps_and_spatial_neighbors(self):
        cfg=Config(); earth=LocalEarth(35,-97); m=FrameMetadata('a',800,500,35,-97,0,pitch_deg=-70,gps_reference='image_center')
        H=ground_homography(m,cfg,earth); np.testing.assert_allclose(project(H,[[399.5,249.5]]),0,atol=1e-8)
        distant=FrameMetadata('b',800,500,36,-97,0)
        pairs,_,_=neighbor_pairs([m,m,distant],cfg); self.assertEqual(pairs,[(0,1)])

    def test_tile_multimap_color_and_partial_edges(self):
        source=self.root/'source'; source.mkdir(); out=self.root/'out'; out.mkdir()
        rgb=np.zeros((23,25,3),np.uint8); rgb[:]=[17,68,201]
        for name in ['a.png','b.png']: Image.fromarray(rgb).save(source/name)
        index=build_index(source,['a.png','b.png'],[np.eye(3),np.eye(3)],Config(),out)
        rows=index.lookup(10,10); self.assertEqual({r[0] for r in rows},{0,1}); self.assertEqual(index.pixels(0,0)[0,0].tolist(),[17,68,201])
        edge=index.db.execute('select valid_w,valid_h from tiles where image_id=0 and tile_id=3').fetchone(); self.assertEqual(edge,(4,2)); self.assertEqual(index.info['independent_geographic_prior'],False); index.close()

    def test_delta_corner_translation_and_bad_geo(self):
        rng=np.random.default_rng(3); image=rng.integers(40,180,(200,240,3),dtype=np.uint8)
        target=np.zeros((250,300,3),np.uint8); target[17:217,23:263]=image+15
        H=np.array([[1,0,23],[0,1,17],[0,0,1.]])
        matches=match_corners(image,target,H,Config()); self.assertEqual(sum(m['accepted'] for m in matches),4)
        self.assertLess(max(m['delta_rmse'] for m in matches),1e-6)
        bad=match_corners(image,target,H,Config(),np.eye(3),np.eye(3)); self.assertFalse(any(m['accepted'] for m in bad))
        np.testing.assert_allclose(delta_map(image),delta_map(image+15),atol=1e-6)

    def test_flat_normals_shadows_and_color_roundtrip(self):
        height=np.zeros((70,80),np.float32); height[:,40:43]=10; valid=np.ones_like(height,bool)
        cfg=Config(sun_azimuth_deg=90,shadow_max_distance_m=30,shadow_steps=60)
        visibility=visibility_from_height(height,valid,1,cfg)
        self.assertLess(visibility[30,30],.1); self.assertGreater(visibility[30,60],.9)
        n=normals_from_height(np.zeros((10,10),np.float32),1); np.testing.assert_allclose(n[0,0],[0,0,1])
        x=np.linspace(0,1,101,dtype=np.float32); np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)),x,atol=1e-6)

    def test_filter_exact_duplicate_and_overlay(self):
        rgb=np.random.default_rng(1).integers(40,160,(180,300,3),dtype=np.uint8)
        Image.fromarray(rgb).save(self.root/'a.png'); (self.root/'b.png').write_bytes((self.root/'a.png').read_bytes())
        rgb[20:40,20:140]=[0,0,255]; Image.fromarray(rgb).save(self.root/'c.png')
        names,rows=select_frames(self.root,Config(skip_first=0)); self.assertEqual(names,['a.png']); self.assertIn('duplicate',rows[1]['excluded']); self.assertIn('label',rows[2]['excluded'])

    def test_height_lighting_export_and_input_validation(self):
        height=np.zeros((32,40),np.float32); height[:,20]=4; path=self.root/'height.npy'; np.save(path,height)
        rgba=np.full((32,40,4),130,np.uint8); rgba[:,:,3]=255
        report=render_lighting(rgba,Config(lighting_candidates=3),self.root,path,1)
        self.assertIn('height-field',report['mode']); self.assertTrue((self.root/'map_golden_linear.hdr').exists())
        with self.assertRaises(ValueError): render_lighting(rgba,Config(),self.root,path,None)
        with self.assertRaises(ValueError): read_metadata(self.root/'map_golden.png',Config(),{'latitude':float('nan')})

    def test_gps_center_alignment_ignores_incorrect_yaw(self):
        earth=LocalEarth(35,-97); known=np.array([[.05,0,100],[0,-.05,30],[0,0,1.]])
        hs=[]; metadata=[]
        for i,(x,y) in enumerate([(0,0),(100,0),(100,100),(0,100)]):
            H=np.array([[1,0,x],[0,1,y],[0,0,1.]])
            lat,lon=earth.latlon(project(known@H,[[399.5,249.5]])[0])
            hs.append(H); metadata.append(FrameMetadata(str(i),800,500,float(lat),float(lon),123))
        corrected=align_centers_to_geography(hs,metadata,earth,Config(),self.root)
        for expected,actual in zip(hs,corrected): np.testing.assert_allclose(actual,known@expected,atol=1e-6)
        report=json.loads((self.root/'geographic_alignment.json').read_text()); self.assertLess(report['leave_one_out_p95_m'],1e-6)

    def test_geographic_pipeline_with_real_exif(self):
        source=self.root/'input'; source.mkdir(); out=self.root/'result'
        rng=np.random.default_rng(40); base=rng.integers(30,210,(500,1100,3),dtype=np.uint8)
        base=cv2.GaussianBlur(base,(3,3),.6)
        cfg=Config(skip_first=0,registration_width=800,max_output_dimension=700,feature_count=3000)
        earth=LocalEarth(35,-97); dummy=FrameMetadata('a',800,500,35,-97,0); gsd=cfg.altitude_agl_m/intrinsics(dummy,cfg)[0,0]
        for i in range(3):
            lat,lon=earth.latlon(np.array([i*100*gsd,0.])); write_exif(source/f'{i}.jpg',base[:,i*100:i*100+800],lat,lon,0)
        config=self.root/'config.json'; config.write_text(json.dumps(cfg.to_dict()))
        args=SimpleNamespace(source=source,output=out,config=config,altitude=None,max_dim=None,strict_corners=False,exclusions=None,metadata=None,allow_local=False,resume=False,heightmap=None,heightmap_spacing=None,backend='cpu')
        run(args)
        report=json.loads((out/'report.json').read_text()); self.assertTrue(report['georeferenced']); self.assertEqual(report['retained_images'],3)
        alignment=json.loads((out/'geographic_alignment.json').read_text()); self.assertLess(alignment['median_footprint_prior_residual_m'],.2)
        rgba=cv2.imread(str(out/'map_golden.png'),cv2.IMREAD_UNCHANGED); self.assertEqual(rgba.shape[2],4); self.assertTrue((rgba[:,:,3]>0).any())

        # Same known scene with explicitly wrong recorded yaw: GPS-center mode
        # should still position the map and label the hash as inferred geometry.
        for i in range(3):
            lat,lon=earth.latlon(np.array([i*100*gsd,0.])); write_exif(source/f'{i}.jpg',base[:,i*100:i*100+800],lat,lon,123)
        args.gps_centers=True; args.output=self.root/'gps_centers'; run(args)
        report=json.loads((args.output/'report.json').read_text()); self.assertEqual(report['geographic_alignment_mode'],'GPS camera centers')
        tile_report=json.loads((args.output/'tile_report.json').read_text()); self.assertFalse(tile_report['independent_geographic_prior'])

if __name__=='__main__': unittest.main()
