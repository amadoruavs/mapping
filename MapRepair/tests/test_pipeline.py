import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from georeference import GeoReference,load_georeference
from map_geo_checker import Checker,Settings,edge_weights,valid_homography
from native import Native,plane
from run import pipeline,relight,highlight_bloom


def fixture(folder):
    """Known planar scene, overlapping source crops, and one shifted map region."""
    rng=np.random.default_rng(312)
    gray=cv2.GaussianBlur(rng.integers(35,180,(720,960),np.uint8),(3,3),.7)
    scene=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGRA);scene[:,:,3]=255
    for i in range(120):
        x,y=rng.integers([20,20],[930,690]);color=tuple(int(c) for c in rng.integers(30,240,3))+(255,)
        cv2.circle(scene,(int(x),int(y)),int(rng.integers(3,14)),color,2)
    cv2.line(scene,(50,250),(900,480),(240,240,240,255),7)
    damaged=scene.copy();damaged[256:448,320:576]=scene[256:448,307:563]
    folder=Path(folder);photos=folder/'photos';photos.mkdir()
    for i,(x,y) in enumerate(((0,0),(100,0),(50,100))):cv2.imwrite(str(photos/f'{i}.png'),scene[y:y+600,x:x+800])
    cv2.imwrite(str(folder/'map.png'),damaged)
    return scene,damaged,photos


class GeometryTests(unittest.TestCase):
    def test_world_file_crs_and_pixel_centers(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'map.png';p.touch()
            g=load_georeference(p,(100,200),(-96,36,-95,37))
            np.testing.assert_allclose(g.latlon([0,0]),[36.995,-95.9975])
            H=np.array([[.0001,.00002,-96],[.00003,-.0001,37],[0,0,1]])
            geo=GeoReference(H,'EPSG:4326','fixture');point=np.array([35.2,78.1])
            ll=geo.latlon(point);np.testing.assert_allclose(geo.pixel(*ll),point,atol=1e-8)
            self.assertGreater(geo.error_m(point,point+[1,0]),5)
            (p.with_suffix('.pgw')).write_text('1\n0\n0\n-1\n500000\n4000000\n')
            with self.assertRaisesRegex(ValueError,'CRS'):load_georeference(p,(100,200))

    def test_georeference_survives_final_2x(self):
        with tempfile.TemporaryDirectory() as td:
            g=GeoReference([[.0001,0,-96],[0,-.0001,37],[0,0,1]],'EPSG:4326','fixture')
            data=g.export(td,'final_2x',2);new=GeoReference(data['pixel_to_world'],data['crs'],'fixture')
            np.testing.assert_allclose(new.latlon([100.5,80.5]),g.latlon([50,40]),atol=1e-12)

    def test_weak_homography_and_real_feather_boundary(self):
        self.assertFalse(valid_homography(np.diag([.00001,.00001,1]),(100,100),(500,500)))
        self.assertFalse(valid_homography(np.array([[1,0,0],[0,1,0],[.03,0,-1]]),(100,100),(500,500)))
        weights=edge_weights((40,60),8)
        self.assertTrue(np.all(weights[0]==0));self.assertTrue(np.all(weights[:,-1]==0))
        self.assertGreater(weights[4,30],weights[1,30]);self.assertEqual(weights[20,30],1)

    def test_known_local_error_is_repaired_and_unrelated_pixels_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            truth,damaged,photos=fixture(td)
            checker=Checker(damaged,None,Settings(map_width=960,photo_width=800,min_inliers=25,photo_features=4000,map_features=15000),lambda s:None)
            checker.scan(sorted(photos.glob('*.png')));repaired,report,_=checker.repair()
            patch=np.s_[280:420,344:552,:3]
            before=np.mean(np.abs(damaged[patch].astype(float)-truth[patch]))
            after=np.mean(np.abs(repaired[patch].astype(float)-truth[patch]))
            print('Known-error repair:',dict(before=before,after=after,tiles=report['repaired_tiles']),flush=True)
            self.assertGreater(report['repaired_tiles'],0);self.assertLess(after,before*.65)
            np.testing.assert_array_equal(repaired[:100],damaged[:100]);np.testing.assert_array_equal(repaired[:,:,3],damaged[:,:,3])


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.native=Native(4)

    def test_fsr_flat_alpha_and_thread_repeatability(self):
        rng=np.random.default_rng(88);image=rng.integers(0,256,(47,65,4),np.uint8);image[:,:,3]=255;image[:4]=0
        with tempfile.TemporaryDirectory() as td:
            a=self.native.fsr2x(image,Path(td)/'a',progress=lambda s:None)
            b=Native(1).fsr2x(image,Path(td)/'b',row_chunk=19,progress=lambda s:None)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(a[:,:,3],np.repeat(np.repeat(image[:,:,3],2,axis=0),2,axis=1))
            self.assertFalse(a[:8].any());a._mmap.close();b._mmap.close()
            image[:]=[81,120,175,255]
            c=self.native.fsr2x(image,Path(td)/'c',progress=lambda s:None)
            np.testing.assert_array_equal(c,np.broadcast_to(image[0,0],c.shape));c._mmap.close()

    def test_ray_plane_neutral_and_occluder_casts_shadow(self):
        shape=(160,200);ground=plane(shape);neutral=self.native.trace(ground,shape,grid=80)
        np.testing.assert_allclose(neutral,1,atol=1e-6)
        wall=np.array([[[90,40,0],[90,115,0],[90,115,30]],[[90,40,0],[90,115,30],[90,40,30]]],np.float32)
        scene=np.concatenate([ground,wall]);light=self.native.trace(scene,shape,grid=80,elevation=30,azimuth=270)
        self.assertLess(float(light[:,:,0].min()),.9)
        np.testing.assert_array_equal(light,self.native.trace(scene,shape,grid=80,elevation=30,azimuth=270))
        image=np.full((*shape,4),100,np.uint8);image[:,:,3]=255
        output,report=relight(image,self.native);np.testing.assert_array_equal(output,image)

    def test_chroma_strips_equal_original_ism_reference(self):
        from presentation import chroma_clean_reference,_chroma_full
        rng=np.random.default_rng(91);a=rng.integers(0,256,(281,173,4),np.uint8);a[:,:,3]=255;a[:3]=0
        np.testing.assert_array_equal(chroma_clean_reference(a),_chroma_full(a))

    def test_full_pipeline_writes_final_2x_and_reports(self):
        with tempfile.TemporaryDirectory() as td:
            truth,damaged,photos=fixture(td);out=Path(td)/'result'
            import rasterio
            from rasterio.transform import Affine
            input_path=Path(td)/'map.tif';damaged[:10,:,3]=0;damaged[600:620,700:720,3]=0
            with rasterio.open(input_path,'w',driver='GTiff',width=960,height=720,count=4,dtype='uint8',photometric='RGB',alpha='YES',
                transform=Affine(.12,.01,500000,.01,-.12,4000000),crs='EPSG:32615') as ds:
                ds.write(damaged[:,:,[2,1,0,3]].transpose(2,0,1))
            original_bytes=input_path.read_bytes()
            result=pipeline(input_path,photos,out,cfg=Settings(map_width=960,photo_width=800,min_inliers=25),save_stages=False)
            self.assertEqual(result['status'],'complete')
            final=cv2.imread(str(out/'final_2x.jpg'),cv2.IMREAD_UNCHANGED)
            self.assertEqual(final.shape,(1440,1920,3));self.assertTrue((out/'geometry_report.json').exists())
            self.assertTrue((out/'final_2x_alpha.tif').exists());self.assertFalse((out/'final_2x.png').exists())
            with rasterio.open(out/'final_2x_alpha.tif') as ds:
                np.testing.assert_array_equal(ds.read(1),np.repeat(np.repeat(damaged[:,:,3],2,0),2,1))
            self.assertTrue((out/'final_2x.jgw').exists())
            self.assertFalse((out/'_work').exists());self.assertGreater(result['repaired_tiles'],0)
            self.assertEqual(input_path.read_bytes(),original_bytes)


if __name__=='__main__':unittest.main(verbosity=2)
