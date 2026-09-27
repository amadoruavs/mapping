"""Connection recovery uses verified imagery and preserves registration units."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import cv2
import numpy as np
from adm.config import Config
from adm.geometry import project
from adm.match_graph import components,recover,require_connected,valid_transform


class MatchGraphTests(unittest.TestCase):
    def test_degenerate_pair_is_rejected_before_projection(self):
        self.assertTrue(valid_transform(np.eye(3),640,360,360))
        self.assertFalse(valid_transform(np.zeros((3,3)),640,360,360))
        self.assertFalse(valid_transform(np.array([[1.,0,0],[0,1,0],[.01,0,-3]]),640,360,360))
    def test_connected_graph_skips_all_recovery_work(self):
        edges=[dict(i=0,j=1)]
        result,report=recover(None,['a','b'],None,None,edges,Config(),None)
        self.assertIs(result,edges);self.assertFalse(report['attempted'])

    def test_higher_resolution_bridge_uses_original_coordinate_system(self):
        rng=np.random.default_rng(91)
        image=cv2.GaussianBlur(rng.integers(0,256,(721,1280,3),np.uint8),(5,5),1.)
        for _ in range(250):
            x,y=rng.integers(30,680,2);cv2.circle(image,(int(x*1.7),int(y)),int(rng.integers(3,15)),tuple(int(x) for x in rng.integers(0,256,3)),-1)
        names=['a.png','b.png','c.png'];metadata=[SimpleNamespace(width=1280,height=721) for _ in names]
        cfg=Config(registration_width=640,feature_count=1000)
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            for i,name in enumerate(names):cv2.imwrite(str(folder/name),cv2.warpAffine(image,np.float32([[1,0,20*i],[0,1,0]]),(1280,721)))
            original=dict(i=0,j=1);edges=[original]
            result,report=recover(folder,names,metadata,[(0,1),(1,2)],edges,cfg,folder)
            self.assertTrue(report['connected']);self.assertEqual(components(3,result),[[0,1,2]])
            self.assertIs(result[0],original);bridge=result[-1]
            self.assertEqual((bridge['i'],bridge['j']),(1,2))
            np.testing.assert_allclose(np.median(bridge['b']-bridge['a'],axis=0),[10,0],atol=.2)
            self.assertLess(np.median(np.linalg.norm(project(bridge['H'],bridge['a'])-bridge['b'],axis=1)),.2)

    def test_unmatchable_images_are_reported_without_fake_edges(self):
        names=['blank1.png','blank2.png'];metadata=[SimpleNamespace(width=128,height=96) for _ in names]
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            for name in names:cv2.imwrite(str(folder/name),np.full((96,128,3),120,np.uint8))
            edges,report=recover(folder,names,metadata,[(0,1)],[],Config(registration_width=640),folder)
            self.assertEqual(edges,[]);self.assertFalse(report['connected'])
            self.assertEqual(report['components'],[[names[0]],[names[1]]])
            with self.assertRaisesRegex(RuntimeError,'No frames were dropped'):require_connected(names,edges,folder)

    def test_faint_texture_recovers_repeatably(self):
        rng=np.random.default_rng(91)
        noise=cv2.GaussianBlur(rng.integers(0,256,(721,1280,3),np.uint8),(5,5),1.)
        image=np.uint8(128+(noise.astype(float)-128)*.25)
        self.assertLess(len(cv2.SIFT_create(contrastThreshold=.025).detect(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))),35)
        names=['faint1.png','faint2.png'];meta=[SimpleNamespace(width=1280,height=721) for _ in names]
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);cv2.imwrite(str(folder/names[0]),image)
            cv2.imwrite(str(folder/names[1]),cv2.warpAffine(image,np.float32([[1,0,20],[0,1,0]]),(1280,721)))
            results=[]
            for _ in range(2):
                edges,report=recover(folder,names,meta,[(0,1)],[],Config(registration_width=640,feature_count=1000),folder)
                self.assertTrue(report['connected']);self.assertGreater(edges[0]['n'],100);results.append(edges[0])
            for key in ('H','a','b'):np.testing.assert_array_equal(results[0][key],results[1][key])
