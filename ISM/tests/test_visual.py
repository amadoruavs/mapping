import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('ism_visual',ROOT/'run_visual.py')
visual=importlib.util.module_from_spec(spec);spec.loader.exec_module(visual)
sys.path.pop(0)


class VisualTests(unittest.TestCase):
    def test_disabled_visual_routes_directly_to_base(self):
        with patch.object(visual.base,'run',return_value=0) as base,patch.object(visual,'VisualProcess',side_effect=AssertionError('preview installed')):
            self.assertEqual(visual.run('input','output',False),0)
            base.assert_called_once_with('input','output')

    @unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
    def test_observation_preserves_presentation_pixels_and_restores_hooks(self):
        from adm.metal import get_backend
        from adm.presentation_metal import PresentationMetal
        gpu=get_backend();engine=PresentationMetal(gpu,'siyi_low_light')
        rng=np.random.default_rng(77);image=rng.integers(30,180,(85,129,4),np.uint8);image[:,:,3]=255
        expected=engine.grade(engine.enhance(image,1.5),'pro_camera')
        original=PresentationMetal.enhance
        with tempfile.TemporaryDirectory() as folder:
            recorder=visual.VisualProcess(folder);recorder.native_shape=image.shape[:2];recorder.install()
            try:actual=engine.grade(engine.enhance(image,1.5),'pro_camera')
            finally:recorder.finish(True)
            np.testing.assert_array_equal(actual,expected)
            self.assertIs(PresentationMetal.enhance,original)
            report=json.loads((Path(folder)/'visual_process/stages.json').read_text())
            self.assertEqual(report['status'],'complete');self.assertGreaterEqual(len(report['stages']),7)
            self.assertNotIn('http-equiv="refresh"',(Path(folder)/'visual_process/process.html').read_text())
