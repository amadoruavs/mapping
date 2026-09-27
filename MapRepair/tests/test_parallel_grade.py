import json,os,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from advanced_grade import render

class ParallelGradeTests(unittest.TestCase):
    def test_serial_parallel_encoded_export(self):
        image=np.random.default_rng(365).integers(0,256,(777,257,4),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);image.tofile(p/'grade_master.bgra');(p/'final_geometry.json').write_text(json.dumps({'width':257,'height':777}))
            for settings in [{},{'exposure':.3,'temperature':-20,'sharpen':90,'radius':3}]:
                with patch.dict(os.environ,{'ADM_PARALLEL':'0','ADM_PIXEL_BACKEND':'cpu'}):serial=render(p,settings)
                with patch.dict(os.environ,{'ADM_PARALLEL':'1','ADM_PARALLEL_WORKERS':'4','ADM_PIXEL_BACKEND':'cpu'}):parallel=render(p,settings)
                self.assertEqual(serial,parallel)
