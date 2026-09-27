"""GPS validation must reject bad geometry and retain actionable diagnostics."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from adm.config import Config
from adm.geometry import LocalEarth,project
from adm.metadata import FrameMetadata
from adm.pipeline import align_centers_to_geography


class GeographicFitTests(unittest.TestCase):
    def fixture(self):
        earth=LocalEarth(40.,-90.);metadata=[];transforms=[];targets=[]
        for i,(x,y) in enumerate(((-30,-30),(0,-30),(30,-30),(-30,0),(0,0),(30,0),(-30,30),(0,30),(30,30))):
            lat,lon=earth.latlon([x,y]);metadata.append(FrameMetadata(str(i),100,80,latitude=float(lat),longitude=float(lon)))
            transforms.append(np.array([[1,0,x*10-49.5],[0,1,-y*10-39.5],[0,0,1.]]));targets.append([x,y])
        return earth,metadata,transforms,targets

    def test_valid_fit_preserves_geometry(self):
        earth,metadata,hs,targets=self.fixture()
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            actual=align_centers_to_geography(hs,metadata,earth,Config(export_profile='png'),Path(folder))
            np.testing.assert_allclose([project(H,[[49.5,39.5]])[0] for H in actual],targets,atol=1e-6)
            self.assertLess(json.loads((Path(folder)/'geographic_alignment.json').read_text())['median_center_residual_m'],1e-6)

    def test_failed_fit_writes_report_before_raising(self):
        earth,metadata,hs,_=self.fixture()
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'median .* exceeds 5.00 m; see'):
                align_centers_to_geography([hs[0]]*len(hs),metadata,earth,Config(export_profile='png'),Path(folder))
            report=json.loads((Path(folder)/'geographic_alignment.json').read_text())
            self.assertGreater(report['median_center_residual_m'],5)

    def test_explicit_bypass_continues_and_reports_failure(self):
        earth,metadata,hs,_=self.fixture()
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            result=align_centers_to_geography([hs[0]]*len(hs),metadata,earth,
                Config(export_profile='png',allow_gps_mismatch=True),Path(folder))
            self.assertEqual(len(result),len(hs))
            report=json.loads((Path(folder)/'geographic_alignment.json').read_text())
            self.assertGreater(report['median_center_residual_m'],5)
            self.assertFalse(report['gps_fit_within_tolerance'])
            self.assertTrue(report['gps_mismatch_bypassed'])

    def test_bypass_setting_requires_boolean(self):
        for value in ('true',1,None):
            with self.assertRaisesRegex(ValueError,'allow_gps_mismatch'):
                Config(allow_gps_mismatch=value).validate()
