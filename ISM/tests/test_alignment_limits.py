"""A larger iteration allowance must retain the solver's early exit."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from adm.config import Config
from adm.geometry import project
from adm.registration import solve


class AlignmentLimitTests(unittest.TestCase):
    def test_symmetric_config_and_dispatch(self):
        Config(gpu_alignment=True,alignment_symmetric_transfer=True).validate()
        for invalid in (1,'true',None):
            with self.assertRaises(ValueError):Config(alignment_symmetric_transfer=invalid).validate()
        with self.assertRaises(ValueError):Config(alignment_symmetric_transfer=True).validate()
        with patch('adm.gpu_alignment.solve',return_value='forwarded') as device:
            result=solve(['a','b'],[dict(i=0,j=1)],Path('.'),640,object(),symmetric=True)
            self.assertEqual(result,'forwarded');self.assertTrue(device.call_args.kwargs['symmetric'])

    def test_limit_validation_and_dispatch(self):
        cfg=Config(alignment_max_iterations=600);cfg.validate()
        for invalid in (0,601,60.5,True):
            with self.assertRaisesRegex(ValueError,'iteration limit'):
                Config(alignment_max_iterations=invalid).validate()
        gpu=object()
        with patch('adm.gpu_alignment.solve',return_value='forwarded') as device:
            result=solve(['a','b'],[dict(i=0,j=1)],Path('.'),640,gpu,
                         iterations=cfg.alignment_max_iterations,workspace_budget_mb=1024)
            self.assertEqual(result,'forwarded')
            self.assertEqual(device.call_args.kwargs,dict(iterations=600,workspace_budget_mb=1024))

    @unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
    def test_600_cap_stops_early_with_same_solution(self):
        from adm.metal import get_backend
        rng=np.random.default_rng(9);width=600
        transforms=[np.array([[1,.02,70],[-.01,.98,20],[.00003,-.00002,1.]]),np.eye(3),
                    np.array([[.99,-.01,-85],[.03,1.02,-15],[-.00002,.00001,1.]])]
        edges=[]
        for i,j in ((0,1),(1,2),(0,2)):
            a=rng.uniform([60,60],[420,330],(90,2));H=np.linalg.inv(transforms[j])@transforms[i];H/=H[2,2]
            b=project(H,a)+rng.normal(0,.03,a.shape);initial=H.copy();initial[0,2]+=1.3
            edges.append(dict(i=i,j=j,H=initial,a=a,b=b,n=90))
        results=[];reports=[]
        with tempfile.TemporaryDirectory() as folder:
            for cap in (60,600):
                with contextlib.redirect_stdout(io.StringIO()):
                    results.append(solve(['a','b','c'],edges,Path(folder),width,get_backend(),iterations=cap))
                report=json.loads((Path(folder)/'registration_report.json').read_text());reports.append(report)
                self.assertTrue(report['solver_success']);self.assertEqual(report['iterations_budget'],cap)
                self.assertLess(report['iterations_encoded'],60)
                self.assertLessEqual(report['iterations_encoded']-report['iterations_executed'],3)
                self.assertLess(report['median_registration_error_px'],.06)
            for i in range(3):np.testing.assert_array_equal(results[0][i],results[1][i])
            self.assertEqual(reports[0]['iterations_executed'],reports[1]['iterations_executed'])
            print('600-iteration cap: converged after',reports[1]['iterations_executed'],
                  'iterations;',reports[1]['iterations_encoded'],'encoded; identical to cap 60',flush=True)
