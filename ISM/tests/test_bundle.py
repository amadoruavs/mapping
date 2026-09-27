"""Relocation and launcher safeguards are part of the standalone contract."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BundleTests(unittest.TestCase):
    def test_one_runtime_shader_source(self):
        # User-created backup.metal is not a runtime dependency.
        manifest=json.loads((ROOT/'runtime/bundle_manifest.json').read_text())
        self.assertEqual(manifest['shader'],'ISM.metal')
        self.assertTrue((ROOT/manifest['shader']).is_file())
        self.assertEqual(list((ROOT/'runtime').rglob('*.metal')),[])

    def test_nonempty_output_is_preserved(self):
        spec = importlib.util.spec_from_file_location('ism_runner', ROOT / 'run.py')
        runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'images';source.mkdir()
            output=Path(folder)/'output';output.mkdir();sentinel=output/'keep.txt';sentinel.write_text('keep')
            with self.assertRaises(ValueError):runner.run(source,output)
            self.assertEqual(sentinel.read_text(),'keep')

    @unittest.skipUnless(os.environ.get('ADM_TEST_METAL')=='1','requires Metal')
    def test_relocated_package_compiles_without_parent_checkout(self):
        with tempfile.TemporaryDirectory(prefix='ism-relocation-') as folder:
            destination=Path(folder)/'ISM';destination.mkdir()
            for name in ('run.py','ISM.metal'):shutil.copy2(ROOT/name,destination/name)
            shutil.copytree(ROOT/'runtime',destination/'runtime',ignore=shutil.ignore_patterns('__pycache__'))
            environment=os.environ.copy();environment.pop('PYTHONPATH',None)
            completed=subprocess.run([sys.executable,str(destination/'run.py'),'--compile-only'],
                cwd=folder,env=environment,capture_output=True,text=True,timeout=60)
            self.assertEqual(completed.returncode,0,completed.stderr[-5000:])
            report=json.loads((destination/'build/compile_report.json').read_text())
            self.assertTrue(report['all_expected_kernels_exported'])
            self.assertEqual(report['shader_kernel_count'],json.loads((ROOT/'runtime/bundle_manifest.json').read_text())['kernel_count'])
            self.assertTrue(Path(report['cpu_runtime']).is_relative_to(destination.resolve()))
