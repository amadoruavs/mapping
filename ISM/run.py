#!/usr/bin/env python3
"""Run ISM. Edit the two paths below, then run: python3 run.py"""
from pathlib import Path

# --------------------- EDIT THESE PATHS ---------------------
INPUT_REPO = Path('/Users/avneh/Downloads/mapv2')
OUTPUT_DESTINATION = Path('/Users/avneh/Code/ADM/final_test')
# -----------------------------------------------------------

# SIYI sharpening followed by the supplied pro_camera color grader.
# AGL is height ABOVE ground.
SETTINGS = {
    'altitude_agl_m': 51.816,
    'allow_gps_mismatch': False,
    'skip_first': 0,
    'registration_width': 1280,
    'coherent_budget_mb': 128,
    'alignment_budget_mb': 1024,
    'alignment_max_iterations': 600,  # Maximum; existing convergence checks stop early.
    'alignment_symmetric_transfer': True,  # Source-pixel errors cannot improve by shrinking the map.
    'geographic_solver': 'visual',  # Best verified default; 'opensfm_planar' enables the new geographic refinement.
    'geographic_gps_sigma_m': 10.0,
    'geographic_altitude_sigma_m': 10.0,
    'geographic_orientation_sigma_deg': 10.0,
    'interpolation_levels': 0,  # Real photos: measured faster and avoids synthetic road fragments.
    'fsr': False,
    'presentation': True,
    'presentation_fsr_passes': 0,
    'presentation_scale': 1.0,
    'presentation_raytrace': False,
    'presentation_profile': 'siyi_low_light',
    'color_grade': 'pro_camera',  # Exact supplied preset; all seven presets supported.
    'regional_grade': 'none',  # e.g. 'aerial_documentary_regional'
    'seed': 2026,  # Reproducible grain for presets that include it.
    'fsr_frame_budget_mb': 1024,
    'lasso_crop': True,  # Final image operation, after sharpening and color grading.
    'lasso_inset_ratio': 0.005,  # Was .01; keep more perimeter detail.
    'lasso_opening_ratio': 0.06,  # Was .09.
    'lasso_angular_radius': 12,  # Was 18.
}
MAX_OUTPUT_MB = 768
ROOT = Path(__file__).resolve().parent

import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
import os
import resource
import subprocess
import sys
import time


@contextmanager
def timer(label, results):
    """Measure elapsed wall time, including time spent waiting for GPU work."""
    started = time.perf_counter()
    try:
        yield
    finally:
        results[label] = time.perf_counter() - started
        print(f'{label}: {results[label]:.3f} seconds', flush=True)


def load_runtime():
    # Absolute path derived from this file; no parent ADM checkout is imported.
    sys.path.insert(0, str(ROOT / 'runtime'))


def compile_shaders():
    load_runtime()
    from adm.metal import get_backend
    from adm.fsr_framegen import FrameGenerator
    times = {}
    with timer('compile_and_initialize_seconds', times):
        gpu = get_backend()
        framegen = FrameGenerator()
        gpu.sync()
        exported = json.loads(gpu.lib.adm_function_names(gpu.handle))
        expected = json.loads((ROOT / 'runtime/bundle_manifest.json').read_text())['entrypoints']
        missing = sorted(set(expected) - set(exported))
        if missing: raise RuntimeError(f'Missing consolidated kernels: {missing}')
    report = {
        **times, 'device': gpu.device,
        'shader': 'ISM.metal',
        'shader_sha256': hashlib.sha256((ROOT / 'ISM.metal').read_bytes()).hexdigest(),
        'shader_kernel_count': json.loads((ROOT / 'runtime/bundle_manifest.json').read_text())['kernel_count'],
        'fsr_pass_count': len(framegen.passes),
        'all_expected_kernels_exported': True,
        'compilation': 'One runtime-compiled Metal 4 library shared by ADM and FSR',
        'cpu_runtime': str(Path(sys.modules['adm'].__file__).resolve()),
    }
    (ROOT / 'build').mkdir(exist_ok=True)
    (ROOT / 'build/compile_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def self_test():
    report = compile_shaders()
    import unittest
    os.environ['ADM_TEST_METAL'] = '1'
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report['tests'] = {'count': result.testsRun, 'failures': len(result.failures),
                       'errors': len(result.errors), 'skipped': len(result.skipped),
                       'seconds': time.perf_counter() - started, 'passed': result.wasSuccessful()}
    (ROOT / 'build/validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if result.wasSuccessful() else 1


def photo_preview(source,output,check_fsr=False):
    """Inspect the selected presentation settings on one photograph."""
    load_runtime()
    import cv2
    import numpy as np
    from adm.metal import get_backend
    from adm.presentation_metal import PresentationMetal
    from adm.raytracing import relight_map
    image=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim!=3:raise ValueError('Photo preview expects one readable color image')
    if image.shape[2]==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Choose a new photo-preview folder')
    output.mkdir(parents=True,exist_ok=True);times={};gpu=get_backend()
    engine=PresentationMetal(gpu,SETTINGS.get('presentation_profile','standard'))
    with timer('presentation_and_export_seconds',times):
        enhanced=engine.enhance(image,SETTINGS['presentation_scale'])
        lit,ray=relight_map(enhanced,gpu) if SETTINGS['presentation_raytrace'] else (enhanced,{'enabled':False})
        regional=SETTINGS.get('regional_grade','none')
        final=engine.grade(lit,SETTINGS['color_grade'],region_preset=None if regional=='none' else regional,seed=SETTINGS.get('seed',2026))
        if not cv2.imwrite(str(output/'photo_graded.png'),final):raise OSError('Failed to save photo preview')
    report={**times,'source':str(source),'source_size':[image.shape[1],image.shape[0]],
        'output_size':[final.shape[1],final.shape[0]],'settings':SETTINGS,'sharpen_stages':engine.timings,
        'raytracing':ray,'kind':'Single-photo presentation preview; no mapping or registration'}
    if check_fsr:
        from adm.fsr_framegen import FrameGenerator
        fsr=FrameGenerator(SETTINGS['altitude_agl_m'],81.,SETTINGS.get('fsr_frame_budget_mb',512))
        with timer('full_size_fsr_identity_seconds',times):
            generated,_=fsr.interpolate(image,image,np.eye(3),np.eye(3))
        difference=np.abs(generated.astype(np.int16)-image.astype(np.int16))
        report['full_size_fsr_check']={'seconds':times['full_size_fsr_identity_seconds'],
            'mean_abs_error':float(difference.mean()),'max_abs_error':int(difference.max()),
            'resource_budget_mb':fsr.resource_budget_mb,'dimensions':[image.shape[1],image.shape[0]]}
    (output/'preview_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Photo preview: {output / "photo_graded.png"}')
    return 0


def worker(source, output, config):
    load_runtime()
    from adm.pipeline import main as pipeline_main
    sys.argv = ['ism-pipeline', str(source), '--output', str(output), '--config', str(config),
                '--gps-centers', '--backend', 'metal', '--gpu-optimized', '--export-profile', 'png',
                '--max-output-mb', str(MAX_OUTPUT_MB)]
    pipeline_main()


def run(source=INPUT_REPO, output=OUTPUT_DESTINATION, geographic_solver=None):
    """Time a fresh child process, including imports, decoding, GPU and export."""
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f'Input photo directory does not exist: {source}')
    if output == source or output in source.parents or source in output.parents:
        raise ValueError('Keep the output directory separate from the input photo directory')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Output must be empty. Choose a new directory to preserve previous results: {output}')
    output.mkdir(parents=True, exist_ok=True)
    config = output / 'input_config.json'
    settings=dict(SETTINGS)
    if geographic_solver is not None:settings['geographic_solver']=geographic_solver
    config.write_text(json.dumps(settings, indent=2) + '\n')
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--input', str(source),
               '--output', str(output), '--config', str(config)]
    times = {}
    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f'Input: {source}\nOutput: {output}\nProgress log: {output / "run.log"}', flush=True)
    with (output / 'run.log').open('w') as log, timer('wall_seconds', times):
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = {**times, 'start_utc': start_utc, 'exit_code': completed.returncode,
              'input': str(source), 'output': str(output),
              'cpu_user_seconds': usage.ru_utime, 'cpu_system_seconds': usage.ru_stime,
              'peak_child_rss_bytes': usage.ru_maxrss * (1024 if sys.platform.startswith('linux') else 1),
              'rss_note': 'Maximum child process RSS, not total unified GPU memory',
              'includes': 'Fresh worker startup/imports, runtime GPU setup, raw-photo processing and final PNG export',
              'shader_sha256': hashlib.sha256((ROOT / 'ISM.metal').read_bytes()).hexdigest()}
    (output / 'timing.json').write_text(json.dumps(report, indent=2) + '\n')
    if completed.returncode:
        print((output / 'run.log').read_text()[-8000:], file=sys.stderr)
        return completed.returncode
    pipeline = json.loads((output / 'report.json').read_text())
    print(f"Retained {pipeline['retained_images']} originals; generated {pipeline.get('virtual_views', 0)} views.")
    print(f"Final image: {output / 'map_graded.png'}")
    print(f"Full benchmark: {output / 'timing.json'}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT_REPO, help='Optional override of INPUT_REPO')
    parser.add_argument('--output', type=Path, default=OUTPUT_DESTINATION, help='Optional override of OUTPUT_DESTINATION')
    parser.add_argument('--geographic-solver',choices=('visual','opensfm_planar'),help='Override the top-level solver setting for this run')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--compile-only', action='store_true', help='Compile the unified shader and initialize all FSR passes')
    mode.add_argument('--self-test', action='store_true', help='Run the included CPU/GPU parity and ray-tracing checks')
    mode.add_argument('--photo-preview',action='store_true',help='Apply the current presentation settings to one --input photo')
    parser.add_argument('--check-photo-fsr',action='store_true',help='With --photo-preview, also validate FSR at the full photo size')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--config', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.config is None: parser.error('Worker requires --config')
        worker(args.input, args.output, args.config)
        return 0
    if args.self_test: return self_test()
    if args.photo_preview:return photo_preview(args.input,args.output,args.check_photo_fsr)
    if args.compile_only:
        print(json.dumps(compile_shaders(), indent=2))
        return 0
    return run(args.input, args.output,args.geographic_solver)


if __name__ == '__main__':
    raise SystemExit(main())
