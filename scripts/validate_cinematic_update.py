"""Check the supplied grade on a saved neutral map; never runs flight mapping."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ISM/runtime'))
import cv2
import numpy as np
from adm.cinematic_grade import grade_bgra,PRESETS,REGION_PRESETS
from adm.lasso import crop_in_place
from adm.metal import get_backend
from adm.presentation_metal import PresentationMetal


class UnbatchedPresentation(PresentationMetal):
    """Same recipe, but submit/wait after each blur pass as before batching."""
    def pillow_blur(self,source,radius):
        dispatch=self.gpu.dispatch
        def immediate(*args,**kwargs):
            dispatch(*args,**kwargs);self.gpu.sync()
        self.gpu.dispatch=immediate
        try:return super().pillow_blur(source,radius)
        finally:self.gpu.dispatch=dispatch


def difference(a,b):
    histogram=np.zeros(256,np.int64)
    for y in range(0,len(a),128):
        d=np.abs(a[y:y+128,:,:3].astype(np.int16)-b[y:y+128,:,:3].astype(np.int16))
        histogram+=np.bincount(d.ravel(),minlength=256)
    n=int(histogram.sum());nz=np.flatnonzero(histogram)
    return dict(mean_rgb_code_error=float(histogram@np.arange(256)/n),
                max_rgb_code_error=int(nz[-1]),
                fraction_rgb_channels_identical=float(histogram[0]/n),
                p999_rgb_code_error=int(np.searchsorted(histogram.cumsum(),n*.999)))


def timed(fn):
    t=time.perf_counter();result=fn();return result,time.perf_counter()-t


def main():
    source=ROOT/'final_test'
    out=ROOT/'ISM/benchmarks/grader_update';out.mkdir(parents=True,exist_ok=True)
    gpu=get_backend();engine=PresentationMetal(gpu,'siyi_low_light')
    raw=cv2.imread(str(source/'map_neutral.png'),cv2.IMREAD_UNCHANGED)
    if raw is None:raise ValueError('Saved 94-photo neutral map is missing')
    geometry=json.loads((source/'map_geometry.json').read_text())
    if len(geometry['input_names'])!=94:raise ValueError('Expected the saved 94-photo map')
    enhanced,enhance_seconds=timed(lambda:engine.enhance(raw,1.))
    print('Sharpened saved neutral map',enhance_seconds,flush=True)
    before=gpu.stats()
    graded,grade_seconds=timed(lambda:engine.grade(enhanced,'pro_camera'))
    after=gpu.stats()
    unbatched,unbatched_seconds=timed(lambda:UnbatchedPresentation(gpu).grade(enhanced,'pro_camera'))
    assert np.array_equal(graded,unbatched),'Batching changed grade pixels'
    cpu,cpu_seconds=timed(lambda:grade_bgra(enhanced,'pro_camera'))
    error=difference(cpu,graded)
    print('Full-map CPU/Metal parity',error,flush=True)
    # A one-code tone-rounding difference can cross Pillow's threshold twice
    # (clarity, then fine unsharp). Check the unsharpened grade separately.
    base_cpu=grade_bgra(enhanced,'pro_camera',overrides={'sharpness':0,'clarity':0})
    base_gpu=engine.grade(enhanced,'pro_camera',overrides={'sharpness':0,'clarity':0})
    base_error=difference(base_cpu,base_gpu)
    assert base_error['max_rgb_code_error']<=1
    assert error['mean_rgb_code_error']<.002 and error['max_rgb_code_error']<=5
    np.testing.assert_array_equal(graded[:,:,3],enhanced[:,:,3])
    # Warm repeat, also used to verify deterministic behavior.
    repeated,repeat_seconds=timed(lambda:engine.grade(enhanced,'pro_camera'))
    assert np.array_equal(graded,repeated),'Grade is not deterministic'
    # CPU copies ensure both lasso variants receive identical uncropped pixels.
    old,old_crop_seconds=timed(lambda:crop_in_place(np.array(graded),gpu,
        inset_ratio=.01,opening_ratio=.09,angular_radius=18))
    new,new_crop_seconds=timed(lambda:crop_in_place(np.array(graded),gpu))
    assert new.report['retained_pixels']>old.report['retained_pixels']
    for y in range(0,len(graded),128):
        keep=new.image[y:y+128,:,3]>0
        np.testing.assert_array_equal(new.image[y:y+128][keep],graded[y:y+128][keep])
    def save():
        if not cv2.imwrite(str(out/'map_graded.png'),new.image):raise OSError('PNG write failed')
        preview=cv2.resize(new.image,(1300,round(len(raw)*1300/raw.shape[1])),interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(out/'preview.png'),preview):raise OSError('Preview write failed')
    _,export_seconds=timed(save)
    geometry.update(covered_pixels=new.report['retained_pixels'],lasso_crop='lasso_crop.json',applies_to=['map_graded.png'])
    (out/'presentation_geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    (out/'lasso_crop.json').write_text(json.dumps(new.report,indent=2)+'\n')
    report=dict(kind='Presentation only on saved 94-photo neutral map; not a raw-photo E2E run',
        source=str(source/'map_neutral.png'),runtime=str(sys.modules['adm'].__file__),
        source_sha256=hashlib.sha256((source/'map_neutral.png').read_bytes()).hexdigest(),
        shader_sha256=hashlib.sha256((ROOT/'ISM/ISM.metal').read_bytes()).hexdigest(),
        size=[raw.shape[1],raw.shape[0]],preset=PRESETS['pro_camera'],
        sharpen_seconds=enhance_seconds,grade_seconds=grade_seconds,warm_grade_seconds=repeat_seconds,
        unbatched_grade_seconds=unbatched_seconds,cpu_grade_seconds=cpu_seconds,
        grade_command_submissions=after['command_submissions']-before['command_submissions'],
        grade_metal_dispatches=after['msl_dispatches']-before['msl_dispatches'],
        batching_bit_exact=True,repeat_bit_exact=True,cpu_metal_error=error,tone_before_unsharp_error=base_error,
        old_crop=dict(seconds=old_crop_seconds,**old.report),new_crop=dict(seconds=new_crop_seconds,**new.report),
        cropped_pixels_exact=True,png_and_preview_export_seconds=export_seconds,
        timing_scope='Excludes input decode and GPU initialization; single-machine samples, not E2E speed claims')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('old_crop','new_crop','preset')},indent=2))
    print('Retained coverage:',old.report['retained_fraction'],'->',new.report['retained_fraction'])


if __name__=='__main__':main()
