from pathlib import Path
import hashlib
import http.client
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web import Jobs, RequestError, make_server, TERMINAL
from run import file_sha256


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.photos = self.root / 'photos with spaces'
        self.photos.mkdir()
        (self.photos / 'PHOTO.JPG').write_bytes(b'photo')
        self.worker = self.root / 'worker.py'
        self.worker.write_text('''import argparse,json,time,sys
from pathlib import Path
p=argparse.ArgumentParser()
for n in ('map','photos','output'):p.add_argument('--'+n,required=True)
a=p.parse_args();src=Path(a.map).read_bytes()
print('[preflight] starting',flush=True)
if src==b'fail':sys.exit('Deliberate worker failure')
if src==b'slow':time.sleep(30)
out=Path(a.output);out.mkdir()
(out/'final_2x.jpg').write_bytes(b'full jpeg bytes')
(out/'preview.jpg').write_bytes(b'preview bytes')
(out/'timing.json').write_text(json.dumps(dict(total_seconds=.1)))
(out/'final_geometry.json').write_text(json.dumps(dict(width=10,height=8)))
''')
        self.jobs = Jobs(self.root / 'runs', worker=self.worker, max_upload=1024**3)
        self.server = make_server(self.jobs)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.jobs.shutdown()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp.cleanup()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=15)
        supplied = {'X-MapRepair': '1'}
        if isinstance(body, dict):
            body = json.dumps(body).encode()
            supplied['Content-Type'] = 'application/json'
        supplied.update(headers or {})
        conn.request(method, path, body, supplied)
        response = conn.getresponse()
        data = response.read()
        info = dict(response.getheaders())
        status = response.status
        conn.close()
        if 'application/json' in info.get('Content-Type', ''):
            data = json.loads(data)
        return status, data, info

    def create(self, content=b'valid', **overrides):
        data = dict(filename='a map.tif', photos=f'"{self.photos}"', size=len(content))
        data.update(overrides)
        status, job, _ = self.request('POST', '/api/jobs', data)
        self.assertEqual(status, 201, job)
        return job['id']

    def wait(self, job_id, limit=10):
        start = time.monotonic()
        while time.monotonic() - start < limit:
            status, job, _ = self.request('GET', f'/api/jobs/{job_id}')
            self.assertEqual(status, 200)
            if job['state'] in TERMINAL:
                return job
            time.sleep(.04)
        self.fail('Job did not finish: ' + str(job))

    def start(self, content=b'valid'):
        job_id = self.create(content)
        status, _, _ = self.request('PUT', f'/api/jobs/{job_id}/input', content)
        self.assertEqual(status, 200)
        status, _, _ = self.request('POST', f'/api/jobs/{job_id}/start')
        self.assertEqual(status, 200)
        return job_id

    def test_http_run_refresh_download_and_persistent_result(self):
        status, page, _ = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn(b'Drop your TIFF', page)
        job_id = self.start()
        job = self.wait(job_id)
        self.assertEqual(job['state'], 'complete', job)
        self.assertEqual(job['dimensions'], [10, 8])
        self.assertEqual(self.request('GET', '/api/state')[1]['latest'], job_id)
        status, data, headers = self.request('GET', job['download'])
        self.assertEqual(status, 200)
        self.assertEqual(data, b'full jpeg bytes')
        self.assertEqual(headers['Content-Type'], 'image/jpeg')
        self.assertIn('attachment;', headers['Content-Disposition'])
        self.jobs.shutdown()
        recovered = Jobs(self.root / 'runs', worker=self.worker)
        self.assertEqual(recovered.snapshot(job_id)['state'], 'complete')
        recovered.shutdown()

    def test_bad_folder_empty_folder_extension_and_payload_rejected(self):
        base = dict(filename='map.tif', photos=str(self.photos), size=10)
        for overrides in (dict(photos='/does/not/exist'), dict(photos='relative'), dict(filename='script.py'), dict(size=0)):
            status, data, _ = self.request('POST', '/api/jobs', base | overrides)
            self.assertGreaterEqual(status, 400)
            self.assertIn('error', data)
        empty = self.root / 'empty';empty.mkdir()
        self.assertEqual(self.request('POST', '/api/jobs', base | dict(photos=str(empty)))[0], 400)
        self.assertFalse(self.jobs.jobs)

    def test_single_active_run_and_directory_owner(self):
        job_id = self.create()
        with self.assertRaises(RuntimeError):
            Jobs(self.root / 'runs', worker=self.worker)
        self.assertEqual(self.jobs.snapshot(job_id)['state'], 'awaiting_upload')
        status, _, _ = self.request('POST', '/api/jobs', dict(filename='a.tif', photos=str(self.photos), size=5))
        self.assertEqual(status, 409)
        self.assertEqual(self.request('GET', f'/api/jobs/{job_id}/download')[0], 409)
        self.assertEqual(self.request('POST', f'/api/jobs/{job_id}/start')[0], 409)

    def test_partial_upload_removed_and_retry_possible(self):
        job_id = self.create(b'12345678')
        with self.assertRaises(RequestError):
            self.jobs.upload(job_id, io.BytesIO(b'123'), 8)
        self.assertEqual(self.jobs.snapshot(job_id)['state'], 'failed')
        self.assertEqual(list((self.jobs.root / job_id / 'input').iterdir()), [])
        self.assertNotEqual(self.create(), job_id)

    def test_fail_cancel_and_retry_do_not_reuse_output(self):
        failed_id = self.start(b'fail')
        failed = self.wait(failed_id)
        self.assertEqual(failed['state'], 'failed')
        self.assertIn('Deliberate worker failure', failed['error'])
        slow_id = self.start(b'slow')
        self.assertEqual(self.request('POST', f'/api/jobs/{slow_id}/cancel')[0], 200)
        self.assertEqual(self.wait(slow_id)['state'], 'cancelled')
        deadline = time.monotonic() + 3
        while self.jobs.processes and time.monotonic() < deadline:
            time.sleep(.02)
        good_id = self.start()
        self.assertEqual(self.wait(good_id)['state'], 'complete')
        self.assertEqual(len({failed_id, slow_id, good_id}), 3)

    def test_local_origin_and_download_routes(self):
        data = dict(filename='map.tif', photos=str(self.photos), size=10)
        self.assertEqual(self.request('POST', '/api/jobs', data, {'Origin': 'https://example.com'})[0], 403)
        self.assertEqual(self.request('POST', '/api/jobs', data, {'X-MapRepair': ''})[0], 403)
        self.assertEqual(self.request('GET', '/', headers={'Host': 'example.com'})[0], 403)
        self.assertEqual(self.request('GET', '/../../run.py')[0], 404)

    def test_restart_marks_incomplete_jobs_interrupted(self):
        job_id = self.create()
        # Simulate a lost process without gracefully cancelling its job.
        self.jobs.directory_lock.close()
        recovered = Jobs(self.root / 'runs', worker=self.worker)
        self.assertEqual(recovered.snapshot(job_id)['state'], 'interrupted')
        recovered.shutdown()

    def test_folder_upload_requires_all_photos_and_preserves_duplicate_names(self):
        status, job, _ = self.request('POST', '/api/jobs', dict(filename='map.tif', size=5,
            photo_files=[dict(name='sub/a.JPG', size=3), dict(name='other/a.JPG', size=4)]))
        self.assertEqual(status, 201, job)
        job_id = job['id']
        self.assertEqual(self.request('PUT', f'/api/jobs/{job_id}/input', b'valid')[0], 200)
        self.assertEqual(self.request('POST', f'/api/jobs/{job_id}/start')[0], 409)
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda pair: self.request('PUT', f'/api/jobs/{job_id}/photos/{pair[0]}', pair[1]), [(0, b'one'), (1, b'two2')]))
        self.assertTrue(all(r[0] == 200 for r in results), results)
        state = self.jobs.snapshot(job_id)
        self.assertEqual(state['state'], 'ready')
        self.assertEqual(state['received'], 12)
        self.assertEqual(state['uploaded_photo_count'], 2)
        self.assertEqual(sorted(f.read_bytes() for f in Path(state['photos']).iterdir()), [b'one', b'two2'])
        self.assertEqual(self.request('PUT', f'/api/jobs/{job_id}/photos/0', b'one')[0], 409)
        self.assertEqual(self.request('POST', f'/api/jobs/{job_id}/start')[0], 200)
        self.assertEqual(self.wait(job_id)['state'], 'complete')

    def test_partial_photo_upload_fails_whole_job_without_starting(self):
        _, job, _ = self.request('POST', '/api/jobs', dict(filename='map.tif', size=5,
            photo_files=[dict(name='a.JPG', size=8)]))
        job_id = job['id']
        with self.assertRaises(RequestError):
            self.jobs.upload(job_id, io.BytesIO(b'123'), 8, photo_index=0)
        self.assertEqual(self.jobs.snapshot(job_id)['state'], 'failed')
        self.assertFalse(list((self.jobs.root / job_id / 'source_photos').iterdir()))
        self.assertEqual(self.request('POST', f'/api/jobs/{job_id}/start')[0], 409)

    def test_actual_tiff_to_downloadable_jpeg_and_exact_alpha(self):
        self.jobs.worker = Path(__file__).resolve().parents[1] / 'web_worker.py'
        rng = np.random.default_rng(123)
        rgb = rng.integers(30, 210, (160, 220, 3), dtype=np.uint8)
        alpha = np.full((160, 220), 255, np.uint8)
        alpha[:12] = 0;alpha[60:80, 80:100] = 128
        transform = from_origin(500000, 4000000, .05, .05)
        input_path = self.root / 'real.tif'
        with rasterio.open(input_path, 'w', driver='GTiff', width=220, height=160, count=4,
                           dtype='uint8', crs='EPSG:32615', transform=transform) as ds:
            ds.write(np.dstack([rgb, alpha]).transpose(2, 0, 1))
            ds.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
        Image.fromarray(rgb).save(self.photos / 'PHOTO.JPG')
        payload = input_path.read_bytes()
        job_id = self.start(payload)
        result = self.wait(job_id, limit=45)
        self.assertEqual(result['state'], 'complete', result)
        status, jpeg, headers = self.request('GET', result['download'])
        self.assertEqual(status, 200)
        with Image.open(io.BytesIO(jpeg)) as im:
            self.assertEqual(im.format, 'JPEG');self.assertEqual(im.size, (440, 320))
        output = self.jobs.root / job_id / 'output'
        with rasterio.open(output / 'final_2x_alpha.tif') as ds:
            np.testing.assert_array_equal(ds.read(1), np.repeat(np.repeat(alpha, 2, 0), 2, 1))
            self.assertEqual(ds.crs.to_epsg(), 32615)
            self.assertAlmostEqual(ds.transform.a, .025)
            self.assertEqual(ds.transform.c, transform.c)
        self.assertEqual(file_sha256(input_path), hashlib.sha256(payload).hexdigest())


if __name__ == '__main__':
    unittest.main()
