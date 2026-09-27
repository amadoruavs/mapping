#!/usr/bin/env python3
"""Local MapRepair UI. Run: python web.py [--port 8765] [--no-browser]."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid
import webbrowser

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
PHOTO_EXTENSIONS = EXTENSIONS | {'.webp'}
MAX_UPLOAD_BYTES = 8 * 1024**3
CHUNK = 1024**2
TERMINAL = {'complete', 'failed', 'cancelled', 'interrupted'}


class RequestError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def tail(path, size=48 * 1024):
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - size))
            return f.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        else:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass


class Jobs:
    def __init__(self, root, worker=None, max_upload=MAX_UPLOAD_BYTES):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.directory_lock = (self.root / '.server.lock').open('a+b')
        if self.directory_lock.tell() == 0:
            self.directory_lock.write(b'0')
            self.directory_lock.flush()
        self.directory_lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.directory_lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.directory_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.directory_lock.close()
            raise RuntimeError('MapRepair is already using this runs folder. Open the existing server or choose another --runs-dir.') from exc
        self.worker = Path(worker or ROOT / 'web_worker.py')
        self.max_upload = max_upload
        self.lock = threading.RLock()
        self.processes = {}
        self.jobs = {}
        for path in self.root.glob('*/job.json'):
            try:
                job = json.loads(path.read_text())
                if not re.fullmatch(r'[a-f0-9]{32}', path.parent.name) or job['id'] != path.parent.name:
                    continue
                if job['state'] not in TERMINAL:
                    job.update(state='interrupted', error='The server stopped before this run finished. Select the map and run again.', ended=time.time())
                self.jobs[job['id']] = job
                self.save(job)
            except (OSError, ValueError, KeyError):
                continue

    def save(self, job):
        directory = self.root / job['id']
        temp = directory / 'job.json.tmp'
        temp.write_text(json.dumps(job, indent=2) + '\n')
        temp.replace(directory / 'job.json')

    def get(self, job_id):
        if not re.fullmatch(r'[a-f0-9]{32}', job_id) or job_id not in self.jobs:
            raise RequestError('Run not found.', 404)
        return self.jobs[job_id]

    def create(self, data):
        filename = str(data.get('filename', '')).replace('\\', '/').rsplit('/', 1)[-1]
        if Path(filename).suffix.lower() not in EXTENSIONS:
            raise RequestError('Choose a TIFF, PNG or JPEG map.')
        filename = re.sub(r'[^\w. -]', '_', Path(filename).stem)[:160] + Path(filename).suffix.lower()
        manifest = data.get('photo_files')
        if manifest is not None:
            if not isinstance(manifest, list) or not 1 <= len(manifest) <= 10000:
                raise RequestError('Choose a folder with between 1 and 10,000 supported photos.')
            files = []
            for index, entry in enumerate(manifest):
                if not isinstance(entry, dict):
                    raise RequestError('Invalid photo list.')
                name = str(entry.get('name', '')).replace('\\', '/').rsplit('/', 1)[-1]
                if Path(name).suffix.lower() not in PHOTO_EXTENSIONS:
                    raise RequestError(f'Unsupported photo: {name}')
                if type(entry.get('size')) is not int or not 0 < entry['size'] <= self.max_upload:
                    raise RequestError(f'Invalid photo size: {name}')
                safe = re.sub(r'[^\w. -]', '_', Path(name).stem)[:140] + Path(name).suffix.lower()
                files.append(dict(name=name, stored=f'{index:05d}_{safe}', size=entry['size'], state='pending', received=0))
            count = len(files)
            photos = None
        else:
            files = []
            # Browser paths pasted with surrounding shell quotes are common.
            folder = str(data.get('photos', '')).strip().strip('"\'')
            if not folder:
                raise RequestError('Drop a photos folder or enter its full path.')
            photos = Path(folder).expanduser()
            if not photos.is_absolute():
                raise RequestError('Use the full folder path, starting with / on Mac/Linux or a drive letter on Windows.')
            if not photos.is_dir():
                raise RequestError('Photos folder not found on this computer. Check the path.')
            try:
                count = sum(p.is_file() and p.suffix.lower() in PHOTO_EXTENSIONS for p in photos.iterdir())
            except OSError as exc:
                raise RequestError(f'Cannot read the photos folder: {exc}') from exc
            if not count:
                raise RequestError('No supported photos found directly inside this folder.')
        size = data.get('size')
        if type(size) is not int or not 0 < size <= self.max_upload:
            raise RequestError(f'Map must be nonempty and no larger than {self.max_upload / 1024**3:g} GB.', 413)
        total_bytes = size + sum(f['size'] for f in files)
        if shutil.disk_usage(self.root).free < total_bytes + 128 * CHUNK:
            raise RequestError('Not enough free disk space to upload the selected files.', 507)
        with self.lock:
            if self.processes or any(j['state'] not in TERMINAL for j in self.jobs.values()):
                raise RequestError('Another run is active. Wait for it to finish or stop it first.', 409)
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            directory.mkdir()
            (directory / 'input').mkdir()
            (directory / 'run.log').touch()
            if manifest is not None:
                photos = directory / 'source_photos'
                photos.mkdir()
            job = dict(id=job_id, filename=filename, photos=str(photos.resolve()), photo_count=count,
                       size=size, total_bytes=total_bytes, received=0, map_received=0, map_uploaded=False,
                       uploaded_photo_count=0, photo_files=files, source_mode='upload' if manifest is not None else 'path',
                       state='awaiting_upload', created=time.time(), error=None)
            self.jobs[job_id] = job
            self.save(job)
            return self.snapshot(job_id)

    def snapshot(self, job_id):
        with self.lock:
            job = dict(self.get(job_id))
            job.pop('photo_files', None)  # Do not poll a large per-file manifest.
        directory = self.root / job_id
        log = tail(directory / 'run.log')
        job['log'] = log
        job['elapsed'] = max(0, job.get('ended', time.time()) - job.get('started', job['created']))
        job['stage'] = 'Preparing'
        names = {'preflight': 'Checking input and dependencies', 'load': 'Reading map',
                 'scan_and_source_cross_checks': 'Checking source photos', 'repair': 'Repairing supported areas',
                 'native_build': 'Preparing image processor', 'openrt_build': 'Preparing lighting',
                 'ism_sharpen': 'Sharpening', 'ism_color_grade': 'Color grading',
                 'cinematic_finish': 'Color grading', 'adaptive_finish': 'Enhancing tone and fine detail', 'raytrace': 'Applying lighting',
                 'highlight_bloom': 'Finishing highlights', 'fsr_2x': 'Upscaling 2×', 'final_export': 'Saving full JPEG'}
        for line in log.splitlines():
            m = re.match(r'\[([\w_]+)\] starting', line)
            if m:
                job['stage'] = names.get(m[1], m[1].replace('_', ' '))
            m = re.match(r'\[scan (\d+)/(\d+)\]', line)
            if m:
                job['stage'] = f'Checking source photos · {m[1]} / {m[2]}'
        job['download'] = f'/api/jobs/{job_id}/download' if job['state'] == 'complete' else None
        job['preview'] = f'/api/jobs/{job_id}/preview' if job['state'] == 'complete' else None
        job['log_url'] = f'/api/jobs/{job_id}/log'
        job['output_folder'] = str(directory / 'output')
        if job['state'] == 'complete':
            try:
                timing = json.loads((directory / 'output/timing.json').read_text())
                job['pipeline_seconds'] = timing['total_seconds']
                geometry = json.loads((directory / 'output/final_geometry.json').read_text())
                job['dimensions'] = [geometry['width'], geometry['height']]
            except (OSError, ValueError, KeyError):
                pass
        return job

    def upload(self, job_id, stream, length, photo_index=None):
        with self.lock:
            job = self.get(job_id)
            if job['state'] not in ('awaiting_upload', 'uploading'):
                raise RequestError('This run is not waiting for files.', 409)
            if photo_index is not None:
                if not 0 <= photo_index < len(job['photo_files']):
                    raise RequestError('Photo not found in this run.', 404)
                entry = job['photo_files'][photo_index]
                if entry['state'] != 'pending':
                    raise RequestError('This photo is already uploaded or uploading.', 409)
                expected = entry['size']
                path = self.root / job_id / 'source_photos' / entry['stored']
            else:
                if job['map_uploaded'] or job.get('map_uploading'):
                    raise RequestError('This map is already uploaded or uploading.', 409)
                entry = None
                expected = job['size']
                path = self.root / job_id / 'input' / job['filename']
            if length != expected:
                raise RequestError('Upload length does not match the selected file.')
            if entry is not None:
                entry['state'] = 'uploading'
            else:
                job['map_uploading'] = True
            job['state'] = 'uploading'
            self.save(job)
        part = path.with_name(path.name + '.part')
        try:
            with part.open('wb') as dst:
                remaining = length
                while remaining:
                    with self.lock:
                        if job['state'] in TERMINAL:
                            raise RequestError('Upload stopped.')
                    chunk = stream.read(min(CHUNK, remaining))
                    if not chunk:
                        raise RequestError('Upload interrupted. Select the map and try again.')
                    dst.write(chunk)
                    remaining -= len(chunk)
                    with self.lock:
                        job['received'] += len(chunk)
                        if entry is not None:
                            entry['received'] += len(chunk)
                        else:
                            job['map_received'] += len(chunk)
            with self.lock:
                if job['state'] in TERMINAL:
                    raise RequestError('Upload stopped.')
                part.replace(path)
                if entry is not None:
                    entry['state'] = 'complete'
                    job['uploaded_photo_count'] += 1
                else:
                    job['map_uploaded'] = True
                    job['map_uploading'] = False
                if job['map_uploaded'] and job['uploaded_photo_count'] == len(job['photo_files']):
                    job['state'] = 'ready'
                self.save(job)
        except Exception as exc:
            part.unlink(missing_ok=True)
            with self.lock:
                if job['state'] not in TERMINAL:
                    job.update(state='failed', error=str(exc), ended=time.time())
                    self.save(job)
            raise

    def start(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job['state'] != 'ready':
                raise RequestError('Upload the map completely before starting.', 409)
            job.update(state='running', started=time.time())
            self.save(job)
            threading.Thread(target=self._work, args=(job_id,), daemon=True).start()

    def _work(self, job_id):
        directory = self.root / job_id
        try:
            with (directory / 'run.log').open('ab', buffering=0) as log:
                with self.lock:
                    job = self.get(job_id)
                    if job['state'] == 'cancelled':
                        return
                    command = [sys.executable, '-u', str(self.worker), '--map',
                               str(directory / 'input' / job['filename']), '--photos', job['photos'],
                               '--output', str(directory / 'output')]
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=os.name != 'nt')
                    self.processes[job_id] = process
                code = process.wait()
            with self.lock:
                if job['state'] != 'cancelled':
                    complete = code == 0 and (directory / 'output/final_2x.jpg').is_file()
                    lines = tail(directory / 'run.log').strip().splitlines()
                    error = lines[-1] if lines else f'Image processor exited with code {code}.'
                    if code < 0:
                        error = f'Image processor was stopped by the operating system (signal {-code}). Check available memory and the run log.'
                    job.update(state='complete' if complete else 'failed', error=None if complete else error, ended=time.time())
                    self.save(job)
        except Exception as exc:
            with self.lock:
                job = self.get(job_id)
                if job['state'] != 'cancelled':
                    job.update(state='failed', error=str(exc), ended=time.time())
                    self.save(job)
        finally:
            with self.lock:
                self.processes.pop(job_id, None)

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job['state'] in TERMINAL:
                return
            job.update(state='cancelled', error='Stopped. You can start a new run.', ended=time.time())
            self.save(job)
            process = self.processes.get(job_id)
        stop_process(process)

    def shutdown(self):
        with self.lock:
            active = [j['id'] for j in self.jobs.values() if j['state'] not in TERMINAL]
        for job_id in active:
            self.cancel(job_id)
        self.directory_lock.close()


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, jobs, **kwargs):
        self.jobs = jobs
        super().__init__(*args, **kwargs)

    def log_message(self, *_):
        pass

    def send_headers(self, status, kind, length, download=None):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(length))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if download:
            self.send_header('Content-Disposition', f'attachment; filename="{download}"')
        self.end_headers()

    def json(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_headers(status, 'application/json', len(data))
        self.wfile.write(data)

    def file(self, path, kind, download=None):
        if not path.is_file():
            raise RequestError('File is not available.', 404)
        with path.open('rb') as f:
            self.send_headers(200, kind, os.fstat(f.fileno()).st_size, download)
            shutil.copyfileobj(f, self.wfile, CHUNK)

    def dispatch(self):
        # Loopback binding + Host/Origin checks keep a random website from
        # starting local filesystem jobs. No authentication setup is needed.
        port = self.server.server_address[1]
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host') not in hosts:
            raise RequestError('Open the local URL printed by web.py.', 403)
        if self.command != 'GET':
            origin = self.headers.get('Origin')
            if self.headers.get('X-MapRepair') != '1' or (origin and origin not in {f'http://{h}' for h in hosts}):
                raise RequestError('This request must come from the local MapRepair page.', 403)
        path = urlsplit(self.path).path
        assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                  '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                  '/grading.js': ('grading.js', 'text/javascript; charset=utf-8'),
                  '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if self.command == 'GET' and path in assets:
            name, kind = assets[path]
            return self.file(ROOT / 'web_ui' / name, kind)
        if self.command == 'GET' and path == '/api/state':
            with self.jobs.lock:
                latest = sorted(self.jobs.jobs.values(), key=lambda j: j['created'], reverse=True)
                job_id = latest[0]['id'] if latest else None
            return self.json(dict(latest=job_id, max_upload_bytes=self.jobs.max_upload))
        if self.command == 'POST' and path == '/api/jobs':
            length = self.content_length(4 * 1024 * 1024)
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise RequestError('Invalid run settings.')
            return self.json(self.jobs.create(data), 201)
        match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})(?:/(input|start|cancel|download|preview|grade|auto-grade|log|photos/\d+))?', path)
        if not match:
            raise RequestError('Not found.', 404)
        job_id, action = match.groups()
        with self.jobs.lock:
            self.jobs.get(job_id)
        if self.command == 'GET' and action is None:
            return self.json(self.jobs.snapshot(job_id))
        if self.command == 'PUT' and (action == 'input' or (action and action.startswith('photos/'))):
            self.connection.settimeout(120)
            index = int(action.split('/')[1]) if action.startswith('photos/') else None
            self.jobs.upload(job_id, self.rfile, self.content_length(self.jobs.max_upload), photo_index=index)
            return self.json(self.jobs.snapshot(job_id))
        if self.command == 'POST' and action in ('start', 'cancel'):
            getattr(self.jobs, action)(job_id)
            return self.json(self.jobs.snapshot(job_id))
        if self.command == 'POST' and action == 'auto-grade':
            from advanced_grade import PREVIEW_LOCK, auto_grade
            if self.jobs.snapshot(job_id)['state'] != 'complete':
                raise RequestError('Finish the map before grading.', 409)
            if not PREVIEW_LOCK.acquire(blocking=False):
                raise RequestError('Preview is rendering. Try Auto cinema again in a moment.', 409)
            try:
                return self.json(auto_grade(self.jobs.root / job_id / 'output'))
            finally:
                PREVIEW_LOCK.release()
        if self.command == 'POST' and action == 'grade':
            from advanced_grade import LOCK, PREVIEW_LOCK, render
            if self.jobs.snapshot(job_id)['state'] != 'complete':
                raise RequestError('Finish the map before grading.', 409)
            data = json.loads(self.rfile.read(self.content_length(8192)))
            if not isinstance(data, dict):
                raise RequestError('Invalid grading request.')
            preview = data.get('preview', False) is True
            grade_lock = PREVIEW_LOCK if preview else LOCK
            if not grade_lock.acquire(blocking=False):
                raise RequestError('Another grade is rendering. Try again shortly.', 409)
            try:
                preview = data.get('preview', False) is True
                output_format=data.get('format','jpeg')
                crop=data.get('crop')
                result = render(self.jobs.root / job_id / 'output', data.get('settings', {}), preview, output_format,crop)
                extension='png' if crop is not None or (output_format=='png' and not preview) else 'jpg'
                self.send_headers(200, 'image/png' if extension=='png' else 'image/jpeg', len(result), None if preview else 'maprepair_graded.'+extension)
                self.wfile.write(result)
            finally:
                grade_lock.release()
            return
        if self.command == 'GET' and action in ('download', 'preview', 'log'):
            job = self.jobs.snapshot(job_id)
            directory = self.jobs.root / job_id
            if action == 'log':
                return self.file(directory / 'run.log', 'text/plain; charset=utf-8', 'maprepair-run.log')
            if job['state'] != 'complete':
                raise RequestError('The JPEG will be available after the run completes.', 409)
            if action == 'preview':
                return self.file(directory / 'output/preview.jpg', 'image/jpeg')
            name = re.sub(r'[^a-zA-Z0-9_-]', '_', Path(job['filename']).stem)[:100] + '_full_2x.jpg'
            return self.file(directory / 'output/final_2x.jpg', 'image/jpeg', name)
        raise RequestError('Not found.', 404)

    def content_length(self, limit):
        if self.headers.get('Transfer-Encoding'):
            raise RequestError('Upload must include a Content-Length header.')
        try:
            length = int(self.headers.get('Content-Length', ''))
        except ValueError:
            raise RequestError('Missing upload size.')
        if not 0 < length <= limit:
            raise RequestError('Request is empty or exceeds the size limit.', 413)
        return length

    def handle_request(self):
        try:
            self.dispatch()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except RequestError as exc:
            self.json(dict(error=str(exc)), exc.status)
        except (ValueError, OSError) as exc:
            self.json(dict(error=str(exc)), 400)
        except Exception as exc:
            print(f'[web] {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
            self.json(dict(error=f'Local server error: {exc}'), 500)

    do_GET = handle_request
    do_POST = handle_request
    do_PUT = handle_request


def make_server(jobs, port=0):
    server = ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, jobs=jobs))
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765, help='0 chooses an available port')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'web_runs')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('Port must be between 0 and 65535.')
    try:
        jobs = Jobs(args.runs_dir)
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f'{exc}\n')
    try:
        server = make_server(jobs, args.port)
    except OSError as exc:
        jobs.shutdown()
        parser.exit(1, f'Could not start local server: {exc}\nTry: {sys.executable} {__file__} --port 0\n')
    url = f'http://127.0.0.1:{server.server_address[1]}'
    print(f'MapRepair: {url}\nResults: {jobs.root}\nKeep this terminal open. Ctrl+C stops the server and active run.', flush=True)
    if not args.no_browser:
        threading.Timer(.3, lambda: webbrowser.open(url)).start()
    def stop_signal(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_signal)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        print('\nStopping MapRepair...', flush=True)
    finally:
        jobs.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
