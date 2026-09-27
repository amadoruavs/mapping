'use strict';
const $ = id => document.getElementById(id);
let selectedFile = null, selectedPhotos = [], currentJob = null, polling = null, busy = false;
const uploads = new Set();
let maxUpload = 8 * 1024 ** 3, stopping = false;
const terminal = new Set(['complete', 'failed', 'cancelled', 'interrupted']);
const remember = (key, value) => { try { localStorage.setItem(key, value); } catch (_) {} };
const recall = key => { try { return localStorage.getItem(key); } catch (_) { return null; } };

function error(message) {
  $('form-error').textContent = message || '';
  $('form-error').hidden = !message;
}
function setBusy(value) {
  busy = value;
  $('inputs').disabled = value;
  $('run').disabled = value;
  $('stop').hidden = !value;
}
function choose(file) {
  if (busy || !file) return;
  if (!/\.(tiff?|png|jpe?g)$/i.test(file.name)) return error('Choose a TIFF, PNG or JPEG map.');
  if (!file.size || file.size > maxUpload) return error(`Choose a nonempty map under ${maxUpload / 1024 ** 3} GB.`);
  selectedFile = file;
  $('file-name').textContent = file.name;
  $('file-info').textContent = `${(file.size / 1024 ** 2).toFixed(1)} MB · ready to upload`;
  error('');
}
function choosePhotos(files, name) {
  if (busy) return;
  const photos = Array.from(files).filter(file => /\.(jpe?g|png|tiff?|webp)$/i.test(file.name));
  if (!photos.length) return error('This folder contains no supported photos.');
  if (photos.length > 10000) return error('Choose a folder with no more than 10,000 photos.');
  if (photos.some(file => !file.size || file.size > maxUpload)) return error('A photo is empty or exceeds the file size limit.');
  photos.sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name));
  selectedPhotos = photos;
  $('folder-name').textContent = name || 'Selected photos';
  const bytes = photos.reduce((sum, f) => sum + f.size, 0);
  $('folder-info').textContent = `${photos.length} photos · ${(bytes / 1024 ** 2).toFixed(1)} MB`;
  $('photos').value = ''; remember('maprepair-photos', '');
  $('path-option').open = false;
  error('');
}
async function readFolder(entry, files) {
  if (entry.isFile) {
    const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
    if (/\.(jpe?g|png|tiff?|webp)$/i.test(file.name)) files.push(file);
    if (files.length > 10000) throw new Error('Choose a folder with no more than 10,000 photos.');
  } else if (entry.isDirectory) {
    const reader = entry.createReader();
    // Chromium returns directories in batches; read until the empty batch.
    for (;;) {
      const entries = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
      if (!entries.length) break;
      for (const child of entries) await readFolder(child, files);
    }
  }
}
async function api(path, method = 'GET', data) {
  const response = await fetch(path, {method, headers: {'X-MapRepair': '1', 'Content-Type': 'application/json'},
    body: data === undefined ? undefined : JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
  return result;
}
function render(job) {
  currentJob = job.id;
  window.currentGradeJob=job;window.MapGrade?.setJob(job);
  $('status-panel').hidden = false;
  const active = !terminal.has(job.state);
  setBusy(active);
  const titles = {awaiting_upload: 'Waiting for upload', uploading: 'Uploading files', ready: 'Files uploaded',
    running: job.stage, complete: 'Complete', failed: 'Run failed', cancelled: 'Stopped', interrupted: 'Run interrupted'};
  $('status').textContent = titles[job.state] || job.state;
  $('elapsed').textContent = `${job.elapsed.toFixed(1)} s`;
  $('progress').hidden = !active;
  if (job.state === 'uploading' || job.state === 'awaiting_upload') {
    $('progress').max = job.total_bytes || job.size; $('progress').value = job.received;
  } else $('progress').removeAttribute('value');
  $('status-note').textContent = job.state === 'running' ? `${job.photo_count} source photos. You can refresh this page; the run will continue.` :
    job.state === 'ready' ? 'Files are uploaded. If processing did not start, stop this run and try again.' : '';
  $('run-error').textContent = job.error || '';
  $('run-error').hidden = !job.error;
  const log = $('log'), anchored = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
  if (log.textContent !== job.log) {
    log.textContent = job.log || 'Waiting for processor output…';
    if (anchored) log.scrollTop = log.scrollHeight;
  }
  $('download-log').href = job.log_url;
  $('result').hidden = job.state !== 'complete';
  $('saved-path').textContent = job.state === 'complete' ? `Saved in ${job.output_folder}` : '';
  if (job.state === 'complete') {
    $('download').href = job.download;
    if ($('preview').getAttribute('src') !== job.preview) $('preview').src = job.preview;
    $('result-info').textContent = `${job.dimensions ? job.dimensions.join(' × ') + ' pixels · ' : ''}Full-resolution JPEG. Original map footprint preserved.`;
  }
  if (job.error) $('log-details').open = true;
  return active;
}
function watch(id) {
  clearTimeout(polling);
  const poll = async () => {
    if (currentJob !== id) return;
    try {
      const job = await api(`/api/jobs/${id}`);
      $('connection').hidden = true;
      if (render(job)) polling = setTimeout(poll, 800);
    } catch (_) {
      $('connection').textContent = 'Connection lost. Checking again… Make sure the local server is still running.';
      $('connection').hidden = false;
      polling = setTimeout(poll, 2000);
    }
  };
  poll();
}
function upload(path, file, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    uploads.add(request);
    request.open('PUT', path);
    request.setRequestHeader('X-MapRepair', '1');
    request.setRequestHeader('Content-Type', 'application/octet-stream');
    request.upload.onprogress = event => {
      if (!event.lengthComputable) return;
      onProgress(event.loaded, event.total);
    };
    request.onload = () => {
      uploads.delete(request);
      try {
        const result = JSON.parse(request.responseText);
        if (request.status < 200 || request.status >= 300) throw new Error(result.error || 'Upload failed.');
        resolve(result);
      } catch (err) { reject(err); }
    };
    request.onerror = () => { uploads.delete(request); reject(new Error('Upload failed. Check that the local server is running, then try again.')); };
    request.onabort = () => { uploads.delete(request); reject(new Error('Upload stopped.')); };
    request.send(file);
  });
}
$('map-file').addEventListener('change', event => choose(event.target.files[0]));
for (const eventName of ['dragenter', 'dragover']) $('drop-zone').addEventListener(eventName, event => {
  event.preventDefault(); if (!busy) $('drop-zone').classList.add('dragging');
});
async function uploadPhotos(id, photos) {
  const loaded = new Array(photos.length).fill(0);
  const total = photos.reduce((sum, f) => sum + f.size, 0);
  let next = 0, finished = 0;
  const update = () => {
    $('status').textContent = `Uploading photos · ${finished} / ${photos.length}`;
    $('progress').max = total;
    $('progress').value = loaded.reduce((sum, size) => sum + size, 0);
  };
  const worker = async () => {
    while (next < photos.length && !stopping) {
      const index = next++;
      await upload(`/api/jobs/${id}/photos/${index}`, photos[index], amount => { loaded[index] = amount; update(); });
      loaded[index] = photos[index].size; finished++; update();
    }
  };
  try { await Promise.all([worker(), worker()]); }
  catch (err) { for (const request of uploads) request.abort(); throw err; }
}
$('photo-folder').addEventListener('change', event => {
  const files = event.target.files;
  choosePhotos(files, files[0]?.webkitRelativePath.split('/')[0]);
});
for (const eventName of ['dragenter', 'dragover']) $('folder-drop-zone').addEventListener(eventName, event => {
  event.preventDefault(); if (!busy) $('folder-drop-zone').classList.add('dragging');
});
for (const eventName of ['dragleave', 'drop']) $('folder-drop-zone').addEventListener(eventName, event => {
  event.preventDefault(); $('folder-drop-zone').classList.remove('dragging');
});
$('folder-drop-zone').addEventListener('drop', async event => {
  if (busy) return;
  const entries = Array.from(event.dataTransfer.items).filter(item => item.kind === 'file').map(item => item.webkitGetAsEntry?.());
  if (entries.length !== 1 || !entries[0]?.isDirectory) return error('Drop one photos folder, or use Choose folder.');
  $('run').disabled = true;
  $('folder-name').textContent = 'Reading folder…';
  try {
    const files = []; await readFolder(entries[0], files); choosePhotos(files, entries[0].name);
  } catch (err) { error(`Could not read this folder: ${err.message}. Try Choose folder or the folder path.`); }
  finally { $('run').disabled = busy; }
});
for (const eventName of ['dragleave', 'drop']) $('drop-zone').addEventListener(eventName, event => {
  event.preventDefault(); $('drop-zone').classList.remove('dragging');
});
$('drop-zone').addEventListener('drop', event => {
  if (event.dataTransfer.files.length !== 1) return error('Drop one map at a time.');
  choose(event.dataTransfer.files[0]);
});
// Dropping a file outside the box must not navigate away from a running job.
window.addEventListener('dragover', event => event.preventDefault());
window.addEventListener('drop', event => event.preventDefault());
$('photos').value = recall('maprepair-photos') || '';
if ($('photos').value) $('path-option').open = true;
$('photos').addEventListener('input', () => {
  if ($('photos').value.trim()) {
    selectedPhotos = []; $('folder-name').textContent = 'Using folder path below';
    $('folder-info').textContent = 'Photos will be read directly from this computer';
  }
  remember('maprepair-photos', $('photos').value);
});
$('run-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (busy) return;
  if (!selectedFile) return error('Choose a map image first.');
  if (!selectedPhotos.length && !$('photos').value.trim()) return error('Drop a photos folder, choose one, or paste its path.');
  clearTimeout(polling); stopping = false; currentJob = null;
  $('status-panel').hidden = true; $('connection').hidden = true; error(''); setBusy(true);
  try {
    const settings = {filename: selectedFile.name, size: selectedFile.size};
    if (selectedPhotos.length) settings.photo_files = selectedPhotos.map(file => ({name: file.name, size: file.size}));
    else settings.photos = $('photos').value;
    const job = await api('/api/jobs', 'POST', settings);
    currentJob = job.id;
  window.currentGradeJob=job;window.MapGrade?.setJob(job); remember('maprepair-job', job.id); remember('maprepair-photos', $('photos').value);
    render(job);
    if (stopping) { await api(`/api/jobs/${job.id}/cancel`, 'POST'); return watch(job.id); }
    render(await upload(`/api/jobs/${job.id}/input`, selectedFile, (loaded, total) => {
      $('status').textContent = `Uploading map · ${Math.round(loaded / total * 100)}%`;
      $('progress').max = total; $('progress').value = loaded;
    }));
    if (selectedPhotos.length && !stopping) await uploadPhotos(job.id, selectedPhotos);
    if (stopping) return watch(job.id);
    render(await api(`/api/jobs/${job.id}/start`, 'POST'));
    watch(job.id);
  } catch (err) {
    if (!stopping) error(err.message);
    // The server may have accepted a start even if its response was lost.
    // Inspect its persisted state; never automatically launch a duplicate.
    if (currentJob) watch(currentJob);
    else { setBusy(false); $('status-panel').hidden = true; }
  }
});
$('stop').addEventListener('click', async () => {
  stopping = true; $('stop').disabled = true;
  for (const request of uploads) request.abort();
  try {
    if (currentJob) render(await api(`/api/jobs/${currentJob}/cancel`, 'POST'));
  } catch (err) { error(err.message); }
  finally { $('stop').disabled = false; if (currentJob) watch(currentJob); }
});
(async () => {
  setBusy(true); $('stop').hidden = true;
  try {
    const state = await api('/api/state');
    maxUpload = state.max_upload_bytes;
    const id = state.latest || recall('maprepair-job');
    if (id) {
      const job = await api(`/api/jobs/${id}`);
      if (render(job)) watch(id);
    } else setBusy(false);
  } catch (err) { error(`Cannot connect to the local server: ${err.message}`); setBusy(false); }
})();
