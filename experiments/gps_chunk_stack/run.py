#!/usr/bin/env python3
"""CPU-only GPS tile stacking experiment. Requires numpy, opencv-python, pillow."""
from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

INPUT_DIRECTORY = Path('/Users/avneh/Downloads/mapv2')
OUTPUT_DIRECTORY = Path('/Users/avneh/Code/ADM/output/gps_chunk_stack_10')
IMAGE_COUNT = 10
CHUNK_SIZE = 50  # source pixels, not metres
ALTITUDE_AGL_M = 51.816
HORIZONTAL_FOV_DEG = 81.0  # inherited flight assumption; not a fitted calibration
MAX_MAP_DIMENSION = 4096
SAVE_CHUNK_PIXELS = True  # padded uint8 RGB .npy, usable directly as a memory map


def log(message):
    print(message, flush=True)


class Timer:
    def __init__(self):
        self.start = self.last = time.perf_counter()
        self.stages = {}

    def mark(self, name):
        now = time.perf_counter()
        self.stages[name] = now - self.last
        self.last = now
        log(f'[{name}] {self.stages[name]:.3f}s; total {now - self.start:.3f}s')


class LocalEarth:
    """WGS84 local tangent approximation, east/north in metres at one small site."""
    def __init__(self, lat, lon):
        self.lat, self.lon = float(lat), float(lon)
        phi = np.deg2rad(lat)
        den = np.sqrt(1 - 6.69437999014e-3 * np.sin(phi) ** 2)
        self.east_radius = 6378137 / den * np.cos(phi)
        self.north_radius = 6378137 * (1 - 6.69437999014e-3) / den ** 3

    def xy(self, lat, lon):
        return np.stack([np.deg2rad((np.asarray(lon) - self.lon + 180) % 360 - 180)
                         * self.east_radius,
                         np.deg2rad(np.asarray(lat) - self.lat) * self.north_radius], axis=-1)

    def latlon(self, xy):
        xy = np.asarray(xy)
        return np.stack([self.lat + np.rad2deg(xy[..., 1] / self.north_radius),
                         (self.lon + np.rad2deg(xy[..., 0] / self.east_radius) + 180)
                         % 360 - 180], axis=-1)


def metadata(path, agl):
    """Read actual JPEG EXIF/XMP. Never use flight yaw as camera heading."""
    fields = {}
    with path.open('rb') as f:
        if f.read(2) != b'\xff\xd8':
            raise ValueError(f'{path}: expected JPEG')
        while True:
            marker = f.read(2)
            if len(marker) != 2 or marker[0] != 255 or marker[1] in (0xDA, 0xD9):
                break
            while marker[1] == 255:
                marker = b'\xff' + f.read(1)
            if marker[1] == 1 or 0xD0 <= marker[1] <= 0xD7:
                continue
            length = int.from_bytes(f.read(2), 'big')
            if length < 2:
                raise ValueError(f'{path}: malformed JPEG marker')
            block = f.read(length - 2)
            if block.startswith(b'http://ns.adobe.com/xap/1.0/\x00'):
                root = ET.fromstring(block.split(b'\x00', 1)[1])
                for node in root.iter():
                    fields.update({k.split('}')[-1]: v for k, v in node.attrib.items()})
                    if node.text and node.text.strip():
                        fields[node.tag.split('}')[-1]] = node.text.strip()

    def xmp_float(*keys):
        for key in keys:
            if key in fields:
                value = float(fields[key])
                if not math.isfinite(value):
                    raise ValueError(f'{path}: nonfinite {key}')
                return value
        return None

    def gps_coordinate(value, ref):
        if value is None or ref is None:
            return None
        if isinstance(ref, bytes):
            ref = ref.decode()
        d, m, s = map(float, value)
        return (d + m / 60 + s / 3600) * (-1 if ref in ('S', 'W') else 1)

    with Image.open(path) as im:
        width, height = im.size
        exif = im.getexif()
        if exif.get(274, 1) != 1:
            raise ValueError(f'{path}: normalize EXIF orientation and camera axes first')
        gps = exif.get_ifd(34853)
        lat = xmp_float('GpsLatitude', 'GPSLatitude')
        lon = xmp_float('GpsLongitude', 'GpsLongtitude', 'GPSLongitude')
        yaw = xmp_float('GimbalYawDegree', 'CameraYawDegree')
        lat_source, lon_source, yaw_source = 'XMP', 'XMP', 'XMP camera yaw'
        if lat is None:
            lat = gps_coordinate(gps.get(2), gps.get(1)); lat_source = 'EXIF GPS'
        if lon is None:
            lon = gps_coordinate(gps.get(4), gps.get(3)); lon_source = 'EXIF GPS'
        if yaw is None and gps.get(16) in ('T', b'T') and gps.get(17) is not None:
            yaw = float(gps[17]); yaw_source = 'EXIF true-north image direction'
    if any(v is None or not math.isfinite(v) for v in (lat, lon, yaw)):
        raise ValueError(f'{path}: latitude, longitude and absolute camera heading required')
    if not -85 < lat < 85 or not -180 <= lon <= 180:
        raise ValueError(f'{path}: coordinates outside supported local projection')
    pitch = xmp_float('GimbalPitchDegree', 'CameraPitchDegree')
    roll = xmp_float('GimbalRollDegree', 'CameraRollDegree')
    return dict(name=path.name, width=width, height=height, latitude=lat, longitude=lon,
                yaw_deg=yaw, pitch_deg=-90.0 if pitch is None else pitch,
                roll_deg=0.0 if roll is None else roll, altitude_agl_m=agl,
                provenance=dict(latitude=lat_source, longitude=lon_source, yaw=yaw_source,
                                pitch='nadir assumption' if pitch is None else 'XMP',
                                roll='zero assumption' if roll is None else 'XMP',
                                altitude='configured AGL'),
                gps_status=fields.get('GpsStatus'),
                reported_relative_altitude_m=xmp_float('RelativeAltitude'))


def project(H, points):
    p = np.asarray(points, dtype=np.float64)
    q = np.concatenate([p, np.ones((*p.shape[:-1], 1))], axis=-1) @ H.T
    if np.any(np.abs(q[..., 2]) < 1e-12) or not np.isfinite(q).all():
        raise ValueError('Projection reaches horizon or contains nonfinite coordinates')
    return q[..., :2] / q[..., 2:]


def camera_to_ground(meta, earth, hfov):
    """Ray/plane intersection: C + (-C.z / ray.z) * ray, ground z=0."""
    w, h = meta['width'], meta['height']
    f = w / (2 * np.tan(np.deg2rad(hfov) / 2))
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.]])
    yaw, pitch, roll = np.deg2rad([meta['yaw_deg'], meta['pitch_deg'], meta['roll_deg']])
    forward = np.array([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
    right = np.array([np.cos(yaw), -np.sin(yaw), 0.])
    down = np.cross(forward, right)
    R = np.column_stack([right * np.cos(roll) + down * np.sin(roll),
                         -right * np.sin(roll) + down * np.cos(roll), forward])
    rays = R @ np.linalg.inv(K)
    boundary = np.array([[-.5, -.5, 1], [w - .5, -.5, 1],
                         [w - .5, h - .5, 1], [-.5, h - .5, 1]])
    if np.any(boundary @ rays[2] >= -1e-5):
        raise ValueError(f"{meta['name']}: footprint intersects or rises above horizon")
    east, north = earth.xy(meta['latitude'], meta['longitude'])
    agl = meta['altitude_agl_m']
    H = np.vstack([east * rays[2] - agl * rays[0], north * rays[2] - agl * rays[1], rays[2]])
    return H / H[2, 2]


def chunk_grid(width, height, size):
    x, y = np.meshgrid(np.arange(0, width, size), np.arange(0, height, size))
    x, y = x.ravel(), y.ravel()
    tw, th = np.minimum(size, width - x), np.minimum(size, height - y)
    centers = np.column_stack([x + (tw - 1) / 2, y + (th - 1) / 2])
    corners = np.stack([np.column_stack([x - .5, y - .5]),
                        np.column_stack([x + tw - .5, y - .5]),
                        np.column_stack([x + tw - .5, y + th - .5]),
                        np.column_stack([x - .5, y + th - .5])], axis=1)
    return np.column_stack([x, y, tw, th]).astype(np.int32), centers, corners


def composite_tile(image, rect, H, inverse, canvases, output_bounds=None):
    """Place one source tile. Ownership is half-open; filter across tile borders."""
    x, y, tw, th = map(int, rect)
    chosen, best, mean_sum, weights, coverage, source_ids = canvases[:6]
    source_id = canvases[6]
    h, w = chosen.shape[:2]
    if output_bounds is None:
        corners = project(H, [[x - .5, y - .5], [x + tw - .5, y - .5],
                              [x + tw - .5, y + th - .5], [x - .5, y + th - .5]])
        output_bounds = (*np.floor(corners.min(0)).astype(int),
                         *np.ceil(corners.max(0)).astype(int) + 1)
    x0, y0, x1, y1 = map(int, output_bounds)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    denominator = inverse[2, 0] * xx + inverse[2, 1] * yy + inverse[2, 2]
    u = (inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]) / denominator
    v = (inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]) / denominator
    valid = (u >= x - .5) & (u < x + tw - .5) & (v >= y - .5) & (v < y + th - .5)
    if not valid.any():
        return
    # Sampling original neighbours gives identical filtering at both sides of a tile cut.
    color = cv2.remap(image, u.astype(np.float32), v.astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ih, iw = image.shape[:2]
    radius2 = ((u - (iw - 1) / 2) / (iw / 2)) ** 2 + ((v - (ih - 1) / 2) / (ih / 2)) ** 2
    score = (1 / (1 + 2 * radius2)).astype(np.float32)
    region = np.s_[y0:y1, x0:x1]
    replace = valid & (score > best[region])
    chosen[region][replace] = color[replace]
    best[region][replace] = score[replace]
    source_ids[region][replace] = source_id
    weight = np.where(valid, score, 0)
    mean_sum[region] += color.astype(np.float32) * weight[..., None]
    weights[region] += weight
    coverage[region] += valid.astype(np.uint16)


def write_png(path, array):
    if not cv2.imwrite(str(path), array, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f'Could not save {path}')


def preview(image, width=1100):
    return cv2.resize(image, (width, max(1, round(image.shape[0] * width / image.shape[1]))),
                      interpolation=cv2.INTER_AREA)


def labeled(image, title, subtitle=''):
    canvas = np.full((image.shape[0] + 65, image.shape[1], 3), 245, np.uint8)
    canvas[65:] = image[..., :3]
    if image.shape[2] == 4:
        canvas[65:][image[..., 3] == 0] = (34, 30, 28)
    cv2.putText(canvas, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .62, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, .42, (70, 70, 70), 1, cv2.LINE_AA)
    return canvas


def write_viewer(path, centers, layers, colors, filenames):
    payload = json.dumps(dict(xy=np.round(centers, 3).tolist(), layers=layers.tolist(),
                              colors=colors.tolist(), filenames=filenames), separators=(',', ':'))
    html = '''<!doctype html><meta charset="utf-8"><title>GPS tile stack</title>
<style>body{margin:0;background:#111a20;color:#edf1f2;font:15px system-ui}header{padding:20px}button{padding:8px 15px;margin-right:12px}canvas{display:block;touch-action:none}p{margin:8px 0;color:#b9c7ce}</style>
<header><b>50 x 50 pixel tiles: coordinate stack</b>
<p>Drag to rotate; scroll to zoom. East/north are metres. Vertical spacing is photo index, NOT terrain elevation. Every ground point assumes z = 0 m.</p>
<button id="flatten">Flatten layers</button><span id="info"></span></header><canvas id="c"></canvas>
<script>const data=PAYLOAD;const c=document.getElementById('c'),ctx=c.getContext('2d');
let yaw=-.45,pitch=.7,zoom=1,flat=false,drag=null;
let minx=Infinity,maxx=-Infinity,miny=Infinity,maxy=-Infinity;
for(const [x,y] of data.xy){minx=Math.min(minx,x);maxx=Math.max(maxx,x);miny=Math.min(miny,y);maxy=Math.max(maxy,y)}
const cx=(minx+maxx)/2,cy=(miny+maxy)/2;
const span=Math.max(maxx-minx,maxy-miny),n=data.filenames.length;
const colors=data.colors.map(p=>`rgb(${p[0]},${p[1]},${p[2]})`);
function draw(){c.width=innerWidth;c.height=Math.max(400,innerHeight-c.offsetTop);const s=Math.min(c.width,c.height)*.72/span*zoom,co=Math.cos(yaw),si=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
ctx.fillStyle='#111a20';ctx.fillRect(0,0,c.width,c.height);
let order=[];for(let i=0;i<data.xy.length;i++){const x=data.xy[i][0]-cx,y=data.xy[i][1]-cy,z=flat?0:(data.layers[i]-(n-1)/2)*span*.06;
const xx=co*x-si*y,yy=si*x+co*y;order.push([i,c.width/2+xx*s,c.height/2+(-yy*cp-z*sp)*s,yy*sp-z*cp]);}
order.sort((a,b)=>a[3]-b[3]);for(const [i,x,y] of order){ctx.fillStyle=colors[i];ctx.fillRect(x,y,2.3,2.3);}
document.getElementById('info').textContent=`${data.xy.length.toLocaleString()} tiles / ${n} photographs / ${flat?'ground plane':'photo layers'}`;}
c.onpointerdown=e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};
c.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw()}};
c.onpointerup=()=>drag=null;c.onwheel=e=>{e.preventDefault();zoom=Math.max(.1,Math.min(15,zoom*Math.exp(-e.deltaY*.001)));draw()};
document.getElementById('flatten').onclick=e=>{flat=!flat;e.target.textContent=flat?'Separate photo layers':'Flatten layers';draw()};onresize=draw;draw();</script>'''
    path.write_text(html.replace('PAYLOAD', payload.replace('</', '<\\/')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT_DIRECTORY)
    parser.add_argument('--output', type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument('--count', type=int, default=IMAGE_COUNT, help='First N sorted JPEGs; 0 means all')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--altitude', type=float, default=ALTITUDE_AGL_M)
    parser.add_argument('--hfov', type=float, default=HORIZONTAL_FOV_DEG)
    parser.add_argument('--max-dimension', type=int, default=MAX_MAP_DIMENSION)
    parser.add_argument('--no-save-chunks', action='store_true', default=not SAVE_CHUNK_PIXELS)
    args = parser.parse_args()
    if (args.count < 0 or args.chunk_size < 1 or not math.isfinite(args.altitude)
            or args.altitude <= 0 or not 0 < args.hfov < 175 or args.max_dimension < 64
            or args.max_dimension > 12000):
        parser.error('Invalid count, tile size, altitude, FOV or output dimension (64..12000)')
    timer = Timer()
    cv2.setNumThreads(6)
    cv2.ocl.setUseOpenCL(False)
    paths = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg'))
    available = len(paths)
    if args.count > available:
        parser.error(f'Requested {args.count} photos but found {available}')
    if args.count:
        paths = paths[:args.count]
    if not paths:
        parser.error('No input JPEGs')
    if len(paths) > 65534:
        parser.error('Source index supports at most 65534 images')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new or empty output directory to preserve prior experiments')
    args.output.mkdir(parents=True, exist_ok=True)
    frames = [metadata(path, args.altitude) for path in paths]
    earth = LocalEarth(frames[0]['latitude'], frames[0]['longitude'])
    homographies = [camera_to_ground(m, earth, args.hfov) for m in frames]
    grids = [chunk_grid(m['width'], m['height'], args.chunk_size) for m in frames]
    counts = [len(g[0]) for g in grids]
    offsets = np.r_[0, np.cumsum(counts)]
    rects = np.concatenate([g[0] for g in grids])
    layers = np.repeat(np.arange(len(paths), dtype=np.uint16), counts)
    centers = np.concatenate([project(H, g[1]) for H, g in zip(homographies, grids)])
    corners = np.concatenate([project(H, g[2]) for H, g in zip(homographies, grids)])
    latlon = earth.latlon(centers)
    minimum, maximum = corners.min(axis=(0, 1)), corners.max(axis=(0, 1))
    natural_gsd = min(2 * args.altitude * np.tan(np.deg2rad(args.hfov) / 2) / m['width'] for m in frames)
    gsd = max(natural_gsd, float(max(maximum - minimum) / (args.max_dimension - 4)))
    width, height = np.ceil((maximum - minimum) / gsd).astype(int) + 4
    # Pixel centres, north-up. Two-pixel padding to keep boundary coverage intact.
    map_from_ground = np.array([[1 / gsd, 0, -minimum[0] / gsd + 1.5],
                                [0, -1 / gsd, maximum[1] / gsd + 1.5], [0, 0, 1.]])
    image_to_map = np.stack([map_from_ground @ H for H in homographies])
    camera_centers = earth.xy([m['latitude'] for m in frames], [m['longitude'] for m in frames])
    np.savez(args.output / 'coordinates.npz', center_lat_lon_ground_z=np.c_[latlon, np.zeros(len(latlon))],
             center_east_north_up=np.c_[centers, np.zeros(len(centers))], photo_layer=layers,
             source_rect_xywh=rects, corners_east_north=corners, image_to_ground=np.stack(homographies),
             image_to_map=image_to_map, camera_east_north_up=np.c_[camera_centers, np.full(len(paths), args.altitude)],
             filenames=np.array([p.name for p in paths]))
    (args.output / 'metadata.json').write_text(json.dumps(frames, indent=2))
    with (args.output / 'chunk_centers.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['chunk_id', 'photo', 'photo_layer', 'source_x', 'source_y', 'valid_width', 'valid_height',
                         'latitude', 'longitude', 'ground_z_m', 'east_m', 'north_m'])
        for i, (rect, layer, ll, xy) in enumerate(zip(rects, layers, latlon, centers)):
            writer.writerow([i, paths[layer].name, int(layer), *map(int, rect),
                             f'{ll[0]:.10f}', f'{ll[1]:.10f}', 0, f'{xy[0]:.6f}', f'{xy[1]:.6f}'])
    log(f'{len(paths)}/{available} photos; {len(rects):,} tiles; {width}x{height} map; {gsd:.5f} m/pixel')
    suspect = [m['name'] for m in frames if 'no gps' in (m['gps_status'] or '').lower()]
    if suspect:
        log(f'Metadata note: {len(suspect)} photos contain GPS coordinates but also a "NO GPS connected" tag; positions remain unverified.')
    timer.mark('metadata_projection_and_coordinate_export')

    chunks = None if args.no_save_chunks else np.lib.format.open_memmap(
        args.output / 'chunks_rgb.npy', mode='w+', dtype=np.uint8,
        shape=(len(rects), args.chunk_size, args.chunk_size, 3))
    colors = np.zeros((len(rects), 3), np.uint8)
    chosen = np.zeros((height, width, 3), np.uint8)
    best = np.zeros((height, width), np.float32)
    mean_sum = np.zeros((height, width, 3), np.float32)
    weights = np.zeros((height, width), np.float32)
    coverage = np.zeros((height, width), np.uint16)
    source_ids = np.full((height, width), 65535, np.uint16)
    thumbnails = []
    image_seconds = []
    for index, (path, grid, H) in enumerate(zip(paths, grids, image_to_map)):
        began = time.perf_counter()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise RuntimeError(f'Could not decode {path}')
        if image.shape[:2] != (frames[index]['height'], frames[index]['width']):
            raise RuntimeError(f'{path}: decoded dimensions differ from metadata')
        start, end = offsets[index:index + 2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if chunks is not None:
            rows = math.ceil(image.shape[0] / args.chunk_size)
            cols = math.ceil(image.shape[1] / args.chunk_size)
            padded = np.zeros((rows * args.chunk_size, cols * args.chunk_size, 3), np.uint8)
            padded[:rgb.shape[0], :rgb.shape[1]] = rgb
            chunks[start:end] = padded.reshape(rows, args.chunk_size, cols, args.chunk_size, 3).transpose(0, 2, 1, 3, 4).reshape(-1, args.chunk_size, args.chunk_size, 3)
            del padded
        center_pixels = np.rint(grid[1]).astype(int)
        colors[start:end] = rgb[center_pixels[:, 1], center_pixels[:, 0]]
        del rgb
        thumbnails.append(labeled(preview(image, 400), path.name,
                                  f'Yaw {frames[index]["yaw_deg"]:.1f} deg / {counts[index]:,} tiles'))
        if index == 0:
            grid_preview = preview(image, 1200)
            scale = grid_preview.shape[1] / image.shape[1]
            for x in range(0, image.shape[1], args.chunk_size):
                cv2.line(grid_preview, (round(x * scale), 0), (round(x * scale), grid_preview.shape[0]), (80, 220, 240), 1)
            for y in range(0, image.shape[0], args.chunk_size):
                cv2.line(grid_preview, (0, round(y * scale)), (grid_preview.shape[1], round(y * scale)), (80, 220, 240), 1)
            write_png(args.output / 'source_chunk_grid.png', labeled(grid_preview, 'Source image cut into 50-pixel tiles' if args.chunk_size == 50 else f'Source image cut into {args.chunk_size}-pixel tiles'))
        inverse = np.linalg.inv(H)
        projected_corners = project(map_from_ground, corners[start:end])
        bounds = np.c_[np.floor(projected_corners.min(axis=1)), np.ceil(projected_corners.max(axis=1)) + 1].astype(int)
        for rect, bound in zip(grid[0], bounds):
            composite_tile(image, rect, H, inverse,
                           (chosen, best, mean_sum, weights, coverage, source_ids, index), bound)
        image_seconds.append(time.perf_counter() - began)
        log(f'[composite] {index + 1}/{len(paths)} {path.name}: {counts[index]:,} tiles, {image_seconds[-1]:.3f}s')
    if chunks is not None:
        chunks.flush()
        del chunks
    timer.mark('decode_tile_storage_and_composite')

    valid = coverage > 0
    mean = np.clip(np.rint(mean_sum / np.maximum(weights[..., None], 1e-12)), 0, 255).astype(np.uint8)
    del mean_sum
    alpha = (valid * 255).astype(np.uint8)
    selected_bgra = np.dstack([chosen, alpha])
    mean_bgra = np.dstack([mean, alpha])
    write_png(args.output / 'map_selected.png', selected_bgra)
    write_png(args.output / 'map_mean.png', mean_bgra)
    write_png(args.output / 'coverage.png', coverage)
    write_png(args.output / 'source_ids.png', source_ids)
    write_png(args.output / 'preview.png', labeled(preview(selected_bgra), 'GPS tile mosaic - sharp source selection',
              f'{len(paths)} photos / {len(rects):,} tiles / flat ground / metadata placement'))
    comparison = np.concatenate([labeled(preview(selected_bgra, 800), 'Select the most central source', 'Preserves individual-source detail; positional seams remain'),
                                 labeled(preview(mean_bgra, 800), 'Average overlapping observations', 'GPS or heading errors appear as double edges')], axis=1)
    write_png(args.output / 'comparison.png', comparison)
    # Contact sheet supports mixed source aspect ratios.
    th = max(x.shape[0] for x in thumbnails)
    contact = np.full((math.ceil(len(thumbnails) / 5) * th, 2000, 3), 245, np.uint8)
    for i, thumb in enumerate(thumbnails):
        y, x = i // 5 * th, i % 5 * 400
        contact[y:y + thumb.shape[0], x:x + 400] = thumb
    write_png(args.output / 'input_contact_sheet.png', contact)
    heat = cv2.applyColorMap(np.rint(coverage.astype(float) / max(1, coverage.max()) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat[~valid] = 0
    write_png(args.output / 'coverage_preview.png', labeled(preview(heat), 'Number of overlapping photographs', f'Maximum {coverage.max()} observations per pixel'))
    write_viewer(args.output / 'stack.html', centers, layers, colors, [p.name for p in paths])
    timer.mark('map_and_preview_export')

    overlap = coverage >= 2
    g0 = project(np.linalg.inv(map_from_ground), [[0, 0]])[0]
    geometry = dict(width=int(width), height=int(height), meters_per_pixel=gsd,
                    local_origin_lat_lon=[earth.lat, earth.lon], pixel_0_center_east_north_m=g0.tolist(),
                    map_from_ground=map_from_ground.tolist(), ground_plane_up_m=0,
                    horizontal_fov_deg=args.hfov, altitude_agl_m=args.altitude,
                    coordinate_model='local WGS84 tangent approximation; camera rays intersect flat z=0 plane',
                    calibration='square pixels, image-centred principal point, configured FOV; no distortion correction')
    (args.output / 'map_geometry.json').write_text(json.dumps(geometry, indent=2))
    # Bound diagnostic scratch rather than allocating float64 copies of the full overlap.
    absolute_difference, overlap_channels = 0., 0
    for row in range(0, height, 128):
        mask = overlap[row:row + 128]
        delta = mean[row:row + 128][mask].astype(np.int16) - chosen[row:row + 128][mask].astype(np.int16)
        absolute_difference += float(np.abs(delta).sum(dtype=np.float64))
        overlap_channels += delta.size
    pixel_counts = np.bincount(source_ids.ravel(), minlength=65536)
    timer.mark('report_metrics')
    report = dict(input=str(args.input.resolve()), output=str(args.output.resolve()),
                  input_count=len(paths), available_input_count=available, chunk_count=len(rects),
                  chunk_size=args.chunk_size, padded_edge_chunks=int(np.count_nonzero((rects[:, 2:] != args.chunk_size).any(axis=1))),
                  chunk_pixels_saved=not args.no_save_chunks, image_seconds=image_seconds,
                  stage_seconds=timer.stages, processing_wall_seconds=time.perf_counter() - timer.start,
                  peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  map_size=[int(width), int(height)], covered_pixels=int(valid.sum()),
                  overlap_pixels=int(overlap.sum()), max_overlap=int(coverage.max()),
                  source_pixel_counts={p.name: int(pixel_counts[i]) for i, p in enumerate(paths)},
                  overlap_mean_vs_selected_mae=absolute_difference / overlap_channels if overlap_channels else None,
                  gps_status_conflicts=suspect, backend='CPU: Python, NumPy, OpenCV; OpenCL disabled',
                  limitations=['Positions estimated from GPS, gimbal angles, FOV and configured AGL; not surveyed coordinates.',
                               'Ground height is assumed zero. Photo layers are observation indices, not recovered 3D.',
                               'No feature alignment, motion estimation, parallax correction or lens calibration.',
                               'Overlap MAE is disagreement between compositing modes, not geolocation accuracy.'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    log(f'Done: {args.output / "map_selected.png"}; {report["processing_wall_seconds"]:.3f}s processing')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
