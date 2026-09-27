"""Persistent Metal/MPS backend with shared NumPy views and explicit fences.

No PyObjC dependency, subprocess per kernel, or file-based pixel exchange.
Importing the CPU pipeline never builds or opens a GPU device.
"""
from pathlib import Path
import ctypes as C
import hashlib
import os
import subprocess
import sys
import threading
import time
import weakref
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'msl-mps-1'


def build_library():
    if sys.platform != 'darwin':
        raise RuntimeError('The Metal/MPS backend requires macOS; use --backend cpu')
    source = ROOT / 'shaders' / 'ADMMetal.m'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    binary = ROOT / 'build' / f'libADMMetal-{digest}.dylib'
    binary.parent.mkdir(exist_ok=True)
    if not binary.exists():
        temporary = binary.with_suffix(f'.{os.getpid()}.tmp.dylib')
        result = subprocess.run(['xcrun', 'clang', '-O3', '-fobjc-arc', '-fno-modules',
                        '-dynamiclib', '-framework', 'Foundation', '-framework', 'Metal',
                        '-framework', 'MetalPerformanceShaders', str(source), '-o', str(temporary)],
                       capture_output=True, text=True)
        if result.returncode: raise RuntimeError('Metal host compilation failed:\n' + result.stderr)
        temporary.replace(binary)
    return binary


class Params(C.Structure):
    _fields_ = [('u', C.c_uint32 * 16), ('f', C.c_float * 32)]

    def __init__(self, u=(), f=()):
        super().__init__()
        self.u[15] = int(cv2.__version__.split('.')[0]) >= 5
        if len(u) > 16 or len(f) > 32: raise ValueError('Kernel parameter block overflow')
        for i, v in enumerate(u): self.u[i] = int(v) & 0xffffffff
        for i, v in enumerate(f): self.f[i] = float(v)


class SharedArray(np.ndarray):
    """Keep the native allocation alive if its Buffer wrapper is dropped."""
    def __array_finalize__(self, obj):
        self._owner = getattr(obj, '_owner', None)


class Buffer:
    def __init__(self, gpu, shape, dtype=np.float32, data=None, zero=False):
        self.gpu = gpu
        self.shape = tuple(shape) if np.iterable(shape) else (int(shape),)
        self.dtype = np.dtype(dtype)
        size = int(np.prod(self.shape)) * self.dtype.itemsize
        live = gpu.allocated_bytes()
        if live + size > gpu.memory_limit_bytes:
            raise MemoryError(f'Metal allocation of {size >> 20} MiB would exceed the {gpu.memory_limit_bytes >> 20} MiB '
                              f'device memory limit ({live >> 20} MiB live). Set ADM_METAL_MEMORY_LIMIT_MB to override.')
        self.handle = gpu.lib.adm_buffer(gpu.handle, size)
        if not self.handle: gpu.fail()
        self._release = weakref.finalize(self, gpu.lib.adm_release, self.handle)
        pointer = gpu.lib.adm_contents(self.handle)
        self._memory = (C.c_ubyte * size).from_address(pointer)
        if data is not None: np.copyto(self.view(), data, casting='no')
        elif zero: self.view().fill(0)

    def view(self):
        """Unsafe before sync for GPU-written data; use explicit fences."""
        a = np.frombuffer(self._memory, self.dtype).reshape(self.shape).view(SharedArray)
        a._owner = self
        return a

    def numpy(self, copy=True):
        self.gpu.sync()
        return np.array(self.view()) if copy else self.view()


class MetalBackend:
    def __init__(self):
        started = time.perf_counter()
        self.lib = C.CDLL(str(build_library()))
        self._thread = threading.get_ident()
        signatures = {
            'adm_create': (C.c_void_p, [C.c_char_p]),
            'adm_error': (C.c_char_p, [C.c_void_p]),
            'adm_device': (C.c_char_p, [C.c_void_p]),
            'adm_max_buffer': (C.c_uint64, [C.c_void_p]),
            'adm_allocated': (C.c_uint64, [C.c_void_p]),
            'adm_release': (None, [C.c_void_p]),
            'adm_buffer': (C.c_void_p, [C.c_void_p, C.c_uint64]),
            'adm_contents': (C.c_void_p, [C.c_void_p]),
            'adm_sync': (C.c_int, [C.c_void_p]),
            'adm_stats': (None, [C.c_void_p, C.POINTER(C.c_double)]),
            'adm_dispatch': (C.c_int, [C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p), C.c_uint32,
                                      C.POINTER(Params), C.c_uint32, C.c_uint32]),
            'adm_matmul': (C.c_int, [C.c_void_p] * 4 + [C.c_uint32] * 3 + [C.c_uint64]),
            'adm_gaussian': (C.c_int, [C.c_void_p] * 3 + [C.c_uint32] * 2 + [C.POINTER(C.c_float), C.c_uint32]),
            'adm_align': (C.c_int, [C.c_void_p, C.POINTER(C.c_void_p)] + [C.c_uint32]*5 + [C.c_float]),
        }
        for name, (result, args) in signatures.items():
            fn = getattr(self.lib, name); fn.restype = result; fn.argtypes = args
        self.handle = self.lib.adm_create(os.fsencode(ROOT / 'shaders' / 'pipeline.metal'))
        if not self.handle: self.fail()
        self._release = weakref.finalize(self, self.lib.adm_release, self.handle)
        self.device = self.lib.adm_device(self.handle).decode()
        shader_bytes=b''.join(p.read_bytes() for p in [ROOT/'shaders'/name for name in ('pipeline.metal','alignment.metal','coherent.metal','lasso.metal')] if p.exists())
        self.fingerprint = hashlib.sha256(shader_bytes + (ROOT/'shaders/ADMMetal.m').read_bytes() +
                                         Path(__file__).read_bytes() + cv2.__version__.encode()).hexdigest()
        self.max_buffer_bytes = self.lib.adm_max_buffer(self.handle)
        # Fail with an exception well before unified memory is exhausted; the OS
        # watchdog otherwise panics the whole machine instead of this process.
        physical = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        override = int(os.environ.get('ADM_METAL_MEMORY_LIMIT_MB', '0') or 0)
        self.memory_limit_bytes = override * 1024**2 if override > 0 else int(physical * .6)
        self.initialization_seconds = time.perf_counter() - started

    def fail(self):
        message = self.lib.adm_error(self.handle)
        raise RuntimeError(message.decode() if message else 'Metal operation failed')

    def check(self, status):
        if status: self.fail()

    def thread_check(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError('A MetalBackend must be used on its creating thread')

    def sync(self):
        self.thread_check(); self.check(self.lib.adm_sync(self.handle))

    def allocated_bytes(self):
        """Live device allocations, including buffers still referenced by pending commands."""
        return int(self.lib.adm_allocated(self.handle))

    def buffer(self, shape, dtype=np.float32, zero=False):
        self.thread_check()
        return Buffer(self, shape, dtype, zero=zero)

    def upload(self, array, dtype=None):
        self.thread_check(); a = np.asarray(array, dtype=dtype)
        return Buffer(self, a.shape, a.dtype, data=a)

    def shared_input(self,array):
        owner=getattr(array,'_owner',None)
        if isinstance(owner,Buffer) and owner.gpu is self and owner.shape==array.shape and array.flags.c_contiguous and array.ctypes.data==C.addressof(owner._memory):
            return owner
        return self.upload(array)

    def dispatch(self, name, buffers, count, u=(), f=(), group=0):
        self.thread_check()
        handles = (C.c_void_p * len(buffers))(*(b.handle for b in buffers))
        self.check(self.lib.adm_dispatch(self.handle, name.encode(), handles, len(buffers),
                                        C.byref(Params(u, f)), count, group))

    def stats(self):
        self.sync(); v = (C.c_double * 4)(); self.lib.adm_stats(self.handle, v)
        return dict(backend='metal', version=VERSION, device=self.device,
                    initialization_seconds=self.initialization_seconds, gpu_seconds=v[0],
                    command_submissions=int(v[1]), msl_dispatches=int(v[2]), mps_operations=int(v[3]),
                    max_buffer_bytes=self.max_buffer_bytes,
                    device_allocated_bytes=self.allocated_bytes(), memory_limit_bytes=self.memory_limit_bytes,
                    cpu_stages=['image codecs, metadata and selection', 'SIFT feature extraction',
                                'RANSAC and sparse global alignment', 'geographic coordinates and SQLite', 'output codecs'])

    def gaussian(self, src, sigma, dst=None):
        self.thread_check()
        h, w, channels = src.shape
        if src.dtype != np.float32 or channels != 4 or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('MPS Gaussian expects H×W×4 float32 and a positive sigma')
        size = int(round(sigma * 8 + 1)) | 1
        weights = cv2.getGaussianKernel(size, sigma, cv2.CV_32F).ravel()
        dst = dst or self.buffer(src.shape)
        if dst.handle == src.handle or dst.shape != src.shape or dst.dtype != src.dtype:
            raise ValueError('Gaussian destination must be a distinct buffer with matching shape and dtype')
        self.check(self.lib.adm_gaussian(self.handle, src.handle, dst.handle, w, h,
                                        weights.ctypes.data_as(C.POINTER(C.c_float)), size))
        return dst

    def warp(self, image, transform, shape):
        h, w = image.shape[:2]; oh, ow = shape
        rgb = self.buffer((oh, ow, 3), np.uint8); confidence = self.buffer((oh, ow))
        inverse = np.linalg.inv(transform); inverse /= inverse[2, 2]
        self.dispatch('warp_confidence', [image, rgb, confidence], ow * oh,
                      [w, h, ow, oh], inverse.ravel())
        return rgb, confidence

    def corner_search(self, image, patch, transform, origin, radius):
        h, w = image.shape[:2]; size = patch.shape[0]; extent = size + 2 * radius
        warped = self.buffer((extent, extent, 3), np.uint8)
        valid = self.buffer((extent, extent), np.uint8)
        scores = self.buffer((2 * radius + 1, 2 * radius + 1))
        matrix = transform / transform[2, 2]
        self.dispatch('rectify', [image, warped, valid], extent * extent,
                      [w, h, extent, extent], [*matrix.ravel(), origin[0] - radius, origin[1] - radius])
        self.dispatch('ncc', [warped, self.upload(patch), valid, scores], scores.shape[0] ** 2 * 128,
                      [size, extent], group=128)
        return warped.numpy(False), scores.numpy(False)

    def descriptors(self, data):
        data = np.asarray(data, np.float32)
        if data.ndim != 2 or min(data.shape) < 1 or not np.isfinite(data).all():
            raise ValueError('Descriptors must be a nonempty finite 2D matrix')
        values = self.upload(data, np.float32); norm = self.buffer(len(data))
        self.dispatch('descriptor_norm', [values, norm], len(data), [len(data), data.shape[1]])
        return values, norm

    def match_descriptors(self, a, b, batch=512):
        self.thread_check()
        av, an = a; bv, bn = b; rows, inner = av.shape; cols = bv.shape[0]
        if inner != bv.shape[1] or batch < 1 or max(rows, cols) > 2**24:
            raise ValueError('Incompatible descriptor shapes or batch size')
        if cols < 2: return np.empty((0, 4), np.float32)
        result = self.buffer((rows, 4)); product = self.buffer((min(batch, rows), cols))
        for start in range(0, rows, batch):
            count = min(batch, rows - start)
            self.check(self.lib.adm_matmul(self.handle, av.handle, bv.handle, product.handle,
                                          count, cols, inner, start * inner * 4))
            self.dispatch('nearest_two', [product, an, bn, result], count * 32,
                          [count, cols, start], group=32)
        return result.numpy()

    def tiles(self, bgr, size):
        h, w = bgr.shape[:2]; count = ((h + size - 1) // size) * ((w + size - 1) // size)
        result = self.buffer((count, size, size, 3), np.uint8)
        self.dispatch('pack_tiles', [self.upload(bgr), result], count * size * size,
                      [w, h, size, count * size * size])
        return result.numpy(False)

    def lighting(self, rgba, cfg, height=None, spacing=None):
        from .lighting import sun_vector
        h, w = rgba.shape[:2]; count = h * w
        source = self.upload(rgba); linear = self.buffer((h, w, 3)); encoded = self.buffer(rgba.shape, np.uint8)
        if height is None:
            # Every environment has the same flat-ground equation.
            self.dispatch('flat_light', [source, linear, encoded], count, [count],
                          [*sun_vector(cfg.sun_elevation_deg, cfg.sun_azimuth_deg), cfg.light_strength])
        else:
            terrain = self.upload(height, np.float32)
            offsets = np.linspace(-4, 4, cfg.lighting_candidates) if cfg.lighting_candidates > 1 else [0.]
            radius = (int(round(.8 * 8 + 1)) | 1) // 2
            # Bound environment storage by stripe size, never candidates × full map.
            for y in range(0, h, 128):
                th = min(128, h-y); n = th*w
                normal = self.buffer((th, w, 4)); mean = self.buffer((th, w, 4), zero=True)
                self.dispatch('terrain_normals', [terrain, normal], n, [w, h, y, th], [spacing])
                visibility = self.buffer((th + 2*radius, w, 4)); blurred = self.buffer(visibility.shape)
                candidates = []
                for offset in offsets:
                    el = float(np.clip(cfg.sun_elevation_deg + offset, 1, 89)); az = cfg.sun_azimuth_deg + offset*2
                    angle = np.deg2rad(az)
                    step = (cfg.shadow_max_distance_m - spacing) / (cfg.shadow_steps - 1) if cfg.shadow_steps > 1 else 0
                    self.dispatch('terrain_visibility', [terrain, source, visibility], (th + 2*radius)*w,
                                  [w, h, y-radius, th+2*radius, cfg.shadow_steps],
                                  [spacing, step, np.sin(angle)/spacing, -np.cos(angle)/spacing, np.tan(np.deg2rad(el))])
                    self.gaussian(visibility, .8, blurred)
                    candidate = self.buffer((th, w, 4)); candidates.append(candidate)
                    self.dispatch('light_candidate', [source, normal, blurred, candidate, mean], n,
                                  [n, y*w, radius*w], [*sun_vector(el, az), cfg.light_strength, 1/len(offsets)])
                best = self.buffer((th, w)); chosen = self.buffer((th, w, 4))
                for environment, candidate in enumerate(candidates):
                    # Preserve the CPU reference's environment-major PCG64 draws.
                    rng = np.random.default_rng(cfg.seed + 1); rng.bit_generator.advance(environment*count + y*w)
                    uniform = self.upload(rng.uniform(.0001, .9999, (th, w)).astype(np.float32))
                    self.dispatch('choose_light', [candidate, mean, uniform, best, chosen], n, [n, environment])
                self.dispatch('export_light', [chosen, source, linear, encoded], n, [n, y*w])
                self.sync()
        return linear.numpy(False), encoded.numpy(False)


_local = threading.local()


def get_backend():
    if not hasattr(_local, 'backend'): _local.backend = MetalBackend()
    return _local.backend


def shade_metal(rgb, normals, visibility, elevation, azimuth, strength):
    from .lighting import sun_vector
    gpu = get_backend(); rgb = np.asarray(rgb, np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or np.shape(normals) != rgb.shape or np.shape(visibility) != rgb.shape[:2]:
        raise ValueError('Expected H×W×3 RGB/normals and H×W visibility')
    result = gpu.buffer(rgb.shape)
    gpu.dispatch('shade_reference', [gpu.upload(rgb), gpu.upload(normals, np.float32),
                                   gpu.upload(visibility, np.float32), result], rgb.shape[0]*rgb.shape[1],
                 [rgb.shape[0]*rgb.shape[1]], [*sun_vector(elevation, azimuth), strength])
    return result.numpy()
