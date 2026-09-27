"""Metal spatial FSR1. Generated pixels are not independent observations."""
import numpy as np


def _source(image, gpu):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError('FSR expects an encoded uint8 BGR or BGRA image')
    return gpu.shared_input(image)


def easu(image, shape, gpu):
    """One 1–2x-per-axis edge-adaptive spatial upscale, preserving pixel centers."""
    h, w = image.shape[:2]
    oh, ow = map(int, shape)
    if not (h <= oh <= 2*h and w <= ow <= 2*w):
        raise ValueError('Each EASU pass must scale each axis by 1 to 2')
    src = _source(image, gpu)
    dst = gpu.buffer((oh, ow, image.shape[2]), np.uint8)
    gpu.dispatch('fsr_easu', [src, dst], ow*oh, [w, h, ow, oh, image.shape[2]])
    return dst.numpy(False)


def rcas(image, strength, gpu):
    """Noise-attenuated RCAS; strength 0 disables, 1 is AMD's maximum lobe."""
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('RCAS strength must be in [0,1]')
    src = _source(image, gpu)
    if strength == 0:
        return image
    h, w, channels = image.shape
    dst = gpu.buffer(image.shape, np.uint8)
    gpu.dispatch('fsr_rcas', [src, dst], w*h, [w, h, channels], [strength])
    return dst.numpy(False)


def upscale(image, passes, strength, gpu):
    """Bounded recursive experiment; returns image and native-to-output transform."""
    if not isinstance(passes, int) or not 0 <= passes <= 3:
        raise ValueError('Upscale passes must be an integer from 0 to 3')
    for _ in range(passes):
        image = rcas(easu(image, (image.shape[0]*2, image.shape[1]*2), gpu), strength, gpu)
    scale = 2**passes
    return image, np.array([[scale, 0, (scale-1)/2], [0, scale, (scale-1)/2], [0, 0, 1.]])


def interpolate(a, b, a_to_mid, b_to_mid, fraction, gpu, gated=True):
    """ADM planar virtual view, with no invented depth or independent measurements."""
    if a.shape != b.shape or a.shape[2] != 4 or not 0 < fraction < 1:
        raise ValueError('Interpolation needs equal BGRA shapes and 0 < fraction < 1')
    ha, hb = np.linalg.inv(a_to_mid), np.linalg.inv(b_to_mid)
    ha /= ha[2, 2]; hb /= hb[2, 2]
    if not np.isfinite([ha,hb]).all():raise ValueError('Nonfinite virtual view transform')
    h, w = a.shape[:2]
    dst = gpu.buffer(a.shape, np.uint8)
    gpu.dispatch('photo_interpolate', [_source(a,gpu), _source(b,gpu), dst], w*h,
                 [w,h,int(gated)], [*ha.ravel(),*hb.ravel(),fraction,24/255])
    return dst.numpy(False)
