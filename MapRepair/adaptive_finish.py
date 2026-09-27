"""Offline photographic finishing. Enhances measured texture, never generates it.

One scene analysis feeds every strip: no tile-dependent tone mapping or seams.
Local extrema constrain detail; a native-resolution noise estimate prevents
sharpening flat-area noise. All spatial filters respect coverage alpha.
"""
import cv2
import numpy as np
from parallel_runtime import bounded_map, worker_count

LUMA = np.array([.2126, .7152, .0722], np.float32)
HALO = 36  # Explicit 65-tap broad filter plus local extrema support.


def proxy(image, edge=1024):
    h, w = image.shape[:2]
    scale = min(1., edge / max(h, w))
    pw, ph = max(1, round(w * scale)), max(1, round(h * scale))
    if (pw, ph) == (w, h):
        return image.copy()
    result = np.empty((ph, pw, 4), np.uint8)
    for y in range(ph):
        lo, hi = int(np.floor(y*h/ph)), int(np.ceil((y+1)*h/ph))
        rows = image[lo:hi].astype(np.float32)
        weights = (np.minimum(np.arange(lo, hi)+1, (y+1)*h/ph)
                   - np.maximum(np.arange(lo, hi), y*h/ph)).astype(np.float32)
        rows[:, :, :3] *= rows[:, :, 3:4] / 255
        row = np.sum(rows * weights[:, None, None], axis=0) / (h/ph)
        row = cv2.resize(row[None], (pw, 1), interpolation=cv2.INTER_AREA)[0]
        row[:, :3] /= np.maximum(row[:, 3:4]/255, 1e-6)
        result[y] = np.clip(row+.5, 0, 255).astype(np.uint8)
    return result


def estimate_noise(image):
    """Robust diagonal Haar residual on the smoothest sampled native patches."""
    h, w = image.shape[:2]
    samples = []
    for y in np.unique(np.linspace(0, max(0, h-64), 5).astype(int)):
        for x in np.unique(np.linspace(0, max(0, w-64), 6).astype(int)):
            patch = image[y:y+64, x:x+64]
            a = patch[:, :, 3] > 250
            v = patch[:, :, 2::-1].astype(np.float32) @ LUMA / 255
            nh, nw = v.shape[0]//2*2, v.shape[1]//2*2
            if min(nh, nw) < 8:
                continue
            valid = a[:nh:2, :nw:2] & a[1:nh:2, :nw:2] & a[:nh:2, 1:nw:2] & a[1:nh:2, 1:nw:2]
            if valid.sum() < 32:
                continue
            residual = (v[:nh:2, :nw:2]-v[1:nh:2, :nw:2]-v[:nh:2, 1:nw:2]+v[1:nh:2, 1:nw:2])*.5
            sigma = float(np.median(np.abs(residual[valid]-np.median(residual[valid])))/.67449)
            activity = float(np.std(v[a]))
            samples.append((activity, sigma))
    samples.sort()
    return float(np.clip(np.median([s[1] for s in samples[:max(1, len(samples)//4)]])
                         if samples else 1/255, .5/255, 12/255))


def analyze(image):
    small = proxy(image)
    rgb = small[:, :, 2::-1].astype(np.float32)/255
    valid = small[:, :, 3] > 250
    if not valid.any():
        valid = small[:, :, 3] > 0
    y = rgb @ LUMA
    gain = np.ones(3, np.float32)
    # Only near-neutral bright surfaces qualify; vegetation is never a gray card.
    neutral = valid & (np.ptp(rgb, axis=2) < .055) & (y > .55) & (y < .90)
    if neutral.sum() >= 64:
        median = np.median(rgb[neutral], axis=0)
        gain = np.clip(float(median @ LUMA)/np.maximum(median, .01), .975, 1.025)
    percentiles = np.percentile(y[valid], [1, 10, 50, 90, 99]) if valid.any() else np.array([0, .1, .5, .9, 1])
    p1, p10, p50, p90, p99 = map(float, percentiles)
    spread = p99-p1
    # Low-contrast scenes get a bounded adjustment, not an arbitrary full stretch.
    confidence = float(np.clip(spread/.35, 0, 1))
    exposure = float(np.clip(np.log2(.46/max(p50, .06))*.24, -.18, .22)*confidence)
    saturation = np.ptp(rgb, axis=2)
    chroma = float(np.median(saturation[valid])) if valid.any() else 0.
    return dict(version=1, offline=True, luminance_percentiles=percentiles.tolist(),
                noise_sigma_codes=estimate_noise(image)*255, neutral_reference_pixels=int(neutral.sum()),
                white_balance_rgb_gain=gain.tolist(), exposure_stops=exposure,
                contrast_strength=float((.07+.07*np.clip((.75-spread)/.4, 0, 1))*confidence),
                saturation_gain=float(1+.055*np.clip((.28-chroma)/.20, 0, 1)),
                geometry='unchanged', synthetic_detail=False)


def masked_blur(y, alpha, sigma):
    size = 2*int(np.ceil(4*sigma))+1
    weight = cv2.GaussianBlur(alpha, (size, size), sigma)
    return cv2.GaussianBlur(y*alpha, (size, size), sigma)/np.maximum(weight, 1e-6)


def detail_luma(y, alpha, noise):
    """Multiscale sharpening bounded by existing local extrema (no edge overshoot)."""
    fine = y-masked_blur(y, alpha, .7)
    medium = y-masked_blur(y, alpha, 1.6)
    broad = y-masked_blur(y, alpha, 8.)
    kernel = np.ones((5, 5), np.uint8)
    lo = cv2.erode(np.where(alpha > 0, y, 1).astype(np.float32), kernel)
    hi = cv2.dilate(np.where(alpha > 0, y, 0).astype(np.float32), kernel)
    # Strong roof/runway edges get less gain than low-contrast natural texture.
    edge_guard = 1/(1+((hi-lo)/.18)**4)
    threshold = max(.6/255, 2.6*noise)
    signal = np.clip((np.abs(fine)-threshold)/np.maximum(np.abs(fine), 1e-6), 0, 1)
    medium_signal = np.clip((np.abs(medium)-threshold)/np.maximum(np.abs(medium), 1e-6), 0, 1)
    delta = edge_guard*(.80*fine*signal+.28*medium*medium_signal)
    # Broad local contrast remains bounded and cannot create new local extrema.
    delta += .12*np.sign(broad)*np.maximum(np.abs(broad)-threshold, 0)*edge_guard
    return np.clip(y+np.clip(delta, -.045, .045), lo, hi)


def in_gamut(y, chroma):
    """Compress chroma around luminance rather than clip individual RGB channels."""
    positive = np.maximum(chroma.max(2), 1e-6)
    negative = np.maximum(-chroma.min(2), 1e-6)
    amount = np.minimum(1, np.minimum((1-y)/positive, y/negative))
    return y[:, :, None]+chroma*amount[:, :, None]


def finish_strip(image, analysis):
    rgb = image[:, :, 2::-1].astype(np.float32)/255
    rgb *= np.asarray(analysis['white_balance_rgb_gain'], np.float32)
    y = np.clip(rgb @ LUMA, 0, 1)
    # Anchored, monotone tone curve: exact black/white, gentle midtone exposure.
    target = y + analysis['exposure_stops']*.32*4*y*(1-y)
    target += analysis['contrast_strength']*(target-.5)*(1-np.abs(2*target-1))
    target = np.clip(target, 0, 1)
    alpha = image[:, :, 3].astype(np.float32)/255
    target = detail_luma(target, alpha, analysis['noise_sigma_codes']/255)
    # Retain near-white/black code separation instead of rounding extra pixels
    # into clipping after contrast. Exact endpoints remain anchored.
    target = y+(target-y)*np.clip(np.minimum(y,1-y)/.03,0,1)
    chroma = rgb-y[:, :, None]
    # Soft per-pixel color response retains grass hue and neutral roof markings.
    chroma *= (1+(analysis['saturation_gain']-1)*(1-np.clip(np.ptp(rgb, axis=2)/.55, 0, 1)))[:, :, None]
    # Subtle warm upper tones/cool shadows; zero luminance bias and no vignette/glow.
    warmth = (.004*np.maximum(2*y-1, 0)-.002*np.maximum(1-2*y, 0))*4*y*(1-y)
    tint = np.array([1., .10, -.65], np.float32);tint -= tint @ LUMA
    chroma += warmth[:, :, None]*tint
    result = image.copy()
    result[:, :, :3] = np.rint(np.clip(in_gamut(target, chroma)[:, :, ::-1], 0, 1)*255).astype(np.uint8)
    result[result[:, :, 3] == 0, :3] = 0
    return result


def finish_adaptive(image, strip_rows=128, workers=None):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4 or min(image.shape[:2]) < 1:
        raise ValueError('Adaptive finish expects nonempty BGRA8')
    if strip_rows < 1:
        raise ValueError('strip_rows must be positive')
    analysis = analyze(image)
    h, w = image.shape[:2]
    output = np.empty_like(image)
    def process(y):
        end = min(h, y+strip_rows);lo = max(0, y-HALO);hi = min(h, end+HALO)
        return y, finish_strip(image[lo:hi], analysis)[y-lo:end-lo]
    for y, strip in bounded_map(process, range(0, h, strip_rows), workers=workers or worker_count(),
                                bytes_per_item=max(1, w*(strip_rows+2*HALO)*192)):
        output[y:y+len(strip)] = strip
    analysis.update(alpha_preserved=bool(np.array_equal(image[:, :, 3], output[:, :, 3])),
                    operations=['scene-adaptive tone', 'neutral-only white balance', 'noise-gated multiscale luminance detail',
                                'local extrema halo protection', 'restrained cinematic color', 'luminance-preserving gamut compression'])
    return output, analysis


def suggest_grade(image, already_finished=False):
    """Editable starting point for the existing grading controls, computed offline."""
    a = analyze(image)
    p1, p10, p50, p90, p99 = a['luminance_percentiles']
    gain = a['white_balance_rgb_gain']
    strength = .55 if already_finished else 1.
    contrast = a['contrast_strength']*strength
    values = dict(exposure=round(a['exposure_stops']*strength, 2),
                  temperature=round(125*np.log2(gain[0]/gain[2]), 1),
                  tint=round(-300*np.log2(gain[1]), 1),
                  highlights=round(-12*np.clip((p99-.90)/.1, 0, 1)*strength),
                  shadows=round(8*np.clip((.25-p10)/.25, 0, 1)*strength),
                  vibrance=round((a['saturation_gain']-1)*150*strength),
                  clarity=round(9*strength), texture=round(12*strength),
                  sharpen=round(32*strength), radius=.7,
                  threshold=round(min(20, max(2, a['noise_sigma_codes']*2.6))),
                  detail_protection=100,
                  curves={'master':[[0, 0], [.25, .25-.18*contrast], [.75, .75+.18*contrast], [1, 1]]})
    return dict(settings=values, analysis=a,
                note='Offline scene analysis. Settings remain editable; fine detail is limited by the source image.')
