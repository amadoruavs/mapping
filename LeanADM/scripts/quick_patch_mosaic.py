"""Quick GPS-seeded, corner-patch-refined averaging mosaic (no SfM).

Per frame: centre from GPS, heading from GPS course (image top = forward),
scale from altitude above takeoff. Two diagonal 50x50 corner patches are
template-matched into the next frame; if both match with a sane scale, the
frame's affine is refit through (centre, patch A, patch B) so it lands on the
neighbour. Overlaps are averaged.
"""
import json, math, sys, time
from pathlib import Path
import cv2, numpy as np
from PIL import Image

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/avneh/Downloads/101SIYI_IMG")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "output/quick_patch_mosaic")
OUT.mkdir(parents=True, exist_ok=True)
DOWN, P, INSET, NCC_MIN = 4, 50, 40, 0.55
MOVE_MIN = 1.0           # metres: pairs used for heading/scale estimation
SCALE_OK = (0.85, 1.18)  # accepted pair scale ratio (rejects false patch hits)
t0 = time.time()

def gps(path):
    ex = Image.open(path).getexif().get_ifd(0x8825)
    dms = lambda v, ref: (float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600) * (-1 if ref in "SW" else 1)
    return dms(ex[2], ex[1]), dms(ex[4], ex[3]), float(ex[6])

files = sorted(SRC.glob("*.jpg"))
imgs = [cv2.imread(str(f), cv2.IMREAD_REDUCED_COLOR_4) for f in files]
grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
H, W = grays[0].shape; N = len(imgs); C = np.array([W / 2, H / 2])
g = np.array([gps(f) for f in files])
lat0 = math.radians(g[:, 0].mean())
east = (g[:, 1] - g[:, 1].mean()) * 111320 * math.cos(lat0)
north = (g[:, 0] - g[:, 0].mean()) * 110540
agl = np.maximum(g[:, 2] - g[0, 2], 0.5)          # altitude above first (takeoff) frame
print(f"loaded {N} frames {W}x{H} in {time.time()-t0:.1f}s; AGL {agl.min():.1f}..{agl.max():.1f} m")

# ---- corner patches of frame i matched into frame i+1 ----------------------
corners = {"TL": (INSET, INSET), "BR": (W - INSET - P, H - INSET - P),
           "TR": (W - INSET - P, INSET), "BL": (INSET, H - INSET - P)}
def match_pair(a, b):
    hits = {}
    for name, (x, y) in corners.items():
        res = cv2.matchTemplate(grays[b], grays[a][y:y + P, x:x + P], cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score >= NCC_MIN:
            hits[name] = (score, np.array([x + P / 2, y + P / 2]), np.array([loc[0] + P / 2, loc[1] + P / 2]))
    combos = [("TL", "BR"), ("TR", "BL"), ("TL", "TR"), ("BL", "BR"), ("TL", "BL"), ("TR", "BR")]  # diagonals first
    best = None
    for n0, n1 in combos:
        if n0 in hits and n1 in hits:
            (s0, a0, b0), (s1, a1, b1) = hits[n0], hits[n1]
            ratio = np.linalg.norm(b1 - b0) / np.linalg.norm(a1 - a0)
            if SCALE_OK[0] <= ratio <= SCALE_OK[1]:
                cand = {"ok": True, "names": (n0, n1), "ncc": [round(s0, 3), round(s1, 3)], "src": [a0, a1],
                        "dst": [b0, b1], "ratio": ratio, "shift": ((b0 - a0) + (b1 - a1)) / 2, "diag": n0[0] != n1[0] and n0[1] != n1[1]}
                if best is None or (cand["diag"] and not best["diag"]) or (cand["diag"] == best["diag"] and sum(cand["ncc"]) > sum(best["ncc"])):
                    best = cand
    return best or {"ok": False}

pair = [match_pair(i, i + 1) for i in range(N - 1)]
print(f"patch matching {time.time()-t0:.1f}s; {sum(p['ok'] for p in pair)}/{N-1} pairs accepted, "
      f"{sum(p.get('diag', False) for p in pair)} of them diagonal")

# ---- heading per frame from GPS course; scale from moving, matched pairs ---
d_e, d_n = np.diff(east), np.diff(north)
step = np.hypot(d_e, d_n)
course = np.full(N, np.nan)
for i in range(N):
    segs = [k for k in (i - 1, i) if 0 <= k < N - 1 and step[k] > 0.5]
    if segs:
        course[i] = math.atan2(np.mean(d_e[segs]), np.mean(d_n[segs]))   # bearing, clockwise from north
idx = np.where(~np.isnan(course))[0]
course = np.array([course[idx[np.argmin(abs(idx - i))]] for i in range(N)])  # nearest moving frame
# canvas: x = east, y = south (north up). Frame rotation: image "up" (-y) points along the bearing.
def frame_R(i):
    b = course[i]; s = S_REF * agl[i] / AGL_REF   # canvas px per image px
    # image (0,-1) -> canvas (sin b, -cos b)  =>  (0,1) -> (-sin b, cos b);  (1,0) -> (cos b, sin b)
    return s * np.array([[math.cos(b), -math.sin(b)], [math.sin(b), math.cos(b)]])
mov = [i for i, p in enumerate(pair) if p["ok"] and step[i] > MOVE_MIN]
AGL_REF = float(np.mean([agl[i] for i in mov]))
# camera moved by -shift (image px) ; in metres it moved step[i]  ->  px/m at the working res
S_PX_PER_M = np.median([np.linalg.norm(pair[i]["shift"]) / step[i] for i in mov])
S_REF = 1.0                     # canvas px per image px at AGL_REF (canvas GSD == image GSD at cruise)
GSD_FULL = 100 / (S_PX_PER_M * DOWN)
print(f"{len(mov)} moving matched pairs; {S_PX_PER_M:.1f} px/m at 1/{DOWN} -> {GSD_FULL:.2f} cm/px full res, AGL_REF {AGL_REF:.1f} m")
# sanity: forward direction in the image should be -y
fwd = np.mean([-pair[i]["shift"] for i in mov], 0)
print(f"mean camera motion in image px per moving pair = {fwd.round(1)} (expect ~(0,-100) if image top = forward)")

centre = np.c_[east, -north] * S_PX_PER_M * (S_REF)          # canvas centre of every frame (px)
def gps_affine(i):
    R = frame_R(i); return np.c_[R, centre[i] - R @ C]

# ---- per-frame placement: chain through patch matches, snap to GPS at breaks
# Frame i+1 is placed so that its two matched patch points land exactly where
# frame i's corresponding corner patches landed (2-point similarity). Where no
# match exists, the frame is placed from GPS centre + course heading + altitude.
def sim_from_2(src, dst):
    (x0, y0), (x1, y1) = src; (u0, v0), (u1, v1) = dst
    z = complex(u1 - u0, v1 - v0) / complex(x1 - x0, y1 - y0); a, b = z.real, z.imag
    return np.array([[a, -b, u0 - (a * x0 - b * y0)], [b, a, v0 - (b * x0 + a * y0)]])
final, mode, resid = [gps_affine(0)], ["gps"], []
for i in range(1, N):
    p = pair[i - 1]
    if p["ok"]:
        landed = [final[i - 1] @ [*q, 1] for q in p["src"]]      # where frame i-1's patches sit on the canvas
        M = sim_from_2(p["dst"], landed)                          # frame i pixels -> canvas
        resid.append(float(np.linalg.norm(M @ [*C, 1] - centre[i])))  # drift of chained centre vs GPS centre
        final.append(M); mode.append("patch")
    else:
        final.append(gps_affine(i)); mode.append("gps")
print(f"chained centre vs GPS centre: median {np.median(resid):.1f}px, max {max(resid):.1f}px (canvas px, 1px = {1/S_PX_PER_M*100:.1f} cm)")

# ---- composite: average every covering pixel ------------------------------
corn = np.float32([[[0, 0], [W, 0], [W, H], [0, H]]])
allc = np.vstack([cv2.transform(corn, M)[0] for M in final])
mn = np.floor(allc.min(0)).astype(int); mx = np.ceil(allc.max(0)).astype(int)
CW, CH = int(mx[0] - mn[0]), int(mx[1] - mn[1])
acc = np.zeros((CH, CW, 3), np.float32); cnt = np.zeros((CH, CW), np.float32)
for im, M in zip(imgs, final):
    M2 = M.copy(); M2[:, 2] -= mn
    acc += cv2.warpAffine(im, M2, (CW, CH), flags=cv2.INTER_LINEAR).astype(np.float32) * \
           cv2.warpAffine(np.ones((H, W), np.float32), M2, (CW, CH), flags=cv2.INTER_NEAREST)[..., None]
    cnt += cv2.warpAffine(np.ones((H, W), np.float32), M2, (CW, CH), flags=cv2.INTER_NEAREST)
mosaic = (acc / np.maximum(cnt, 1)[..., None]).astype(np.uint8)
cv2.imwrite(str(OUT / "mosaic_avg.png"), np.dstack([mosaic, (cnt > 0).astype(np.uint8) * 255]))
cv2.imwrite(str(OUT / "coverage.png"), (cnt / cnt.max() * 255).astype(np.uint8))
prev = mosaic.copy()
for i, M in enumerate(final):
    c = (M @ [*C, 1] - mn).astype(int)
    cv2.circle(prev, tuple(c), 5, (0, 0, 255) if mode[i] == "patch" else (255, 0, 255), -1)
    cv2.putText(prev, files[i].stem[-2:], tuple(c + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
cv2.imwrite(str(OUT / "mosaic_centres.jpg"), prev, [cv2.IMWRITE_JPEG_QUALITY, 88])
json.dump({"canvas": [CW, CH], "gsd_cm_px_full_res": GSD_FULL, "px_per_m_working": S_PX_PER_M,
           "agl_ref_m": AGL_REF, "frames": [{"file": f.name, "mode": m, "agl_m": float(a), "bearing_deg": math.degrees(b) % 360,
           "centre_px": (M @ [*C, 1] - mn).tolist()} for f, m, a, b, M in zip(files, mode, agl, course, final)],
           "pairs": [{k: (v if not isinstance(v, (np.ndarray, list)) else np.asarray(v).round(2).tolist()) for k, v in p.items()} for p in pair],
           "chain_vs_gps_centre_px": resid}, open(OUT / "report.json", "w"), indent=1, default=float)
print(f"canvas {CW}x{CH}; done in {time.time()-t0:.1f}s -> {OUT}")
