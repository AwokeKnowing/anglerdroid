"""Bake the old, stretched carpet's lengthwise ridges into the carpet textures (normal map; colour/roughness re-tiled).
Ridges (James): wave crests ~1 cm tall x ~8 cm wide x ~100 cm long running lengthwise; measured on 21 Oct (RS2 raw depth,
3 stationary spots, 0.56 m2 each): band-passed (3-25 cm) floor relief RMS 1.0-1.7 mm, p1-p99 4.6-11 mm, no clear
orientation -> the default amplitude here is 5 mm (between the measurement and the estimate), all values in dressing.json
"carpet_ridges".  Output: a 2 m x 2 m tileable texture set (4096 px, ~0.5 mm/px, same detail density as the 1K 0.5 m
carpet tile) = the carpet tile repeated 4x4 with the ridge height field folded into the normal map (whiteout blend).
Shading only (normal map): at <= 1 cm the silhouette change is invisible from the robot cameras.
usage: carpet_ridges.py <textures_dir> <dressing.json>"""
import json, math, os, sys
import numpy as np, cv2

tex, dj = sys.argv[1:3]; P = json.load(open(dj))["carpet_ridges"]
S, L = 4096, 2.0; px = L / S; rng = np.random.default_rng(P.get("seed", 7))
H = np.zeros((S, S), np.float32)
yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) * px
th = math.radians(P.get("dir_deg", 90.0))          # ridge long axis in the texture (= world, UV is planar x,y) frame
n = int(P.get("density_per_m2", 2.0) * L * L)
for _ in range(n):
    cx, cy = rng.uniform(0, L, 2); ln = P["length_m"] * rng.uniform(0.6, 1.4); wd = P["width_m"] * rng.uniform(0.7, 1.3)
    a = P["amp_m"] * rng.uniform(0.5, 1.0); t = th + math.radians(rng.normal(0, P.get("dir_jitter_deg", 6)))
    for ox in (-L, 0, L):                           # wrap -> tileable
        for oy in (-L, 0, L):
            dx, dy = xx - (cx + ox), yy - (cy + oy)
            u = dx * math.cos(t) + dy * math.sin(t); v = -dx * math.sin(t) + dy * math.cos(t)
            m = (np.abs(u) < ln / 2) & (np.abs(v) < wd / 2)
            if m.any():
                H[m] += a * (0.5 + 0.5 * np.cos(2 * math.pi * v[m] / wd)) * (0.5 + 0.5 * np.cos(2 * math.pi * u[m] / ln))
gy, gx = np.gradient(H, px)
def load(sfx, flag=cv2.IMREAD_COLOR): return cv2.imread(os.path.join(tex, "Carpet011_1K-JPG" + sfx), flag)
def tile4(im): return cv2.resize(np.tile(im, (4, 4, 1) if im.ndim == 3 else (4, 4)), (S, S), interpolation=cv2.INTER_LINEAR)
nd = tile4(load("_NormalGL.jpg")).astype(np.float32)[..., ::-1] / 127.5 - 1          # RGB, OpenGL (+y up = -image rows)
# ridge normal in the same (OpenGL) convention: image rows grow downward = -v
nr = np.stack([-gx, gy, np.ones_like(gx)], -1)
# whiteout blend of detail + ridge normals
nb = np.stack([nd[..., 0] + nr[..., 0], nd[..., 1] + nr[..., 1], nd[..., 2] * nr[..., 2]], -1)
nb /= np.linalg.norm(nb, axis=-1, keepdims=True)
cv2.imwrite(os.path.join(tex, "CarpetRidged_NormalGL.jpg"), ((nb + 1) * 127.5).clip(0, 255).astype(np.uint8)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite(os.path.join(tex, "CarpetRidged_Beige.jpg"), tile4(load("_Beige.jpg")), [cv2.IMWRITE_JPEG_QUALITY, 90])
cv2.imwrite(os.path.join(tex, "CarpetRidged_Roughness.jpg"), tile4(load("_Roughness.jpg", cv2.IMREAD_GRAYSCALE)), [cv2.IMWRITE_JPEG_QUALITY, 90])
np.save(os.path.join(tex, "CarpetRidged_height.npy"), H.astype(np.float16))
print(json.dumps(dict(ridges=n, max_height_mm=round(float(H.max()) * 1e3, 2), slope_deg_max=round(math.degrees(math.atan(float(np.hypot(gx, gy).max()))), 1), coverage=round(float((H > 0.0005).mean()), 3))))
