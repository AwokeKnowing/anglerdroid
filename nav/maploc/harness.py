import sys, os, json, time, math, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maploc import StructMap, Localizer, se2, image_dt, bilinear, R_MAST, AXLE_H
mi = json.load(open("mlx/mast.json")); S = int(os.environ.get("ML_DOWN", 2)); LENS = os.environ.get("ML_LENS", "truth")
if LENS == "truth":
    mt = json.load(open("mlx/mast_truth.json")); K0 = np.array([[mt["fx"], 0, mt["cx"]], [0, mt["fy"], mt["cy"]], [0, 0, 1]]); DIST = np.array([mt["k1"], mt["k2"], 0, 0])
else:
    K0 = np.array([[mi["intr"]["fx"], 0, mi["intr"]["ppx"]], [0, mi["intr"]["fy"], mi["intr"]["ppy"]], [0, 0, 1]]); DIST = None
K = np.array([[K0[0, 0] / S, 0, (K0[0, 2] + 0.5) / S - 0.5], [0, K0[1, 1] / S, (K0[1, 2] + 0.5) / S - 0.5], [0, 0, 1]]); W, H = 640 // S, 480 // S
_UMAP = cv2.initUndistortRectifyMap(K0, DIST, None, K0, (640, 480), cv2.CV_32FC1) if DIST is not None else None
Tbc = np.eye(4); Tbc[:3, :3] = R_MAST; Tbc[:3, 3] = [mi["pos"][0], mi["pos"][1], mi["pos"][2] - AXLE_H]
smap = StructMap("mlx/casitahouse-walls-fixed.stl", zmin=float(os.environ.get("ML_ZMIN", -1)))
MAP_DISTORT = float(os.environ.get("ML_MAP_DISTORT", 1.0))     # synthetic: model built this factor too large (true metric scale = 1/k)
if MAP_DISTORT != 1.0: smap.mesh.apply_scale(MAP_DISTORT)
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def rel(a, b):   # SE2 b expressed in a's frame
    c, s = math.cos(a[2]), math.sin(a[2]); dx, dy = b[0] - a[0], b[1] - a[1]; return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])
def compose(a, d):
    c, s = math.cos(a[2]), math.sin(a[2]); return np.array([a[0] + c * d[0] - s * d[1], a[1] + s * d[0] + c * d[1], a[2] + d[2]])
def anchor(tr, g0):
    out = [g0.copy()]
    for i in range(1, len(tr)): out.append(compose(out[-1], rel(tr[i - 1], tr[i])))
    return np.array(out)
def load(n):
    z = np.load(f"mlx/{n}/poses.npz"); return z
def gray(n, f):
    im = cv2.imread(f"{n}/{f}.jpg", cv2.IMREAD_GRAYSCALE)
    if _UMAP is not None: im = cv2.remap(im, _UMAP[0], _UMAP[1], cv2.INTER_LINEAR)
    return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) if S > 1 else im
