import sys, os, json, time, math, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maploc import StructMap, Localizer, LocalizerSeg, se2, image_dt, bilinear, R_MAST, AXLE_H_STOCK, axle_height
mi = json.load(open("mlx/mast.json")); S = int(os.environ.get("ML_DOWN", 1))
# lens = what the robot code reads (info["mast"] via slamlog Log.mast_lens: the twin truth lens, intr + dist; see export.py)
K0 = np.array([[mi["intr"]["fx"], 0, mi["intr"]["ppx"]], [0, mi["intr"]["fy"], mi["intr"]["ppy"]], [0, 0, 1]])
DIST = np.array(mi["dist"], float) if any(mi.get("dist", [])) else None
K = np.array([[K0[0, 0] / S, 0, (K0[0, 2] + 0.5) / S - 0.5], [0, K0[1, 1] / S, (K0[1, 2] + 0.5) / S - 0.5], [0, 0, 1]]); W, H = 640 // S, 480 // S
_UMAP = cv2.initUndistortRectifyMap(K0, DIST, None, K0, (640, 480), cv2.CV_32FC1) if DIST is not None else None
# mast pos is relative to base_link (twin frames.rgb.xyz); base_link sits at the axle = the log's tyre loaded radius above
# the floor (stock 0.08565 -> camera 0.97615 m; soft_v3_2 0.10065 -> 0.99115 m).  v1-v3 wrongly subtracted it (camera
# 8.6 cm too low -> ~-9 % range bias).  ML_BASE_Z overrides the base height (all logs).
def log_tire(n):
    """tyre profile a log was recorded with: poses.npz "tire" (export.py copies info["tire"]); none -> stock (pre-7fcec63)."""
    z = np.load(f"mlx/{n}/poses.npz"); return str(z["tire"]) if "tire" in z.files and str(z["tire"]) else "stock"
def log_axle(n=None): return AXLE_H_STOCK if n is None else axle_height(log_tire(n))
def tbc(n=None):
    """base_link -> camera for log n (n=None: stock).  Use with se2(..., z=log_axle(n))."""
    a = log_axle(n); bz = float(os.environ.get("ML_BASE_Z", a))
    T = np.eye(4); T[:3, :3] = _Ry @ R_MAST; T[:3, 3] = _Ry @ np.array([mi["pos"][0], mi["pos"][1], mi["pos"][2] + bz - a]); return T
PITCH = math.radians(float(os.environ.get("ML_PITCH_DEG", 0.0)))   # static body pitch (+ = nose down, about base y)
_Ry = np.array([[math.cos(PITCH), 0, math.sin(PITCH)], [0, 1, 0], [-math.sin(PITCH), 0, math.cos(PITCH)]])
Tbc = tbc()   # stock default (diagnostics use tbc(n) / log_axle(n) per log)
smap = StructMap(None, zmin=float(os.environ.get("ML_ZMIN", -1)))   # ML_STL or the resolved world mesh (world_params)
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
METHOD = os.environ.get("ML_METHOD", "seg")
def make_loc(n=None, **kw):
    """localiser for log n (camera height from that log's tyre; n=None: stock)"""
    kw.setdefault("axle_h", log_axle(n))
    return LocalizerSeg(smap, K, W, H, tbc(n), **kw) if METHOD == "seg" else Localizer(smap, K, W, H, tbc(n), **kw)
