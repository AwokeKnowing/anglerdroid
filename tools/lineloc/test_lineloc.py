"""Unit tests (python3 -m pytest -q test_lineloc.py)."""
import math, json, numpy as np, pytest, cv2
from lineloc import Camera, LineMap, LineLoc, wrap, se2_cam, world_stl
import os
STL = world_stl()   # LL_STL or the resolved world mesh
mi = json.load(open(os.environ.get("LL_MAST", "/workspace/vl/mlx/mast.json")))
_CAM_H = float(np.load("casita_lines.npz")["cam_h"]) if os.path.exists("casita_lines.npz") else None   # the map's camera height
def twin_cam(): return Camera(mi["intr"]["fx"], mi["intr"]["fy"], mi["intr"]["ppx"], mi["intr"]["ppy"], mi["dist"], t_bc=None if _CAM_H is None else (0.0291, 0, _CAM_H))

def test_resect_exact():
    rng = np.random.default_rng(1)
    for _ in range(50):
        P = rng.uniform(-5, 5, (3, 2)); cp = np.r_[rng.uniform(-1, 1, 2), rng.uniform(-3, 3)]
        d = P - cp[:2]; beta = wrap(np.arctan2(d[:, 1], d[:, 0]) - cp[2])
        sols = LineLoc.resect(P, beta); assert any(np.allclose(s[:2], cp[:2], atol=1e-6) and abs(wrap(s[2] - cp[2])) < 1e-6 for s in sols)

def test_undistort_roundtrip():
    cam = twin_cam(); xy = np.random.default_rng(0).uniform(-1.2, 1.2, (100, 2)); uv = cam.distort_norm(xy)
    assert np.abs(cam.undistort(uv, it=20) - xy).max() < 1e-6

def test_vertical_classification_with_tilt():
    cam = twin_cam(); cam.R_bc = Camera.tilt(math.radians(2.2), math.radians(-3.2)) @ cam.R_bc
    L = LineLoc(None, cam); pose = np.array([3.8, 1.4, math.radians(95.8)]); c, R = se2_cam(pose, cam); Rcw = (R @ cam.R_bc).T
    rng = np.random.default_rng(2); segs = []; truth = []
    for b in np.linspace(-0.8, 0.8, 40) + rng.uniform(-0.005, 0.005, 40):
        r = rng.uniform(1, 5); p = c[:2] + r * np.array([math.cos(pose[2] + b), math.sin(pose[2] + b)])
        X = np.array([[p[0], p[1], 0.2], [p[0], p[1], 2.0]]); Xc = (X - c) @ Rcw.T; segs.append(cam.distort_norm(Xc[:, :2] / Xc[:, 2:]).ravel()); truth.append(b)
    V, Hh = L.observe(np.array(segs)); assert len(Hh) == 0 and len(V) == 40
    assert np.abs(np.sort(V[:, 0]) - np.sort(truth)).max() < 1e-6

@pytest.fixture(scope="module")
def setup():
    from synth import Renderer
    return Renderer(STL), LineMap("casita_lines.npz"), twin_cam()

def test_synthetic_poses(setup):
    R, M, cam = setup; gt = np.load(os.environ.get("LL_GT", "/workspace/vl/mlx/L20/poses.npz"))["gt"]; rng = np.random.default_rng(3); errs = []
    # exact map (synthetic): all horizontal edges, no map-error term, full-res-ish LSD
    loc = LineLoc(M, cam, lsd_scale=0.8, sigma_map=0.0, h_junction=False, h_minlen=0)
    for k, i in enumerate(np.linspace(0, len(gt) - 1, 8).astype(int)):
        p = gt[i]; img = R.render(cam, p, seed=k)
        P = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(2) ** 2]); prior = p + rng.multivariate_normal(np.zeros(3), P)
        x, Pp, info = loc.update(img, prior, P)
        e = math.hypot(*(x[:2] - p[:2])); ey = math.degrees(abs(wrap(x[2] - p[2]))); errs.append((e, ey, info.get("ok")))
        print(i, round(e * 100, 2), "cm", round(ey, 3), "deg", info)
    e = np.array([q[:2] for q in errs]); assert all(q[2] for q in errs); assert np.median(e[:, 0]) < 0.01 and e[:, 0].max() < 0.03 and e[:, 1].max() < 0.3

def test_synthetic_defaults_smoke(setup):
    """deployed defaults (LSD 0.5, junction lines, sigma_map 2 cm): from a 5 cm / 2 deg prior every accepted fix is
    within 7 cm and most frames are accepted (single frame, no tracking)."""
    R, M, cam = setup; gt = np.load(os.environ.get("LL_GT", "/workspace/vl/mlx/L20/poses.npz"))["gt"]; rng = np.random.default_rng(3); loc = LineLoc(M, cam); ok = 0
    for k, i in enumerate(np.linspace(0, len(gt) - 1, 8).astype(int)):
        p = gt[i]; P = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(2) ** 2]); x, _, info = loc.update(R.render(cam, p, seed=k), p + rng.multivariate_normal(np.zeros(3), P), P)
        if info.get("ok"): ok += 1; assert math.hypot(*(x[:2] - p[:2])) < 0.07
    assert ok >= 6
