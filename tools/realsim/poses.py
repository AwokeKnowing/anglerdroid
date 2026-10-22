"""Camera poses of real pilot looks in the casita frame, for photoreal-sim comparison (tools/realsim).

Chain (all in the resolved world frame of src/world_params.py, see tools/casita_fit/camgeo.py):
  * anchor = the mast-camera fit 'b_zfix' of the parked 12:13 frame (casita_fit, camera z fixed 0.99115):
    the robot had not been moved since, and the 21:16 'pre' look shows the same view (checked by edge score below);
  * base_link pose of the anchor = camera xy minus the twin mast offset (0.0291 m forward), heading = camera yaw;
  * later looks = anchor (+) the wheel/IMU odometry the look recorded (odometry resets at stack restarts are
    chained by the caller: segments list which look starts a new odometry origin);
  * optional refinement: model-edge registration (dx, dy, dyaw of the base, mount pitch/roll fixed), scored by the
    truncated chamfer distance of rendered model edges (depth/normal discontinuities, raycast) to the real frame's
    long LSD line segments (default; architecture-dominated) or Canny edges.  Groups of in-place-turn looks share one
    position (local_pose.py joint fit), each look's heading is then refined alone (final_poses.py).  Reported before/after so the refinement is visible, never silent.
Mount (twin_geometry soft_v3_2): mast rgb (0.0291, 0, 0.8905) above the axle, height 0.99115; rs2_color
(-0.130066, 0.015, 0.382213), height 0.482863, pitch 27 deg down.
"""
import json, math, os, sys
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools", "casita_fit")); sys.path.insert(0, os.path.join(REPO, "src"))
import camgeo  # noqa: E402

ANCHOR = dict(x=3.8006249153953555, y=1.393381162892552, z=0.99115, yaw=1.6706337464446281, pitch=0.027961467909011526,
              roll=-0.05430334296121204, f=505.90498860925373, k1=-0.028701410450345806, k2=0.0, cx=799.5, cy=599.5)  # @1600x1200
MAST_FWD = 0.0291
RS2 = dict(fwd=-0.130066, left=0.015, z=0.482863, pitch=math.radians(27.0), fy_640=240.0 / math.tan(math.radians(21.0)))


def compose(a, d):
    x, y, t = a; dx, dy, dt = d
    return (x + dx * math.cos(t) - dy * math.sin(t), y + dx * math.sin(t) + dy * math.cos(t), t + dt)


def anchor_base():
    a = ANCHOR; return (a["x"] - MAST_FWD * math.cos(a["yaw"]), a["y"] - MAST_FWD * math.sin(a["yaw"]), a["yaw"])


def mast_cam(base, w=800, h=600):
    s = w / 1600.0; a = ANCHOR; x, y, t = base
    return dict(x=x + MAST_FWD * math.cos(t), y=y + MAST_FWD * math.sin(t), z=a["z"], yaw=t, pitch=a["pitch"], roll=a["roll"],
                f=a["f"] * s, cx=(a["cx"] + 0.5) * s - 0.5, cy=(a["cy"] + 0.5) * s - 0.5, k1=a["k1"], k2=0.0, w=w, h=h)


def rs2_cam(base, w=640, h=480):
    x, y, t = base; s = w / 640.0
    return dict(x=x + RS2["fwd"] * math.cos(t) - RS2["left"] * math.sin(t), y=y + RS2["fwd"] * math.sin(t) + RS2["left"] * math.cos(t),
                z=RS2["z"], yaw=t, pitch=RS2["pitch"], roll=0.0, f=RS2["fy_640"] * s, cx=(w - 1) / 2, cy=(h - 1) / 2, k1=0.0, k2=0.0, w=w, h=h)


def model_edges(mesh, p, step=2):
    cam = camgeo.Cam(p, p["w"], p["h"]); dist, fid = camgeo.raycast(mesh, cam, step=step)
    n = np.zeros(fid.shape + (3,)); ok = fid >= 0; n[ok] = mesh.face_normals[fid[ok]]
    n[fid == -1] = (0, 0, 1); n[fid == -2] = (0, 0, -1)
    d = np.where(np.isfinite(dist), dist, 50.0)
    e = np.zeros(fid.shape, bool)
    for ax in (0, 1):
        dd = np.abs(np.diff(d, axis=ax)) / np.minimum(d[:-1] if ax == 0 else d[:, :-1], d[1:] if ax == 0 else d[:, 1:])
        dn = 1 - (n[:-1] * n[1:]).sum(-1) if ax == 0 else 1 - (n[:, :-1] * n[:, 1:]).sum(-1)
        m = (dd > 0.06) | (dn > 0.3)
        if ax == 0: e[:-1] |= m
        else: e[:, :-1] |= m
    v, u = np.nonzero(e); return np.stack([u * step + step / 2, v * step + step / 2], 1), dist


def real_dt(img, mode="lsd", min_len=40):
    """distance transform to the real frame's edges: 'lsd' = long straight segments only (architecture-dominated;
    clutter edges are mostly short), 'canny' = all edges."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "canny":
        E = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.2), 40, 100)
    else:
        L = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g)[0]; E = np.zeros_like(g)
        if L is not None:
            for x0, y0, x1, y1 in L.reshape(-1, 4):
                if np.hypot(x1 - x0, y1 - y0) >= min_len: cv2.line(E, (int(x0), int(y0)), (int(x1), int(y1)), 255, 1)
    return cv2.distanceTransform(255 - E, cv2.DIST_L2, 3)


def edge_points(mesh, p, step=2):
    """3D world points of the model edge pixels seen from camera p (raycast once)."""
    cam = camgeo.Cam(p, p["w"], p["h"]); uv, dist = model_edges(mesh, p, step=step)
    vi = (uv[:, 1] // step).astype(int); ui = (uv[:, 0] // step).astype(int); d = dist[vi, ui]; ok = np.isfinite(d) & (d < 30)
    D, _ = cam.rays(step); D = D[vi[ok], ui[ok]]
    return cam.t + D * d[ok, None]


def score_pts(X, base, DT, kind="mast", trunc=12.0):
    p = mast_cam(base) if kind == "mast" else rs2_cam(base)
    uv, z = camgeo.Cam(p, p["w"], p["h"]).project(X)
    ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < DT.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < DT.shape[0])
    if ok.sum() < 50: return 1e9
    return float(np.minimum(DT[uv[ok, 1].astype(int), uv[ok, 0].astype(int)], trunc).mean() * ok.mean() + trunc * (1 - ok.mean()))


def score(mesh, base, DT, kind="mast"):
    p = mast_cam(base) if kind == "mast" else rs2_cam(base)
    return score_pts(edge_points(mesh, p), base, DT, kind)


def refine(mesh, base, DT, kind="mast", span=(0.25, 0.25, math.radians(8)), n=(11, 11, 17), iters=2):
    """Coarse-to-fine grid over (dx, dy, dyaw) of the base; edge points re-raycast at each iteration's best pose."""
    s0 = score(mesh, base, DT, kind); cur = base
    for it in range(iters):
        p = mast_cam(cur) if kind == "mast" else rs2_cam(cur); X = edge_points(mesh, p)
        bs, bd = score_pts(X, cur, DT, kind), (0.0, 0.0, 0.0)
        for sc in (1.0, 0.3, 0.1):
            c = bd
            for dx in np.linspace(-span[0], span[0], n[0]) * sc + c[0]:
                for dy in np.linspace(-span[1], span[1], n[1]) * sc + c[1]:
                    for dt in np.linspace(-span[2], span[2], n[2]) * sc + c[2]:
                        s = score_pts(X, (cur[0] + dx, cur[1] + dy, cur[2] + dt), DT, kind)
                        if s < bs: bs, bd = s, (dx, dy, dt)
        cur = (cur[0] + bd[0], cur[1] + bd[1], cur[2] + bd[2]); span = tuple(v * 0.3 for v in span)
    return cur, s0, score(mesh, cur, DT, kind), (cur[0] - base[0], cur[1] - base[1], cur[2] - base[2])


def overlay(img, mesh, p, col=(0, 0, 255)):
    uv, _ = model_edges(mesh, p, step=1); o = img.copy()
    ui = np.clip(uv[:, 0].astype(int), 0, o.shape[1] - 1); vi = np.clip(uv[:, 1].astype(int), 0, o.shape[0] - 1); o[vi, ui] = col
    return o
