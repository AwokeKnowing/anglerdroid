"""Model-free checks from vanishing points (Manhattan directions of the casita walls)."""
from __future__ import annotations
import numpy as np


def _norm_seg(s, lens):
    """segment endpoints -> undistorted pinhole pixel coords (centred on the principal point)."""
    f, cx, cy, k1, k2 = lens
    out = []
    for u, v in ((s[0], s[1]), (s[2], s[3])):
        xd, yd = (u - cx) / f, (v - cy) / f; xn, yn = xd, yd
        for _ in range(30):
            r2 = xn * xn + yn * yn; sc = 1 + k1 * r2 + k2 * r2 * r2; xn, yn = xd / sc, yd / sc
        out.append((xn * f, yn * f))
    return np.array(out)


def vanishing_point(segs, lens, scale=1000.0):
    """Length-weighted algebraic LSQ VP (homogeneous, centred px / scale) + per-segment angular residual (deg)."""
    Ls, P = [], []
    for s in segs:
        p = _norm_seg(s, lens) / scale; P.append(p)
        l = np.cross([p[0, 0], p[0, 1], 1.0], [p[1, 0], p[1, 1], 1.0]); n = np.linalg.norm(l[:2])
        Ls.append(l / n * np.linalg.norm(p[1] - p[0]))
    _, _, Vt = np.linalg.svd(np.array(Ls)); v = Vt[-1]
    res = []
    for p in P:   # angle between the segment and the ray from its midpoint to the VP
        m = p.mean(0); d = p[1] - p[0]
        t = v[:2] - m * v[2]
        res.append(np.degrees(np.arcsin(abs(d[0] * t[1] - d[1] * t[0]) / (np.linalg.norm(d) * np.linalg.norm(t) + 1e-12))))
    return v, np.array(res)


def manhattan_checks(ann, lens, W, H):
    groups = {"vertical": [], "x": [], "y": []}
    for ln in ann["lines"]:
        d = np.abs(np.array(ln["B"]) - np.array(ln["A"]))
        if d[0] < 1e-6 and d[1] < 1e-6: groups["vertical"] += ln["segs"]
        elif d[1] < 1e-6 and d[2] < 1e-6: groups["x"] += ln["segs"]
        elif d[0] < 1e-6 and d[2] < 1e-6: groups["y"] += ln["segs"]
    f = lens[0]; out = {}
    vps = {}
    for k, segs in groups.items():
        if len(segs) < 2: continue
        v, r = vanishing_point(segs, lens); vps[k] = v
        out[k] = dict(n=len(segs), resid_deg_rms=float(np.sqrt(np.mean(r ** 2))))
    if "vertical" in vps:
        v = vps["vertical"]; d = np.array([v[0] * 1000.0, v[1] * 1000.0, v[2] * f]); d /= np.linalg.norm(d)
        if d[1] > 0: d = -d                      # world up = image up (-y)
        # camgeo convention: up . forward = -sin(pitch) (pitch + = nose down); up . right = -sin(roll) cos(pitch)
        pitch = -np.arcsin(np.clip(d[2], -1, 1)); roll = np.arctan2(-d[0], -d[1])
        out["vertical"].update(pitch_deg_nose_down=float(np.degrees(pitch)), roll_deg=float(np.degrees(roll)))
    if "x" in vps and "y" in vps:
        a, b = vps["x"], vps["y"]
        if abs(a[2]) > 1e-9 and abs(b[2]) > 1e-9:
            pa, pb = a[:2] / a[2] * 1000.0, b[:2] / b[2] * 1000.0   # NB ill-conditioned when one VP is near infinity
            f2 = -float(pa @ pb); out["f_from_xy_vps_px"] = float(np.sqrt(f2)) if f2 > 0 else None
            out["vp_x_px"] = (pa + [lens[1], lens[2]]).tolist(); out["vp_y_px"] = (pb + [lens[1], lens[2]]).tolist()
    return out
