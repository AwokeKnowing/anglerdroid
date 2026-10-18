"""2-D safety frame render (RS1 top-down + RS2 forward cone, as the safety core sees it NOW, robot frame) with the
hard bands, the core's stopping-envelope prediction and the planner's 2x threshold.  Input = what the core publishes
on nav.bus (sc_viz + sc_state) and the brain's governor tuple, so it works on the real robot's frame stream as well
as in sim (no ground truth used).  GET /api/safety.png on the nav server."""
from __future__ import annotations
import math
import numpy as np
import cv2
from nav.safety_core import FOOTPRINT_SAFE, MARGIN, need_dist, need_angle, pose_on_arc

X0, X1, Y0, Y1, PX = -1.0, 2.2, -1.3, 1.3, 0.008          # metres, m/px; image up = +x (forward), left = +y


def _px(P):
    P = np.asarray(P, float).reshape(-1, 2)
    return np.stack([(Y1 - P[:, 1]) / PX, (X1 - P[:, 0]) / PX], 1).astype(np.int32)


def _swept(v, w, dist_or_ang, rotate=False, n=8):
    polys = []
    for k in range(n + 1):
        s = dist_or_ang * k / n
        x, y, th = (0.0, 0.0, math.copysign(s, w)) if rotate else pose_on_arc(s, 1 if v >= 0 else -1, w / v if abs(v) > 1e-3 else 0.0)
        c, si = math.cos(th), math.sin(th); F = FOOTPRINT_SAFE
        polys.append(np.stack([x + c * F[:, 0] - si * F[:, 1], y + si * F[:, 0] + c * F[:, 1]], 1))
    return polys


def render(pts, bands, core, gov):
    W, H = int((Y1 - Y0) / PX), int((X1 - X0) / PX)
    img = np.full((H, W, 3), 24, np.uint8)
    for r in (0.5, 1.0, 1.5, 2.0):                                   # range rings
        cv2.circle(img, tuple(_px([[0, 0]])[0]), int(r / PX), (60, 60, 60), 1, cv2.LINE_AA)
    if pts is not None and len(pts):
        q = _px(pts); ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H); img[q[ok, 1], q[ok, 0]] = (60, 60, 255)
    c = core or {}
    v_c, w_c = c.get("v_cmd") or 0.0, c.get("w_cmd") or 0.0
    # planner 2x threshold (governor: what the brain asked before governing -> need2), cyan
    if gov:
        vp, wp, vg, wg, need2 = gov[:5]
        if need2 and abs(vg) >= 1e-3:
            for P in _swept(vg, wg, need2): cv2.polylines(img, [_px(P)], True, (200, 200, 0), 1, cv2.LINE_AA)
        elif need2 and abs(wg) >= 1e-3:
            for P in _swept(0.0, wg, need2, rotate=True): cv2.polylines(img, [_px(P)], True, (200, 200, 0), 1, cv2.LINE_AA)
    # core stopping-envelope prediction for the current command: one tick + latency + measured max-brake distance, orange
    vm = c.get("v_m") or 0.0
    if abs(v_c) >= 1e-3:
        ve = max(abs(v_c), abs(vm) if vm * v_c > 0 else 0.0)
        for P in _swept(v_c, w_c, need_dist(ve)): cv2.polylines(img, [_px(P)], True, (0, 140, 255), 1, cv2.LINE_AA)
    elif abs(w_c) >= 1e-3:
        for P in _swept(0.0, w_c, need_angle(w_c), rotate=True): cv2.polylines(img, [_px(P)], True, (0, 140, 255), 1, cv2.LINE_AA)
    cv2.polylines(img, [_px(FOOTPRINT_SAFE)], True, (230, 230, 230), 2, cv2.LINE_AA)
    if bands is not None:
        for i, (xa, ya, xb, yb) in enumerate(bands):
            hit = bool(c.get("band_f" if i == 0 else "band_r"))
            q = _px([[xb, yb], [xa, ya]]); ov = img.copy()
            cv2.rectangle(ov, tuple(q[0]), tuple(q[1]), (0, 0, 255) if hit else (0, 220, 255), -1); img[:] = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
    txt = ["core %s  ramp %.1f" % (c.get("code", "-"), c.get("ramp") or 0),
           "age rs1 %s rs2 %s ms  period %s ms" % tuple(("%.0f" % (1e3 * c[k])) if c.get(k) is not None else "-" for k in ("age1", "age2", "period")),
           "cmd %.2f %.2f  meas %.2f %.2f  out %.2f %.2f" % (v_c, w_c, vm, c.get("w_m") or 0, c.get("v_out") or 0, c.get("w_out") or 0),
           "white footprint  yellow 5cm bands (red=occupied)", "orange core stop envelope  cyan planner 2x"]
    for i, t in enumerate(txt): cv2.putText(img, t, (6, 14 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def png(brain):
    hw = brain.hw; vz = hw.core_viz() if hasattr(hw, "core_viz") else None
    pts, bands = (vz[0], vz[1]) if vz else (None, None)
    img = render(pts, bands, hw.core_state() if hasattr(hw, "core_state") else None, getattr(brain, "_gov", None))
    return cv2.imencode(".png", img)[1].tobytes()
