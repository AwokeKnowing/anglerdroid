#!/usr/bin/env python3
"""Fit the casita wall model to one mast-RGB frame by line reprojection (robust least squares).

Unknowns: camera pose in the casita frame (x, y, z, yaw, pitch, roll) + lens (f, k1 [, k2, cx, cy]).
Data: hand-matched model 3D lines <-> image line segments (annotation json; LSD segments).
Residual: each detected segment is sampled every ~15 px; a sample q is undistorted with the current lens, its
closest point on the projected (pinhole) model line is found and distorted back; the residual is that 2D pixel
offset.  One robust loss (soft-L1, scale 1.5 px) and nothing else: no per-line weights.
Independent check: the vanishing point of the image verticals (after undistortion) gives pitch/roll without
the model.  Scale: the model scale and camera height are one degree of freedom (similarity), so the fit
reports the camera height in model units; model scale factor = true mast height / fitted height.
"""
from __future__ import annotations
import argparse, json, os, sys, numpy as np, cv2
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camgeo import Cam, load_casita
from vp import manhattan_checks

KEYS = ["x", "y", "z", "yaw", "pitch", "roll", "f", "k1", "k2", "cx", "cy", "sz"]


def lsd_segments(gray, min_len=45):
    L = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0].reshape(-1, 4)
    return L[np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]) > min_len]


def samples(seg, step=15.0):
    a, b = np.array(seg[:2]), np.array(seg[2:])
    n = max(2, int(np.linalg.norm(b - a) / step) + 1)
    return a + np.linspace(0, 1, n)[:, None] * (b - a)


def line_residuals(cam, A, B, Q):
    """2D px residual of image samples Q (n,2) to the projected 3D line AB (lens-aware)."""
    Pc = cam.to_cam(np.stack([A, B]))
    if np.any(Pc[:, 2] <= 0.05):           # line end behind camera: clip to the visible part along the line
        d = Pc[1] - Pc[0]
        if Pc[0, 2] <= 0.05: Pc[0] = Pc[0] + d * ((0.05 - Pc[0, 2]) / d[2])
        else: Pc[1] = Pc[0] + d * ((0.05 - Pc[0, 2]) / d[2])
    a, b = Pc[0, :2] / Pc[0, 2], Pc[1, :2] / Pc[1, 2]
    xn, yn = cam.pix_to_norm(Q); q = np.stack([xn, yn], -1)
    u = (b - a) / np.linalg.norm(b - a); c = a + ((q - a) @ u)[:, None] * u      # closest points (pinhole)
    xd, yd = cam.distort(c[:, 0], c[:, 1])
    pc = np.stack([cam.p["f"] * xd + cam.p["cx"], cam.p["f"] * yd + cam.p["cy"]], -1)
    return (pc - Q)


def build(ann, roles):
    out = []
    for ln in ann["lines"]:
        if ln["role"] not in roles: continue
        Q = np.vstack([samples(s) for s in ln["segs"]])
        out.append((ln["name"], np.array(ln["A"], float), np.array(ln["B"], float), Q))
    return out


def run_fit(data, p0, free, W, H):
    def unpack(v):
        p = dict(p0); p.update({k: v[i] for i, k in enumerate(free)}); return p
    def fun(v):
        p = unpack(v); cam = Cam(p, W, H); S = np.array([1.0, 1.0, p.get("sz", 1.0)])
        return np.concatenate([line_residuals(cam, A * S, B * S, Q).ravel() for _, A, B, Q in data])
    v0 = np.array([p0[k] for k in free], float)
    r = least_squares(fun, v0, loss="soft_l1", f_scale=1.5, x_scale="jac", max_nfev=2000)
    p = unpack(r.x)
    # covariance from the robust-weighted Jacobian at the optimum (Gauss-Newton approximation)
    res = fun(r.x); dof = max(1, len(res) - len(free))
    J = r.jac; s2 = float(np.sum(r.fun ** 2) / dof) if hasattr(r, "fun") else 1.0
    try: cov = np.linalg.inv(J.T @ J) * s2; sd = dict(zip(free, np.sqrt(np.diag(cov))))
    except np.linalg.LinAlgError: sd = {}
    return p, sd, r


def errs(p, data, W, H):
    cam = Cam(p, W, H); S = np.array([1.0, 1.0, p.get("sz", 1.0)])
    return [np.linalg.norm(line_residuals(cam, A * S, B * S, Q), axis=1) for _, A, B, Q in data]


def per_line(p, data, W, H):
    rows = []
    for (name, A, B, Q), e in zip(data, errs(p, data, W, H)):
        rows.append((name, len(Q), float(np.sqrt(np.mean(e ** 2))), float(np.median(e)), float(e.max())))
    return rows


def vertical_vp(cam_lens, segs_list, W, H):
    """Least-squares VP of the vertical segments (homogeneous, undistorted normalised coords) -> world-up in cam."""
    Ls = []
    for s in segs_list:
        a = np.array(cam_lens.pix_to_norm(np.array(s[:2]))); b = np.array(cam_lens.pix_to_norm(np.array(s[2:])))
        l = np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0]); Ls.append(l / np.linalg.norm(l[:2]) * np.linalg.norm(b - a))
    Ls = np.array(Ls); _, _, Vt = np.linalg.svd(Ls); v = Vt[-1]
    v = v / np.linalg.norm(v)
    if v[1] > 0: v = -v                    # world-up points to image -y
    # cam axes x right, y down, z fwd; world up u = v.  pitch (+ nose down) = asin(u_z) ; roll from u_x
    pitch = np.arcsin(np.clip(v[2], -1, 1)); roll = np.arctan2(v[0], -v[1])
    return np.degrees(pitch), np.degrees(roll), v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True); ap.add_argument("--annot", required=True)
    ap.add_argument("--stl", default=None, help="default: the resolved world mesh (src/world_params.py)"); ap.add_argument("--out", required=True)
    ap.add_argument("--init", default='{"x":3.3,"y":1.0,"yaw_deg":85}')
    ap.add_argument("--mast-height", type=float, default=None, help="m; default = resolved mast_rgb_height_m (src/robot_params)")
    ap.add_argument("--f0", type=float, default=None, help="initial focal px at this resolution (default twin truth DFOV 118)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mast_height is None:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
            import robot_params as _RP; a.mast_height = float(_RP.V["mast_rgb_height_m"])
        except Exception as e:
            a.mast_height = 0.99115; print("robot_params unavailable (%r): mast height %.5f m" % (e, a.mast_height))
    img = cv2.imread(a.image); H, W = img.shape[:2]
    ann = json.load(open(a.annot))
    if any("segs" not in ln for ln in ann["lines"]):
        L = lsd_segments(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        for ln in ann["lines"]: ln.setdefault("segs", [[round(float(v), 2) for v in L[i]] for i in ln["lsd"]])
        json.dump(ann, open(a.annot, "w"), indent=1)
    diag = np.hypot(W, H); f_twin = (diag / 2) / np.tan(np.radians(118) / 2)
    init = json.loads(a.init)
    p0 = dict(x=3.3, y=1.0, z=a.mast_height, yaw=np.radians(85), pitch=0.0, roll=0.0, f=a.f0 or f_twin,
              cx=(W - 1) / 2, cy=(H - 1) / 2, k1=-0.05, k2=0.008, sz=1.0)
    for k, v in init.items():
        if k.endswith("_deg"): p0[k[:-4]] = np.radians(v)
        else: p0[k] = v
    fitd = build(ann, {"fit"}); chk = build(ann, {"check"})
    stages = [("A pose6+f+k1 (k2=0, pp centre)", ["x", "y", "z", "yaw", "pitch", "roll", "f", "k1"], dict(k2=0.0)),
              ("B pose6+f+k1+k2", ["x", "y", "z", "yaw", "pitch", "roll", "f", "k1", "k2"], {}),
              ("C pose6+f+k1+k2+pp", ["x", "y", "z", "yaw", "pitch", "roll", "f", "k1", "k2", "cx", "cy"], {}),
              ("D pose6 with twin-truth lens fixed", ["x", "y", "z", "yaw", "pitch", "roll"], dict(f=f_twin, k1=-0.05, k2=0.008, cx=(W - 1) / 2, cy=(H - 1) / 2)),
              ("E height fixed at mast height: pose5+f+k1", ["x", "y", "yaw", "pitch", "roll", "f", "k1"], dict(z=a.mast_height, k2=0.0)),
              ("F pose6+f+k1 + model vertical scale sz", ["x", "y", "z", "yaw", "pitch", "roll", "f", "k1", "sz"], dict(k2=0.0)),
              ("G twin lens fixed + sz", ["x", "y", "z", "yaw", "pitch", "roll", "sz"], dict(f=f_twin, k1=-0.05, k2=0.008, cx=(W - 1) / 2, cy=(H - 1) / 2))]
    report = dict(image=a.image, W=W, H=H, f_twin_px=f_twin, mast_height_m=a.mast_height, stages={})
    p = dict(p0)
    for name, free, fixed in stages:
        q = dict(p); q.update(fixed)
        if name.startswith("A"): q = dict(p0); q.update(fixed)
        pf, sd, r = run_fit(fitd, q, free, W, H)
        rows = per_line(pf, fitd, W, H); rc = per_line(pf, chk, W, H)
        allf = np.concatenate(errs(pf, fitd, W, H)); allc = np.concatenate(errs(pf, chk, W, H)) if chk else np.zeros(1)
        st = dict(free=free, params={k: float(pf[k]) for k in KEYS}, sd={k: float(v) for k, v in sd.items()},
                  fit_rms_px=float(np.sqrt(np.mean(allf ** 2))), fit_median_px=float(np.median(allf)), n_fit_samples=int(len(allf)),
                  check_rms_px=float(np.sqrt(np.mean(allc ** 2))), check_median_px=float(np.median(allc)),
                  per_line_fit=rows, per_line_check=rc, cost=float(r.cost))
        report["stages"][name] = st
        if name.startswith("A") or name.startswith("B"): p = pf
        print("\n== %s: fit rms %.2f px (med %.2f, n=%d) | held-out rms %.2f px (med %.2f)" % (name, st["fit_rms_px"], st["fit_median_px"], len(allf), st["check_rms_px"], st["check_median_px"]))
        print("   sz %.4f | x %.3f y %.3f z %.3f m | yaw %.2f pitch %.2f roll %.2f deg | f %.1f px k1 %.4f k2 %.4f cx %.1f cy %.1f" % (
            pf.get("sz", 1.0), pf["x"], pf["y"], pf["z"], np.degrees(pf["yaw"]), np.degrees(pf["pitch"]), np.degrees(pf["roll"]), pf["f"], pf["k1"], pf["k2"], pf["cx"], pf["cy"]))
        print("   1-sigma: " + " ".join("%s %.3g" % (k, np.degrees(v) if k in ("yaw", "pitch", "roll") else v) for k, v in sd.items()))
        for nm, n, rms, med, mx in rows + [("[check] " + x[0],) + tuple(x[1:]) for x in rc]:
            print("   %-62s n=%3d rms %6.2f med %6.2f max %6.2f" % (nm, n, rms, med, mx))
    # model-free check: vanishing point of the image verticals -> pitch / roll (lens from stage A)
    pa = next(v for k, v in report["stages"].items() if k.startswith("A"))["params"]
    report["vp_check"] = manhattan_checks(ann, (pa["f"], pa["cx"], pa["cy"], pa["k1"], pa["k2"]), W, H)
    report["vp_check_twin_lens"] = manhattan_checks(ann, (f_twin, (W - 1) / 2, (H - 1) / 2, -0.05, 0.008), W, H)
    vc = report["vp_check"]["vertical"]
    print("\n== vertical vanishing point (no model): pitch %.2f deg nose-down, roll %.2f deg, n=%d segs, rms %.2f deg" % (
        vc["pitch_deg_nose_down"], vc["roll_deg"], vc["n"], vc["resid_deg_rms"]))
    json.dump(report, open(os.path.join(a.out, "fit_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
