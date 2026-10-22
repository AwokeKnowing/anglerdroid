#!/usr/bin/env python3
"""Plan view of a casita fit: model walls + key lines (red), image lines back-projected with the fitted pose/lens
(blue), Kevin's footprint, camera, heading, horizontal FOV cone.  Prints per-line plan discrepancies (cm).

Vertical lines: the image segments + camera centre span a plane; its horizontal trace is a ray from the camera,
intersected with the wall the model corner lies on (along-wall offset) and compared with the corner (lateral miss).
Horizontal lines at known model height h: each sample's ray is cut with z = h (x sz) -> plan points, compared with
the model line (signed perpendicular offset).  Read-only, CPU-light (only a few hundred rays)."""
from __future__ import annotations
import argparse, json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camgeo import Cam, load_casita
from fit import samples

# plan wall (2 points) each vertical model corner sits on, for the along-wall offset
WALLS = {
    "diag wall right end": ((3.589, 4.497), (5.320, 2.989)),
    "diag wall left silhouette": ((3.550, 4.531), (3.550, 4.667)),
    "diag wall left: image edge": ((3.589, 4.497), (5.320, 2.989)),
    "partition column SW": ((0.533, 4.426), (2.366, 4.426)),
    "partition column SE": ((0.533, 4.426), (2.366, 4.426)),
    "partition column NE": ((2.366, 4.426), (2.366, 4.948)),
    "door-wall / step": ((0.533, 3.051), (0.533, 4.426)),
    "front door": ((0.533, 3.051), (0.533, 4.426)),
    "door-wall / partition": ((0.533, 3.051), (0.533, 4.426)),
    "pass-through left jamb": ((0.533, 4.426), (2.366, 4.426)),
    "niche right edge": ((0.533, 4.426), (2.366, 4.426)),
}
FOOT = [[-0.27371, -0.01022], [-0.1995, -0.15], [-0.1038, -0.2033], [-0.08009, -0.20998], [0.08319, -0.20998], [0.1038, -0.2033],
        [0.12372, -0.15418], [0.12629, -0.13818], [0.12629, 0.13818], [0.12372, 0.15418], [0.1038, 0.2033], [0.08319, 0.20998],
        [-0.08009, 0.20998], [-0.1038, 0.2033], [-0.1995, 0.15], [-0.27371, 0.01022]]   # resolved footprint_safe (base frame)
MAST_X = 0.0291                                                                         # twin frames.rgb x in base


def world_rays(cam, Q):
    xn, yn = cam.pix_to_norm(Q); d = np.stack([xn, yn, np.ones_like(xn)], -1) @ cam.R
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def wall_for(name):
    for k, w in WALLS.items():
        if name.startswith(k): return np.array(w, float)
    return None


def ray_seg(o, d, a, b):
    """ray o + t d (t>0) x infinite line a-b -> (t, s along a->b in m from a)"""
    u = b - a; M = np.array([d, -u]).T
    t, s = np.linalg.solve(M, a - o); return t, s * np.linalg.norm(u)


def analyse(p, ann, W, H):
    cam = Cam(p, W, H); sz = p.get("sz", 1.0); c = cam.t[:2]; out = []
    for ln in ann["lines"]:
        A, B = np.array(ln["A"]), np.array(ln["B"]); Q = np.vstack([samples(s) for s in ln["segs"]])
        D = world_rays(cam, Q); vert = np.allclose(A[:2], B[:2])
        r = dict(name=ln["name"], role=ln["role"], vertical=bool(vert))
        if vert:
            _, _, Vt = np.linalg.svd(D); n = Vt[-1]                    # plane through camera containing the rays
            d = np.cross(n, [0, 0, 1.0])[:2]; d /= np.linalg.norm(d)
            if d @ D.mean(0)[:2] < 0: d = -d
            P = A[:2]; v = P - c
            r["lateral_cm"] = float(100 * (d[0] * v[1] - d[1] * v[0]))  # + = model corner left of the camera ray
            r["ray_dir"] = d.tolist(); w = wall_for(ln["name"])
            if w is not None:
                t, s = ray_seg(c, d, w[0], w[1]); u = (w[1] - w[0]) / np.linalg.norm(w[1] - w[0])
                s0 = (P - w[0]) @ u
                r["hit"] = (c + t * d).tolist(); r["along_wall_cm"] = float(100 * (s - s0)); r["ray_len_m"] = float(t)
            else:
                r["hit"] = (c + d * np.linalg.norm(v)).tolist()
        else:
            h = A[2] * sz; t = (h - cam.t[2]) / D[:, 2]; ok = t > 0
            P = cam.t[None, :2] + t[ok, None] * D[ok, :2]
            u = (B[:2] - A[:2]) / np.linalg.norm(B[:2] - A[:2]); nrm = np.array([-u[1], u[0]])
            off = (P - A[:2]) @ nrm
            # sign: + = image line lands farther from the camera than the model line
            if (c - A[:2]) @ nrm > 0: off = -off
            r.update(points=P.tolist(), perp_cm_mean=float(100 * off.mean()), perp_cm_max=float(100 * np.abs(off).max()), height_m=float(h))
        out.append(r)
    return cam, out


def draw(ax, m, cam, p, ann, res, title, W):
    # faint plan: wall sections at 0.5 / 1.5 m filled at 20 %
    for z in (0.5, 1.5):
        sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None: continue
        for e in sec.discrete:
            if len(e) > 2 and np.allclose(e[0], e[-1]):
                x, y = e[:, 0], e[:, 1]; area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                if area < 1.5:             # wall pieces; big loops = outer ring / cross-room section artefacts (outlined below)
                        ax.add_patch(Polygon(e[:, :2], closed=True, fc=(1, 0, 0, 0.2 if z == 0.5 else 0.0), ec=(0.8, 0, 0, 0.35), lw=0.6))
            for a0, b0 in zip(e[:-1], e[1:]):   # outlines of everything else, minus long non-axis cross-room segments
                d = b0 - a0; L = np.hypot(d[0], d[1])
                if L > 2.5 and min(abs(d[0]), abs(d[1])) > 0.05 * L: continue
                ax.plot([a0[0], b0[0]], [a0[1], b0[1]], color=(0.8, 0, 0, 0.45), lw=0.7)
    for ln in ann["lines"]:
        A, B = np.array(ln["A"]), np.array(ln["B"]); chk = ln["role"] != "fit"
        if np.allclose(A[:2], B[:2]): ax.plot(A[0], A[1], "o", ms=7, mfc="none" if chk else "red", mec="red", mew=1.5)
        else: ax.plot([A[0], B[0]], [A[1], B[1]], "-" if not chk else "--", color="red", lw=2.5)
    c = cam.t[:2]
    for r in res:
        col = "#1060ff" if r["role"] == "fit" else "#00b8c8"
        if r["vertical"]:
            hx, hy = r["hit"]; ax.plot([c[0], hx], [c[1], hy], color=col, lw=1.1, alpha=0.9); ax.plot(hx, hy, "x", color=col, ms=7, mew=1.8)
        else:
            P = np.array(r["points"])
            if len(P): ax.plot(P[:, 0], P[:, 1], ".", color=col, ms=3.5)
    # FOV cone at the image centre row, heading, camera, footprint
    Q = np.array([[0.0, p["cy"]], [W - 1.0, p["cy"]]]); D = world_rays(cam, Q)[:, :2]; D /= np.linalg.norm(D, axis=1, keepdims=True)
    hf = np.degrees(np.arccos(np.clip(D[0] @ D[1], -1, 1)))
    ax.add_patch(Polygon([c, c + 6 * D[0], c + 6 * D[1]], closed=True, fc=(1, 0.8, 0, 0.12), ec=(0.9, 0.6, 0, 0.8), lw=1, ls=":"))
    yaw = p["yaw"]; R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]); base = c - MAST_X * np.array([np.cos(yaw), np.sin(yaw)])
    ax.add_patch(Polygon(base + np.array(FOOT) @ R.T, closed=True, fc=(0.2, 0.2, 0.2, 0.35), ec="k", lw=1.2))
    ax.annotate("", xy=base + 0.6 * np.array([np.cos(yaw), np.sin(yaw)]), xytext=base, arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
    ax.plot(*c, "*", color="orange", ms=14, mec="k")
    ax.text(base[0] + 0.15, base[1] - 0.35, "Kevin base (%.2f, %.2f)\nheading %.1f deg\ncam z %.2f model-m\nHFOV %.0f deg" % (base[0], base[1], np.degrees(yaw), cam.t[2], hf), fontsize=8)
    # scale bar
    ax.plot([0.3, 1.3], [0.3, 0.3], "k-", lw=4); ax.text(0.8, 0.38, "1 m", ha="center", fontsize=9)
    ax.set_xlim(-0.3, 7.5); ax.set_ylim(0.0, 6.2); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    ax.set_xlabel("casita x (m, east)"); ax.set_ylabel("casita y (m, north)"); ax.set_title(title, fontsize=10)
    return hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True); ap.add_argument("--stl", required=True); ap.add_argument("--report", required=True)
    ap.add_argument("--stages", default="A,E"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ann = json.load(open(a.annot)); rep = json.load(open(a.report)); W, H = rep["W"], rep["H"]; m = load_casita(a.stl)
    st = a.stages.split(","); fig, axs = plt.subplots(1, len(st), figsize=(9 * len(st), 8.2)); axs = np.atleast_1d(axs)
    allres = {}
    for ax, s in zip(axs, st):
        k, v = next((k, v) for k, v in rep["stages"].items() if k.startswith(s)); p = v["params"]
        cam, res = analyse(p, ann, W, H); allres[k] = res
        draw(ax, m, cam, p, ann, res, "fit %s (image rms %.1f px)" % (k, v["fit_rms_px"]), W)
        print("\n== %s  cam (%.3f, %.3f, z %.3f) yaw %.2f" % (k, p["x"], p["y"], p["z"], np.degrees(p["yaw"])))
        for r in res:
            tag = "" if r["role"] == "fit" else "[check] "
            if r["vertical"]:
                print("  %-70s lateral %+7.1f cm  along-wall %s  (ray %.2f m)" % (tag + r["name"][:62], r["lateral_cm"],
                      "%+7.1f cm" % r["along_wall_cm"] if "along_wall_cm" in r else "   n/a", r.get("ray_len_m", 0)))
            else:
                print("  %-70s perp mean %+7.1f cm  max %6.1f cm  (z %.2f)" % (tag + r["name"][:62], r["perp_cm_mean"], r["perp_cm_max"], r["height_m"]))
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="red", lw=2.5, label="model key line (dashed = held-out)"),
         Line2D([], [], color="red", marker="o", ls="", label="model vertical corner (hollow = held-out)"),
         Line2D([], [], color="#1060ff", lw=1.2, marker="x", label="image vertical -> camera ray, x = hit on matched wall"),
         Line2D([], [], color="#1060ff", marker=".", ls="", label="image horizontal line cut at its model height"),
         Line2D([], [], color="#00b8c8", lw=1.2, marker="x", label="same, held-out lines"),
         Line2D([], [], color="orange", marker="*", ls="", ms=12, label="mast camera"),
         Polygon([[0, 0]], fc=(0.2, 0.2, 0.2, 0.35), ec="k", label="Kevin footprint_safe + heading"),
         Polygon([[0, 0]], fc=(1, 0.8, 0, 0.12), ec=(0.9, 0.6, 0, 0.8), ls=":", label="horizontal FOV (centre row)"),
         Polygon([[0, 0]], fc=(1, 0, 0, 0.2), ec=(0.8, 0, 0, 0.35), label="casita walls (section 0.5 m, 20 %)")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=9, frameon=True)
    fig.suptitle("Casita plan view: mast RGB 2026-10-21 12:13 PDT back-projected through the fitted pose + lens", fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 0.96)); fig.savefig(a.out, dpi=130)
    json.dump(allres, open(os.path.splitext(a.out)[0] + "_discrepancy.json", "w"), indent=1)


if __name__ == "__main__":
    main()
