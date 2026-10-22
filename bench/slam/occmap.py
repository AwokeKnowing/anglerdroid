"""One 2D occupancy mapper for every method (bench/slam), so map quality differs only by the trajectory.
Forward RS2 depth (every 3rd frame = 5 Hz) -> points in base (twin extrinsic) -> obstacle band 0.05..1.2 m above the
floor / floor points -> 2D log-odds ray casting (free along the ray, hit at the end, max range 4 m) on a 4 cm grid (2x the 2 cm twin GT raster),
posed by the method's trajectory anchored at the GT start pose (the robot knows where it starts; no map, no fit).
Score vs the twin GT occupancy (casita walls STL + counters + furniture, emu.envs, max-pooled 2 cm -> 4 cm), on the
cells the GT-pose map observed:
  occ_iou   IoU(est occupied, GT occupied)          free_iou  IoU(est free, GT free)
  occ_iou_t IoU with 1-cell (4 cm) tolerance: est hit counts if a GT wall is within 1 cell, and vice versa
python bench/slam/occmap.py DATA/<seq> RESULTS/<seq>  [--gt DATA/<seq>/gt_occ.npz]  -> RESULTS/<seq>/maps.json + map PNGs"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
import numpy as np, cv2
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from eval import load_tum, load_gt, interp_gt, T_from  # noqa: E402
RES = 0.04; L_HIT, L_FREE, L_MAX = 0.85, -0.4, 4.0; RMAX = 4.0


class Grid:
    def __init__(self, org, shape): self.org = np.asarray(org, float); self.L = np.zeros(shape, np.float32)
    def rc(self, xy): return np.floor((xy[:, ::-1] - self.org[::-1]) / RES).astype(int)
    def add(self, o, hits, frees):
        """o: (2,) sensor xy; hits/frees: (N,2) endpoints (world xy).  Endpoints are reduced to unique cells first, then
        each cell is updated at most once per scan (free along the rays, hit at the hit cells)."""
        L = self.L; H, W = L.shape; Lf = L.reshape(-1)
        def cells(P):
            rc = self.rc(P); ok = (rc[:, 0] >= 0) & (rc[:, 1] >= 0) & (rc[:, 0] < H) & (rc[:, 1] < W); return np.unique(rc[ok, 0] * W + rc[ok, 1])
        hit = cells(hits) if len(hits) else np.zeros(0, int); fr = cells(frees) if len(frees) else np.zeros(0, int)
        ends = np.r_[hit, fr]
        if not len(ends): return
        E = np.stack([(ends % W + 0.5) * RES + self.org[0], (ends // W + 0.5) * RES + self.org[1]], 1)
        d = E - o; n = np.hypot(*d.T); k = np.maximum(1, np.ceil(n / (RES * 0.5)).astype(int)); K = int(k.max())
        s = (np.arange(K)[None, :] + 0.5) / k[:, None]; pts = o + d[:, None, :] * np.minimum(s, 1.0)[..., None]
        m = s < 1.0; m[:len(hit)] &= s[:len(hit)] < (1 - 0.75 / k[:len(hit), None])      # stop short of hit cells
        free = np.zeros(H * W, bool); fc = cells(pts[m]); free[fc] = True; free[hit] = False
        Lf[free] += L_FREE; Lf[hit] += L_HIT
        np.clip(L, -L_MAX, L_MAX, out=L)
    def occ(self): return self.L > 0.5
    def free(self): return self.L < -0.5
    def known(self): return np.abs(self.L) > 0.5


class DepthScan:
    def __init__(self, meta, sub=6):
        I = meta["intr"]["rs2_depth"]; e = meta["extr_twin_base_from_optical"]["rs2_depth"]
        self.R, self.t = np.array(e["R"]), np.array(e["t"]); self.sub = sub
        v, u = np.mgrid[0:I["h"]:sub, 0:I["w"]:sub]; self.u, self.v = u.ravel(), v.ravel()
        self.rx, self.ry = (self.u - I["ppx"]) / I["fx"], (self.v - I["ppy"]) / I["fy"]
        self.zb0 = None
    def base_points(self, d):
        z = d[self.v, self.u].astype(np.float32) / 1000.0; ok = (z > 0.2) & (z < RMAX)
        P = np.stack([self.rx[ok] * z[ok], self.ry[ok] * z[ok], z[ok]], 1) @ self.R.T + self.t
        return P
    def split(self, P, floor_z):
        h = P[:, 2] - floor_z; return P[(h > 0.05) & (h < 1.2)], P[np.abs(h) < 0.03]


def build(seq, T_base, times, every=3, gt_shape=None, gt_org=None, cb=None):
    meta = json.load(open(os.path.join(seq, "meta.json"))); F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True)
    ds = DepthScan(meta); g = Grid(gt_org, gt_shape); floor_z = -float(meta["twin"]["axle_h_m"])   # base_link sits at the axle
    per_frame = len(times) > 0.6 * len(F["t"])        # keyframe-rate estimates (viw_fg: every 3rd frame, any phase) use all their poses
    for k, (fi, tf) in enumerate(zip(F["i"].astype(int), F["t"])):
        if per_frame and k % every: continue
        j = np.searchsorted(times, tf - 1e-4)
        if j >= len(times) or abs(times[j] - tf) > (0.04 if per_frame else 0.02): continue
        d = cv2.imread(os.path.join(seq, "rs2_depth", "%06d.png" % fi), cv2.IMREAD_UNCHANGED)
        if d is None: continue
        P = ds.base_points(d); ob, fl = ds.split(P, floor_z); T = T_base[j]
        o = (T @ np.r_[ds.t, 1])[:2]
        w = lambda Q: (Q[:, :2] @ T[:2, :2].T) + T[:2, 3]
        g.add(o, w(ob), w(fl[::2]))
        if cb: cb(k, fi, tf, g, T)
    return g


def gt_grid(path):
    z = np.load(path); occ, org, r = z["occ"], z["origin"], float(z["res"]); f = int(round(RES / r))
    assert abs(f * r - RES) < 1e-9, "map RES must be an integer multiple of the GT raster"
    H, W = occ.shape[0] // f, occ.shape[1] // f
    return occ[:H * f, :W * f].reshape(H, f, W, f).any(axis=(1, 3)), org


def score(est: Grid, ref_known, gt_occ):
    m = ref_known; eo, ef = est.occ() & m, est.free() & m; go = gt_occ & m; gf = (~gt_occ) & m
    iou = lambda a, b: float((a & b).sum() / max(1, (a | b).sum()))
    gd, ed = ndimage.binary_dilation(go), ndimage.binary_dilation(eo)
    tp_e = (eo & gd).sum(); tp_g = (go & ed).sum()
    occ_t = float((tp_e + tp_g) / max(1, eo.sum() + go.sum() - 0.5 * (tp_e + tp_g)) / 2 * 2) if (eo.sum() + go.sum()) else 0.0
    prec, rec = tp_e / max(1, eo.sum()), tp_g / max(1, go.sum())
    return dict(occ_iou=round(iou(eo, go), 3), free_iou=round(iou(ef, gf), 3), occ_prec_t=round(float(prec), 3), occ_rec_t=round(float(rec), 3),
                occ_f1_t=round(float(2 * prec * rec / max(1e-9, prec + rec)), 3), known_frac=round(float(est.known()[m].mean()), 3))


def anchored(seq, tum):
    """method trajectory (base frame) anchored at the GT pose of its first frame; returns times, T (N,4,4)."""
    tg, Tg = load_gt(seq); t, Te = load_tum(tum)
    if len(t) < 2: return t, Te
    T0, _ = interp_gt(tg, Tg, t[:1]); A = T0[0] @ np.linalg.inv(Te[0]); return t, np.einsum("ij,njk->nik", A, Te)


def png(g: Grid, gt_occ, path, traj=None, gtraj=None):
    img = np.full(g.L.shape + (3,), 128, np.uint8); img[g.free()] = (235, 235, 235); img[g.occ()] = (30, 30, 30)
    gt_e = gt_occ & ~ndimage.binary_erosion(gt_occ); img[gt_e & ~g.occ()] = (60, 60, 230)
    img = cv2.resize(img[::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    def draw(T, col):
        if T is None or not len(T): return
        rc = g.rc(T[:, :2, 3]); p = np.stack([rc[:, 1] * 2, (g.L.shape[0] - 1 - rc[:, 0]) * 2], 1).astype(np.int32)
        cv2.polylines(img, [p], False, col, 1, cv2.LINE_AA)
    draw(gtraj, (0, 160, 0)); draw(traj, (0, 0, 255))
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("res"); ap.add_argument("--gt"); ap.add_argument("--only")
    a = ap.parse_args(); seq = os.path.abspath(a.seq); gtp = a.gt or os.path.join(seq, "gt_occ.npz")
    gt_occ, org = gt_grid(gtp); tg, Tg = load_gt(seq)
    F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True)
    Tf, _ = interp_gt(tg, Tg, F["t"]); ref = build(seq, Tf, F["t"], gt_shape=gt_occ.shape, gt_org=org)
    out = {"_gt_pose_map": score(ref, ref.known(), gt_occ)}; png(ref, gt_occ, os.path.join(a.res, "map__gt_pose.png"), gtraj=Tf)
    print(json.dumps(dict(method="_gt_pose_map", **out["_gt_pose_map"])), flush=True)
    for p in sorted(glob.glob(os.path.join(a.res, "*.tum"))):
        name = os.path.basename(p)[:-4]
        if a.only and name not in a.only.split(","): continue
        mj = p[:-4] + ".json"; mm = json.load(open(mj)) if os.path.exists(mj) else {}
        if mm.get("frame", "base") != "base" and not os.path.exists(p[:-4] + ".base.txt"): print("skip (not base frame)", name); continue
        t, T = anchored(seq, p[:-4] + ".base.txt" if mm.get("frame", "base") != "base" else p)
        if len(t) < 10: continue
        g = build(seq, T, t, gt_shape=gt_occ.shape, gt_org=org); s = score(g, ref.known(), gt_occ); out[name] = s
        png(g, gt_occ, os.path.join(a.res, "map_%s.png" % name), traj=T, gtraj=Tf); print(json.dumps(dict(method=name, **s)), flush=True)
    old = json.load(open(os.path.join(a.res, "maps.json"))) if os.path.exists(os.path.join(a.res, "maps.json")) else {}
    old.update(out); json.dump(old, open(os.path.join(a.res, "maps.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
