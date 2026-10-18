"""Score SLAM/odometry trajectories against the sim ground truth (bench/slam).

Each method writes  RESULTS/<seq>/<method>.tum  (t x y z qx qy qz qw) and optional <method>.json with
{"frame": "base" | "cam:<twin frame>", "scale_free": bool, "runtime": {...}, ...}.
  base frame = Kevin base_link (axle centre, x fwd, z up) in the method's own world frame
  cam:<f>    = optical frame <f> (rs2_color, rs2_ir1, mast ...) as camera-to-world.  Converted to base with the twin
               extrinsic.  scale_free -> Sim(3) Umeyama fit of camera positions to GT (the scale comes from GT: flagged).

Metrics (xy = floor plane):
  anchored   the estimate is put at the GT pose of its first frame (what a robot experiences: start pose known):
             ATE xy RMSE, final xy error, max |yaw error|, final yaw error, RPE per 1 m of travel (xy, median & p95),
             heading drift per 360 deg turned (yaw error growth / total |rotation|)
  aligned    best-fit SE(3) (or Sim(3) when scale_free) alignment: ATE RMSE  (shape quality, map consistency)
  coverage   fraction of GT frames with an estimate (lost tracking)
python bench/slam/eval.py DATA/<seq> RESULTS/<seq> [--md out.md]"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

wrap = lambda a: (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def T_from(t, q):
    T = np.eye(4); T[:3, :3] = Rot.from_quat(q).as_matrix(); T[:3, 3] = t; return T


def load_gt(seq):
    G = np.genfromtxt(os.path.join(seq, "gt.csv"), delimiter=",", names=True)
    t = G["t"]; T = np.zeros((len(t), 4, 4))
    for i in range(len(t)):
        T[i] = T_from([G["x"][i], G["y"][i], G["z"][i]], Rot.from_euler("xyz", [G["roll"][i], G["pitch"][i], G["yaw"][i]]).as_quat())
    return t, T


def load_tum(p):
    D = np.loadtxt(p, ndmin=2)
    if len(D) == 0: return np.zeros(0), np.zeros((0, 4, 4))
    return D[:, 0], np.stack([T_from(r[1:4], r[4:8]) for r in D])


def interp_gt(tg, Tg, t):
    """GT pose at times t (nearest within 6 ms; GT is 100 Hz)."""
    i = np.clip(np.searchsorted(tg, t), 1, len(tg) - 1); i = np.where(np.abs(tg[i - 1] - t) < np.abs(tg[i] - t), i - 1, i)
    return Tg[i], np.abs(tg[i] - t) < 0.006


def umeyama(X, Y, scale):
    """min || s R X + t - Y ||  (X, Y: N x 3) -> s, R, t."""
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: D[2, 2] = -1
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / Xc.var(0).sum() if scale else 1.0
    return s, R, my - s * R @ mx


def yaw_of(T): return np.arctan2(T[..., 1, 0], T[..., 0, 0])


def score(seq, tum, meta_m, meta_seq):
    tg, Tg = load_gt(seq)
    t, Te = load_tum(tum)
    if len(t) < 5: return dict(ok=False, n=len(t))
    Tgt, ok = interp_gt(tg, Tg, t); t, Te, Tgt = t[ok], Te[ok], Tgt[ok]
    frame = meta_m.get("frame", "base"); sf = bool(meta_m.get("scale_free", False)); s = 1.0
    if frame.startswith("cam:"):
        e = meta_seq["extr_twin_base_from_optical"][frame[4:]]; Tbc = np.eye(4); Tbc[:3, :3] = e["R"]; Tbc[:3, 3] = e["t"]; Tcb = np.linalg.inv(Tbc)
        Tgc = Tgt @ Tbc                                                       # GT camera poses
        if sf:                                                                # bring the estimate to metric GT world
            s, R, tt = umeyama(Te[:, :3, 3], Tgc[:, :3, 3], True)
            A = np.eye(4); A[:3, :3] = R; A[:3, 3] = tt
            Tm = Te.copy(); Tm[:, :3, 3] *= s; Te = np.einsum("ij,njk->nik", A, Tm)
        Te = Te @ Tcb                                                          # camera -> base
    # anchored at the first estimate's GT pose
    Ta = np.einsum("ij,njk->nik", Tgt[0] @ np.linalg.inv(Te[0]), Te)
    exy = np.hypot(*(Ta[:, :2, 3] - Tgt[:, :2, 3]).T); eyaw = np.degrees(np.abs(wrap(yaw_of(Ta) - yaw_of(Tgt))))
    # RPE per 1 m of GT travel
    path = np.r_[0, np.cumsum(np.hypot(*np.diff(Tgt[:, :2, 3], axis=0).T))]
    rpe = []
    for i in range(len(t)):
        j = np.searchsorted(path, path[i] + 1.0)
        if j >= len(t): break
        dG = np.linalg.inv(Tgt[i]) @ Tgt[j]; dE = np.linalg.inv(Ta[i]) @ Ta[j]; rpe.append(np.hypot(*(dE[:2, 3] - dG[:2, 3])))
    rot = float(np.abs(np.diff(np.unwrap(yaw_of(Tgt)))).sum())
    # aligned SE(3)
    _, R, tt = umeyama(Te[:, :3, 3], Tgt[:, :3, 3], False); Al = (R @ Te[:, :3, 3].T).T + tt
    ate_al = float(np.sqrt((np.hypot(*(Al[:, :2] - Tgt[:, :2, 3]).T) ** 2).mean()))
    nfr = sum(1 for _ in open(os.path.join(seq, "frames.csv"))) - 1
    return dict(ok=True, n=len(t), coverage=round(min(1.0, len(t) / max(nfr, 1)), 3) if meta_m.get("per_frame", True) else None,
                path_m=round(float(path[-1]), 2), turned_deg=round(math.degrees(rot), 0),
                ate_xy=round(float(np.sqrt((exy ** 2).mean())), 3), final_xy=round(float(exy[-1]), 3), max_xy=round(float(exy.max()), 3),
                yaw_max=round(float(eyaw.max()), 2), yaw_final=round(float(eyaw[-1]), 2),
                yaw_per_360=round(float(eyaw.max()) / max(math.degrees(rot) / 360, 1e-6), 2),
                rpe1m_med=round(float(np.median(rpe)), 3) if rpe else None, rpe1m_p95=round(float(np.percentile(rpe, 95)), 3) if rpe else None,
                ate_aligned=round(ate_al, 3), scale=round(float(s), 4) if sf else None, frame=frame, scale_free=sf)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("res"); ap.add_argument("--md")
    a = ap.parse_args(); meta = json.load(open(os.path.join(a.seq, "meta.json"))); rows = []
    for p in sorted(glob.glob(os.path.join(a.res, "*.tum"))):
        name = os.path.basename(p)[:-4]; mj = p[:-4] + ".json"; mm = json.load(open(mj)) if os.path.exists(mj) else {}
        try: r = score(a.seq, p, mm, meta)
        except Exception as e: r = dict(ok=False, err=repr(e))
        r.update(method=name, runtime=mm.get("runtime"), note=mm.get("note")); rows.append(r); print(json.dumps(r))
    json.dump(rows, open(os.path.join(a.res, "scores.json"), "w"), indent=1)
    if a.md:
        cols = ["method", "coverage", "ate_xy", "final_xy", "yaw_max", "yaw_final", "yaw_per_360", "rpe1m_med", "ate_aligned", "scale"]
        L = ["| " + " | ".join(cols + ["runtime"]) + " |", "|" + "---|" * (len(cols) + 1)]
        for r in rows: L.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " | %s |" % (r.get("runtime") or ""))
        open(a.md, "w").write("seq %s  path %s m, turned %s deg\n\n" % (os.path.basename(a.seq.rstrip("/")), rows[0].get("path_m") if rows else "?", rows[0].get("turned_deg") if rows else "?") + "\n".join(L) + "\n")


if __name__ == "__main__":
    main()
