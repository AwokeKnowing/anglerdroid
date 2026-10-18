"""Render a nav stuck dump (KEVIN_STUCK_DUMP, nav.run._stuck_dump) -> <dump>.png
python emu/tools/stuck_render.py <dump.npz> [...]   (crop: +-R m around the robot, R = STUCK_R env, default 2.5)

Panels: (1) SLAM map memory + current RS1 observation + trail / global / local path / goal / footprint;
(2) planning window as MPPI sees it (map + observation layer): axle clearance (EDT) with the R_PASS / R_ROTATE contours,
cells not observed free for reversing, MPPI proposals (best = thick); (3) last ~15 s of control: planner command,
after governor, issued, governor scale, MPPI min cost."""
import json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import twin_geometry as TG
R = float(os.environ.get("STUCK_R", "2.5"))

def fp(pose):
    x, y, th = pose; c, s = math.cos(th), math.sin(th); F = np.asarray(TG.FOOTPRINT_SAFE, float)
    P = np.c_[x + c * F[:, 0] - s * F[:, 1], y + s * F[:, 0] + c * F[:, 1]]; return np.r_[P, P[:1]]

def render(fn):
    d = np.load(fn, allow_pickle=True); M = json.loads(str(d["meta"])); x, y, th = M["pose"]; org = np.asarray(M["org"])
    fig, ax = plt.subplots(1, 3, figsize=(21, 7)); ext = [x - R, x + R, y - R, y + R]
    # (1) map memory + observation
    L = d["mapL"]; mo = np.asarray(M["map_origin"]); res = M["map_res"]
    a0, b0 = int((x - R - mo[0]) / res), int((y - R - mo[1]) / res); n = int(2 * R / res)
    sub = L[max(b0, 0):b0 + n, max(a0, 0):a0 + n]
    img = np.full(sub.shape + (3,), 0.6); img[sub < -0.5] = 1.0; img[sub > 1.5] = 0.0
    ax[0].imshow(img, origin="lower", extent=[mo[0] + max(a0, 0) * res, mo[0] + (max(a0, 0) + sub.shape[1]) * res, mo[1] + max(b0, 0) * res, mo[1] + (max(b0, 0) + sub.shape[0]) * res])
    ax[0].set_title("map memory (black occ>=2 hits, white free, grey unknown)\n+ trail (blue), global path (green), local path (magenta), goal")
    # (2) planning window
    occ = d["occ"]; edt = d["edt"].astype(float); rev = d["rev"]
    i0, j0 = int((x - R - org[0]) * 100), int((y - R - org[1]) * 100); n = int(2 * R * 100)
    sl = (slice(max(j0, 0), j0 + n), slice(max(i0, 0), i0 + n)); e2 = [org[0] + max(i0, 0) / 100, org[0] + (max(i0, 0) + edt[sl].shape[1]) / 100, org[1] + max(j0, 0) / 100, org[1] + (max(j0, 0) + edt[sl].shape[0]) / 100]
    im = ax[1].imshow(np.minimum(edt[sl], 0.6), origin="lower", extent=e2, cmap="viridis", vmin=0, vmax=0.6)
    ov = np.zeros(occ[sl].shape + (4,)); ov[occ[sl]] = (0, 0, 0, 1); ov[rev[sl] & ~occ[sl]] = (1, 0.5, 0, 0.18)
    ax[1].imshow(ov, origin="lower", extent=e2)
    X = np.linspace(e2[0], e2[1], edt[sl].shape[1]); Y = np.linspace(e2[2], e2[3], edt[sl].shape[0])
    cs = ax[1].contour(X, Y, edt[sl], levels=[TG.R_PASS, TG.R_ROTATE], colors=["red", "white"], linewidths=1)
    plt.colorbar(im, ax=ax[1], fraction=0.04, label="axle clearance m")
    pr = d["props"]
    if pr.ndim >= 3 and pr.size:
        P = pr.reshape(-1, *pr.shape[-2:]) if pr.ndim > 3 else pr; pc = d["pcost"].reshape(-1) if d["pcost"].size else None
        for k in range(len(P)): ax[1].plot(P[k, :, 0] + org[0], P[k, :, 1] + org[1], lw=0.6, color="w", alpha=0.5)
        if pc is not None and len(pc) == len(P): b = int(np.argmin(pc)); ax[1].plot(P[b, :, 0] + org[0], P[b, :, 1] + org[1], lw=2.5, color="cyan")
    ax[1].set_title("planning window (map + current observation): axle clearance\nred R_PASS %.3f, white R_ROTATE %.3f; orange tint = not observed free (no reversing); MPPI proposals" % (TG.R_PASS, TG.R_ROTATE))
    for a in ax[:2]:
        for k, c, lw in (("trail", "b", 1.5), ("gpath", "g", 1.5), ("lpath", "m", 1.2)):
            P = d[k]
            if P.size: a.plot(P[:, 0], P[:, 1], c, lw=lw)
        F = fp((x, y, th)); a.plot(F[:, 0], F[:, 1], "r", lw=1.5); a.arrow(x, y, 0.25 * math.cos(th), 0.25 * math.sin(th), color="r", width=0.01)
        for k, mk in (("goal", "g*"), ("goal_req", "y*"), ("planner_goal", "m*"), ("back_tgt", "bs")):
            if M.get(k): a.plot(*M[k][:2], mk, ms=14)
        a.set_xlim(x - R, x + R); a.set_ylim(y - R, y + R); a.set_aspect("equal")
    # (3) control history
    rg = M.get("ring") or []
    if rg:
        t = np.array([r[0] for r in rg]); t = t - t[-1]
        col = lambda i: np.array([np.nan if r[i] is None else r[i] for r in rg], float)
        ax[2].plot(t, col(5), label="v planner"); ax[2].plot(t, col(7), label="v after governor"); ax[2].plot(t, col(9), "k", label="v issued")
        ax[2].plot(t, col(15), ":", label="v odom"); ax[2].plot(t, col(6), label="w planner", alpha=0.6); ax[2].plot(t, col(10), "--", label="w issued")
        ax[2].plot(t, col(12), label="governor scale")
        if len(rg[0]) > 17: ax[2].plot(t, col(17), "y", lw=2, alpha=0.6, label="v safety core out"); ax[2].plot(t, col(18), "c", lw=2, alpha=0.4, label="w safety core out")
        ax2 = ax[2].twinx(); ax2.semilogy(t, np.maximum(col(13), 1e-2), "r.", ms=2, label="MPPI min cost"); ax2.set_ylabel("MPPI min cost (>=1e4: all infeasible)")
        modes = [r[1] for r in rg]
        for k in range(1, len(modes)):
            if modes[k] != modes[k - 1]: ax[2].axvline(t[k], color="grey", lw=0.5); ax[2].text(t[k], 0.62, modes[k], fontsize=7, rotation=90)
        ax[2].set_ylim(-0.7, 0.7); ax[2].legend(fontsize=7, loc="lower left"); ax[2].set_xlabel("s before dump")
    g = M.get("gov") or [None] * 9
    ax[2].set_title("%s  mode %s stuck %s best_d %s\naxle clr %.3f (rotate %.3f)  planner %s mincost %s u %s\ngovernor (v,w)->(v,w) %s nearest %s" % (
        M["why"], M["mode"], M["stuck"], None if M["best_d"] is None else round(M["best_d"], 2), M["clearance"], M["R_ROTATE"], M["planner_status"],
        M["timing"].get("mincost"), M["timing"].get("u"), None if g[0] is None else [round(float(v), 2) for v in g[:4]], g[6]), fontsize=8)
    fig.suptitle(os.path.basename(fn) + "  " + " | ".join(e[1] for e in M["events"][-3:])[:220], fontsize=8)
    out = fn.replace(".npz", ".png"); fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig); return out

if __name__ == "__main__":
    for f in sys.argv[1:]: print(render(f))
