"""Mast-camera feature-density probe (offline, in-process twin; no shm/sockets, never touches the live emulator).

Renders the mast RGB through the full sensor pipeline (truth lens, AE settled, noise, vignetting) from a fixed set of
poses (default: 16 evenly spaced ground-truth poses from each of the recorded drives L12/L13/L14 = realistic
viewpoints along real routes; --places: every casita place x 4 headings) and reports trackable-point density with the same metrics as the reference
measurement of real-home frames (TUM RGB-D fr1_room / fr2_pioneer_360):
  cell32_cov  fraction of 32x32 px cells holding >= 1 FAST(threshold 20) corner   (target >= 0.35; TUM fr1_room 0.46)
  fast_kpx    FAST corners per 1000 px;   gftt  Shi-Tomasi corners (quality 0.01, min dist 5)
python -m emu.tools.texprobe --out /tmp/texprobe [--save 6]   (KEVIN_EMU_TEX selects the texture set)"""
import argparse, json, math, os, sys
import numpy as np


def metrics(g):
    import cv2
    kp = cv2.FastFeatureDetector_create(20).detect(g); occ = np.zeros((g.shape[0] // 32, g.shape[1] // 32), bool)
    for k in kp:
        x, y = int(k.pt[0]) // 32, int(k.pt[1]) // 32
        if y < occ.shape[0] and x < occ.shape[1]: occ[y, x] = True
    c = cv2.goodFeaturesToTrack(g, 2000, 0.01, 5, blockSize=7)
    return dict(cell32_cov=float(occ.mean()), fast_kpx=len(kp) / g.size * 1000, gftt=0 if c is None else len(c),
                cell32_cov_bottom3=float(occ[occ.shape[0] * 2 // 3:].mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--save", type=int, default=8)
    ap.add_argument("--poses", default=os.path.expanduser("~/slamlogs")); ap.add_argument("--logs", default="L12,L13,L14"); ap.add_argument("--per", type=int, default=16)
    ap.add_argument("--places", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["KEVIN_EMU_DOG"] = "0"
    import cv2, mujoco
    from emu import world as W, realism
    class Sim(W.World):
        def _setup_io(self): pass
    ns = argparse.Namespace(env="casita", seed=0, tire="stock", ir=1, subpix=0.08, holes=0.004, min_z=0.175, band=True, rt=1.0)
    S = Sim(ns); m, d = S.m, S.d; mt = W.MAST_TRUE
    R = mujoco.Renderer(m, W.MH, W.MW)
    if a.places: poses = [(x, y, k * math.pi / 2) for (x, y) in S.env.places.values() for k in range(4)]
    else:
        from emu.tools.slamlog.log import Log
        poses = []
        for lg in a.logs.split(","):
            L = Log(os.path.join(a.poses, lg)); t = np.array([t for t, _ in L.frames["mast"]]); t = t[(t > L.gt_t[0]) & (t < L.gt_t[-1])]
            for g in L.gt(t[np.linspace(0, len(t) - 1, a.per).astype(int)]): poses.append((float(g[0]), float(g[1]), float(g[2])))
    rows = []
    for i, (x, y, yaw) in enumerate(poses):
        S.teleport(x, y, yaw)
        fx = realism.CamFX(W.MW, W.MH, [[mt["fx"], 0, mt["cx"]], [0, mt["fy"], mt["cy"]], [0, 0, 1]], dist=(mt["k1"], mt["k2"]), render_K=W.MAST_RK, exposure_s=0.015, vignette=0.35, seed=3)
        R.update_scene(d, "mast_rgb"); raw = R.render()
        for _ in range(16): img = fx(raw)                         # settle auto-exposure on this view
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); r = metrics(g); r.update(i=i, x=x, y=y, yaw=yaw); rows.append(r)
        if i % max(1, len(poses) // a.save) == 0: cv2.imwrite(os.path.join(a.out, "mast_%02d.jpg" % i), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    k = ("cell32_cov", "fast_kpx", "gftt", "cell32_cov_bottom3")
    summ = dict(tex=os.environ.get("KEVIN_EMU_TEX", realism.TEX), n=len(rows), **{f + "_med": float(np.median([r[f] for r in rows])) for f in k},
                **{f + "_mean": float(np.mean([r[f] for r in rows])) for f in k}, frac_views_cov_ge_035=float(np.mean([r["cell32_cov"] >= 0.35 for r in rows])))
    json.dump(dict(summary=summ, views=rows), open(os.path.join(a.out, "texprobe.json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
