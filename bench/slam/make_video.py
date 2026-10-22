"""Watchable SLAM-from-zero video: the map built up from empty by one method (shared occmap mapper, RS2 depth posed by
the method's ONLINE trajectory), estimated (red) vs true (green) trajectory, twin walls (blue, reference only: never
used by the method), side by side with the photoreal RTX mast camera (+ RTX RS2 inset).
python bench/slam/make_video.py DATA/<seq> RESULTS/<seq>/<method>.tum OUT.mp4 [--title ...] [--fps 10]"""
import argparse, json, math, os, subprocess, sys
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import occmap
from eval import load_gt, interp_gt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("tum"); ap.add_argument("out"); ap.add_argument("--title", default=""); ap.add_argument("--fps", type=float, default=10)
    a = ap.parse_args(); seq = os.path.abspath(a.seq)
    gt_occ, org = occmap.gt_grid(os.path.join(seq, "gt_occ.npz")); tg, Tg = load_gt(seq)
    t, T = occmap.anchored(seq, a.tum); F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True)
    Tgf, _ = interp_gt(tg, Tg, F["t"])
    # crop to the house bounding box
    rr, cc = np.nonzero(gt_occ); r0, r1, c0, c1 = max(0, rr.min() - 4), rr.max() + 5, max(0, cc.min() - 4), cc.max() + 5
    Hh = 720; sc = Hh / (r1 - r0); Wm = int((c1 - c0) * sc)
    wall = gt_occ & ~__import__("scipy.ndimage", fromlist=["x"]).binary_erosion(gt_occ)
    tmp = a.out + ".tmp.mp4"; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (960 + Wm, Hh)); est_xy = []; gt_xy = []
    def px(xy):
        rc = np.floor((xy[:, ::-1] - org[::-1]) / occmap.RES).astype(int); return np.stack([(rc[:, 1] - c0 + 0.5) * sc, (r1 - 1 - rc[:, 0] + 0.5) * sc], 1).astype(np.int32)
    def cb(k, fi, tf, g, Tb):
        L = g.L[r0:r1, c0:c1][::-1]; img = np.full(L.shape + (3,), 110, np.uint8); img[L < -0.5] = (238, 238, 238); img[L > 0.5] = (25, 25, 25)
        wv = wall[r0:r1, c0:c1][::-1]; img[wv & ~(L > 0.5)] = (215, 150, 120)
        img = cv2.resize(img, (Wm, Hh), interpolation=cv2.INTER_NEAREST)
        est_xy.append(Tb[:2, 3]); j = min(len(Tgf) - 1, k); gt_xy.append(Tgf[j][:2, 3])
        cv2.polylines(img, [px(np.array(gt_xy))], False, (0, 170, 0), 2, cv2.LINE_AA); cv2.polylines(img, [px(np.array(est_xy))], False, (0, 0, 230), 2, cv2.LINE_AA)
        for Tx, col in ((Tgf[j], (0, 170, 0)), (Tb, (0, 0, 230))):
            p0 = px(Tx[None, :2, 3])[0]; yaw = math.atan2(Tx[1, 0], Tx[0, 0]); p1 = (int(p0[0] + 18 * math.cos(yaw)), int(p0[1] - 18 * math.sin(yaw)))
            cv2.circle(img, tuple(int(v) for v in p0), 6, col, -1); cv2.line(img, tuple(int(v) for v in p0), p1, col, 2)
        e = math.hypot(*(Tb[:2, 3] - Tgf[j][:2, 3])); de = abs((math.degrees(math.atan2(Tb[1, 0], Tb[0, 0])) - math.degrees(math.atan2(Tgf[j][1, 0], Tgf[j][0, 0])) + 180) % 360 - 180)
        mast = cv2.imread(os.path.join(seq, "rtx_mast", "%06d.jpg" % fi)); rs2 = cv2.imread(os.path.join(seq, "rtx_rs2", "%06d.jpg" % fi))
        left = np.zeros((Hh, 960, 3), np.uint8); left[:720] = cv2.resize(mast, (960, 720))
        ins = cv2.resize(rs2, (320, 240)); left[Hh - 250:Hh - 10, 960 - 330:960 - 10] = ins
        cv2.putText(left, "RTX mast camera (photoreal sim)  t=%.1fs" % tf, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(left, "RTX RS2", (960 - 325, Hh - 256), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        for li, (txt, col) in enumerate([(s_, (0, 0, 0)) for s_ in a.title.split("|")] + [("map from zero", (0, 0, 0)), ("red = estimate", (0, 0, 230)),
                                        ("green = truth", (0, 140, 0)), ("blue = twin walls", (170, 90, 40)), ("(reference only)", (170, 90, 40))]):
            cv2.putText(left, txt.strip(), (14, 62 + 24 * li), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(left, txt.strip(), (14, 62 + 24 * li), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col if col != (0, 0, 0) else (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, "err %.2f m %.1f deg" % (e, de), (6, Hh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2, cv2.LINE_AA)
        vw.write(np.hstack([left, img]))
    occmap.build(seq, T, t, gt_shape=gt_occ.shape, gt_org=org, cb=cb); vw.release()
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", a.out]); os.remove(tmp)
    else: os.replace(tmp, a.out)
    print("video", a.out)


if __name__ == "__main__":
    main()
