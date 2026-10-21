"""mastcal: self-calibrate the mast camera intrinsics (focal + radial k1,k2) from ordinary in-place turns.  No target
and no twin map: the robot's own heading estimate (kslam yaw: wheel+gyro+Manhattan) gives the rotation between two
frames.  For a (near-)pure rotation, matched pixels must satisfy  ray2 = R_cam * ray1, so the rays observed through
the unknown lens must be consistent with the known rotation, which pins down fx and the distortion.
Matches come from pyramidal LK on Shi-Tomasi corners, with a forward-backward check.  Solved by robust least
squares (soft-L1).
python bench/slam/mastcal.py DATA/<seq> RESULTS/<seq> [--yaw kslam]   -> RESULTS/<seq>/mastcal.json"""
import argparse, json, math, os
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

def rays(px, f, k1, k2, cx, cy):
    """pixel -> unit ray for the Brown radial model (iterative undistort)"""
    xd = (px - [cx, cy]) / f; x = xd.copy()
    for _ in range(8):
        r2 = (x ** 2).sum(1, keepdims=True); x = xd / (1 + k1 * r2 + k2 * r2 ** 2)
    v = np.concatenate([x, np.ones((len(x), 1))], 1); return v / np.linalg.norm(v, axis=1, keepdims=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("res"); ap.add_argument("--yaw", default="kslam")
    ap.add_argument("--max-pairs", type=int, default=150)
    a = ap.parse_args(); seq, res = os.path.abspath(a.seq), os.path.abspath(a.res)
    M = json.load(open(os.path.join(seq, "meta.json"))); Rbm = np.asarray(M["extr_twin_base_from_optical"]["mast"]["R"])
    F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True)
    K = np.loadtxt(os.path.join(res, a.yaw + ".tum")); yk = np.unwrap(Rot.from_quat(K[:, 4:8]).as_euler("zyx")[:, 0])
    W = np.genfromtxt(os.path.join(seq, "wheel.csv"), delimiter=",", names=True)
    v = np.interp(F["t"], W["t"], 0.5 * (W["vl"] + W["vr"])); w = np.interp(F["t"], W["t"], (W["vr"] - W["vl"]) / M["twin"]["track_m"])
    cand = [k for k in range(len(F["t"]) - 3) if abs(v[k]) < 0.01 and abs(v[k + 3]) < 0.01 and 0.3 < abs(w[k]) and abs(yk[k + 3] - yk[k]) > math.radians(6)]
    sel = cand[:: max(1, len(cand) // a.max_pairs)][: a.max_pairs]
    P1, P2, RR = [], [], []
    for k in sel:
        i0, i1 = int(F["i"][k]), int(F["i"][k + 3])
        g0 = cv2.imread(os.path.join(seq, "mast", "%06d.jpg" % i0), 0); g1 = cv2.imread(os.path.join(seq, "mast", "%06d.jpg" % i1), 0)
        p0 = cv2.goodFeaturesToTrack(g0, 300, 0.01, 12)
        if p0 is None: continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=4)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(21, 21), maxLevel=4)
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pb - p0, axis=2)[:, 0] < 0.5)
        if ok.sum() < 20: continue
        dpsi = yk[k + 3] - yk[k]
        Rb = Rot.from_rotvec([0, 0, dpsi]).as_matrix()                 # base rotation frame0 -> frame1
        Rc = Rbm.T @ Rb.T @ Rbm                                         # ray in cam0 -> ray in cam1
        P1.append(p0[ok, 0]); P2.append(p1[ok, 0]); RR.append(np.repeat(Rc[None], ok.sum(), 0))
    P1, P2, RR = np.concatenate(P1), np.concatenate(P2), np.concatenate(RR)
    cx, cy = (M["intr"].get("mast", {}).get("w", 640) - 1) / 2, (M["intr"].get("mast", {}).get("h", 480) - 1) / 2
    def resid(p):
        f, k1, k2 = p; r1 = rays(P1, f, k1, k2, cx, cy); r2 = rays(P2, f, k1, k2, cx, cy)
        q = np.einsum("nij,nj->ni", RR, r1); return np.cross(q, r2).ravel()
    f0 = 320 / math.tan(math.radians(M["robot_belief"]["mast_hfov_deg"]) / 2)    # start from the robot's belief
    sol = least_squares(resid, [f0, 0.0, 0.0], loss="soft_l1", f_scale=0.003, x_scale=[50, 0.05, 0.01])
    f, k1, k2 = sol.x; T = M["withheld_truth"]["mast"]
    # HFOV implied by the estimated lens: angle of the ray through the image edge pixel
    edge = rays(np.array([[0.0, cy], [639.0, cy]]), f, k1, k2, cx, cy); hfov = math.degrees(math.acos(float(edge[0] @ edge[1])))
    tedge = rays(np.array([[0.0, cy], [639.0, cy]]), T["fx"], T["k1"], T["k2"], cx, cy); thfov = math.degrees(math.acos(float(tedge[0] @ tedge[1])))
    out = dict(pairs=len(sel), matches=int(len(P1)), fx_est=round(f, 2), k1_est=round(k1, 4), k2_est=round(k2, 4), hfov_est_deg=round(hfov, 2),
               fx_true=round(T["fx"], 2), k1_true=T["k1"], k2_true=T["k2"], hfov_true_deg=round(thfov, 2),
               belief_hfov_deg=M["robot_belief"]["mast_hfov_deg"], fx_belief=round(f0, 2), rot_source=a.yaw, cost=float(sol.cost))
    json.dump(out, open(os.path.join(res, "mastcal.json"), "w"), indent=1); print(json.dumps(out))

if __name__ == "__main__":
    main()
