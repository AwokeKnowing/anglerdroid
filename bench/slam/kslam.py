"""kslam: Kevin's own estimator.  Wheel odometry is the motion prior, the gyro handles short windows, and
house structure (dominant wall directions) bounds heading drift.  Everything is self-calibrated online from the
data:
  * RS2 mount pitch/roll + camera height from the floor plane in depth (RANSAC, EMA)
  * robot self-occlusion mask: depth pixels that never change while the robot moves
  * gyro scale + bias in a 3-state EKF [yaw, gyro_scale_err, gyro_bias]
    - ZUPT (wheels stopped) observes the bias
    - wall-direction yaw observations make the scale observable
  * Manhattan frame of the house learned online in the first seconds.  No twin map is needed; the twin gives only
    the nominal (belief) extrinsics.
Position = wheel speed integrated along the fused heading.  Optional extra edge sources (VO / learned / loop
closure) plug in on the i777 pose-graph side (see NOTES.md).
python bench/slam/kslam.py DATA/<seq> RESULTS/<seq> [--name kslam] [--no-manhattan] [--no-selfcal]
-> RESULTS/<seq>/<name>.tum (base frame, metric) + .json (self-calibration estimates vs withheld truth)"""
import argparse, json, math, os, time
import numpy as np, cv2

def rot_x(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def rot_y(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi
def wrap90(a): q = math.pi / 2; return (a + q / 2) % q - q / 2

class FloorCal:
    """RS2 mount correction (pitch about base y, roll about base x) + camera height from the floor plane."""
    def __init__(self, R_bc_belief, t_bc):
        self.Rb, self.t = np.asarray(R_bc_belief), np.asarray(t_bc); self.dp = 0.0; self.dr = 0.0; self.h = float(t_bc[2]); self.n = 0
        self.rng = np.random.default_rng(0)
    def R(self): return rot_y(self.dp) @ rot_x(self.dr) @ self.Rb
    def update(self, P_c):                                   # P_c: Nx3 camera-frame points
        Pb = P_c @ self.R().T + self.t
        cand = Pb[(Pb[:, 2] < 0.25) & (Pb[:, 2] > -0.25)]    # near the believed floor
        if len(cand) < 300: return False
        best, bi = None, 0
        for _ in range(60):
            s = cand[self.rng.choice(len(cand), 3, replace=False)]
            n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
            if nn < 1e-9: continue
            n /= nn
            if n[2] < 0: n = -n
            if n[2] < 0.94: continue                        # within ~20 deg of believed up
            inl = np.abs((cand - s[0]) @ n) < 0.012; c = int(inl.sum())
            if c > bi: bi, best = c, inl
        if best is None or bi < 0.5 * len(cand): return False
        Q = cand[best]; m = Q.mean(0); _, _, Vt = np.linalg.svd(Q - m); n = Vt[2]; n = n if n[2] > 0 else -n
        # n is the floor normal in the *current* believed base frame -> rotate the correction so that n -> +z
        dp = -math.atan2(n[0], n[2]); dr = math.atan2(n[1], n[2])
        a = 1.0 / min(self.n + 1, 20)
        self.dp += a * dp; self.dr += a * dr
        self.h = (1 - a) * self.h + a * float(self.t[2] - (n @ m))   # camera height above the floor
        self.n += 1; return True

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("res"); ap.add_argument("--name", default="kslam")
    ap.add_argument("--no-manhattan", action="store_true"); ap.add_argument("--no-selfcal", action="store_true")
    ap.add_argument("--every", type=int, default=3, help="use every Nth depth frame for structure (15 Hz / 3 = 5 Hz)")
    a = ap.parse_args(); seq, res = os.path.abspath(a.seq), os.path.abspath(a.res); os.makedirs(res, exist_ok=True)
    M = json.load(open(os.path.join(seq, "meta.json"))); E = M["extr_twin_base_from_optical"]; Id = M["intr"]["rs2_depth"]
    TR = M["twin"]["track_m"]
    F = np.genfromtxt(os.path.join(seq, "frames.csv"), delimiter=",", names=True)
    W = np.genfromtxt(os.path.join(seq, "wheel.csv"), delimiter=",", names=True)
    I = np.genfromtxt(os.path.join(seq, "imu.csv"), delimiter=",", names=True)
    R_imu_b = np.asarray(E["rs2_imu"]["R"]); R_dep_b = np.asarray(E["rs2_depth"]["R"]); t_dep = np.asarray(E["rs2_depth"]["t"])
    cal = FloorCal(R_dep_b, t_dep)
    # depth back-projection grid (stride 4)
    S = 4; Hh, Ww = Id["h"], Id["w"]; vv, uu = np.mgrid[0:Hh:S, 0:Ww:S]
    rx = (uu - Id["ppx"]) / Id["fx"]; ry = (vv - Id["ppy"]) / Id["fy"]
    # EKF state
    x = np.array([float(np.genfromtxt(os.path.join(seq, "gt.csv"), delimiter=",", names=True, max_rows=1)["yaw"]), 0.0, 0.0])
    gt0 = np.genfromtxt(os.path.join(seq, "gt.csv"), delimiter=",", names=True, max_rows=1)
    px, py = float(gt0["x"]), float(gt0["y"])              # anchored at the start pose (same as every method in eval)
    P = np.diag([1e-6, 0.02 ** 2, 0.003 ** 2]); QG, QS, QB = 0.004 ** 2, 1e-6 ** 2, 2e-5 ** 2
    psi0 = None; psi_acc = []; dmask = None; var_acc = None; mean_acc = None; nmov = 0
    stats = dict(manhattan_used=0, manhattan_rejected=0, floor_updates=0, zupt=0)
    out = open(os.path.join(res, a.name + ".tum"), "w")
    wi = 0; fi = 0; nf = len(F["t"]); t_prev = I["t"][0]; vel = 0.0; still = False; last_z = -1.0
    def R_imu():                                           # same D435i body: the floor self-cal also corrects the IMU mount
        return (rot_y(cal.dp) @ rot_x(cal.dr) @ R_imu_b) if not a.no_selfcal else R_imu_b
    dbg = open(os.path.join(res, "_" + a.name + "_manhattan.txt"), "w") if os.environ.get("KSLAM_DEBUG") else None
    t0 = time.perf_counter()
    for k in range(len(I["t"])):
        t = I["t"][k]; dt = t - t_prev; t_prev = t
        while wi < len(W["t"]) - 1 and W["t"][wi + 1] <= t: wi += 1
        vl, vr = W["vl"][wi], W["vr"][wi]; vel = 0.5 * (vl + vr); still = abs(vl) < 0.003 and abs(vr) < 0.003
        gz = float((R_imu() @ np.array([I["gx"][k], I["gy"][k], I["gz"][k]]))[2])
        # predict
        w = gz * (1 + x[1]) - x[2]; x[0] = wrap(x[0] + w * dt)
        Fm = np.array([[1, gz * dt, -dt], [0, 1, 0], [0, 0, 1]]); P = Fm @ P @ Fm.T + np.diag([QG * dt, QS * dt, QB * dt])
        px += vel * math.cos(x[0]) * dt; py += vel * math.sin(x[0]) * dt
        if still and t - last_z > 0.05:                      # ZUPT: true yaw rate is 0 -> observes the bias
            last_z = t; Hm = np.array([0.0, gz, -1.0]); r = 0.0 - w; Rz = (0.004 ** 2) * 200 / 20
            Sv = Hm @ P @ Hm + Rz; K = P @ Hm / Sv; x = x + K * r; P = P - np.outer(K, Hm) @ P; stats["zupt"] += 1
        # camera frame(s) due
        while fi < nf and F["t"][fi] <= t:
            tf = F["t"][fi]; i = int(F["i"][fi])
            if fi % a.every == 0 and not (a.no_manhattan and a.no_selfcal):
                d = cv2.imread(os.path.join(seq, "rs2_depth", "%06d.png" % i), cv2.IMREAD_UNCHANGED)[::S, ::S].astype(np.float32) * 1e-3
                moving = abs(vel) > 0.05 or abs(w) > 0.2
                if moving and nmov < 60:                     # learn the self-occlusion mask while moving
                    if mean_acc is None: mean_acc = np.zeros_like(d); var_acc = np.zeros_like(d)
                    nmov += 1; dd = d - mean_acc; mean_acc += dd / nmov; var_acc += dd * (d - mean_acc)
                    if nmov == 60: dmask = np.sqrt(var_acc / nmov) > 0.01; stats["selfmask_frac"] = round(float(1 - dmask.mean()), 3)
                if dmask is not None:
                    valid = (d > 0.2) & (d < 4.0) & dmask
                    Pc = np.stack([rx * d, ry * d, d], -1)
                    if not a.no_selfcal and fi % (a.every * 3) == 0:
                        if cal.update(Pc[valid][::2]): stats["floor_updates"] += 1
                    if not a.no_manhattan:
                        dx = Pc[1:-1, 2:] - Pc[1:-1, :-2]; dy = Pc[2:, 1:-1] - Pc[:-2, 1:-1]; n = np.cross(dx, dy)
                        nn = np.linalg.norm(n, axis=-1); ok = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1] & (nn > 1e-9)
                        ok &= (np.linalg.norm(dx, axis=-1) < 0.15) & (np.linalg.norm(dy, axis=-1) < 0.15)
                        nb = (n[ok] / nn[ok][:, None]) @ cal.R().T; Pb = Pc[1:-1, 1:-1][ok] @ cal.R().T + cal.t
                        sel = (np.abs(nb[:, 2]) < 0.15) & (Pb[:, 2] > 0.12) & (Pb[:, 2] < 2.2)
                        if sel.sum() > 150:
                            ang = np.arctan2(nb[sel, 1], nb[sel, 0]) + x[0]          # world-frame wall-normal angle
                            z4 = np.exp(4j * ang).mean(); Rl = abs(z4); phi = np.angle(z4) / 4
                            if psi0 is None:
                                if Rl > 0.5 and sel.sum() > 400: psi_acc.append(np.exp(4j * phi))
                                if len(psi_acc) >= 5: psi0 = float(np.angle(np.mean(psi_acc)) / 4); stats["psi0_deg"] = round(math.degrees(psi0), 2)
                            elif Rl > 0.45:
                                r = -wrap90(phi - psi0)          # innovation: measured - predicted heading (mod 90)
                                if dbg: dbg.write("%.3f %.5f %.3f %d %.5f %.4f\n" % (tf, r, Rl, int(sel.sum()), x[0], w))
                                sig = math.radians(1.5) / math.sqrt(min(sel.sum(), 4000) / 500.0) / Rl
                                Sv = P[0, 0] + sig ** 2
                                if r * r / Sv < 25.0 and abs(w) < 1.2:        # 3-sigma gate; skip fast spins (blur/latency)
                                    K = P[:, 0] / Sv; x = x + K * r; x[0] = wrap(x[0]); P = P - np.outer(K, P[0]); stats["manhattan_used"] += 1
                                else: stats["manhattan_rejected"] += 1
            out.write("%.4f %.5f %.5f 0 0 0 %.6f %.6f\n" % (tf, px, py, math.sin(x[0] / 2), math.cos(x[0] / 2)))
            fi += 1
    out.close(); wall = time.perf_counter() - t0
    wt = M["withheld_truth"]["rs2_mount"]
    est = dict(rs2_pitch_corr_deg=round(math.degrees(cal.dp), 2), rs2_roll_corr_deg=round(math.degrees(cal.dr), 2), cam_height_m=round(cal.h, 4),
               rs2_pitch_est_deg=round(math.degrees(math.asin(-(cal.R() @ np.array([0, 0, 1.0]))[2])), 2),
               rs2_pitch_belief_deg=round(math.degrees(math.asin(-(R_dep_b @ np.array([0, 0, 1.0]))[2])), 2), rs2_pitch_true_deg=wt["pitch_deg"],
               gyro_scale_err_est=round(float(x[1]), 5), gyro_bias_est=round(float(x[2]), 5), gyro_scale_err_true_z_imu=M["drawn_errors"]["gyro_scale_err"])
    json.dump(dict(frame="base", scale_free=False, per_frame=True, note="kslam: wheel + gyro EKF (scale/bias) + online Manhattan heading + floor self-cal" +
                   (" [no manhattan]" if a.no_manhattan else "") + (" [no selfcal]" if a.no_selfcal else ""),
                   runtime="%.2f ms/frame CPU (python, %d frames)" % (wall / nf * 1e3, nf), selfcal=est, stats=stats),
              open(os.path.join(res, a.name + ".json"), "w"), indent=1)
    print(a.name, json.dumps(est), json.dumps(stats), "%.1f s" % wall)

if __name__ == "__main__":
    main()
