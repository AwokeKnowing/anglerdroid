"""Reader for a sensor log written by emu/tools/sensorlog.py (+ nav KEVIN_RECORD in <dir>/nav).
All times are seconds on the sim/nav monotonic clock.  Poses are SE(2) base_link (axle centre) in the world frame (GT) or
the nav frame (nav 'pose' / 'odom').  Camera extrinsics (optical -> base) come from the digital twin (same numbers the sim
renders with)."""
import glob, json, math, os
import cv2, numpy as np

R_RS1 = np.array([[-1.0, 0, 0], [0, 1, 0], [0, 0, -1]]); T_RS1 = np.array([-0.007, 0.0, 0.8619])
_p = math.radians(27.0); R_RS2 = np.array([[0, -math.sin(_p), math.cos(_p)], [-1, 0, 0], [0, -math.cos(_p), -math.sin(_p)]])
T_RS2 = np.array([-0.1301, 0.0, 0.3822])
R_MAST = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])
AXLE_H = 0.0856

def wrap(a): return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

class Log:
    def __init__(self, d):
        self.d = d; self.info = json.load(open(os.path.join(d, "info.json")))
        g, fr = [], {}
        for l in open(os.path.join(d, "index.jsonl")):
            r = json.loads(l)
            if "gt" in r: g.append([r["t_mono"]] + r["gt"][:3] + [r["t_sim"]])
            else: fr.setdefault(r["cam"], {})[r["stamp_ns"]] = r["file"]
        g = np.array(g); self.gt_xy = g[:, 1:3]; self.gt_yaw = np.unwrap(g[:, 3])
        # GT time: the pose belongs to sim time t_sim; t_mono (query midpoint) carries the UDP round trip (p99 +20 ms).
        # Map t_sim -> monotonic by the lower envelope of (t_mono - t_sim) (latency only adds), fitted per 5 s chunk.
        off = g[:, 0] - g[:, 4]; ch = np.floor((g[:, 4] - g[0, 4]) / 5.0); cx, cy = [], []
        for c in np.unique(ch): m = ch == c; i = np.argmin(off[m]); cx.append(g[m, 4][i]); cy.append(off[m][i])
        self.gt_t = g[:, 4] + (np.polyval(np.polyfit(cx, cy, 1), g[:, 4]) if len(cx) > 2 else off.min())
        self.frames = {c: sorted((s * 1e-9, f) for s, f in v.items()) for c, v in fr.items()}
        mast = self.info["mast"]; T_m = np.array(mast["pos"], float); T_m[2] -= AXLE_H   # mast pos is floor-relative
        self.ext = dict(rs1=(R_RS1, T_RS1), rs2=(R_RS2, T_RS2), mast=(R_MAST, T_m))
        dev = list(self.info["devices"].values()); self.streams = dev[0]["streams"]
        self._nav = None

    def gt(self, t):
        t = np.asarray(t, float)
        return np.stack([np.interp(t, self.gt_t, self.gt_xy[:, 0]), np.interp(t, self.gt_t, self.gt_xy[:, 1]), np.interp(t, self.gt_t, self.gt_yaw)], -1)

    def K(self, cam, stream="color"):
        s = self.info["mast"]["intr"] if cam == "mast" else self.streams[stream]
        return np.array([[s["fx"], 0, s["ppx"]], [0, s["fy"], s["ppy"]], [0, 0, 1.0]]), (s.get("w", 640), s.get("h", 480))

    def rgb(self, f): return cv2.imread(os.path.join(self.d, f + ".jpg"))[..., ::-1]
    def depth(self, f): return cv2.imread(os.path.join(self.d, f + ".png"), cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-3

    def depth_in_color(self, f):
        """RS depth (848x480, depth optical frame) re-projected into the colour camera (640x480, x offset -15 mm), z-buffered."""
        z = self.depth(f); Kd, _ = self.K("rs1", "depth"); Kc, (w, h) = self.K("rs1", "color")
        tc = np.array(self.streams["color"]["pose"][1], float)
        v, u = np.nonzero(z > 0); zz = z[v, u]
        X = np.stack([(u - Kd[0, 2]) / Kd[0, 0] * zz, (v - Kd[1, 2]) / Kd[1, 1] * zz, zz], 1) - tc
        uc = np.round(X[:, 0] / X[:, 2] * Kc[0, 0] + Kc[0, 2]).astype(int); vc = np.round(X[:, 1] / X[:, 2] * Kc[1, 1] + Kc[1, 2]).astype(int)
        ok = (uc >= 0) & (uc < w) & (vc >= 0) & (vc < h); out = np.full((h, w), np.inf, np.float32)
        np.minimum.at(out, (vc[ok], uc[ok]), X[ok, 2].astype(np.float32)); out[np.isinf(out)] = 0
        return out

    def nav(self):
        """nav KEVIN_RECORD ticks: t, vl, vr, gz (EKF gyro z as nav saw it), raw gyro xyz, accel xyz, pose, odom."""
        if self._nav is None:
            cache = os.path.join(self.d, "nav_ticks.npz")
            if os.path.exists(cache): self._nav = dict(np.load(cache))
            else:
                R = []
                for f in sorted(glob.glob(os.path.join(self.d, "nav", "*.npz"))):
                    z = np.load(f); m = json.loads(str(z["meta"]))
                    R.append([m["t"], m["vl"], m["vr"], m["gz"]] + list(z["gyro"]) + list(z["accel"]) + list(m["pose"]) + list(m["odom"]) + [m.get("imu_t") or m["t"]])
                R = np.array(R, float)
                self._nav = dict(t=R[:, 0], vl=R[:, 1], vr=R[:, 2], gz=R[:, 3], gyro=R[:, 4:7], accel=R[:, 7:10], pose=R[:, 10:13], odom=R[:, 13:16], imu_t=R[:, 16])
                np.savez(cache, **self._nav)
        return self._nav
