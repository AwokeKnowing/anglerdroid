"""Forward D435i mount auto-calibration (pitch, roll, height) from two standard stationary measurements:

  gravity : mean accelerometer while parked -> camera 'up' vs gravity (base assumed level, i.e. on a level floor)
  floor   : RANSAC plane on the forward depth while parked -> camera pitch/roll and height relative to the floor

Each stationary window gives one estimate per source; the running estimate is the inverse-variance weighted mean
of all window estimates (per source sample mean + its standard error, then the two sources fused).  The result is a
base-frame correction rotation dR (roll about base x, pitch about base y; yaw is unobservable) plus dz, written to
src/calibration.json frames.rs2_depth / rs2_imu (twin stays nominal).  Warns when the correction exceeds 1 deg."""
from __future__ import annotations
import json, math, os, time
import numpy as np
import twin_geometry as TG

G_UP = np.array([0.0, 0.0, 1.0])
TOL_DEG, TOL_DZ = 5.0, 0.03   # mechanical mount tolerance (spec 24-30 deg pitch, +-1 deg roll, +-1 cm): estimates outside are not this camera's floor/gravity
STILL_V = 0.01     # m/s: "wheels stopped" = below encoder jitter (sim + real: <= ~3.3 mm/s parked)


def rot_to_up(v):
    """minimal rotation taking unit vector v to +z; returns (roll, pitch) in rad of R = Ry(pitch) Rx(roll) form."""
    v = v / np.linalg.norm(v)
    # Rx(roll) then Ry(pitch) mapping v -> z:  solve roll from (y,z), pitch from (x, z')
    roll = math.atan2(v[1], v[2]); zp = math.hypot(v[1], v[2])
    pitch = -math.atan2(v[0], zp)
    return roll, pitch


def R_rp(roll, pitch):
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return Ry @ Rx


def ransac_plane(P, thr=0.01, iters=200, rng=np.random.default_rng(0)):
    """standard RANSAC plane + least-squares refit on inliers; thr = D435 depth noise scale (~1 cm at 1-2 m)."""
    best, bi = None, None
    n = len(P)
    if n < 50: return None
    for _ in range(iters):
        a, b, c = P[rng.integers(0, n, 3)]
        nv = np.cross(b - a, c - a); L = np.linalg.norm(nv)
        if L < 1e-9: continue
        nv /= L; inl = np.abs((P - a) @ nv) < thr
        if bi is None or inl.sum() > bi.sum(): bi = inl
    Q = P[bi]; m = Q.mean(0); _, _, Vt = np.linalg.svd(Q - m, full_matrices=False); nv = Vt[2]
    return nv, float(nv @ m), int(bi.sum())


class _Mean:
    def __init__(self): self.x = []
    def add(self, v): self.x.append(np.asarray(v, float))
    def est(self):
        if not self.x: return None, None
        X = np.array(self.x); m = X.mean(0)
        se2 = (X.var(0, ddof=1) / len(X)) if len(X) > 1 else np.full(X.shape[1], np.inf)
        return m, se2


class AutoCal:
    def __init__(self, intr, stride=8, path=None, log=print):
        w, h = intr.get("width", 848), intr.get("height", 480)
        u, v = np.meshgrid(np.arange(0, w, stride, dtype=np.float32), np.arange(0, h, stride, dtype=np.float32))
        self.rx = ((u - intr.get("ppx", w / 2)) / intr.get("fx", 430.0)).ravel(); self.ry = ((v - intr.get("ppy", h / 2)) / intr.get("fy", 430.0)).ravel()
        self.s = stride; self.log = log
        dflt = os.path.expanduser("~/.kevin/emu/calib_sim.json") if os.environ.get("KEVIN_HAL") == "sim" else os.path.join(os.path.dirname(TG.__file__), "calibration.json")
        self.path = path or os.environ.get("KEVIN_CALIB", dflt)          # the sim never writes the robot's src/calibration.json
        # nominal (twin) poses, no calibration applied
        f = TG._G["frames"]; self.T0 = np.array(f["rs2_depth"]["xyz"], float); self.R0 = np.array(f["rs2_depth"]["R"], float)
        self.Ri0 = np.array(f["rs2_imu"]["R"], float)
        self.grav, self.floor = _Mean(), _Mean()
        self._acc = []; self._dep = []; self.still = False; self.n_win = 0
        self.est = dict(roll_deg=0.0, pitch_deg=0.0, dz=0.0)

    # --- feed every control tick
    def update(self, still, accel=None, depth=None):
        if still:
            if accel is not None and np.all(np.isfinite(accel)) and np.linalg.norm(accel) > 1: self._acc.append(np.asarray(accel, float))
            if depth is not None and len(self._dep) < 5: self._dep.append(depth)
        elif self.still:
            self._close()
        self.still = still

    def _close(self):
        """end of a stationary window: one estimate per source."""
        if len(self._acc) >= 10:
            g = self.Ri0 @ np.mean(self._acc, 0)           # specific force at rest = 'up' in the (nominal) base frame
            r, p = rot_to_up(g)
            if max(abs(r), abs(p)) < math.radians(TOL_DEG): self.grav.add([r, p])
            else: self.log("autocal REJECTED gravity window: roll %+.1f pitch %+.1f deg outside the %.0f deg mount tolerance -- check the camera mount" % (math.degrees(r), math.degrees(p), TOL_DEG))
        for d in self._dep[-1:]:
            z = d[::self.s, ::self.s].ravel().astype(np.float32) * 1e-3; ok = (z > 0.2) & (z < 3.0)
            Pc = np.stack([self.rx[ok] * z[ok], self.ry[ok] * z[ok], z[ok]], 1)
            Pb = Pc @ self.R0.T                            # camera-centred, nominal base orientation
            res = ransac_plane(Pb)
            if res is None or res[2] < 0.3 * len(Pb): continue   # floor must be the dominant plane in view
            nv, dd, _ = res
            if nv[2] < 0: nv, dd = -nv, -dd
            r, p = rot_to_up(nv)
            # floor at z = -AXLE_H in base: camera height above floor = -dd (camera-centred) -> dz
            dz = (-dd) - (self.T0[2] + TG.AXLE_H_M)
            if max(abs(r), abs(p)) < math.radians(TOL_DEG) and abs(dz) < TOL_DZ: self.floor.add([r, p, dz])
            else: self.log("autocal REJECTED floor window: roll %+.1f pitch %+.1f deg dz %+.3f m outside tolerance" % (math.degrees(r), math.degrees(p), dz))
        self._acc, self._dep = [], []; self.n_win += 1
        self._fuse()

    def _fuse(self):
        if not self.grav.x and not self.floor.x: return
        mg, vg = self.grav.est(); mf, vf = self.floor.est()
        rp = []
        for i in range(2):
            num = den = 0.0
            for m, v in ((mg, vg), (mf, vf)):
                if m is None: continue
                w = 1.0 / v[i] if np.isfinite(v[i]) and v[i] > 0 else 1e-12   # single sample: tiny weight until a 2nd arrives
                num += w * m[i]; den += w
            rp.append(num / den if den else 0.0)
        old = dict(self.est)
        self.est = dict(roll_deg=math.degrees(rp[0]), pitch_deg=math.degrees(rp[1]), dz=float(mf[2]) if mf is not None else 0.0,
                        n_grav=len(self.grav.x), n_floor=len(self.floor.x))
        dR = self.dR()
        self.log("autocal: roll %+.2f pitch %+.2f deg dz %+.3f m (windows grav %d floor %d)" % (self.est["roll_deg"], self.est["pitch_deg"], self.est["dz"], self.est["n_grav"], self.est["n_floor"]))
        if max(abs(self.est["roll_deg"]), abs(self.est["pitch_deg"])) > 1.0:
            self.log("autocal WARNING: forward D435i mount differs from the twin by >1 deg (roll %+.2f, pitch %+.2f)" % (self.est["roll_deg"], self.est["pitch_deg"]))
        self.write()

    def dR(self):
        # correction maps the nominal-base 'up' measurement onto true up: R_true = dR @ R0
        return R_rp(math.radians(self.est["roll_deg"]), math.radians(self.est["pitch_deg"]))

    def pose(self):
        """calibrated (T, R) optical->base for the forward depth camera."""
        return self.T0 + np.array([0, 0, self.est["dz"]]), self.dR() @ self.R0

    def write(self):
        try: C = json.load(open(self.path)) if os.path.exists(self.path) else {}
        except Exception: C = {}
        rpy = self._rpy_of(self.dR())
        fr = C.setdefault("frames", {})
        for k in ("rs2_depth", "rs2_imu"):
            fr[k] = {"dxyz": [0.0, 0.0, round(self.est["dz"], 4)], "drpy_deg": [round(math.degrees(a), 3) for a in rpy]}
        C["rs2_autocal"] = dict(self.est, t=time.time())
        tmp = self.path + ".tmp"; json.dump(C, open(tmp, "w"), indent=1); os.replace(tmp, self.path)

    @staticmethod
    def _rpy_of(R):   # inverse of twin_geometry._rpy (ZYX)
        return (math.atan2(R[2, 1], R[2, 2]), -math.asin(max(-1, min(1, R[2, 0]))), math.atan2(R[1, 0], R[0, 0]))
