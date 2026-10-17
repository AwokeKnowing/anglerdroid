"""cuVSLAM pose source for nav (KEVIN_SLAM=cuv).  Baseline per James 2026-10-13: out-of-box cuVSLAM (pycuvslam,
default OdometryConfig with rectified stereo, default SlamConfig) on the RS2 IR pair gives the pose; the 2D grid is
built from slam2's mapping path with scan matching and loop closure OFF.  Wheel odometry only bridges the gap
between camera frames and tracking loss (pose = cuVSLAM pose at its frame time, then odom delta) - no weighting.
After a tracking loss cuVSLAM restarts in a new frame; it is re-anchored at the current pose."""
import math, threading, time, numpy as np
from nav.slam2 import Slam2
from nav.slam import compose, inverse
import os
import cuvslam
import cuvslam_tracker as CT
from cuvslam_tracker import rig_from_twin, _quat
USE_IMU = os.environ.get("KEVIN_CUV_IMU") == "1"   # stereo-inertial (cuVSLAM OdometryMode.Inertial, D435i default noise params)


class CuvSlam(Slam2):
    def __init__(self, grid=None, match=False):
        super().__init__(grid, match=False)
        self._tr = None; self._A = None; self._fq = None; self._fcv = threading.Condition(); self._intr = None; self._imu = None; self._isince = None; self._it = 0.0
        self.stats.update(vslam="cuvslam", v_ok=0, v_lost=0, v_ms=0.0)
        threading.Thread(target=self._vloop, daemon=True).start()

    def _try_loop(self): pass                       # loop closure is cuVSLAM's job

    def _make(self, w, h, intr):
        cams = []
        for k in (1, 2):
            c = cuvslam.Camera(size=[w, h], principal=[intr["ppx"], intr["ppy"]], focal=[intr["fx"], intr["fy"]])
            R, t = rig_from_twin("rs2_ir%d" % k); c.rig_from_camera = cuvslam.Pose(rotation=_quat(R), translation=t.tolist()); cams.append(c)
        rig = cuvslam.Rig(cams); kw = dict(rectified_stereo_camera=True)
        if USE_IMU and self._imu is not None:
            ic = cuvslam.ImuCalibration(gyroscope_noise_density=CT.IMU_GYRO_NOISE_DENSITY, gyroscope_random_walk=CT.IMU_GYRO_RANDOM_WALK,
                                        accelerometer_noise_density=CT.IMU_ACCEL_NOISE_DENSITY, accelerometer_random_walk=CT.IMU_ACCEL_RANDOM_WALK, frequency=CT.IMU_FREQUENCY)
            Ri, ti = rig_from_twin("rs2_imu"); ic.rig_from_imu = cuvslam.Pose(rotation=_quat(Ri), translation=ti.tolist())
            rig.imus = [ic]; kw["odometry_mode"] = cuvslam.Tracker.OdometryMode.Inertial; self.stats["vslam"] = "cuvslam-inertial"
        self._tr = cuvslam.Tracker(rig, cuvslam.Tracker.OdometryConfig(**kw), cuvslam.Tracker.SlamConfig())

    def feed(self, t, ir1, ir2, intr, imu=None):
        if imu is not None: self._imu = imu
        if ir1 is None or ir2 is None: return
        with self._fcv: self._fq = (t, ir1, ir2, intr); self._fcv.notify()

    def _vloop(self):
        while self._run:
            with self._fcv:
                while self._fq is None: self._fcv.wait()
                t, a, b, intr = self._fq; self._fq = None
            try:
                if self._tr is None: self._make(a.shape[1], a.shape[0], intr)
                if USE_IMU and self._imu is not None:                       # all IMU samples up to this frame, in order
                    if self._isince is None: self._isince = int(self._imu.drain_ring(0, max_n=1)[3])
                    g, ac, ti, self._isince = self._imu.drain_ring(self._isince, max_n=400)
                    self._ibuf = getattr(self, "_ibuf", []) + [(float(ti[k]), ac[k].tolist(), g[k].tolist()) for k in range(len(ti))]
                    keep = []
                    for tk, ak, gk in self._ibuf:                            # strict ascending: only samples <= frame time
                        if tk > t: keep.append((tk, ak, gk)); continue
                        if tk <= self._it: continue
                        self._tr.register_imu_measurement(0, cuvslam.ImuMeasurement(timestamp_ns=int(tk * 1e9), linear_accelerations=ak, angular_velocities=gk))
                        self._it = tk
                    self._ibuf = keep[-400:]
                    self.stats["imu_lag_ms"] = round((t - self._it) * 1e3, 1); self.stats["imu_buf"] = len(self._ibuf)
                    self.stats["imu_n"] = self.stats.get("imu_n", 0) + len(ti); self.stats["frames"] = self.stats.get("frames", 0) + 1
                    if len(ti): self.stats["imu_last"] = [round(float(v), 2) for v in list(ac[-1]) + list(g[-1])]
                t0 = time.perf_counter(); pe, sp = self._tr.track(int(t * 1e9), [a, b])
                self.stats["v_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
                p = sp if (sp is not None and getattr(sp, "translation", None) is not None) else (pe.world_from_rig.pose if pe.world_from_rig is not None else None)
                if p is None:
                    self.stats["v_lost"] += 1; self._A = None; continue      # odom carries T until re-acquired
                tx, ty, tz = p.translation; qx, qy, qz, qw = p.rotation
                q = (tz, -tx, -math.atan2(2 * (qx * qz + qw * qy), 1 - 2 * (qx * qx + qy * qy)))
                po = self.odom.at(t)
                if self._A is None: self._A = compose(compose(self.T, po), inverse(q))   # anchor vslam frame to map frame
                if self.frozen: continue
                self.T = compose(compose(self._A, q), inverse(po)); self.stats["v_ok"] += 1
            except Exception as e:
                print("cuvslam error: %r" % e); self.stats["v_err"] = self.stats.get("v_err", 0) + 1
