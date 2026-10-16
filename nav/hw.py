"""Robot I/O: the only module that touches drivers.  30 Hz clock = RS1 (top-down) frames."""
from __future__ import annotations
import math, os, threading, time
import numpy as np
import nav  # noqa: F401  (src on sys.path)

# geometry: digital twin (src/twin_geometry.json + calibration.json hook), base_link = axle centre
import twin_geometry as TG
RS2_T = TG.RS2_T.astype(np.float32); RS2_R = TG.RS2_R.astype(np.float32)       # optical -> base
RS2_PITCH = math.radians(TG.RS2_PITCH_DOWN_DEG)
_s, _c = math.sin(RS2_PITCH), math.cos(RS2_PITCH)
AXLE_H = TG.AXLE_H_M                                                             # per KEVIN_TIRE profile


class Hardware:
    def __init__(self, drive=True, stereo=True):
        import pyrealsense2 as rs
        from cameras import RSCamera
        from imu import IMUPipeline
        ser = TG.realsense_roles(rs)                  # by product (D435 top-down / D435i forward), not sort order
        if not all(ser):
            raise RuntimeError("need RS1 (D435 top-down) + RS2 (D435i forward), found %s" % (ser,))
        self.rs1 = RSCamera(ser[0], compute_pointcloud=True)
        self.rs2 = RSCamera(ser[1], compute_pointcloud=False, capture_ir=stereo, emitter=not stereo)
        self.imu = IMUPipeline(ser[1])
        self.rs2_intr = self.rs2.depth_intrinsics()
        self.wb = None
        if drive:
            import wheelbase
            cwd = os.getcwd(); os.chdir(nav.SRC)          # WheelBase reads flat_endpoints.json relative to src/
            try: self.wb = wheelbase.WheelBase()
            finally: os.chdir(cwd)
        self._rs2_lock = threading.Lock(); self.rs2_seq = 0; self.rs2_stamp = 0.0
        self._run = True
        threading.Thread(target=self._rs2_pump, daemon=True, name="rs2").start()
        # command heartbeat (M1): the brain loop's period jitters (p99 ~550 ms seen) past wheelbase's 0.5 s stale
        # watchdog, which then IDLEs the axes mid-goal.  Re-send the LAST (already safety-filtered + capped) command
        # at HB_HZ, but only while the brain is fresh (last drive() < HB_FRESH_S ago).  Once stale we send ONE zero
        # and stop sending, so wheelbase's 0.5 s watchdog and the ODrive watchdog remain the safety net underneath.
        # Never invents motion: it only repeats what drive() was given, and goes silent if drive() stops.
        self._drv_lock = threading.Lock(); self._cmd = (0.0, 0.0, 0.0)          # v, w, monotonic stamp
        self._hb_zeroed = True; self.hb_stats = dict(sent=0, zeroed=0)
        if self.wb is not None and os.environ.get("KEVIN_HB", "1") == "1":
            threading.Thread(target=self._hb_loop, daemon=True, name="cmd-heartbeat").start()

    def _rs2_pump(self):
        while self._run:
            fs = self.rs2.wait_frameset(200)
            if fs is None: continue
            with self._rs2_lock:
                if self.rs2.ingest(fs):
                    self.rs2.publish(); self.rs2_seq += 1; self.rs2_stamp = self.rs2.depth_stamp

    def rs2_snapshot(self):
        """(seq, stamp_s, depth u16, color rgb, ir_left, ir_right) — copies, safe across threads."""
        with self._rs2_lock:
            c = self.rs2
            return (self.rs2_seq, self.rs2_stamp, c.depth.copy(), c.color.copy(),
                    None if c.ir_left is None else c.ir_left.copy(), None if c.ir_right is None else c.ir_right.copy())

    def wait_rs1(self, timeout_ms=200):
        """Block on the RS1 camera clock (30 Hz). Returns (stamp_s, verts) or (None, None) on timeout."""
        fs = self.rs1.wait_frameset(timeout_ms)
        if fs is None or not self.rs1.ingest(fs):
            return None, None
        self.rs1.publish()
        return self.rs1.depth_stamp, self.rs1.verts

    def wheel_vels(self):
        return self.wb.get_wheel_velocities_mps() if self.wb else (0.0, 0.0)

    def yaw_rate(self):
        """Robot yaw rate (rad/s, CCW+) from the D435i gyro (optical frame) — up axis in optical = (0,-c,-s)."""
        if os.environ.get("KEVIN_NO_IMU") == "1": return None          # ablation: wheel-only yaw
        self.imu.grab()
        # mean of ALL gyro samples since the last call (200 Hz ring), not just the latest: a stalled loop
        # (CUDA warm-up, GC) during a spin otherwise lost the rotation (maze: 0.5 rad lost on the first turn,
        # 39 deg yaw error for the whole tour -> SLAM built a rotated map and never recovered)
        G = None
        if hasattr(self.imu, "drain_ring"):
            try:
                gs, _, ts, self._ring_since = self.imu.drain_ring(getattr(self, "_ring_since", 0), max_n=400)
                if len(gs): G = gs.astype(float)
            except Exception: G = None
        if G is None: G = np.asarray(self.imu.gyro, float)[None]
        g = (G.mean(0) - TG.GYRO_BIAS) * TG.GYRO_SCALE
        return float((TG.IMU_R @ g)[2])                    # gyro (optical) -> base z via the twin IMU frame

    HB_HZ = 20.0; HB_FRESH_S = 0.3

    def drive(self, v, w):
        if not self.wb: return
        with self._drv_lock:
            self._cmd = (float(v), float(w), time.monotonic()); self._hb_zeroed = False
            self.wb.twist(float(v), float(w))

    def _hb_loop(self):
        dt = 1.0 / self.HB_HZ
        while self._run:
            time.sleep(dt)
            with self._drv_lock:
                v, w, t = self._cmd
                age = time.monotonic() - t
                try:
                    if age < self.HB_FRESH_S:
                        if age >= dt * 0.5: self.wb.twist(v, w); self.hb_stats["sent"] += 1     # skip if drive() just sent
                    elif not self._hb_zeroed:
                        self.wb.twist(0.0, 0.0); self._hb_zeroed = True; self.hb_stats["zeroed"] += 1
                except Exception:
                    pass

    def close(self):
        self._run = False
        if self.wb:
            try:
                with self._drv_lock: self._cmd = (0.0, 0.0, 0.0); self.wb.twist(0.0, 0.0)
                self.wb.shutdown()
            except Exception: pass
