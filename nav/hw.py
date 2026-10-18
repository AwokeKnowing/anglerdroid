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
        # safety core (nav/safety_core.py): separate process, owns the wheels; this process publishes camera frames to
        # it over nav.bus from dedicated pump threads and sends desired (v, w).  KEVIN_SAFETY_CORE=0: old in-process path.
        self.core = drive and os.environ.get("KEVIN_SAFETY_CORE", "1") == "1"
        self.rs1_intr = self.rs1.depth_intrinsics() if hasattr(self.rs1, "depth_intrinsics") else None
        if self.core:
            from nav import bus
            self.bus = bus.open_all((self.rs2_intr["height"], self.rs2_intr["width"]), create=True)
            self._rs1_cv = threading.Condition(); self._rs1_seq = 0; self._rs1_got = 0
            threading.Thread(target=self._rs1_pump, daemon=True, name="rs1").start()
            import multiprocessing as mp
            from nav.safety_core import core_main
            ctx = mp.get_context("spawn")
            self.core_proc = ctx.Process(target=core_main, name="safety_core", daemon=True,
                                         args=(self.rs1_intr, self.rs2_intr, (self.rs2_intr["height"], self.rs2_intr["width"]), os.getpid(), os.environ.get("KEVIN_SAFETY_CORE_LOG")))
            self.core_proc.start(); drive = False                    # the core creates WheelBase
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
            t_recv = time.monotonic()
            with self._rs2_lock:
                if self.rs2.ingest(fs):
                    self.rs2.publish(); self.rs2_seq += 1; self.rs2_stamp = self.rs2.depth_stamp
                    if getattr(self, "core", False): self.bus["rs2"].write(depth=self.rs2.depth, stamp=self.rs2_stamp, t_recv=t_recv)

    def _rs1_pump(self):
        """core mode: every RS1 frame goes to the safety core at once, independent of the brain loop's timing."""
        while self._run:
            fs = self.rs1.wait_frameset(200)
            if fs is None: continue
            t_recv = time.monotonic()
            with self._rs1_cv:
                if not self.rs1.ingest(fs): continue
                self.rs1.publish(); v = self.rs1.verts
                self.bus["rs1"].write(verts=v, n=len(v), stamp=self.rs1.depth_stamp, t_recv=t_recv)
                self._rs1_seq += 1; self._rs1_cv.notify_all()

    def rs2_snapshot(self):
        """(seq, stamp_s, depth u16, color rgb, ir_left, ir_right) — copies, safe across threads."""
        with self._rs2_lock:
            c = self.rs2
            return (self.rs2_seq, self.rs2_stamp, c.depth.copy(), c.color.copy(),
                    None if c.ir_left is None else c.ir_left.copy(), None if c.ir_right is None else c.ir_right.copy())

    def wait_rs1(self, timeout_ms=200):
        """Block on the RS1 camera clock (30 Hz). Returns (stamp_s, verts) or (None, None) on timeout."""
        if getattr(self, "core", False):                 # latest frame from the pump (skips frames if the brain is slow)
            with self._rs1_cv:
                if self._rs1_seq == self._rs1_got and not self._rs1_cv.wait_for(lambda: self._rs1_seq != self._rs1_got, timeout_ms * 1e-3):
                    return None, None
                self._rs1_got = self._rs1_seq
                return self.rs1.depth_stamp, self.rs1.verts.copy()
        fs = self.rs1.wait_frameset(timeout_ms)
        if fs is None or not self.rs1.ingest(fs):
            return None, None
        self.rs1.publish()
        return self.rs1.depth_stamp, self.rs1.verts

    def wheel_vels(self):
        if getattr(self, "core", False):
            _, r = self.bus["wheel"].read(["vl", "vr"]); return float(r["vl"][0]), float(r["vr"][0])
        return self.wb.get_wheel_velocities_mps() if self.wb else (0.0, 0.0)

    def core_state(self):
        """last safety-core decision as a dict (nav.safety_core.STATE_FIELDS), or None (no core)."""
        if not getattr(self, "core", False): return None
        from nav.safety_core import STATE_FIELDS, CODES
        f = self.bus["state"].read()[1]["f"]; d = dict(zip(STATE_FIELDS, [float(x) if math.isfinite(x) else None for x in f]))
        d["code"] = CODES[int(d["code"])] if d["code"] is not None and 0 <= d["code"] < len(CODES) else "?"; d["alive"] = self.core_proc.is_alive()
        return d

    def core_viz(self):
        if not getattr(self, "core", False): return None
        r = self.bus["viz"].read()[1]; n = int(r["n"][0]); return r["pts"][:n].copy(), r["bands"].reshape(2, 4).copy(), float(r["t"][0])

    def set_rs2_pose(self, T, R):
        if getattr(self, "core", False): self.bus["rs2pose"].write(T=np.asarray(T, float), R=np.asarray(R, float).ravel(), set=1)

    def yaw_rate(self):
        """Robot yaw rate (rad/s, CCW+) from the D435i gyro (optical frame) — up axis in optical = (0,-c,-s)."""
        if os.environ.get("KEVIN_NO_IMU") == "1": return None          # ablation: wheel-only yaw
        self.imu.grab()
        # mean of ALL gyro samples since the last call (200 Hz ring), not just the latest: a stalled loop
        # (CUDA warm-up, GC) during a spin otherwise lost the rotation (maze: 0.5 rad lost on the first turn,
        # 39 deg yaw error for the whole tour -> SLAM built a rotated map and never recovered)
        G = None; A = None
        if hasattr(self.imu, "drain_ring"):
            try:
                gs, acs, ts, self._ring_since = self.imu.drain_ring(getattr(self, "_ring_since", 0), max_n=400)
                if len(gs): G = gs.astype(float); A = acs.astype(float)
            except Exception: G = None
        if G is None: G = np.asarray(self.imu.gyro, float)[None]
        if A is not None and len(A): self.accel_batch = (TG.IMU_R @ A.T).T      # base-frame specific force (bump detector)
        g = (G.mean(0) - TG.GYRO_BIAS) * TG.GYRO_SCALE
        return float((TG.IMU_R @ g)[2])                    # gyro (optical) -> base z via the twin IMU frame

    def motor_currents(self):
        """(i_left, i_right) A, robot-forward sense, or None (no drive / not streamed)."""
        if getattr(self, "core", False):
            _, r = self.bus["wheel"].read(["il", "ir"]); l, rr = float(r["il"][0]), float(r["ir"][0])
            return None if (math.isnan(l) or math.isnan(rr)) else (l, rr)
        f = getattr(self.wb, "get_motor_currents", None) if self.wb else None
        r = f() if f else None
        return None if r is None else (r[0], r[1])

    HB_HZ = 20.0; HB_FRESH_S = 0.3

    def drive(self, v, w):
        if getattr(self, "core", False):
            self.bus["cmd"].write(v=float(v), w=float(w), t=time.monotonic()); return
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
        if getattr(self, "core", False):
            self.bus["cmd"].write(v=0.0, w=0.0, t=0.0); time.sleep(0.2)          # stale -> core zeroes
            try: self.core_proc.terminate(); self.core_proc.join(2.0)
            except Exception: pass
        if self.wb:
            try:
                with self._drv_lock: self._cmd = (0.0, 0.0, 0.0); self.wb.twist(0.0, 0.0)
                self.wb.shutdown()
            except Exception: pass
