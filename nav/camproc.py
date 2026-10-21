"""RS2 camera process (KEVIN_RS2_PROC=1, default with the safety core): owns the forward D435i -- depth/colour/IR AND
the motion module -- and publishes every frame over nav.bus: depth first to the safety core (sc_rs2), then the full
frame to the brain (sc_rs2img), IMU samples to a ring (sc_imu).

Why (2026-10-17 latency trace): the brain's RS2 pump thread shared the GIL with SLAM/MPPI/VLN; the safety core saw
20 % of RS2 frames miss the 30 Hz beat (frame interval > 50 ms) although the camera (sim) had delivered all of them.
In its own process the pump runs at the camera clock.  On the Orin this is also the layout librealsense's RSUSB backend
needs: one process owns the whole device (video + motion module)."""
from __future__ import annotations
import os, threading, time
import numpy as np


def cam_main(serial, stereo, parent_pid, cpus):
    if cpus:
        try: os.sched_setaffinity(0, {int(c) for c in cpus.split(",")})
        except Exception as e: print("rs2cam: affinity %s failed: %s" % (cpus, e), flush=True)
    import nav  # noqa: F401  (src on sys.path)
    from nav import bus
    from cameras import RSCamera
    from imu import IMUPipeline
    from cameras import RS_DEPTH_H, RS_DEPTH_W
    B = bus.open_all((RS_DEPTH_H, RS_DEPTH_W), create=False)
    cam = RSCamera(serial, compute_pointcloud=False, capture_ir=stereo, emitter=not stereo)
    # IMU = motion sensor of the device the video pipeline already runs (a second pipeline on it fails on the Orin's
    # RSUSB build and wedges the IMU until a hardware reset).  KEVIN_IMU_SHARED=0: old separate pipeline.
    shared = os.environ.get("KEVIN_IMU_SHARED", "1") == "1" and getattr(cam, "profile", None) is not None
    imu = IMUPipeline(serial, device=cam.profile.get_device() if shared else None)
    i = cam.depth_intrinsics()
    B["rs2info"].write(f=np.array([i["fx"], i["fy"], i["ppx"], i["ppy"], i["width"], i["height"]], float), set=1)
    print("rs2cam: pid %d cpus %s, %s, IMU %s" % (os.getpid(), sorted(os.sched_getaffinity(0)), serial, "ok" if imu.ok else "off"), flush=True)

    def imu_loop():
        N = bus.IMU_RING; t = np.zeros(N); g = np.zeros((N, 3), np.float32); a = np.zeros((N, 3), np.float32); w = 0; since = 0
        while True:
            if imu.ok and imu.grab():
                gs, acs, ts, since = imu.drain_ring(since, max_n=256)
                for k in range(len(ts)):
                    j = w % N; g[j] = gs[k]; a[j] = acs[k]; t[j] = ts[k]; w += 1
                B["imu"].write(t=t, g=g, a=a, w=w, ok=1)
            time.sleep(0.002)
    threading.Thread(target=imu_loop, daemon=True, name="rs2cam-imu").start()
    n = 0; ring = bus.Rs2Ring(B["rs2"])
    while True:
        if os.getppid() != parent_pid: return
        fs = cam.wait_frameset(200)
        if fs is None: continue
        t_recv = time.monotonic()
        if not cam.ingest(fs): continue
        cam.publish(); n += 1
        ring.push(cam.depth, cam.depth_stamp, t_recv)                                   # safety core first
        has_ir = cam.ir_left is not None and cam.ir_right is not None
        B["rs2img"].write(depth=cam.depth, color=cam.color, stamp=cam.depth_stamp, t_recv=t_recv, n=n, has_ir=int(has_ir),
                          **(dict(ir1=cam.ir_left, ir2=cam.ir_right) if has_ir else {}))


class Rs2Bus:
    """The brain's view of RS2 (same attributes the brain used on RSCamera): latest full frame from sc_rs2img."""
    def __init__(self, B):
        self.B = B; i = B["rs2info"].read()[1]["f"]
        self._intr = dict(fx=float(i[0]), fy=float(i[1]), ppx=float(i[2]), ppy=float(i[3]), width=int(i[4]), height=int(i[5]))
        self.depth = np.zeros((self._intr["height"], self._intr["width"]), np.uint16); self.color = np.zeros((480, 640, 3), np.uint8)
        self.ir_left = self.ir_right = None; self.depth_stamp = 0.0; self._seq = B["rs2img"].seq; self._out = None
    def depth_intrinsics(self): return dict(self._intr)
    def wait(self, timeout):
        """block until a new frame; returns True and updates the attributes (fresh arrays per frame)"""
        b = self.B["rs2img"]; t_end = time.monotonic() + timeout
        while True:
            s = b.seq
            if s != self._seq and not (s & 1): break
            if time.monotonic() > t_end: return False
            time.sleep(0.0005)
        self._seq, r = b.read()
        self.depth, self.color, self.depth_stamp = r["depth"], r["color"], float(r["stamp"][0])
        if int(r["has_ir"][0]): self.ir_left, self.ir_right = r["ir1"], r["ir2"]
        return True


class ImuBus:
    """IMUPipeline-compatible view of the RS2 camera process's IMU ring (grab / gyro / accel / timestamp / drain_ring)."""
    def __init__(self, B):
        self.B = B; self.ok = True        # until the camera process reports otherwise (grab)
        self.gyro = np.zeros(3, np.float32); self.accel = np.zeros(3, np.float32); self.timestamp = 0.0
        self._r = None; self._seq = -1
    def grab(self):
        b = self.B["imu"]
        if b.seq == self._seq: return False
        self._seq, self._r = b.read(out=self._r)
        w = int(self._r["w"][0])
        if w <= 0: return False
        j = (w - 1) % len(self._r["t"]); self.gyro = self._r["g"][j].copy(); self.accel = self._r["a"][j].copy(); self.timestamp = float(self._r["t"][j])
        self.ok = bool(int(self._r["ok"][0])); return True
    def drain_ring(self, since, max_n=16):
        if self._r is None: self.grab()
        if self._r is None: return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros(0), since
        w = int(self._r["w"][0]); N = len(self._r["t"]); avail = min(max_n, max(0, w - int(since)), N)
        if avail <= 0: return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros(0), w
        idx = np.arange(w - avail, w) % N
        return self._r["g"][idx].copy(), self._r["a"][idx].copy(), self._r["t"][idx].copy(), w
    def stop(self): pass
