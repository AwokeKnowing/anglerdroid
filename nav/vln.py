"""LightNav-0 (or any server speaking its WebSocket protocol: login/reset/next) as a high-level goal source.

ask <instruction> -> stream MAST RGB frames (USB16MP01 at the mast tip, pitch 0, ~0.98 m: inside LightNav-0's training
camera range; NOT the 27-deg-pitched D435i) at a fixed FPS (4, the model's training rate: the server keeps the frame
history, so frames must arrive at a steady clock), get its (10, 3) robot-frame waypoints [forward_m, lateral_m(+left),
yaw_rad(+ccw)] (cumulative, relative to the pose at that frame), and hand the planner the chunk pose the robot is meant
to reach BEFORE THE NEXT PREDICTION replaces it (receding horizon).  The model replans every frame and the rest of the
chunk is look-ahead only (LightNav docs/DEPLOYMENT.md: "execute the first waypoint ... the remaining rows are only a
look-ahead"; robot_deploy/vln_mpc tracks the rows at 0.1 s each).  Handing over the far end instead committed the
robot to the whole ~1 s chunk every ~0.4 s: a turn-in-place chunk is 150 deg, so the robot spun 150 deg per call and
the model's frequent left/right flips became circling.  The 30 Hz loop, MPPI and the safety envelope are unchanged:
the VLM only ever proposes goals.  'stop' (or the model's stop flag) ends it.

state() exposes the last calls for the viewer (waypoints in the map frame, using the pose at the frame the image was
taken); every call is appended to ~/.kevin/emu/vln_log/<session>.jsonl for offline evaluation."""
from __future__ import annotations
import base64, collections, json, math, os, threading, time
import numpy as np
import cv2

LOGDIR = os.path.expanduser(os.environ.get("KEVIN_VLN_LOG", "~/.kevin/emu/vln_log"))
FPS = float(os.environ.get("KEVIN_VLN_FPS", "4"))
ROW_DT = 0.1                    # s per chunk row (LightNav robot_deploy vln_mpc WAYPOINT_DT_S: 10 rows = 1 s)


def commit_row(n_rows, period_s):
    """Index of the chunk row the robot should reach before the next prediction (period_s from now)."""
    return int(min(n_rows - 1, max(0, math.ceil(period_s / ROW_DT - 1e-9) - 1)))


class MastCam:
    """Mast RGB source.  sim: shm kevin_mast (emu/world.py);  real: the USB webcam (cameras.find_rgb_device / KEVIN_RGB_DEV).
    Falls back to the RS2 colour stream (flagged in state) only if no mast camera exists."""
    def __init__(self, hw):
        self.hw = hw; self.kind = None; self.blk = None; self.cap = None; self.err = None

    def _open(self):
        if os.environ.get("KEVIN_HAL") == "sim":
            try:
                from emu import shm
                m = shm.read_info(wait_s=0.5)["mast"]; self.blk = shm.Block(m["shm"], m["layout"]); self.kind = "mast_sim"; self.meta = m; return
            except Exception as e: self.err = "no sim mast shm: %r" % e
        else:
            try:
                from cameras import find_rgb_device
                dev = find_rgb_device()
                if dev is not None:
                    self.cap = cv2.VideoCapture(dev); self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # newest frame, not a stale buffered one
                    if self.cap.isOpened(): self.kind = "mast_usb"; return
                self.err = "no USB mast camera"
            except Exception as e: self.err = "mast camera: %r" % e
        self.kind = "rs2_fallback"

    def grab(self):
        """-> RGB uint8 (H, W, 3) or None."""
        if self.kind is None: self._open()
        if self.kind == "mast_sim": return self.blk.read()[2]["color"]
        if self.kind == "mast_usb":
            from cameras import mast_upright
            ok, f = self.cap.read(); return mast_upright(f[..., ::-1]) if ok else None   # mounted upside down
        _, _, _, color, *_ = self.hw.rs2_snapshot(); return color


def to_map(pose, wp):
    """robot-frame SE(2) waypoints (forward, left[, yaw]) at pose (x, y, th) -> map-frame [[x, y], ...]."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return [[round(pose[0] + c * w[0] - s * w[1], 3), round(pose[1] + s * w[0] + c * w[1], 3)] for w in wp]


class VLN:
    def __init__(self, url, brain, lookahead=None, connect=None):
        self.url = url; self.b = brain; self.text = None; self.stats = dict(calls=0, last_ms=0.0, mean_ms=0.0, errors=0)
        self.last = None; self.sets = collections.deque(maxlen=5); self.error = None; self._ev = threading.Event(); self._lock = threading.Lock()
        self._connect = connect; self._log = None; self.inflight = None; self.cam = MastCam(brain.hw); self.fps = 0.0
        threading.Thread(target=self._loop, daemon=True, name="vln").start()

    def instruct(self, text):
        self.text = text; self._ev.set(); self.b.log("VLN: '%s'" % text)

    def cancel(self): self.text = None

    # Stop vs late answer (2026-10-20): every prediction request is tagged with the brain's goal epoch at the moment it is
    # SENT (Brain.epoch, bumped by every stop / bump / core-alarm cancel).  An answer whose request predates the latest
    # stop is dropped here, and the command it would have issued carries the same epoch so Brain._apply drops it too if a
    # stop lands between this check and the command being applied (the brain's pending queue).  No timeout involved.
    def _epoch(self): return getattr(self.b, "epoch", 0)

    def state(self):
        with self._lock:
            return dict(url=self.url, camera=self.cam.kind, camera_err=self.cam.err, fps=round(self.fps, 2), target_fps=FPS, instruction=self.text, active=bool(self.text), inflight=self.inflight, epoch=self._epoch(), error=self.error, stats=dict(self.stats),
                        last=self.last, sets=list(self.sets))

    def _jsonl(self, rec):
        try:
            if self._log is None:
                os.makedirs(LOGDIR, exist_ok=True); self._log = open(os.path.join(LOGDIR, time.strftime("%Y%m%d-%H%M%S") + ".jsonl"), "a")
            self._log.write(json.dumps(rec) + "\n"); self._log.flush()
        except Exception: pass

    def _loop(self):
        if self._connect is None: from websockets.sync.client import connect
        else: connect = self._connect
        while True:
            self._ev.wait(); self._ev.clear()
            try:
                with connect(self.url, max_size=None) as ws:
                    rq = lambda a, d: (ws.send(json.dumps(dict(action=a, data=d))), json.loads(ws.recv()))[1]["data"]
                    rq("login", dict(clientId="kevin")); rq("reset", {}); seq = 0; self.error = None
                    # frames are CAPTURED on a fixed 4 fps clock (own thread, pose at image time) and queued; the socket
                    # sends them in order.  A prediction (~300-400 ms on i777) spans >1 frame period, so queued older
                    # frames go buffer-only (instruction "") and only the newest one asks for a prediction: the model's
                    # history stays evenly spaced at FPS even though predictions run slower.
                    q = collections.deque(maxlen=8); run = [True]; tlast = None; period = 0.0
                    def capture():
                        tick = time.monotonic(); tprev = None
                        while run[0] and self.text:
                            dt = tick - time.monotonic()
                            if dt > 0: time.sleep(dt)
                            tick = max(tick + 1.0 / FPS, time.monotonic() - 0.5 / FPS)
                            color = self.cam.grab()
                            if color is None: continue
                            pose = tuple(float(v) for v in self.b.navpose())
                            ok, jpg = cv2.imencode(".jpg", color[..., ::-1] if color.ndim == 3 else color, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            tn = time.monotonic()
                            if tprev: self.fps = (0.8 * self.fps + 0.2 / max(tn - tprev, 1e-3)) if self.fps else 1.0 / max(tn - tprev, 1e-3)
                            tprev = tn; q.append((jpg, pose, tn))
                    threading.Thread(target=capture, daemon=True, name="vln-cap").start()
                    try:
                      while self.text:
                        if not q: time.sleep(0.01); continue
                        jpg, pose, tcap = q.popleft(); seq += 1; instr = self.text
                        if q:                                                     # a newer frame is waiting: buffer only
                            rq("next", dict(seq=seq, image=base64.b64encode(jpg.tobytes()).decode(), instruction="")); self.stats["buffered"] = self.stats.get("buffered", 0) + 1; continue
                        t0 = time.monotonic(); ep = self._epoch(); self.inflight = (seq, ep)
                        d = rq("next", dict(seq=seq, image=base64.b64encode(jpg.tobytes()).decode(), instruction=instr))
                        ms = (time.monotonic() - t0) * 1e3; self.inflight = None
                        if d.get("rc", 0) != 0: raise RuntimeError("lightnav rc=%s %s" % (d.get("rc"), d.get("msg")))
                        raw = (d.get("actions") or {}).get("actions") or []
                        wp = np.asarray(raw, float); wp = wp.reshape(-1, wp.shape[-1]) if wp.size else np.zeros((0, 2))
                        stop = bool(d.get("stop")) or len(wp) == 0
                        wmap = to_map(pose, wp.tolist()); goal = None if stop else wmap[-1]
                        if tlast: period = 0.8 * period + 0.2 * min(2.0, time.monotonic() - tlast) if period else min(2.0, time.monotonic() - tlast)
                        tlast = time.monotonic(); k = commit_row(len(wp), period or 1.0 / FPS) if len(wp) else 0
                        rec = dict(t=time.time(), seq=seq, age_ms=round((time.monotonic() - tcap) * 1e3, 1), instruction=instr, ms=round(ms, 1), server_ms=d.get("latency_ms"), cam=self.cam.kind, fps=round(self.fps, 2), visible=d.get("visible"), pose=[round(v, 3) for v in pose],
                                   wp=wp.round(3).tolist(), wp_map=wmap, goal=goal, stop=stop, commit_row=k, period_s=round(period, 3))
                        with self._lock:
                            self.stats["calls"] += 1; self.stats["last_ms"] = round(ms, 1); self.stats["period_s"] = round(period, 3)
                            self.stats["mean_ms"] = round(self.stats["mean_ms"] + (ms - self.stats["mean_ms"]) / self.stats["calls"], 1)
                            self.last = rec; self.sets.append(dict(seq=seq, t=rec["t"], wp_map=wmap, goal=goal, stop=stop, commit=None if stop else wmap[k]))
                        late = ep != self._epoch(); rec["late"] = late
                        self._jsonl(rec)
                        if late:                       # issued before the latest stop: never acted on (not even the model's stop)
                            with self._lock: self.stats["late_dropped"] = self.stats.get("late_dropped", 0) + 1
                            self.b.log("VLN: dropped a late answer (seq %d, request epoch %d < %d)" % (seq, ep, self._epoch())); continue
                        if stop:
                            self.b.command("stop"); self.b.log("VLN: model says stop"); self.text = None; break
                        # proposal only (planner + MPPI + safety unchanged).  Receding horizon: the goal is the chunk pose
                        # at the time the next prediction is due (measured prediction period), with the model's heading
                        # there; a turn-in-place prefix (translation < 0.15 m) becomes 'face' so the turn is not lost.
                        f, l, yw = (list(wp[k]) + [0.0, 0.0, 0.0])[:3]; hd = math.degrees(pose[2] + yw); gk = wmap[k]
                        if math.hypot(f, l) < 0.15:
                            if abs(yw) > 0.2: self.b.command("face %.1f" % hd, epoch=ep)
                        else: self.b.command("go to %.3f %.3f %.1f" % (gk[0], gk[1], hd), frame="nav", epoch=ep)
                    finally: run[0] = False
            except Exception as e:
                with self._lock: self.stats["errors"] += 1; self.error = repr(e)[:200]
                self.b.log("VLN error: %r" % e); self.text = None
