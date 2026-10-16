"""LightNav-0 (or any server speaking its WebSocket protocol: login/reset/next) as a high-level goal source.

ask <instruction> -> stream RS2 color frames to the model (~1-2 Hz on i777), take its 10 SE(2) robot-frame
waypoints, and hand the farthest one to the planner as a map-frame goal.  The 30 Hz loop, MPPI and the
safety envelope are unchanged: the VLM only ever proposes goals.  'stop' (or the model's stop flag) ends it.

state() exposes the last calls for the viewer (waypoints in the map frame, using the pose at the frame the image was
taken); every call is appended to ~/.kevin/emu/vln_log/<session>.jsonl for offline evaluation."""
from __future__ import annotations
import base64, collections, json, math, os, threading, time
import numpy as np
import cv2

LOGDIR = os.path.expanduser(os.environ.get("KEVIN_VLN_LOG", "~/.kevin/emu/vln_log"))


def to_map(pose, wp):
    """robot-frame SE(2) waypoints (forward, left[, yaw]) at pose (x, y, th) -> map-frame [[x, y], ...]."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return [[round(pose[0] + c * w[0] - s * w[1], 3), round(pose[1] + s * w[0] + c * w[1], 3)] for w in wp]


class VLN:
    def __init__(self, url, brain, lookahead=None, connect=None):
        self.url = url; self.b = brain; self.text = None; self.stats = dict(calls=0, last_ms=0.0, mean_ms=0.0, errors=0)
        self.last = None; self.sets = collections.deque(maxlen=5); self.error = None; self._ev = threading.Event(); self._lock = threading.Lock()
        self._connect = connect; self._log = None
        threading.Thread(target=self._loop, daemon=True, name="vln").start()

    def instruct(self, text):
        self.text = text; self._ev.set(); self.b.log("VLN: '%s'" % text)

    def cancel(self): self.text = None

    def state(self):
        with self._lock:
            return dict(url=self.url, instruction=self.text, active=bool(self.text), error=self.error, stats=dict(self.stats),
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
                    while self.text:
                        instr = self.text
                        _, _, _, color, *_ = self.b.hw.rs2_snapshot()
                        ok, jpg = cv2.imencode(".jpg", color[..., ::-1] if color.ndim == 3 else color, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        pose = tuple(float(v) for v in self.b.slam.pose()); t0 = time.monotonic(); seq += 1
                        d = rq("next", dict(seq=seq, image=base64.b64encode(jpg.tobytes()).decode(), instruction=instr))
                        ms = (time.monotonic() - t0) * 1e3
                        raw = (d.get("actions") or {}).get("actions") or []
                        wp = np.asarray(raw, float); wp = wp.reshape(-1, wp.shape[-1]) if wp.size else np.zeros((0, 2))
                        stop = bool(d.get("stop")) or len(wp) == 0
                        wmap = to_map(pose, wp.tolist()); goal = None if stop else wmap[-1]
                        rec = dict(t=time.time(), seq=seq, instruction=instr, ms=round(ms, 1), pose=[round(v, 3) for v in pose],
                                   wp=wp.round(3).tolist(), wp_map=wmap, goal=goal, stop=stop)
                        with self._lock:
                            self.stats["calls"] += 1; self.stats["last_ms"] = round(ms, 1)
                            self.stats["mean_ms"] = round(self.stats["mean_ms"] + (ms - self.stats["mean_ms"]) / self.stats["calls"], 1)
                            self.last = rec; self.sets.append(dict(seq=seq, t=rec["t"], wp_map=wmap, goal=goal, stop=stop))
                        self._jsonl(rec)
                        if stop:
                            self.b.command("stop"); self.b.log("VLN: model says stop"); self.text = None; break
                        self.b.command("go to %.3f %.3f" % tuple(goal))       # proposal only: planner + MPPI + safety unchanged
                        time.sleep(0.05)
            except Exception as e:
                with self._lock: self.stats["errors"] += 1; self.error = repr(e)[:200]
                self.b.log("VLN error: %r" % e); self.text = None
