"""LightNav-0 (or any server speaking its WebSocket protocol: login/reset/next) as a high-level goal source.

ask <instruction> -> stream RS2 color frames to the model (~1-2 Hz on i777), take its 10 SE(2) robot-frame
waypoints, and hand the farthest one to the planner as a map-frame goal.  The 30 Hz loop, MPPI and the
safety envelope are unchanged: the VLM only ever proposes goals.  'stop' (or the model's stop flag) ends it."""
from __future__ import annotations
import base64, json, math, threading, time
import numpy as np
import cv2


class VLN:
    def __init__(self, url, brain, lookahead=None):
        self.url = url; self.b = brain; self.text = None; self.stats = dict(calls=0, last_ms=0.0, mean_ms=0.0)
        self.last = None; self._ev = threading.Event()
        threading.Thread(target=self._loop, daemon=True, name="vln").start()

    def instruct(self, text):
        self.text = text; self._ev.set(); self.b.log("VLN: '%s'" % text)

    def cancel(self): self.text = None

    def _loop(self):
        from websockets.sync.client import connect
        while True:
            self._ev.wait(); self._ev.clear()
            try:
                with connect(self.url, max_size=None) as ws:
                    rq = lambda a, d: (ws.send(json.dumps(dict(action=a, data=d))), json.loads(ws.recv()))[1]["data"]
                    rq("login", dict(clientId="kevin")); rq("reset", {}); seq = 0
                    while self.text:
                        _, _, _, color, *_ = self.b.hw.rs2_snapshot()
                        ok, jpg = cv2.imencode(".jpg", color[..., ::-1] if color.ndim == 3 else color, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        pose = self.b.slam.pose(); t0 = time.monotonic(); seq += 1
                        d = rq("next", dict(seq=seq, image=base64.b64encode(jpg.tobytes()).decode(), instruction=self.text))
                        ms = (time.monotonic() - t0) * 1e3; self.stats["calls"] += 1; self.stats["last_ms"] = ms
                        self.stats["mean_ms"] += (ms - self.stats["mean_ms"]) / self.stats["calls"]
                        wp = np.asarray(d.get("actions", {}).get("actions", []), float); self.last = dict(wp=wp.tolist(), stop=d.get("stop"), ms=ms)
                        if d.get("stop") or len(wp) == 0:
                            self.b.command("stop"); self.b.log("VLN: model says stop"); self.text = None; break
                        f, l = wp[-1, 0], wp[-1, 1]                      # robot frame at the frame's pose
                        c, s = math.cos(pose[2]), math.sin(pose[2])
                        self.b.command("go to %.3f %.3f" % (pose[0] + c * f - s * l, pose[1] + s * f + c * l))
                        time.sleep(0.05)
            except Exception as e:
                self.b.log("VLN error: %r" % e); self.text = None
