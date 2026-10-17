"""Fake LightNav-0 WS server (login/reset/next) + fake brain, for box tests of nav/vln.py and the viewer overlay."""
import json, math, threading, time
import numpy as np
from websockets.sync.server import serve

def lightnav(port, delay=0.05, stop_after=None):
    """robot-frame waypoints: 10 points along a gentle left arc, 0.15 m apart.  stop=True after `stop_after` calls."""
    n = {"next": 0}
    def h(ws):
        for m in ws:
            d = json.loads(m); a = d["action"]
            if a == "next":
                n["next"] += 1; time.sleep(delay); assert d["data"]["image"] and d["data"]["instruction"]
                wp = [[0.15 * i, 0.02 * i * i, 0.04 * i] for i in range(1, 11)]
                stop = stop_after is not None and n["next"] > stop_after
                ws.send(json.dumps(dict(action=a, data=dict(actions=dict(actions=[] if stop else wp), stop=stop))))
            else: ws.send(json.dumps(dict(action=a, data=dict(ok=True))))
    srv = serve(h, "127.0.0.1", port, max_size=None); threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, n

class FakeBrain:
    def __init__(self, pose=(1.0, 2.0, math.pi / 2)):
        self._pose = list(pose); self.cmds = []; self.logs = []
        b = self
        class HW:
            def rs2_snapshot(self): return (0, 0.0, np.zeros((48, 64), np.uint16), np.full((48, 64, 3), 128, np.uint8), None, None)
        class Slam:
            def pose(self): return tuple(b._pose)
        self.hw = HW(); self.slam = Slam()
    def command(self, t): self.cmds.append(t)
    def log(self, s): self.logs.append(s)
