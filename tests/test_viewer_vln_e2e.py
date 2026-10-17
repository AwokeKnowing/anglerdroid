"""Box e2e: fake nav (FakeBrain + real nav/vln.py + fake LightNav) <- real viewer hub <- 3 headless WS clients.
Checks: shared 'Send all to LightNav' mode reaches every client; in that mode chat goes to VLN as 'ask <line>' and an exact
stop stays a stop; VLN waypoints (map + env frame) appear identically in all clients' state.
python tests/test_viewer_vln_e2e.py   (uses ports 18090/18095/18051 and /dev/shm/kevin_sim.json)"""
import json, math, os, subprocess, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, ROOT]
import fake_lightnav as F
from nav import vln as VL
from nav import goals as G
from websockets.sync.client import connect
F.lightnav(18051, delay=0.1)
b = F.FakeBrain(pose=(0.0, 0.0, 0.0)); b.v = VL.VLN("ws://127.0.0.1:18051", b); got = []
def brain_cmd(t):
    got.append(t); c = G.parse(t)
    if c["kind"] == "ask": b.v.instruct(c["text"])
    if c["kind"] == "stop": b.v.cancel()
    return c
b.command = lambda t: (got.append("vln:" + t), G.parse(t))[1]       # VLN's own goal proposals
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _s(self, o):
        x = json.dumps(o).encode(); self.send_response(200); self.send_header("Content-Length", str(len(x))); self.end_headers(); self.wfile.write(x)
    def do_GET(self): self._s(dict(pose=b._pose, mode="idle", places={"kitchen": [1, 1, 0]}, vln=b.v.state(), goal=None, path=None, events=[]))
    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"]))); self._s(brain_cmd(d["text"]))
srv = ThreadingHTTPServer(("127.0.0.1", 18090), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
json.dump(dict(kevin=[2.0, 3.0, math.pi / 2], start=[2.0, 3.0, math.pi / 2], places={}, env="fake", dog=None), open("/dev/shm/kevin_sim.json", "w"))
env = dict(os.environ, KEVIN_LLM_URL="http://127.0.0.1:9/v1")
vw = subprocess.Popen([sys.executable, os.path.join(ROOT, "emu/viewer/viewer.py"), "18095", "18090"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
try:
    time.sleep(3); cl = [connect("ws://127.0.0.1:18095/ws", max_size=None).__enter__() for _ in range(3)]
    def latest(ws, pred, t=5):
        end = time.time() + t
        while time.time() < end:
            m = json.loads(ws.recv(timeout=t))
            if m.get("type") == "state" and pred(m): return m
        raise AssertionError("timeout")
    assert all(latest(c, lambda m: m.get("send_all_vln") is False) for c in cl)
    t0 = time.time(); cl[0].send(json.dumps(dict(type="mode", send_all_vln=True, who="c0")))
    seen = [latest(c, lambda m: m.get("send_all_vln") is True) and time.time() - t0 for c in cl]
    print("mode ON propagated to 3 clients in %s ms" % [round(1e3 * s) for s in seen])
    cl[1].send(json.dumps(dict(type="chat", text="go to the kitchen", who="c1")))       # would be a place goal normally
    ms = [latest(c, lambda m: (m.get("vln") or {}).get("stats", {}).get("calls", 0) >= 3) for c in cl]
    print("nav got:", got[:3])
    assert got[0] == "ask go to the kitchen"
    V = [m["vln"] for m in ms]; s0 = V[0]["sets"][-1]
    print("vln instruction %r calls %d last %.0f ms stop %s; set: %d wp_map, goal %s, goal_env %s" % (V[0]["instruction"], V[0]["stats"]["calls"], V[0]["last"]["ms"], V[0]["last"]["stop"], len(s0["wp_map"]), s0["goal"], s0["goal_env"]))
    assert len(s0["wp_map"]) == 10 and len(s0["wp_env"]) == 10 and s0["goal"] == s0["wp_map"][-1]
    # nav pose (0,0,0) == env pose (2,3,90deg): map (1.5, 2.0) -> env (2-2.0, 3+1.5)
    assert abs(s0["goal_env"][0] - 0.0) < 1e-3 and abs(s0["goal_env"][1] - 4.5) < 1e-3
    assert "vln:go to 1.500 2.000 22.9" in got
    cl[2].send(json.dumps(dict(type="chat", text="halt", who="c2")))                   # exact stop in send-all mode -> stop
    latest(cl[0], lambda m: not (m.get("vln") or {}).get("active"))
    print("after halt nav got:", [g for g in got if not g.startswith("vln:")])
    assert "stop" in got and "ask halt" not in got
    cl[2].send(json.dumps(dict(type="mode", send_all_vln=False, who="c2")))
    assert all(latest(c, lambda m: m.get("send_all_vln") is False) for c in cl)
    log = latest(cl[0], lambda m: True)["log"]; print("shared log tail:", [r[1:] for r in log[-5:]])
    print("PASS")
finally:
    vw.terminate()
