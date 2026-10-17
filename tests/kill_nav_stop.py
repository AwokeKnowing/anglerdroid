"""Sim acceptance: drive, kill nav.run mid-goal, measure time until the body (MuJoCo GT) stops.
python tests/kill_nav_stop.py TERM|KILL|STOP   (needs the live sim stack; restart nav afterwards)
TERM = clean shutdown path; KILL = process gone (only ODrive-emulator watchdog left); STOP = frozen process (all threads)."""
import json, os, signal, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from emu import shm
sig = sys.argv[1] if len(sys.argv) > 1 else "TERM"
gt = shm.Block("kevin_gt", [("pose", (12,), "f8")])
v = lambda: float(gt.read()[2]["pose"][4])
urllib.request.urlopen(urllib.request.Request("http://localhost:8090/api/command", json.dumps({"text": "forward 3"}).encode())).read()
t0 = time.time()
while time.time() - t0 < 5 and abs(v()) < 0.25: time.sleep(0.01)
time.sleep(0.4); v0 = v()
pid = int(subprocess.check_output(["pgrep", "-f", "nav.run --port 8090"]).split()[0])
tk = time.monotonic(); os.kill(pid, getattr(signal, "SIG" + sig))
tr = None
while time.monotonic() - tk < 5:
    if abs(v()) < 0.02: tr = time.monotonic() - tk; break
    time.sleep(0.005)
print("SIG%s at v=%.2f m/s: body stopped (|v|<0.02) after %s" % (sig, v0, "%.0f ms" % (1e3 * tr) if tr is not None else ">5 s"))
if sig == "STOP": os.kill(pid, signal.SIGKILL)
