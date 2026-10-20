"""Stop vs late LightNav answer (2026-10-20): ask -> request in flight -> user stop -> the late answer arrives.
The robot must stay idle: the VLN drops the answer (request epoch < brain epoch), and Brain._apply drops any command
already queued with a stale epoch (stop landed between the VLN check and the command being applied).
python tests/test_vln_stop_race.py"""
import math, os, sys, tempfile, time
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]
os.environ["KEVIN_VLN_LOG"] = tempfile.mkdtemp(); os.environ["KEVIN_RGB_DEV"] = "none"; os.environ.pop("KEVIN_HAL", None)
import fake_lightnav as F
from nav import vln as VL
VL.LOGDIR = os.environ["KEVIN_VLN_LOG"]

# 1) VLN level: answer delayed 1.0 s, user stop 0.2 s after the request went out
srv, n = F.lightnav(18051, delay=1.0)
b = F.FakeBrain(pose=(0.0, 0.0, 0.0)); v = VL.VLN("ws://127.0.0.1:18051", b)
v.instruct("find the kitchen")
t0 = time.time()
while v.inflight is None and time.time() - t0 < 10: time.sleep(0.002)
assert v.inflight is not None, "no request in flight"
time.sleep(0.2); b.stop(v); ts = time.time()
while n["next"] < 1 or v.inflight is not None:
    if time.time() - ts > 5: break
    time.sleep(0.01)
time.sleep(1.5)
S = v.state()
print("cmds", b.cmds, "stats", S["stats"], "last.late", S["last"] and S["last"].get("late"))
assert [c for c in b.cmds if not c.startswith("stop")] == [], b.cmds
assert S["stats"].get("late_dropped") == 1 and S["last"]["late"] and not S["active"]

# 1b) a new ask after the stop works normally (new epoch), and its answer is acted on
srv2, n2 = F.lightnav(18052, delay=0.1, stop_after=2)
b2 = F.FakeBrain(); v2 = VL.VLN("ws://127.0.0.1:18052", b2); b2.stop(v2); v2.instruct("go")
t0 = time.time()
while v2.text and time.time() - t0 < 10: time.sleep(0.05)
gos = [e for t, e in b2.ecmds if t.startswith("go to")]
print("after-stop ask: go-to epochs", gos, "late", v2.state()["stats"].get("late_dropped"))
assert gos and all(e == 1 for e in gos) and not v2.state()["stats"].get("late_dropped")

# 2) Brain._apply level (the real method): a go-to queued with the pre-stop epoch, then the stop, then the go-to
from nav import run as R
from nav import goals as G
br = R.Brain.__new__(R.Brain)
br.epoch = 0; br._goal_lock = __import__("threading").RLock(); br.vln = None; br.planner = None; br.core_alarm = None; br.bump_latched = None
br.mode = "idle"; br.goal = None; br.events = __import__("collections").deque(maxlen=50); br.msg = ""
br.navpose = lambda: (0.0, 0.0, 0.0)
c = G.parse("go to 1.0 2.0 0"); c["frame"] = "nav"; c["epoch"] = br.epoch
br._apply({"kind": "stop"}); br._apply(c)
print("brain after stop + stale go-to: mode", br.mode, "goal", br.goal, "epoch", br.epoch, "late_dropped", br.late_dropped, "log", br.msg)
assert br.mode == "idle" and br.goal is None and br.late_dropped == 1
print("PASS")
