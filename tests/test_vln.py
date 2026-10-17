"""nav/vln.py against a fake LightNav: map-frame waypoints use the pose at image time, farthest -> goal, state + jsonl, stop flag.
python tests/test_vln.py"""
import glob, json, math, os, sys, tempfile, time
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]
os.environ["KEVIN_VLN_LOG"] = tempfile.mkdtemp(); os.environ["KEVIN_RGB_DEV"] = "none"; os.environ.pop("KEVIN_HAL", None)
import fake_lightnav as F
from nav import vln as VL
VL.LOGDIR = os.environ["KEVIN_VLN_LOG"]
srv, n = F.lightnav(18050, delay=0.15, stop_after=4)
b = F.FakeBrain(pose=(1.0, 2.0, math.pi / 2)); v = VL.VLN("ws://127.0.0.1:18050", b)
v.instruct("go to the dog")
while n["next"] < 1: time.sleep(0.005)
b._pose = [5.0, 5.0, 0.0]          # robot moves while the first call is in flight: transform must use the image-time pose
t0 = time.time()
while v.text and time.time() - t0 < 10: time.sleep(0.05)
S = v.state(); first = S["sets"][0]
print("calls", S["stats"], "active", S["active"], "sets", len(S["sets"]))
exp = [1.0 - 0.02 * 100, 2.0 + 1.5]                    # pose (1,2,90deg): forward 1.5 -> +y, left 0.2*... -> -x
print("first set farthest", first["wp_map"][-1], "goal", first["goal"], "expected", [round(e, 3) for e in exp], "cmd", b.cmds[0])
assert abs(first["wp_map"][-1][0] - exp[0]) < 1e-6 and abs(first["wp_map"][-1][1] - exp[1]) < 1e-6 and len(first["wp_map"]) == 10
assert b.cmds[0] == "go to %.3f %.3f" % tuple(exp)
assert S["sets"][1]["wp_map"][-1] == [round(5.0 + 1.5, 3), round(5.0 + 2.0, 3)]      # later calls use the moved pose
assert S["last"]["stop"] and b.cmds[-1] == "stop" and not S["active"] and S["stats"]["calls"] == 5 and len(S["sets"]) == 5
assert all(c.startswith("go to ") or c == "stop" for c in b.cmds)                       # VLN only proposes goals (or stop)
L = [json.loads(l) for f in glob.glob(VL.LOGDIR + "/*.jsonl") for l in open(f)]
print("jsonl records", len(L), "keys", sorted(L[0]))
assert len(L) == 5 and L[0]["instruction"] == "go to the dog" and L[0]["goal"] == first["goal"] and L[0]["ms"] > 100
# unreachable server -> error recorded, inactive, no motion
b2 = F.FakeBrain(); v2 = VL.VLN("ws://127.0.0.1:1", b2); v2.instruct("x"); time.sleep(1.0)
print("dead server:", v2.state()["error"], b2.cmds)
assert v2.state()["error"] and not v2.state()["active"] and b2.cmds == []
print("camera", S["camera"], "fps %.2f (target %.1f)" % (S["fps"], S["target_fps"]))
assert 3.0 < S["fps"] <= 4.4
print("PASS")
