"""LightNav-like aggressive yaw wiggle in place (sim): n x (drive 0 +W T, drive 0 -W T'), then still; SLAM/odom yaw
vs GT (map frame = nav start) measured while standing still (no sampling skew).  usage: wiggle_yaw.py <tag> [n=15] [W=2.5]"""
import json, math, os, sys, time, urllib.request, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from emu import gt
tag = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 15; W = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
def cmd(t): urllib.request.urlopen(urllib.request.Request("http://localhost:8090/api/command", json.dumps({"text": t}).encode(), {"Content-Type": "application/json"}), timeout=5).read()
def err():
    time.sleep(1.5); s = json.load(urllib.request.urlopen("http://localhost:8090/api/state", timeout=5)); g = gt.state()
    st = json.load(open(os.path.expanduser("~/m1_backup/dbg/v2/start.json"))); m = gt.to_map((g["x"], g["y"], g["yaw"]), st)
    f = lambda a: math.degrees((a - m[2] + math.pi) % (2 * math.pi) - math.pi)
    return round(f(s["pose"][2]), 2), round(f(s["odom"][2]), 2)
rnd = random.Random(1); E = [err()]; turned = 0.0
for k in range(n):
    for sgn in (1, -1):
        T = rnd.uniform(0.3, 0.9); cmd("drive 0 %.2f %.2f" % (sgn * W, T)); time.sleep(T + 0.15); turned += W * T
    if k % 5 == 4: E.append(err())
cmd("stop"); E.append(err())
print("wiggle %s: %d cycles at +-%.1f rad/s (~%.0f deg commanded rotation), yaw err (slam, odom) deg: %s" % (tag, n, W, math.degrees(turned), E))
json.dump(dict(tag=tag, n=n, W=W, E=E), open(os.path.expanduser("~/m1_backup/dbg/spin/wiggle_%s.json" % tag), "w"))
