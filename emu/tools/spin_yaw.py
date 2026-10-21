"""python emu/tools/spin_yaw.py <tag> [n_turns_each_dir=2] : spin-in-place yaw tracking, SLAM / odom vs sim ground truth.
Sends 'turn left 360' n times then 'turn right 360' n times to nav (/api/command), samples nav /api/state (SLAM pose,
odom pose) and emu.world GT yaw at ~20 Hz.  Per turn: GT rotation, SLAM and odom rotation error (end) and max |error|
during the turn (both relative to the turn start).  Writes ~/m1_backup/dbg/spin/<tag>.json."""
import json, math, os, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from emu import gt
tag = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
def nav(): return json.load(urllib.request.urlopen("http://localhost:8090/api/state", timeout=5))
def cmd(t):
    r = urllib.request.Request("http://localhost:8090/api/command", data=json.dumps({"text": t}).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=30))
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
S = []
def sample():
    s = nav(); g = gt.state(); S.append([time.monotonic(), g["yaw"], s["pose"][2], s["odom"][2], s.get("mode")])
def unwrap(a):
    out = [a[0]]
    for x in a[1:]: out.append(out[-1] + wrap(x - out[-1]))
    return out
turns = []
for d in ["left"] * n + ["right"] * n:
    i0 = len(S); sample(); cmd("turn %s 360" % d); time.sleep(0.5); t_end = time.monotonic() + 40
    while time.monotonic() < t_end:
        sample(); time.sleep(0.05)
        if S[-1][4] == "idle" and time.monotonic() - S[i0][0] > 3: break
    time.sleep(1.0); sample(); turns.append((d, i0, len(S)))
cmd("stop")
G, SL, OD = (unwrap([r[k] for r in S]) for k in (1, 2, 3))
rows = []
for d, i0, i1 in turns:
    g = [x - G[i0] for x in G[i0:i1]]; s = [x - SL[i0] for x in SL[i0:i1]]; o = [x - OD[i0] for x in OD[i0:i1]]
    es = [math.degrees(b - a) for a, b in zip(g, s)]; eo = [math.degrees(b - a) for a, b in zip(g, o)]
    rows.append(dict(dir=d, gt_deg=round(math.degrees(g[-1]), 1), slam_err_end=round(es[-1], 2), slam_err_max=round(max(map(abs, es)), 2),
                     odom_err_end=round(eo[-1], 2), odom_err_max=round(max(map(abs, eo)), 2), secs=round(S[i1 - 1][0] - S[i0][0], 1)))
tot = dict(gt_deg=round(math.degrees(G[-1] - G[0]), 1), slam_err=round(math.degrees((SL[-1] - SL[0]) - (G[-1] - G[0])), 2), odom_err=round(math.degrees((OD[-1] - OD[0]) - (G[-1] - G[0])), 2))
for r in rows: print(r)
print("total", tot)
json.dump(dict(rows=rows, total=tot, samples=S), open(os.path.expanduser("~/m1_backup/dbg/spin/%s.json" % tag), "w"))
