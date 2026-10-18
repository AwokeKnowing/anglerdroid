"""LightNav run with SLAM-vs-truth logging (sim).  usage: vln_drift.py <tag> "<instruction>" [secs=120]
Sends the instruction through the viewer chat (as James does), samples nav pose vs GT (map frame = nav start pose,
~/m1_backup/dbg/v2/start.json) at 2 Hz, writes ~/m1_backup/dbg/drift_<tag>.jsonl and prints a summary."""
import json, math, os, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from emu import gt
tag, text = sys.argv[1], sys.argv[2]; T = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
def nav(): return json.load(urllib.request.urlopen("http://localhost:8090/api/state", timeout=5))
def chat(t):
    r = urllib.request.Request("http://localhost:8095/chat", data=json.dumps({"text": t, "who": "drift"}).encode(), method="POST", headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=60)).get("reply")
st = json.load(open(os.path.expanduser("~/m1_backup/dbg/v2/start.json"))); c0 = gt.state()["collisions"]
out = open(os.path.expanduser("~/m1_backup/dbg/drift_%s.jsonl" % tag), "w")
print(text, "->", chat(text), flush=True); t0 = time.time(); E = []; path = 0.0; last = None; active_seen = False
while time.time() - t0 < T:
    time.sleep(0.5)
    try: n = nav(); s = gt.state()
    except Exception: continue
    g = gt.to_map((s["x"], s["y"], s["yaw"]), st); p = n["pose"][:3]
    ey = math.degrees((p[2] - g[2] + math.pi) % (2 * math.pi) - math.pi); exy = math.hypot(p[0] - g[0], p[1] - g[1])
    if last: path += math.hypot(g[0] - last[0], g[1] - last[1])
    last = g; E.append((ey, exy))
    sl = n.get("slam") or {}; a = (n.get("vln") or {}).get("active")
    out.write(json.dumps(dict(t=round(time.time() - t0, 2), pose=p, gt=g, ey=ey, exy=exy, path=path, mode=n["mode"], msg=n["msg"], vln=a, walls=sl.get("walls"), scale=sl.get("gyro_scale"), bias=sl.get("gyro_bias"), w=s.get("w"), v=s.get("v"), tm=time.monotonic())) + "\n"); out.flush()
    active_seen |= bool(a)
    if active_seen and not a and n["mode"] == "idle": break
s = gt.state(); n = nav()
ae = [abs(e[0]) for e in E]; xe = [e[1] for e in E]
print("drift %s: %.0f s, GT path %.1f m, |yaw err| max %.1f deg final %.1f deg, xy err max %.2f m final %.2f m, collisions %d, last msg: %s" % (
    tag, time.time() - t0, path, max(ae), ae[-1], max(xe), xe[-1], s["collisions"] - c0, n["msg"][:100]))
json.dump(dict(ctl=n.get('ctl'), timing=n.get('timing'), collisions=s["collisions"] - c0, msg=n["msg"]), open(os.path.expanduser("~/m1_backup/dbg/drift_%s_ctl.json" % tag), "w"))
