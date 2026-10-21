"""Live map-carving check (sim): place a 30x30x70 cm box ahead of Kevin, let the map mark it (small turns), remove it,
and measure occupied map cells in the box footprint vs time; occupied cells elsewhere (walls) before/after.
usage: carve_test.py [dist_m=1.2] [lateral_m=0]"""
import json, math, sys, time, urllib.request, os
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from emu import gt
NAV = "http://localhost:8090"
start = json.load(open(os.path.expanduser("~/m1_backup/dbg/v2/start.json")))
def get(p): return urllib.request.urlopen(NAV + p, timeout=5).read()
def cmd(t): urllib.request.urlopen(urllib.request.Request(NAV + "/api/command", json.dumps({"text": t}).encode(), {"Content-Type": "application/json"}), timeout=5).read()
def occmap():
    m = json.loads(get("/api/map.json")); img = cv2.imdecode(np.frombuffer(get("/api/map.png"), np.uint8), 0)
    return m, np.flipud(img) == 20
def count(m, occ, cx, cy, half):
    res = (m["x1"] - m["x0"]) / occ.shape[1]
    c0, c1 = int((cx - half - m["x0"]) / res), int(math.ceil((cx + half - m["x0"]) / res))
    r0, r1 = int((cy - half - m["y0"]) / res), int(math.ceil((cy + half - m["y0"]) / res))
    return int(occ[r0:r1, c0:c1].sum())
def stats():
    s = json.loads(get("/api/state")); sl = s.get("slam") or {}
    return {k: sl.get(k) for k in ("carved_tot", "carve_ms", "ms")} if isinstance(sl, dict) else {}
d = float(sys.argv[1]) if len(sys.argv) > 1 else 1.2; lat = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
s = gt.state(); bx = s["x"] + d * math.cos(s["yaw"]) - lat * math.sin(s["yaw"]); by = s["y"] + d * math.sin(s["yaw"]) + lat * math.cos(s["yaw"])
bm = gt.to_map((bx, by, 0), start)
print("robot gt %.2f %.2f %.1fdeg  box world %.2f %.2f map %.2f %.2f" % (s["x"], s["y"], math.degrees(s["yaw"]), bx, by, bm[0], bm[1]))
m, occ = occmap(); B0 = count(m, occ, bm[0], bm[1], 0.19); T0 = int(occ.sum()) - B0
print("before place: box-area occ %d, other occ %d" % (B0, T0))
gt.ctl("obstacle", i=0, x=bx, y=by); time.sleep(1.0)
for t in ("turn left 12", "turn right 24", "turn left 12"): cmd(t); time.sleep(2.5)
cmd("stop"); time.sleep(1.5)
m, occ = occmap(); B1 = count(m, occ, bm[0], bm[1], 0.19); T1 = int(occ.sum()) - B1
print("box marked: box-area occ %d (footprint 15x15=225 cells, visible face less), other occ %d  %s" % (B1, T1, stats()))
gt.ctl("clear_obstacles"); t0 = time.time(); series = []
while time.time() - t0 < 6.0:
    m, occ = occmap(); series.append((round(time.time() - t0, 1), count(m, occ, bm[0], bm[1], 0.19))); time.sleep(0.5)
print("removed (standing still): box-area occ vs t:", series)
for t in ("turn left 12", "turn right 24", "turn left 12"): cmd(t); time.sleep(2.5)
cmd("stop"); time.sleep(1.5)
m, occ = occmap(); B2 = count(m, occ, bm[0], bm[1], 0.19); T2 = int(occ.sum()) - B2
print("after small turns: box-area occ %d, other occ %d (before place %d)  %s" % (B2, T2, T0, stats()))
json.dump(dict(box_map=bm, B0=B0, T0=T0, B1=B1, T1=T1, series=series, B2=B2, T2=T2), open(os.path.expanduser("~/m1_backup/dbg/carve_test.json"), "w"))
