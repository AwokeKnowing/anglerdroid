"""Live stop-vs-late-LightNav-answer check (2026-10-20).  Per trial: 'ask <text>', wait until a LightNav prediction
request is in flight (vln.inflight), wait `lag` s, 'stop', then watch 4 s.  Pass = mode idle at every sample, no new
goal, robot (odom pose) still, and the in-flight answer counted as vln.stats.late_dropped (not acted on).
python emu/tools/stoprace_live.py [trials] [text]"""
import json, math, random, sys, time, urllib.request as u
API = "http://localhost:8090/api/"
def cmd(t): return u.urlopen(u.Request(API + "command", data=json.dumps({"text": t}).encode(), headers={"Content-Type": "application/json"}), timeout=10).read()
def st(): return json.load(u.urlopen(API + "state", timeout=5))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5; TEXT = sys.argv[2] if len(sys.argv) > 2 else "find the kitchen"
ok_all = True; rng = random.Random(0)
for k in range(N):
    s0 = st(); ld0 = s0["vln"]["stats"].get("late_dropped", 0); ep0 = s0["vln"]["epoch"]; cmd("ask " + TEXT); t0 = time.monotonic()
    while True:
        s = st()
        if s["vln"]["inflight"] is not None or time.monotonic() - t0 > 20: break
        time.sleep(0.005)
    if s["vln"]["inflight"] is None: print("trial %d: no request in flight" % k); ok_all = False; continue
    lag = rng.uniform(0.0, 0.25); time.sleep(lag); s1 = st(); cmd("stop"); ts = time.monotonic(); p0 = s1["pose"]; modes = set(); maxd = 0.0
    calls_at_stop = s1["vln"]["stats"]["calls"]
    while time.monotonic() - ts < 4.0:
        s = st(); dt = time.monotonic() - ts
        if dt > 0.15: modes.add(s["mode"]); maxd = max(maxd, math.hypot(s["pose"][0] - p0[0], s["pose"][1] - p0[1]))
        time.sleep(0.05)
    ld = s["vln"]["stats"].get("late_dropped", 0) - ld0; answered = s["vln"]["stats"]["calls"] - calls_at_stop
    ev = [e for e in s["events"][-8:] if "late" in str(e) or "stopped" in str(e)]
    ok = modes == {"idle"} and ld >= 1 and maxd < 0.05 and s["goal"] is None
    ok_all &= ok
    print("trial %d: inflight seq %s lag %.0f ms; after stop: modes %s, late answers dropped %d (answers %d), pose moved %.3f m (incl. coast), goal %s, epoch %d->%d  %s"
          % (k, s1["vln"]["inflight"], lag * 1e3, sorted(modes), ld, answered, maxd, s["goal"], ep0, s["vln"]["epoch"], "OK" if ok else "FAIL"))
    print("   events:", ev[-3:])
    time.sleep(1.0)
print("PASS" if ok_all else "FAIL")
