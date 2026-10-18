"""python lattable.py <label> <corelog> <tag>... : frame-latency table of the safety core over goal_bench windows,
split goals (kitchen/bedroom/couch) vs VLN ('ask ...').  Core log columns: 1 period, 2 age1, 3 age2 (at decision,
after the transform), 4/5 v_c/w_c, 10 code, 19 per1, 20 per2 (interval between the capture times of consecutive frames
the core used; > 1.5/30 s = missed beat -> inop_cadence)."""
import json, sys, numpy as np
label, log, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
L = [json.loads(l) for l in open(log) if l.strip()]; T = np.array([r[0] for r in L])
W = {"goals": [], "vln": []}
for tag in tags:
    for r in json.load(open('bench_%s.json' % tag)):
        W["vln" if r["cmd"].startswith("ask") else "goals"] += L[np.searchsorted(T, r['t0m']):np.searchsorted(T, r['t1m'])]
f = lambda x: np.array([np.nan if v is None else v for v in x], float) * 1e3
for k, R in W.items():
    if not R: continue
    per, a1, a2, p1, p2 = (f([x[i] for x in R]) for i in (1, 2, 3, 19, 20))
    mv = np.array([abs(x[4]) > 1e-3 or abs(x[5]) > 1e-3 for x in R]); am = np.fmax(a1, a2)
    q = lambda x: "%.0f/%.0f/%.0f" % (np.nanpercentile(x, 50), np.nanpercentile(x, 95), np.nanmax(x))
    pc = lambda m: "%.2f%%" % (100.0 * np.mean(m))
    ev = {}; prev = True
    for x, m in zip(R, mv):
        ok = x[10] in ("ok", "escape", "idle", "cmd_stale")
        if m and not ok and prev: ev[x[10]] = ev.get(x[10], 0) + 1
        prev = ok or not m
    print("| %s %s | ticks %d (moving %d) | period %s | age1 %s | age2 %s | age>100 all %s moving %s | missed RS1 %s RS2 %s (moving %s / %s) | stops %s |" % (
        label, k, len(R), mv.sum(), q(per), q(a1), q(a2), pc(am > 100), pc(am[mv] > 100) if mv.any() else "-",
        pc(p1 > 50), pc(p2 > 50), pc(p1[mv] > 50) if mv.any() else "-", pc(p2[mv] > 50) if mv.any() else "-", ev))
