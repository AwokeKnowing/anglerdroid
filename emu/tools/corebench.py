"""python corebench.py <label> <kind core|env> <log> <tag>... : per-variant loop/age/event table over the goal windows.
core: KEVIN_SAFETY_CORE_LOG rows; env: KEVIN_SAFETY_LOG rows of the old in-brain Envelope (PR #62).
period = decision-to-decision; age = time.monotonic() at decision - frame capture (core: after the transform; env:
after hw.drive).  Events = transitions into a non-pass state while a motion is commanded."""
import collections, json, sys
import numpy as np
label, kind, log, tags = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
L = [json.loads(l) for l in open(log) if l.strip()]; T = np.array([r[0] for r in L])
per, age, mins, ev, ticks, over, over_t = [], [], 0.0, collections.Counter(), 0, 0, 0.0
PASS = ("ok", "escape", "idle", "cmd_stale")
for tag in tags:
    for r in json.load(open('bench_%s.json' % tag)):
        R = L[np.searchsorted(T, r['t0m']):np.searchsorted(T, r['t1m'])]; prev = True
        for k, x in enumerate(R):
            ticks += 1
            if kind == "core":
                p, a, vc, wc, code = x[1], x[2], x[4], x[5], x[10]
            else:
                p = (x[0] - R[k - 1][0]) if k else None; a = x[2] - x[1]; vc, wc, code = x[3], x[4], x[9]
                code = {"ok": "ok", "limited": "limit", "blocked": "blocked", "no_depth": "no_depth"}.get(code, code)
            if p is not None and p < 1.0: per.append(p)
            if a is not None and a < 5: age.append(a)
            if a is not None and a > 0.1: over += 1; over_t += min(p or 1 / 30, 0.2)
            mv = abs(vc) > 1e-3 or abs(wc) > 1e-3
            if mv: mins += min(p or 1 / 30, 0.2) / 60
            ok = code in PASS
            if mv and not ok and prev: ev[code] += 1
            prev = ok or not mv
per, age = np.array(per) * 1e3, np.array(age) * 1e3
n = sum(ev.values())
print("| %s | ticks %d | period ms p50 %.1f p95 %.1f max %.1f | age@decision ms p50 %.0f p95 %.0f max %.0f | age>100 ms: %d ticks (%.1f%%) %.1f s | moving %.1f min | events %d = %.2f/min %s |" % (
    label, ticks, *np.percentile(per, [50, 95, 100]), *np.percentile(age, [50, 95, 100]), over, 100 * over / max(ticks, 1), over_t, mins, n, n / max(mins, 1e-6),
    {k: "%d (%.2f/min)" % (v, v / max(mins, 1e-6)) for k, v in sorted(ev.items())}))
