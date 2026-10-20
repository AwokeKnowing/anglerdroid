"""Offline check of nav/escape.py on a replayed stuck state: python emu/tools/stuckdiag/escape_offline.py <log> <t_nav_s>
Points = replayed Slam2 map (occupied cells within 1 m) + the live scan at that tick (as footprint_sweep.py).  Prints the
plan, then re-checks the executed sweep independently (footprint clearance every 1 cm / 1 deg) and the end pose."""
import math, sys, time
import numpy as np
sys.path[:0] = ['.', 'src']
exec(open('emu/tools/stuckdiag/footprint_sweep.py').read().split("print(\"occupied")[0])
from nav import escape as E
from nav.safety_core import sd
Pall = np.concatenate([W, live]); p = (x, y, th)
print("start pose %s  footprint clr %.3f  axle %.3f (R_ROTATE %.3f)" % (np.round(p, 3), sd(E.to_robot(Pall, p)).min(), np.hypot(*(Pall - p[:2]).T).min(), TG.R_ROTATE))
segs, info = E.plan(Pall, p); print("plan", segs, info)
if segs:
    q = p; mn = 9
    for k, a in segs:
        n = max(1, int(abs(a) / (0.01 if k == "lin" else math.radians(1))))
        for i in range(1, n + 1):
            r = E._step(q, k, a * i / n); mn = min(mn, sd(E.to_robot(Pall, r)).min())
        q = E._step(q, k, a)
    print("executed sweep min clr %.3f, end %s axle %.3f -> %s" % (mn, np.round(q, 3), np.hypot(*(Pall - q[:2]).T).min(), "OK" if mn >= E.CLEAR - 1e-6 and np.hypot(*(Pall - q[:2]).T).min() >= TG.R_ROTATE else "FAIL"))
