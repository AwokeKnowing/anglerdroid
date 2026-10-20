"""SE(2) footprint search from the L15 stuck pose: is there ANY short manoeuvre (true footprint, 2 cm min clearance) to a
spin-free pose (axle clearance >= R_ROTATE)?  Points: replayed Slam2 map + the live obstacle scan."""
import numpy as np, math, json, glob, sys, heapq
sys.path[:0] = ['.', 'src']
exec(open('emu/tools/stuckdiag/footprint_sweep.py').read().split("print(\"occupied")[0])
from nav.governor import sd
import twin_geometry as TG
def rob(P, p): c, s_ = math.cos(p[2]), math.sin(p[2]); d = P - np.array(p[:2]); return np.stack([c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1]], 1)
Pall = np.concatenate([W, live]); CL = 0.02
def clr(p): return sd(rob(Pall, p)).min()
def axle(p): return np.hypot(Pall[:, 0] - p[0], Pall[:, 1] - p[1]).min()
start = (x, y, th); print("start clr %.3f axle %.3f R_ROTATE %.3f" % (clr(start), axle(start), TG.R_ROTATE))
moves = [("fwd", 0.04, 0), ("rev", -0.04, 0), ("left", 0, math.radians(10)), ("right", 0, -math.radians(10))]
key = lambda p: (round(p[0] / 0.02), round(p[1] / 0.02), round(((p[2] + math.pi) % (2 * math.pi)) / math.radians(10)) % 36)
Q = [(0.0, 0, start, [])]; seen = {key(start)}; n = 0
while Q and n < 60000:
    c, _, p, path = heapq.heappop(Q); n += 1
    if axle(p) >= TG.R_ROTATE: print("FOUND after %d expansions, cost %.2f: %s" % (n, c, " ".join(path))); print("end pose", np.round(p, 3)); break
    for nm, d, a in moves:
        q = (p[0] + d * math.cos(p[2]), p[1] + d * math.sin(p[2]), p[2] + a)
        if clr(q) < CL: continue
        k = key(q)
        if k in seen: continue
        seen.add(k); heapq.heappush(Q, (c + (0.04 if d > 0 else 0.08 if d < 0 else 0.02), n * 10 + len(Q), q, path + [nm]))
else: print("no escape found, expanded", n)
