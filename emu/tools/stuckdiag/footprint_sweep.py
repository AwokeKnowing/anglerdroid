import numpy as np, math, json, glob, sys
sys.path[:0] = ['.', 'src']
from emu.tools.slamlog.slam2_replay import load, replay
from nav.slam import transform
from nav.governor import sd
import twin_geometry as TG
lg = sys.argv[1]; tq = float(sys.argv[2])
J = load(lg + "/slamjobs.pkl"); R = [json.loads(str(np.load(f, allow_pickle=True)["meta"])) for f in sorted(glob.glob(lg + "/nav/t*.npz"))]
r0 = R[0]["t"]; Jq = [j for j in J if j[0] - r0 <= tq]; s, _, _ = replay(Jq, timing=False)
rr = min(R, key=lambda r: abs(r["t"] - r0 - tq)); x, y, th = rr["pose"]; g = s.grid
occ = g.occupied(); iy, ix = np.nonzero(occ[int((y - 1 - g.origin[1]) / g.res):int((y + 1 - g.origin[1]) / g.res), int((x - 1 - g.origin[0]) / g.res):int((x + 1 - g.origin[0]) / g.res)])
W = np.stack([g.origin[0] + (ix + int((x - 1 - g.origin[0]) / g.res) + 0.5) * g.res, g.origin[1] + (iy + int((y - 1 - g.origin[1]) / g.res) + 0.5) * g.res], 1)
j = min(Jq, key=lambda j: abs(j[0] - r0 - tq)); live = transform(np.asarray(j[2], np.float32), (x, y, th))
print("occupied map cells within 1 m:", len(W))
def rob(P, p): c, s_ = math.cos(p[2]), math.sin(p[2]); d = P - np.array(p[:2]); return np.stack([c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1]], 1)
for dd in range(-280, 130, 10):
    p = (x, y, th + math.radians(dd)); print("rot %+3d deg: map clr %.3f  live clr %.3f" % (dd, sd(rob(W, p)).min(), sd(rob(live, p)).min()))
for dv in (-0.05, -0.02, 0.02, 0.05):
    p = (x + dv * math.cos(th), y + dv * math.sin(th), th); print("move %+.2f m: map clr %.3f live clr %.3f" % (dv, sd(rob(W, p)).min(), sd(rob(live, p)).min()))
