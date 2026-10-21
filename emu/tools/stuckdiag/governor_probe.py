import numpy as np, math, json, glob, sys
sys.path[:0] = ['.', 'src']
from emu.tools.slamlog.slam2_replay import load, replay
from nav.governor import governor, sd, MARGIN
from nav import run as RN
import twin_geometry as TG
lg = sys.argv[1]; tq = float(sys.argv[2])
J = load(lg + "/slamjobs.pkl"); R = [json.loads(str(np.load(f, allow_pickle=True)["meta"])) for f in sorted(glob.glob(lg + "/nav/t*.npz"))]
r0 = R[0]["t"]; Jq = [j for j in J if j[0] - r0 <= tq]; s, _, _ = replay(Jq, timing=False)
rr = min(R, key=lambda r: abs(r["t"] - r0 - tq)); x, y, th = rr["pose"]
class B: pass
b = B(); b.navgrid = s.grid
mp = RN.Brain._map_points(b, x, y, th)
j = min(Jq, key=lambda j: abs(j[0] - r0 - tq)); live = np.asarray(j[2], np.float32)
print("pose", np.round(rr["pose"], 3), "map pts", len(mp), "live pts", len(live), "R_ROTATE", TG.R_ROTATE, "MARGIN", MARGIN)
for nm, P in (("live", live), ("map", mp), ("both", np.concatenate([live, mp]))):
    d = sd(P); k = np.argsort(d)[:3]; print(" %-4s min signed dist %.3f at %s" % (nm, d.min(), np.round(P[k], 3).tolist()))
    for v, w in ((0.2, 0), (-0.2, 0), (-0.2, 0.5), (-0.2, -0.5), (0, 0.6), (0, -0.6), (0.15, 0.6), (0.15, -0.6)):
        print("   cmd %5.2f %5.2f -> %s" % (v, w, [round(float(q), 3) if not isinstance(q, str) else q for q in governor(P, v, w)]))
