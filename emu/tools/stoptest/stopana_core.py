"""GT-only stop-test analysis (any nav variant): python stopana_core.py <prefix>... (reads ~/m1_backup/dbg/stop/<prefix>_*.json)
gap = min distance between Kevin's footprint (twin SIL12 hull + deck corners) and the 0.30 m box (GT poses).
first rest = first GT sample after the robot had moved >= 0.2 m with |v| < 0.01 and |w| < 0.02 for >= 0.3 s.
creep = GT travel after first rest (path length) and gap lost after first rest (gap at rest - min gap afterwards)."""
import glob, json, math, os, sys
import numpy as np
GMAX = float(os.environ.get('STOP_GAP_MAX', 'inf'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from twin_geometry import FOOTPRINT_SAFE as F

def seg_d(p, a, b):
    ab = b - a; t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1); return np.linalg.norm(p - (a + t * ab))

def pdist(P, Q):
    def inside(p, R):
        s = [np.cross(R[(i + 1) % len(R)] - R[i], p - R[i]) for i in range(len(R))]; return all(x >= 0 for x in s) or all(x <= 0 for x in s)
    if any(inside(p, Q) for p in P) or any(inside(q, P) for q in Q): return 0.0
    return min(min(seg_d(p, Q[i], Q[(i + 1) % len(Q)]) for p in P for i in range(len(Q))), min(seg_d(q, P[i], P[(i + 1) % len(P)]) for q in Q for i in range(len(P))))

rows = []
for pre in sys.argv[1:]:
    for fn in sorted(glob.glob(os.path.expanduser('~/m1_backup/dbg/stop/%s_*.json' % pre))):
        d = json.load(open(fn)); G = d['gt']; bx, by = d['box']
        BOX = np.array([[bx - .15, by - .15], [bx + .15, by - .15], [bx + .15, by + .15], [bx - .15, by + .15]])
        def gap(g):
            c, s = math.cos(g[4]), math.sin(g[4]); return pdist(F @ np.array([[c, s], [-s, c]]) + [g[2], g[3]], BOX)
        xy = np.array([[g[2], g[3]] for g in G]); tg = np.array([g[0] for g in G]); v = np.array([g[5] for g in G]); w = np.array([g[6] for g in G])
        moved = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        still = (np.abs(v) < 0.01) & (np.abs(w) < 0.02)
        k0 = None; pauses = 0; kp = -99
        for k in range(len(G)):
            if moved[k] >= 0.2 and still[k]:
                j = np.searchsorted(tg, tg[k] + 0.3)
                if j < len(G) and still[k:j].all():
                    # STOP_GAP_MAX (m): only a rest this close to the box counts as the obstacle stop; earlier rests
                    # (e.g. an operability stop mid-approach that later resumes) are counted as pauses, not as creep
                    if gap(G[k]) < GMAX: k0 = k; break
                    if k - kp > 10: pauses += 1
                    kp = k
        vmax = float(np.abs(v).max())
        if k0 is None:
            rows.append(dict(run=os.path.basename(fn)[:-5], v=d['v'], vmax=vmax, rest=False, g_min=min(gap(g) for g in G[::5]), contact=G[-1][7] - G[0][7])); continue
        g_rest = gap(G[k0]); after = G[k0::3]; g_after = min(gap(g) for g in after)
        rows.append(dict(run=os.path.basename(fn)[:-5], v=d['v'], vmax=round(vmax, 3), rest=True, g_rest=round(g_rest, 3), g_min=round(min(g_rest, g_after), 3),
                         creep_path=round(float(moved[-1] - moved[k0]), 3), pauses=pauses, creep_gap=round(g_rest - g_after, 3), t_after=round(float(tg[-1] - tg[k0]), 1),
                         contact=G[-1][7] - G[0][7]))
for r in rows: print(json.dumps(r))
