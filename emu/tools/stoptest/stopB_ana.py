import json, math, sys, glob, numpy as np
exec(open('stopana2.py').read().split('for fn in sys.argv[1:]:')[0])
M = 0.035
def model(v, tr, a): return v * tr + v * v / (2 * a)
out = []
for pre in ('A26', 'B26'):
  for fn in sorted(glob.glob('stop/%s_*.json' % pre)):
    d = json.load(open(fn)); G = d['gt']; L = d['nav']; bx, by = d['box']; Rw = wheel_circ(d); v = d['v']
    BOX = np.array([[bx - .15, by - .15], [bx + .15, by - .15], [bx + .15, by + .15], [bx - .15, by + .15]])
    tg = np.array([g[0] for g in G]); vg = np.array([g[5] for g in G]); wv = np.array([g[8][1][3] for g in G]) * Rw
    def gap(i):
        c, s = math.cos(G[i][4]), math.sin(G[i][4]); P = SIL @ np.array([[c, s], [-s, c]]) + [G[i][2], G[i][3]]; return pdist(P, BOX)
    gi = lambda t: int(np.clip(np.searchsorted(tg, t), 0, len(tg) - 1))
    L = [r for r in L if r[17] == 'drive' or r[7] != 0]
    ib = next(i for i, r in enumerate(L) if r[9] != 'ok'); ic = next(i for i in range(ib, len(L)) if L[i][7] == 0 and L[i][8] == 0)
    tc = L[ic][2]; kc = gi(tc); kf = gi(L[ic][1]); ke = next(k for k in range(kc, len(G)) if abs(wv[k]) < 0.01 and abs(vg[k]) < 0.01)
    # first stop: GT gap when the robot is at rest after the first stop command; later = remainder of the 'drive' command
    g_stop = min(gap(k) for k in range(kc, ke + 1)); g_min = min(gap(k) for k in range(kc, len(G), 3)); g_end = gap(len(G) - 1)
    d_true = gap(kf) - gap(ke)                         # travel toward the box from the decision frame's capture to rest
    age = (L[ic][2] - L[ic][1])
    first_lim = next((i for i, r in enumerate(L) if r[9] != 'ok'), None)
    out.append(dict(run=fn.split('/')[-1][:-5], v=v, w=d.get('w', 0), dl=d.get('dl', 0), gap_cap=gap(kf), clr_est=L[ic][11], age=age,
                    d_true=d_true, g_stop=g_stop, g_min=g_min, g_end=g_end, contact=G[-1][7] - G[0][7], first_state=L[ib][9], scale=L[ib][10]))
for o in out: print(json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in o.items()}))
json.dump(out, open('stopB_ana.json', 'w'))
