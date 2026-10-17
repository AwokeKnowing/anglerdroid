import json, math, sys, numpy as np
R = 0.08565 * 2 * math.pi
SIL = np.array([[-0.27371, -0.01022], [-0.16869, -0.15427], [-0.08009, -0.20998], [0.08319, -0.20998], [0.12372, -0.15418], [0.12629, -0.13818], [0.12629, 0.13818], [0.12372, 0.15418], [0.08319, 0.20998], [-0.08009, 0.20998], [-0.16869, 0.15427], [-0.27371, 0.01022]])
def segd(p, a, b):
    ab = b - a; t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1); return np.linalg.norm(p - (a + t * ab))
def pdist(A, B):
    return min(min(segd(p, B[j], B[(j + 1) % len(B)]) for j in range(len(B)) for p in A), min(segd(p, A[j], A[(j + 1) % len(A)]) for j in range(len(A)) for p in B))
for fn in sys.argv[1:]:
    d = json.load(open(fn)); G = d['gt']; L = d['nav']; bx, by = d['box']
    BOX = np.array([[bx - .15, by - .15], [bx + .15, by - .15], [bx + .15, by + .15], [bx - .15, by + .15]])
    tg = np.array([g[0] for g in G]); vg = np.array([g[5] for g in G]); wg = np.array([g[6] for g in G])
    def gap(i):
        c, s = math.cos(G[i][4]), math.sin(G[i][4]); P = SIL @ np.array([[c, s], [-s, c]]) + [G[i][2], G[i][3]]; return pdist(P, BOX)
    gi = lambda t: int(np.clip(np.searchsorted(tg, t), 0, len(tg) - 1))
    L = [r for r in L if r[17] == 'drive' or r[7] != 0]
    ib = next(i for i, r in enumerate(L) if r[9] != 'ok'); ic = next(i for i in range(ib, len(L)) if L[i][7] == 0 and L[i][8] == 0)
    ia = next((i for i, r in enumerate(L) if r[12] is not None and r[12] < 1.5), None)
    print('\n== %s v=%.2f w=%.2f dl=%.2f collisions=%d contact=%s' % (fn.split('/')[-1], d['v'], d.get('w', 0), d.get('dl', 0), G[-1][7] - G[0][7], G[-1][9]))
    for i in range(max(ib - 2, 0), min(ic + 4, len(L))):
        r = L[i]; k = gi(r[2])
        print('  t%+.3f %-8s s=%s v,w sent=(%.2f,%.2f) pre=(%.2f,%.2f) clr=%.3f age=%3.0fms | GT gap %.3f v %.2f w %.2f' % (r[2] - L[ic][2], r[9], r[10], r[7], r[8], r[3], r[4], r[11] if r[11] is not None else -1, (r[0] - r[1]) * 1e3, gap(k), vg[k], wg[k]))
    k0 = gi(L[ic][2]); ks = range(k0, len(G), 8); mg = min((gap(k), k) for k in range(k0, len(G), 2))
    print('  min GT gap after stop cmd %.3f at t+%.3f (v %.2f w %.2f); final gap %.3f; travelled heading change %.2f rad' % (mg[0], tg[mg[1]] - L[ic][2], vg[mg[1]], wg[mg[1]], gap(len(G) - 1), G[-1][4] - G[k0][4]))
