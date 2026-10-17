import json, math, sys, numpy as np
exec(open(__import__('os').path.join(__import__('os').path.dirname(__file__), 'stopana2.py')).read().split('for fn in sys.argv[1:]:')[0])
Rw = 0.08565 * 2 * math.pi
print('| run | v | first seen (bumper->face) | frame capture: t, gap | (b) decision tick: t, gap, clr_est | (c) stop sent: t, gap | (d) sim input_vel=0: t | (e) wheel=0: t, gap | travel after (c) | eff. decel | min gap | contact |')
allp, alla = [], []
for fn in sys.argv[1:]:
    d = json.load(open(fn)); G = d['gt']; L = d['nav']; bx, by = d['box']
    BOX = np.array([[bx - .15, by - .15], [bx + .15, by - .15], [bx + .15, by + .15], [bx - .15, by + .15]])
    tg = np.array([g[0] for g in G]); vg = np.array([g[5] for g in G]); inp = np.array([g[8][1][1] for g in G]) * Rw; wv = np.array([g[8][1][3] for g in G]) * Rw
    def gap(i):
        c, s = math.cos(G[i][4]), math.sin(G[i][4]); P = SIL @ np.array([[c, s], [-s, c]]) + [G[i][2], G[i][3]]; return pdist(P, BOX)
    gi = lambda t: int(np.clip(np.searchsorted(tg, t), 0, len(tg) - 1))
    L = [r for r in L if r[17] == 'drive' or r[7] != 0]
    T0 = np.array([r[0] for r in L]); TF = np.array([r[1] for r in L]); allp += list(np.diff(T0) * 1e3); alla += list((T0 - TF) * 1e3)
    ia = next(i for i, r in enumerate(L) if r[12] is not None and r[12] < 1.5)
    ib = next(i for i, r in enumerate(L) if r[9] != 'ok'); ic = next(i for i in range(ib, len(L)) if L[i][7] == 0 and L[i][8] == 0)
    tc = L[ic][2]; kc = gi(tc); kd = next(k for k in range(kc, len(G)) if abs(inp[k]) < 1e-3); ke = next(k for k in range(kd, len(G)) if abs(wv[k]) < 0.01)
    ks = next(k for k in range(kc, len(G)) if vg[k] < 0.01)
    trav = sum(math.hypot(G[k + 1][2] - G[k][2], G[k + 1][3] - G[k][3]) for k in range(kc, ks)); v_c = vg[kc]
    mn = min(gap(k) for k in range(kc, len(G), 3))
    print('| %s | %.2f/%.2f | %.2f m | %+.0f ms, %.3f | %+.0f ms, %.3f, %.3f | 0, %.3f | %+.0f ms | %+.0f ms, %.3f | %.3f m | %.2f m/s² | %.3f | %d |' % (
        fn.split('/')[-1][:-5], d['v'], d.get('w', 0), gap(gi(TF[ia])), (TF[ic] - tc) * 1e3, gap(gi(TF[ic])), (T0[ib] - tc) * 1e3, gap(gi(T0[ib])), L[ib][11],
        gap(kc), (tg[kd] - tc) * 1e3, (tg[ke] - tc) * 1e3, gap(ke), trav, v_c ** 2 / (2 * max(trav - 0, 1e-3)), mn, G[-1][7] - G[0][7]))
print('loop period ms p5/p50/p95/max %s ; frame age at decision p5/p50/p95 %s ; n=%d' % (np.percentile(allp, [5, 50, 95, 100]).round(0), np.percentile(alla, [5, 50, 95]).round(0), len(allp)))
