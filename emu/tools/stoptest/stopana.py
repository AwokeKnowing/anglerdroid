import json, math, sys, numpy as np
R = 0.08565 * 2 * math.pi
for fn in sys.argv[1:]:
    d = json.load(open(fn)); G = d['gt']; L = d['nav']; X0, Y0, YAW = d['x0']; bx, by = d['box']
    face = (bx - X0) * math.cos(YAW) + (by - Y0) * math.sin(YAW) - 0.15
    tg = np.array([g[0] for g in G]); s = np.array([(g[2] - X0) * math.cos(YAW) + (g[3] - Y0) * math.sin(YAW) for g in G]); gap = face - s - 0.126
    vg = np.array([g[5] for g in G]); inp = np.array([g[8][1][1] for g in G]) * R; spt = np.array([g[8][1][2] for g in G]) * R; wv = np.array([g[8][1][3] for g in G]) * R
    gat = lambda t: float(np.interp(t, tg, gap))
    L = [r for r in L if r[17] == 'drive' or r[7] != 0]
    T0 = np.array([r[0] for r in L]); TF = np.array([r[1] for r in L]); TE = np.array([r[2] for r in L])
    ia = next(i for i, r in enumerate(L) if r[12] is not None and r[12] < 1.5)
    ib = next(i for i, r in enumerate(L) if r[9] != 'ok')
    ic = next(i for i in range(ib, len(L)) if L[i][7] == 0)
    tc = TE[ic]
    vmax = vg.max(); itop = int(np.argmax(vg))
    idd = next(i for i in range(len(G)) if tg[i] > TE[ib] and inp[i] < inp[:i].max() - 0.01)
    idz = next(i for i in range(len(G)) if tg[i] > tc and abs(inp[i]) < 1e-3)
    ie = next(i for i in range(idz, len(G)) if abs(wv[i]) < 0.01)
    # decel: GT speed slope between 80 % and 20 % of the speed at the stop command
    v_c = float(np.interp(tc, tg, vg)); j1 = next(i for i in range(len(G)) if tg[i] > tc and vg[i] < 0.8 * v_c); j2 = next(i for i in range(j1, len(G)) if vg[i] < 0.2 * v_c)
    dec = (vg[j1] - vg[j2]) / (tg[j2] - tg[j1])
    per = np.diff(T0) * 1e3; age = (T0 - TF) * 1e3
    print('\n== %s  v_cmd=%.2f  v_peak=%.3f' % (fn.split('/')[-1], d['v'], vmax))
    print('loop period ms p50/p95 %.0f/%.0f  frame age at decision p50/p95 %.0f/%.0f  loop compute %.0f ms' % (np.median(per), np.percentile(per, 95), np.median(age), np.percentile(age, 95), np.median((TE - T0) * 1e3)))
    rows = [('a obstacle first in RS1 top-down (frame time)', TF[ia], 'ego fwd dist %.3f (from axle)' % L[ia][12]),
            ('b first safety limit (tick start)', T0[ib], 'state %s scale %s clr %.3f; frame age %.0f ms' % (L[ib][9], L[ib][10], L[ib][11] or -1, (T0[ib] - TF[ib]) * 1e3)),
            ('c stop cmd sent (v=0)', tc, 'frame age %.0f ms; clr %.3f; v_meas(odom) %.3f' % ((T0[ic] - TF[ic]) * 1e3, L[ic][11] or -1, L[ic][16])),
            ('d sim input_vel first drops', tg[idd], 'input %.3f m/s' % inp[idd]),
            ('d\' sim input_vel = 0', tg[idz], ''),
            ('e wheel vel ~0', tg[ie], 'GT v %.3f' % vg[ie])]
    for nm, t, ex in rows:
        print('  %-44s t=%+7.3f s  gap(bumper->face)=%.3f  GTv=%.3f  %s' % (nm, t - tc, gat(t), float(np.interp(t, tg, vg)), ex))
    print('  final gap %.3f  GT stop dist after (c) %.3f m  decel (80->20%%) %.2f m/s^2  collisions %d' % (gap[-1], gat(tc) - gap[-1], dec, G[-1][7] - G[0][7]))
    # scale sequence
    print('  scales:', [(round(T0[i] - tc, 3), L[i][9][0], L[i][10], L[i][7], round(L[i][11] or -1, 3)) for i in range(max(ib - 2, 0), min(ic + 3, len(L)))])
    # setpoint vs input (twin ramp)
    k = [i for i in range(idd, ie + 1, max(1, (ie - idd) // 12))]
    print('  t, input, setpoint, wheel, GTv:', [(round(tg[i] - tc, 3), round(inp[i], 3), round(spt[i], 3), round(wv[i], 3), round(vg[i], 3)) for i in k])
