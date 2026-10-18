"""python latrace_dc.py <camlog>.depthcam [t0 t1]... : per-stage latency of emu/depthcam.py (ms p50/p95/max).
row: cam st_ns stamp t_pick t_render t_noise t_pub t_w side drops"""
import sys, numpy as np
C = np.loadtxt(sys.argv[1], dtype=np.int64, ndmin=2)
win = [(float(a), float(b)) for a, b in zip(sys.argv[2::2], sys.argv[3::2])] or [(0, 1e12)]
P = lambda a: "%6.1f %6.1f %6.1f" % tuple(np.percentile(np.asarray(a, float) / 1e6, [50, 95, 100]))
for w0, w1 in win:
    for cam in (1, 2):
        c = C[(C[:, 0] == cam) & (C[:, 1] >= w0 * 1e9) & (C[:, 1] < w1 * 1e9)]
        if not len(c): continue
        _, st, sp, tp, tr, tn, tpub, tw, side, dr = c.T
        d = np.diff(tw)
        print("RS%d %.0f-%.0f s: frames %d, dropped %d (%.2f%%)" % (cam, w0, w1, len(c), dr[-1] - dr[0], 100 * (dr[-1] - dr[0]) / max(len(c), 1)))
        print("  pick-up after capture %s | render %s | noise %s | hold %s | shm write %s" % (P(tp - st), P(tr - tp), P(tn - tr), P(tpub - tn), P(tw - tpub)))
        print("  age at publish (t_w - stamp) %s | publish interval %s, > 50 ms %.2f%%" % (P(tw - sp), P(d), 100 * np.mean(d > 5e7)))
