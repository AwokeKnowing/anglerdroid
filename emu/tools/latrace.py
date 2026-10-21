"""python latrace.py <camlog> <shimlog> [t0_mono t1_mono]... : per-stage latency of the sim depth path.
camlog (emu.world KEVIN_EMU_CAMLOG): st_ns t_job t_pick t_r1 t_w1 t_r2 t_w2 s1 s2 colour_rideon  (monotonic ns)
shimlog (hal/sim/pyrealsense2 KEVIN_SHIM_LOG): shm stamp t_wake t_read pid
Stages: phys_lag = job posted - physics stamp (physics behind wall clock); queue = render thread pick-up - post;
render1/2 = depth render + noise; write1/2 = shm copy; wake = shm write -> consumer wake; copy = consumer read."""
import sys, collections, numpy as np
C = np.loadtxt(sys.argv[1], dtype=np.int64, ndmin=2)
S = [l.split() for l in open(sys.argv[2])]
win = [(float(a), float(b)) for a, b in zip(sys.argv[3::2], sys.argv[4::2])] or [(0, 1e12)]
def P(a): a = np.asarray(a, float) / 1e6; return "%6.1f %6.1f %6.1f" % tuple(np.percentile(a, [50, 95, 100])) if len(a) else "   -"
for (w0, w1) in win:
    c = C[(C[:, 0] >= w0 * 1e9) & (C[:, 0] < w1 * 1e9)]
    if not len(c): continue
    print("window %.0f-%.0f s: %d camera ticks   (ms p50 p95 max)" % (w0, w1, len(c)))
    st, tj, tp, r1, w1_, r2, w2, s1, s2 = (c[:, i] for i in range(9))
    d = np.diff(st); print("  physics camera tick interval   ", P(d), " ticks > 50 ms: %.2f%%" % (100 * np.mean(d > 5e7)))
    dt = np.diff(w1_); print("  RS1 write interval            ", P(dt), " > 50 ms: %.2f%%" % (100 * np.mean(dt > 5e7)))
    for nm, a in (("phys_lag (post - stamp)", tj - st), ("queue (pick - post)", tp - tj), ("render+noise RS1", r1 - tp), ("shm write RS1", w1_ - r1),
                  ("render+noise RS2", r2 - w1_), ("shm write RS2", w2 - r2), ("stamp jitter RS1", s1 - st), ("age at RS1 write (w1 - s1)", w1_ - s1), ("age at RS2 write (w2 - s2)", w2 - s2)):
        print("  %-31s" % nm, P(a))
    by = collections.defaultdict(list)
    for sh, stp, tw, tr, pid in S: by[(sh, pid)].append((int(stp), int(tw), int(tr)))
    W = {("kevin_rs1"): dict(zip(s1, w1_)), ("kevin_rs2"): dict(zip(s2, w2))}
    for (sh, pid), rows in sorted(by.items()):
        rows = [r for r in rows if w0 * 1e9 <= r[1] < w1 * 1e9]
        if not rows: continue
        wm = W.get(sh, {}); got = [r for r in rows if r[0] in wm]
        wake = [r[1] - wm[r[0]] for r in got]; cp = [r[2] - r[1] for r in rows]; age = [r[2] - r[0] for r in rows]
        seen = set(r[0] for r in rows); written = [s for s in wm if s]
        miss = sum(1 for s in written if s not in seen and rows[0][0] <= s <= rows[-1][0])
        print("  consumer %s pid %s: frames %d, missed %d (%.2f%%)" % (sh, pid, len(rows), miss, 100 * miss / max(len(rows) + miss, 1)))
        print("    wake (write -> consumer)     ", P(wake), "  copy", P(cp), "  age after copy", P(age))
