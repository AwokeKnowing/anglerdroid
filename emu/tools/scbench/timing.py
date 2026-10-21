"""per-call cost of safety_core.decide and governor.governor, old rule vs SC_DECEL, on recorded states (recorded.py).
  python -m emu.tools.scbench.timing slamlogs/L14 slamlogs/L15"""
import sys, time, numpy as np
from nav import safety_core as S, governor as G
from emu.tools.scbench import recorded as RC, bench as B
for lg in sys.argv[1:]:
    R, J = RC.load_log(lg); jt = np.array([j[0] for j in J]); T = {False: [], True: []}; TG = {False: [], True: []}; npts = []
    for r in R:
        v_c, w_c = float(r["v"]), float(r["w"])
        if abs(v_c) < 1e-3 and abs(w_c) < 1e-3: continue
        k = int(np.searchsorted(jt, r["t"], "right")) - 1
        if k < 0 or r["t"] - jt[k] > 0.5: continue
        P = S.to_body(RC.to_world(np.asarray(J[k][2], float), J[k][1]), *r["pose"]).astype(np.float32); npts.append(len(P))
        v_m = 0.5 * (r["vl"] + r["vr"]); w_m = (r["vr"] - r["vl"]) / S.TRACK_M
        for mode in (False, True):
            S.SC_DECEL = G.SC_DECEL = mode
            t0 = time.perf_counter(); S.decide(P, B.StubBands(), v_c, w_c, v_m, w_m); T[mode].append(time.perf_counter() - t0)
            t0 = time.perf_counter(); G.governor(P, v_c, w_c, v_m, w_m, B.StubBands().rects()); TG[mode].append(time.perf_counter() - t0)
    f = lambda a: "p50 %.2f p99 %.2f max %.2f ms" % (*np.percentile(np.array(a) * 1e3, [50, 99]), 1e3 * max(a))
    print(lg, "n", len(npts), "pts p50", int(np.median(npts)))
    print("  core decide  old:", f(T[False]), "| decel:", f(T[True]))
    print("  governor     old:", f(TG[False]), "| decel:", f(TG[True]))
