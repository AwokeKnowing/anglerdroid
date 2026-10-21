"""Heading accuracy of the live SLAM (nav.slam2.Slam2), exact replay of a KEVIN_SLAM_REC job stream against sim
ground truth (sensorlog index 'gt' = [x, y, yaw, ...] at t_mono).  Frames are aligned once at the first job (SE2).
Reports |heading error| of raw odometry and of Slam2 (T o odom) over all jobs and the heading change across each
spin (|GT yaw rate| > 0.6 rad/s segments), plus variants of Slam2 constants (e.g. MAX_W turn skip).
python -m emu.tools.slamlog.slam2_heading <log dir> [out.json] [variants json]"""
import sys, json, math
import numpy as np
from emu.tools.slamlog.slam2_replay import load


def wrap(a): return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def gt_track(d):
    t, g = [], []
    for l in open(d + "/index.jsonl"):
        if '"gt"' not in l: continue
        r = json.loads(l); t.append(r["t_mono"]); g.append(r["gt"][:3])
    t = np.array(t); g = np.array(g); o = np.argsort(t); t, g = t[o], g[o]
    g[:, 2] = np.unwrap(g[:, 2]); return t, g


def at(t, g, q):
    return np.c_[np.interp(q, t, g[:, 0]), np.interp(q, t, g[:, 1]), np.interp(q, t, g[:, 2])]


def _skip_all_cls():
    """Slam2 as it ran 19 Oct 22:20-22:26 (L15 live): the WHOLE step skipped when still (f4097e4 reverted that to match-only)."""
    import nav.slam2 as S2
    class SkipAll(S2.Slam2):
        def _step(self, t, podom, *a):
            q = getattr(self, "_lp_all", None)
            if q is not None and math.hypot(podom[0] - q[0], podom[1] - q[1]) < S2.STILL_D and abs(S2.wrap(podom[2] - q[2])) < S2.STILL_A:
                self.stats["skip_all"] = self.stats.get("skip_all", 0) + 1; return
            self._lp_all = podom; return S2.Slam2._step(self, t, podom, *a)
    return SkipAll


def run(J, variant):
    import nav.slam2 as S2
    from nav.slam import compose
    variant = dict(variant); cls = _skip_all_cls() if variant.pop("_skip_all", False) else None
    old = {k: getattr(S2, k) for k in variant}
    try:
        for k, v in variant.items(): setattr(S2, k, v)
        from emu.tools.slamlog.slam2_replay import replay
        s, Ts, ms = replay(J, cls=cls)
    finally:
        for k, v in old.items(): setattr(S2, k, v)
    # pose after each job = T(after) o odom; T after job i = T before job i+1 (last: final s.T)
    Ta = np.vstack([Ts[1:], np.array(s.T)[None]])
    est = np.array([compose(tuple(Ta[i]), tuple(J[i][1])) for i in range(len(J))])
    live = np.array([j[6] for j in J]); dT = Ts - live; dT[:, 2] = wrap(dT[:, 2])
    s.stats["replay_vs_live_T"] = (round(float(np.hypot(dT[:, 0], dT[:, 1]).max()), 4), round(float(np.degrees(np.abs(dT[:, 2]).max())), 3))
    return est, s.stats, ms


def errors(t, est, gt):
    e = est.copy(); e[:, 2] = np.unwrap(e[:, 2])
    d0 = wrap(gt[0, 2] - e[0, 2]); c, s_ = math.cos(d0), math.sin(d0)
    R = np.array([[c, -s_], [s_, c]]); xy = (e[:, :2] - e[0, :2]) @ R.T + gt[0, :2]
    return np.degrees(np.abs(wrap(e[:, 2] + d0 - gt[:, 2]))), np.hypot(*(xy - gt[:, :2]).T)


def spins(tg, g, thr=0.6, min_turn=math.radians(90)):
    w = np.gradient(g[:, 2], tg); fast = np.abs(w) > thr; seg = []; i = 0
    while i < len(fast):
        if fast[i]:
            j = i
            while j < len(fast) and fast[j]: j += 1
            if abs(g[j - 1, 2] - g[i, 2]) > min_turn: seg.append((tg[i], tg[j - 1], math.degrees(g[j - 1, 2] - g[i, 2])))
            i = j
        else: i += 1
    return seg


def summarize(name, t, est, gt, segs, tg, g):
    eh, ep = errors(t, est, gt); out = dict(heading_err_deg=dict(p50=round(float(np.median(eh)), 2), p95=round(float(np.percentile(eh, 95)), 2), max=round(float(eh.max()), 2), final=round(float(eh[-1]), 2)),
                                           pos_err_m=dict(p50=round(float(np.median(ep)), 3), p95=round(float(np.percentile(ep, 95)), 3), final=round(float(ep[-1]), 3)))
    e = est.copy(); e[:, 2] = np.unwrap(e[:, 2]); sp = []
    for a, b, deg in segs:      # heading change across the spin: estimate vs GT, using the last job before / first job >=1 s after
        i = np.searchsorted(t, a) - 1; k = np.searchsorted(t, b + 1.0)
        if i < 0 or k >= len(t): continue
        dg = gt[k, 2] - gt[i, 2]; de = e[k, 2] - e[i, 2]; sp.append(dict(gt_deg=round(math.degrees(dg), 1), err_deg=round(math.degrees(float(wrap(de - dg))), 2)))
    out["spins"] = sp; ae = np.abs([x["err_deg"] for x in sp]) if sp else np.zeros(1)
    out["spin_err_deg"] = dict(n=len(sp), p50=round(float(np.median(ae)), 2), max=round(float(ae.max()), 2)); return out


if __name__ == "__main__":
    d = sys.argv[1]; J = load(d + "/slamjobs.pkl"); tg, g = gt_track(d); t = np.array([j[0] for j in J])
    if abs(np.median(t) - np.median(tg)) > 1e3: raise SystemExit("job clock and GT clock differ")
    gt = at(tg, g, t); segs = spins(tg, g); res = dict(log=d, jobs=len(J), spins_gt=len(segs))
    odom = np.array([j[1] for j in J]); res["odom"] = summarize("odom", t, odom, gt, segs, tg, g)
    V = [("slam2_live", {}), ("slam2_no_turn_skip", {"MAX_W": 1e9})]
    if len(sys.argv) > 3: V = json.loads(sys.argv[3])       # e.g. '[["slam2_live_L15", {"_skip_all": true}], ...]'
    for nm, var in V:
        est, st, ms = run(J, var); res[nm] = summarize(nm, t, est, gt, segs, tg, g)
        res[nm]["stats"] = {k: st.get(k) for k in ("matches", "accepted", "kfs", "loops", "skip_turn", "skip_still", "skip_all", "rej_chi2", "replay_vs_live_T")}; res[nm]["step_ms_p50"] = round(float(np.median(ms)), 1)
    print(json.dumps({k: (v if not isinstance(v, dict) or "spins" not in v else {kk: vv for kk, vv in v.items() if kk != "spins"}) for k, v in res.items()}, indent=1))
    if len(sys.argv) > 2: json.dump(res, open(sys.argv[2], "w"), indent=1)
