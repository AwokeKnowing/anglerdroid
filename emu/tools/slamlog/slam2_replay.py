"""Exact offline replay of the live SLAM (nav.slam2.Slam2) from a KEVIN_SLAM_REC job stream (pickle records:
(t, odom_pose, obs, free, obs_h, rays, T_before)).  python -m emu.tools.slamlog.slam2_replay <slamjobs.pkl> [out.json]
Reports per-step compute time and whether the replayed correction T reproduces the live one (T before each job)."""
import sys, json, math, time, pickle
import numpy as np


def load(p):
    J = []; f = open(p, "rb")
    while True:
        try: J.append(pickle.load(f))
        except EOFError: return J


def replay(J, timing=True):
    from nav.slam2 import Slam2
    from nav.slam import GridMap
    s = Slam2(GridMap(), match=True); s._run = False
    Ts = []; ms = []
    for j in J:
        Ts.append(tuple(s.T)); t0 = time.perf_counter(); s._step(*j[:6]); ms.append((time.perf_counter() - t0) * 1e3)
    return s, np.array(Ts), np.array(ms)


if __name__ == "__main__":
    J = load(sys.argv[1]); s, Ts, ms = replay(J)
    live = np.array([j[6] for j in J]); d = Ts - live; d[:, 2] = (d[:, 2] + math.pi) % (2 * math.pi) - math.pi
    r = dict(jobs=len(J), span_s=round(J[-1][0] - J[0][0], 1), live_rate_hz=round(len(J) / (J[-1][0] - J[0][0]), 2),
             step_ms=dict(p50=round(float(np.median(ms)), 1), p95=round(float(np.percentile(ms, 95)), 1), max=round(float(ms.max()), 1)),
             replay_vs_live_T=dict(max_xy_m=round(float(np.hypot(d[:, 0], d[:, 1]).max()), 4), max_th_deg=round(float(np.degrees(np.abs(d[:, 2]).max())), 3)),
             stats=s.stats)
    print(json.dumps(r, default=float))
    if len(sys.argv) > 2: json.dump(r, open(sys.argv[2], "w"), default=float, indent=1)
