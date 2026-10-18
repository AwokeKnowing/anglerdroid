"""Offline replay of a KEVIN_PRED_LOG (nav/predict.py rows c=brain cmd, o=core output, m=measurement, p=prediction).
For every logged plan-cycle prediction, recompute the start state at its effect time with
  raw      : latest frame measurement, no prediction
  brain    : v1 = brain commands as the wheel command history (ramp 3 turns/s^2), latency d
  core     : v2 = safety-core outputs (its ramp), latency d
and score each against the measured state at the effect time.  usage: pred_replay.py <predlog.jsonl> [d_ms ...]"""
import json, math, sys, os
import numpy as np
sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", ".."), os.path.join(os.path.dirname(__file__), "..", "..", "src")]
from nav.predict import CmdPredictor, _wrap
import twin_geometry as TG
rows = [json.loads(l) for l in open(sys.argv[1])]
ds = [float(x) / 1e3 for x in sys.argv[2:]] or [0.0, 0.05, 0.09, 0.18, 0.3]
C = np.array([r[1:4] + [3.0] for r in rows if r[0] == "c"]); O = np.array([r[1:5] for r in rows if r[0] == "o"])
M = np.array([r[1:8] for r in rows if r[0] == "m"]); P = [r for r in rows if r[0] == "p"]
print("rows: cmds %d core-outputs %d meas %d predictions %d" % (len(C), len(O), len(M), len(P)))
cp = CmdPredictor()
def actual(t):
    i = min(max(int(np.searchsorted(M[:, 0], t)), 1), len(M) - 1); f = (t - M[i - 1, 0]) / max(M[i, 0] - M[i - 1, 0], 1e-6)
    a = M[i - 1] + f * (M[i] - M[i - 1]); a[5] = M[i - 1, 5] + f * _wrap(M[i, 5] - M[i - 1, 5])
    j = min(max(int(np.searchsorted(M[:, 6], t)), 1), len(M) - 1); g = (t - M[j - 1, 6]) / max(M[j, 6] - M[j - 1, 6], 1e-6)
    vl, vr = M[j - 1, 1] + g * (M[j, 1] - M[j - 1, 1]), M[j - 1, 2] + g * (M[j, 2] - M[j - 1, 2])
    return a[3:6], 0.5 * (vl + vr), (vr - vl) / TG.TRACK_M
res = {}
def add(k, p, a):
    (x, y, th), v, w = p; (xa, ya, tha), va, wa = a
    res.setdefault(k, []).append((math.hypot(x - xa, y - ya), abs(math.degrees(_wrap(th - tha))), abs(v - va), abs(w - wa)))
moving = 0
for r in P:
    _, tnow, t_eff, m_t, t_w, x, y, th, v, w = r
    if t_eff > M[-1, 6] or t_eff < M[0, 0]: continue
    k = int(np.argmin(np.abs(M[:, 0] - m_t))); m = M[k]
    a = actual(t_eff)
    if abs(a[1]) < 0.02 and abs(a[2]) < 0.05 and abs(0.5 * (m[1] + m[2])) < 0.02: continue      # score while moving only
    moving += 1
    add("raw", (m[3:6], 0.5 * (m[1] + m[2]), (m[2] - m[1]) / TG.TRACK_M), a)
    add("core_logged", ((x, y, th), v, w), a)
    for d in ds:
        for nm, H in (("brain", C[C[:, 0] <= tnow]), ("core", O[O[:, 0] <= tnow])):
            if not len(H): continue
            sl, sr, pose = cp._sim(H, m[0], t_eff, m[1], m[2], d, pose=tuple(m[3:6]))     # v1 start: frame time, its speeds
            add("%s_d%03d" % (nm, d * 1e3), (pose, 0.5 * (sl + sr), (sr - sl) / TG.TRACK_M), a)
print("scored plan cycles while moving: %d" % moving)
print("%-16s %18s %18s %18s %18s" % ("start state", "pos cm p50/p95", "yaw deg p50/p95", "v m/s p50/p95", "w rad/s p50/p95"))
for k, L in res.items():
    A = np.array(L)
    print("%-16s %8.2f /%7.2f  %8.2f /%7.2f  %8.3f /%7.3f  %8.3f /%7.3f" % (k, np.median(A[:, 0]) * 100, np.percentile(A[:, 0], 95) * 100, np.median(A[:, 1]), np.percentile(A[:, 1], 95),
                                                                         np.median(A[:, 2]), np.percentile(A[:, 2], 95), np.median(A[:, 3]), np.percentile(A[:, 3], 95)))
# how often the core did not send what the brain asked (the v1 bug): brain cmd in force vs core output at each core tick
if len(C) and len(O):
    idx = np.searchsorted(C[:, 0], O[:, 0], side="right") - 1; ok = idx >= 0
    dv = np.abs(C[idx[ok], 1] - O[ok, 1]); dw = np.abs(C[idx[ok], 2] - O[ok, 2])
    print("core ticks: %d; core output != brain cmd (|dv|>0.02 or |dw|>0.05): %.1f%%; |dv| p95 %.3f m/s, |dw| p95 %.3f rad/s" % (ok.sum(), 100 * np.mean((dv > 0.02) | (dw > 0.05)), np.percentile(dv, 95), np.percentile(dw, 95)))
