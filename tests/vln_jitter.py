"""Goal jitter / latency stats from a nav/vln.py jsonl log.  python tests/vln_jitter.py ~/.kevin/emu/vln_log/<file>.jsonl"""
import json, math, sys, numpy as np
L = [json.loads(l) for l in open(sys.argv[1])]; L = [r for r in L if r.get("goal")]
g = np.array([r["goal"] for r in L]); t = np.array([r["t"] for r in L]); ms = np.array([r["ms"] for r in L])
dg = np.hypot(*np.diff(g, axis=0).T)
p = np.array([r["pose"][:2] for r in L]); dp = np.hypot(*np.diff(p, axis=0).T)
rng = np.array([math.hypot(*r["wp"][-1][:2]) for r in L]); yaw = np.array([r["wp"][-1][2] for r in L])
pc = lambda a, q: float(np.percentile(a, q))
print("calls %d over %.0f s (%.2f pred/s); latency ms p50 %.0f p90 %.0f max %.0f" % (len(L), t[-1] - t[0], len(L) / (t[-1] - t[0]), pc(ms, 50), pc(ms, 90), ms.max()))
print("goal step between successive calls (m): p50 %.3f p90 %.3f max %.2f; robot moved p50 %.3f m/call" % (pc(dg, 50), pc(dg, 90), dg.max(), pc(dp, 50)))
print("farthest-wp range (m): p50 %.2f min %.2f max %.2f; final yaw (deg) p50 %.0f |max| %.0f; frac turn-in-place(<0.15 m) %.2f" % (pc(rng, 50), rng.min(), rng.max(), math.degrees(pc(yaw, 50)), math.degrees(abs(yaw).max()), (rng < 0.15).mean()))
if "age_ms" in L[0]: print("frame age at prediction ms p50 %.0f; fps %.2f; buffered-only frames n/a" % (pc([r["age_ms"] for r in L], 50), L[-1]["fps"]))
