"""python benchage.py <safelog> <label> <tag>... : frame age at command send (KEVIN_SAFETY_LOG: col 2 = monotonic
after hw.drive, col 1 = frame stamp; sim stamps are monotonic) and driving on old frames, per goal window."""
import json, sys, numpy as np
SL = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]; T = np.array([r[0] for r in SL])
label, tags = sys.argv[2], sys.argv[3:]; A = []; B = {0.25: [0, 0.0, 0.0, 0.0], 0.5: [0, 0.0, 0.0, 0.0]}; nmov = 0
for tag in tags:
    for r in json.load(open('bench_%s.json' % tag)):
        i0, i1 = np.searchsorted(T, r['t0m']), np.searchsorted(T, r['t1m']); rows = SL[i0:i1]
        for k, x in enumerate(rows):
            age = x[2] - x[1]; A.append(age); v, w = x[7], x[8]
            if abs(v) < 0.01 and abs(w) < 0.01: continue
            nmov += 1; dt = min(rows[k + 1][0] - x[0], 0.2) if k + 1 < len(rows) else 1 / 30
            for th, b in B.items():
                if age > th: b[0] += 1; b[1] = max(b[1], abs(v)); b[2] += abs(v) * dt; b[3] = max(b[3], abs(v) * age)
A = np.array(A) * 1e3
print('| %s | ticks %d (moving %d) | age ms p50 %.0f p90 %.0f p99 %.0f max %.0f | >0.25 s: %d ticks, max |v| %.2f, %.3f m driven, max v*age %.3f m | >0.5 s: %d ticks, max |v| %.2f, %.3f m, max v*age %.3f m |' % (
    label, len(A), nmov, *np.percentile(A, [50, 90, 99]), A.max(), *B[0.25], *B[0.5]))
