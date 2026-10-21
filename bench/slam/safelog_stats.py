import json, sys, time, numpy as np
f = sys.argv[1]; win = float(sys.argv[2]) if len(sys.argv) > 2 else 60; t_from = float(sys.argv[3]) if len(sys.argv) > 3 else None
L = [json.loads(l) for l in open(f) if l.strip()]
A = np.array([[r[0], r[1], r[2]] for r in L])
sel = A[A[:, 2] >= t_from] if t_from else A[A[:, 2] > A[-1, 2] - win]
age = (sel[:, 2] - sel[:, 1]) * 1000; fr = np.diff(np.unique(sel[:, 1])) * 1000
print(json.dumps(dict(ticks=len(sel), span_s=round(sel[-1, 2] - sel[0, 2], 1), frame_age_at_cmd_ms=dict(p50=round(np.percentile(age, 50), 1), p95=round(np.percentile(age, 95), 1), max=round(age.max(), 1)),
                      frame_interval_ms=dict(p50=round(np.percentile(fr, 50), 1), p95=round(np.percentile(fr, 95), 1)), frames_hz=round(1000 / fr.mean(), 2),
                      last_row_age_s=round(time.monotonic() - A[-1, 2], 1))))
