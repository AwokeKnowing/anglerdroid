"""Safety-core health from its state block (read-only; sim or real Orin): rate, frame age, missed frames, stop codes.
python -m emu.tools.scbench.coreage [secs=10] [--json out.json]     (repo root, same venv as nav; nav must be running)
age1/age2 = decision time - RS1/RS2 capture (the core's own staleness inputs, limit KEVIN_SC_AGE_MAX 0.10 s);
late_50ms = decisions > 50 ms after the previous one (a frame handled late; the next comes early);
rs1_dropped = RS1 frames never decided on (capture-stamp gaps of whole frame periods): the missed 30 Hz deadlines."""
import json, os, sys, time
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")]
import numpy as np
from nav import bus, safety_core as S


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]; T = float(a[0]) if a else 10.0
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    ST = bus.Block("state", bus.layouts((480, 848))["state"]); F = {n: i for i, n in enumerate(S.STATE_FIELDS)}
    A = []; codes = {}; last = None; t0 = time.monotonic()
    while time.monotonic() - t0 < T:
        sq, r = ST.try_read()
        if sq is not None and r["f"][0] != last:
            last = r["f"][0]; f = r["f"]; A.append([f[F[k]] for k in ("age1", "age2", "period", "cap1")])
            c = S.CODES[int(f[F["code"]])] if 0 <= int(f[F["code"]]) < len(S.CODES) else str(int(f[F["code"]])); codes[c] = codes.get(c, 0) + 1
        time.sleep(0.002)
    if not A: raise SystemExit("coreage: no core decisions in %.0f s (nav / safety core not running?)" % T)
    A = np.array(A, float) * 1e3; st = lambda x: dict(zip(("p50", "p90", "p99", "max"), np.round(np.percentile(x, [50, 90, 99, 100]), 1).tolist()))
    R = dict(n=len(A), hz=round(len(A) / T, 2), age1_ms=st(A[:, 0]), age2_ms=st(A[:, 1]), period_ms=st(A[:, 2]),
             late_50ms=int(np.sum(A[:, 2] > 50.0)), rs1_dropped=int(np.sum(np.clip(np.round(np.diff(np.unique(A[:, 3])) / 33.33) - 1, 0, None))),
             over_100ms=int(np.sum(np.maximum(A[:, 0], A[:, 1]) > 100)), codes=codes)
    print(json.dumps(R))
    if out: json.dump(R, open(out, "w"))


if __name__ == "__main__":
    main()
