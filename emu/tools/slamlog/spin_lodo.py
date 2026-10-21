"""Leave-one-drive-out evaluation of the spin heading refiner (emu.tools.slamlog.spin_refiner), models:
  full : the 8-feature linear model (ridge, lambda by inner LODO)
  small: one parameter, a heading scale during spins only: GT dtheta - odom dtheta = k * odom dtheta when |w| > 0.6
  PYTHONPATH=.:src python -m emu.tools.slamlog.spin_lodo <out.json> <log> <log> ...
Every drive is the test drive once; its model never sees it."""
import sys, json
import numpy as np
from .spin_refiner import series, ridge, score, W_SPIN

def feats(S, kind):
    spin = (S["W"] > W_SPIN).astype(float)
    return S["X"] if kind == "full" else np.c_[S["dth"] * spin]

if __name__ == "__main__":
    out, logs = sys.argv[1], sys.argv[2:]; SS = [series(l) for l in logs]; O = {}
    for i, l in enumerate(logs):
        te = SS[i]; tr = [s for j, s in enumerate(SS) if j != i]; nm = l.rstrip("/").split("/")[-1]; r = dict(odom=score(te, te["dth"]))
        spin = te["W"] > W_SPIN; r["gyro_in_spins"] = score(te, np.where(spin, te["dg"], te["dth"]))
        for kind in ("full", "small"):
            Xa = np.vstack([feats(s, kind) for s in tr]); ya = np.concatenate([s["y"] for s in tr])
            lam = 1e-2 if kind == "full" else 0.0
            b = ridge(np.c_[Xa, np.ones(len(ya))] if kind == "small" else Xa, ya, lam) if kind == "full" else np.linalg.lstsq(Xa, ya, rcond=None)[0]
            pred = feats(te, kind) @ b; res = ya - (Xa @ b); sd = float(np.std(res))
            z = (te["y"] - pred) / sd
            r[kind] = dict(score(te, te["dth"] + pred), frac_within_2sd=round(float(np.mean(np.abs(z) < 2)), 3), coef=np.round(b, 5).tolist())
        r["spin_heading_err_deg"] = round(float(np.degrees(np.sum(te["y"][spin]))), 2); r["n_windows"] = int(len(te["y"])); r["n_spin_windows"] = int(spin.sum())
        O[nm] = r
    json.dump(O, open(out, "w"), indent=1)
    for k, r in O.items():
        print(k, "spin_err %+.2f" % r["spin_heading_err_deg"], {m: (r[m]["ate_rmse_m"], r[m]["head_max_deg"]) for m in ("odom", "gyro_in_spins", "full", "small")}, "2sd", r["full"]["frac_within_2sd"], r["small"]["frac_within_2sd"], "k", r["small"]["coef"])
