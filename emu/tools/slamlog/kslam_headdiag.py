"""kslam heading-error decomposition: odom (incl. Manhattan) vs matcher T_theta, by speed.  PYTHONPATH=.:src python -m emu.tools.slamlog.kslam_headdiag <log>..."""
import sys, numpy as np, math
from emu.tools.slamlog.log import Log
from emu.tools.slamlog.metrics import traj
for lg in sys.argv[1:]:
    L = Log(lg); N = L.nav(); t = N["t_od"]; m = (t >= L.gt_t[0]) & (t <= L.gt_t[-1]); t = t[m]
    pose, od = N["pose"][m], N["odom"][m]; g = L.gt(t)
    # anchor both to GT at start (same as metrics: first-pose alignment)
    def herr(th): e = np.unwrap(th) - g[:, 2]; return np.degrees((e - e[0] + np.pi) % (2 * np.pi) - np.pi)
    eo, ek = herr(od[:, 2]), herr(pose[:, 2]); Tth = np.degrees(np.unwrap(pose[:, 2]) - np.unwrap(od[:, 2])); Tth -= Tth[0]
    i = int(np.argmax(abs(ek)))
    gz = N["gz"][m]; w = np.gradient(np.unwrap(g[:, 2]), t)
    # heading change attributable to the matcher while spinning fast vs slow
    dT = np.diff(Tth); fast = abs(w[1:]) > 0.8
    print(lg.split('/')[-1], "kslam worst %.2f deg at t=%.1f s: odom err there %.2f, matcher T_th %.2f | odom worst %.2f | T_th range %.2f..%.2f | matcher dT while |w|>0.8: %.2f deg, slower: %.2f deg (sum of |dT|: %.1f / %.1f)" % (
        ek[i], t[i] - t[0], eo[i], Tth[i], eo[np.argmax(abs(eo))], Tth.min(), Tth.max(), dT[fast].sum(), dT[~fast].sum(), abs(dT[fast]).sum(), abs(dT[~fast]).sum()))
print("--- do matcher heading steps reduce or add error? (by robot speed)")
for lg in sys.argv[1:]:
    L = Log(lg); N = L.nav(); t = N["t_od"]; m = (t >= L.gt_t[0]) & (t <= L.gt_t[-1]); t = t[m]
    pose, od = N["pose"][m], N["odom"][m]; g = L.gt(t)
    ek = np.unwrap(pose[:, 2]) - g[:, 2]; ek -= ek[0]
    Tth = np.unwrap(pose[:, 2]) - np.unwrap(od[:, 2]); dT = np.diff(Tth); e_before = ek[:-1]
    v = np.hypot(*np.diff(g[:, :2], axis=0).T) / np.maximum(np.diff(t), 1e-3); nz = abs(dT) > 1e-6
    for nm, k in (("v<0.1", v < 0.1), ("0.1-0.3", (v >= 0.1) & (v < 0.3)), ("v>=0.3", v >= 0.3)):
        s = nz & k; good = (np.sign(dT[s]) == -np.sign(e_before[s])).mean() if s.any() else float('nan')
        print(lg.split('/')[-1], nm, "steps %d, net %.2f deg, |sum| %.1f deg, frac toward GT %.2f" % (s.sum(), np.degrees(dT[s].sum()), np.degrees(abs(dT[s]).sum()), good))
