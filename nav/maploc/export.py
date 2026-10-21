"""python -m nav.maploc.export <logdir> <outdir>: mast frames + GT / odom / live Slam2 poses (mast frame times) for the
offline map-localisation harness (nav/maploc/run.py).  Writes <outdir>/<log>/{poses.npz, mast/*.jpg} and mlx/mast.json (lens = twin truth, intr + dist)."""
import os, sys, json, shutil, numpy as np
from emu.tools.slamlog.log import Log
d, out = sys.argv[1], sys.argv[2]; n = os.path.basename(d.rstrip("/")); L = Log(d); mf = L.frames["mast"]; ts = np.array([t for t, _ in mf]); nv = L.nav()
g = L.gt(ts); u = lambda a: np.unwrap(a)
od = np.stack([np.interp(ts, nv["t_od"], nv["odom"][:, 0]), np.interp(ts, nv["t_od"], nv["odom"][:, 1]), np.interp(ts, nv["t_od"], u(nv["odom"][:, 2]))], -1)
ps = np.stack([np.interp(ts, nv["t"], nv["pose"][:, 0]), np.interp(ts, nv["t"], nv["pose"][:, 1]), np.interp(ts, nv["t"], u(nv["pose"][:, 2]))], -1)
ok = (ts >= max(nv["t"][0], L.gt_t[0])) & (ts <= min(nv["t"][-1], L.gt_t[-1])); o = os.path.join(out, "mlx", n); os.makedirs(o, exist_ok=True)
files = [f for (t, f), k in zip(mf, ok) if k]
# tire: the profile the log was recorded with (camera height); logs without it predate the soft tyres -> stock
np.savez(o + "/poses.npz", t=ts[ok], gt=g[ok], odom=od[ok], slam2=ps[ok], files=np.array(files), tire=str(L.info.get("tire") or "stock"))
for f in files: os.makedirs(os.path.join(out, n, os.path.dirname(f)), exist_ok=True); shutil.copy(os.path.join(d, f + ".jpg"), os.path.join(out, n, f + ".jpg"))
# lens: the twin truth lens (new logs carry it in info["mast"]; Log.mast_lens supplies it for older logs)
K, dist = L.mast_lens(); m = dict(L.info["mast"]); m["intr"] = dict(m["intr"], fx=float(K[0, 0]), fy=float(K[1, 1]), ppx=float(K[0, 2]), ppy=float(K[1, 2])); m["dist"] = [float(v) for v in dist]
json.dump(m, open(os.path.join(out, "mlx", "mast.json"), "w"))
