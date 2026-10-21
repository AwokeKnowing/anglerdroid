"""At GT pose: per-frame fix error and single-frame scale estimate (weak priors). python gtscale.py L13 [N]"""
from harness import *
n = sys.argv[1]; L = make_loc(n); N = int(sys.argv[2]) if len(sys.argv) > 2 else 40; z = load(n)
yr = np.gradient(np.unwrap(z["odom"][:, 2]), z["t"]); idx = np.linspace(5, len(z["t"]) - 5, N).astype(int)
ST = float(os.environ.get("ML_TRUE_SCALE", 1.0)); out = []; ms = []
for i in idx:
    p0 = np.r_[z["gt"][i], 0.0]; P0 = np.diag([0.3 ** 2, 0.3 ** 2, math.radians(10) ** 2, 0.1 ** 2])
    t0 = time.perf_counter(); p, P, info = L.update(gray(n, z["files"][i]), p0, P0, yaw_rate=yr[i]); ms.append(time.perf_counter() - t0)
    if "p_raw" in info and info.get("why") != "blur":
        q = info["p_raw"]; e = q[:3] - z["gt"][i]; e[2] = wrap(e[2]); Hm = info["Hm"]
        out.append([e[0], e[1], math.degrees(e[2]), math.exp(q[3]), info["inl"], info["n"], 1 / math.sqrt(Hm[3, 3])])
o = np.array(out); print(n, "frames", len(o), "ms med %.0f" % (np.median(ms) * 1e3), "pos err med %.3f" % np.median(np.hypot(o[:, 0], o[:, 1])),
    "yaw med %.2f" % np.median(np.abs(o[:, 2])), "scale med %.4f (true %.4f) mean %.4f sd %.4f  inl %.2f n %d" % (np.median(o[:, 3]), ST, o[:, 3].mean(), o[:, 3].std(), np.median(o[:, 4]), np.median(o[:, 5])))
