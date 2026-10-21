"""Fix-information calibration per log: at GT (truth) with a weak prior, c = median(d' Hm d) / median(chi2_3)."""
from harness import *
CALF = os.environ.get("ML_CALF", "calib.json")
out = json.load(open(CALF)) if os.path.exists(CALF) else {}
for n in sys.argv[1:]:
    L = make_loc(n); z = load(n); yr = np.gradient(np.unwrap(z["odom"][:, 2]), z["t"]); idx = np.linspace(5, len(z["t"]) - 5, 60).astype(int); q = []
    for i in idx:
        P0 = np.diag([0.3 ** 2, 0.3 ** 2, math.radians(10) ** 2])
        p, P, info = L.update(gray(n, z["files"][i]), z["gt"][i], P0, yaw_rate=yr[i])
        if "Hm" in info and info.get("why") != "blur":
            d = np.array(info["p_raw"]) - z["gt"][i]; d[2] = wrap(d[2]); q.append(float(d @ info["Hm"] @ d))
    out[n] = float(2.366 / np.median(q)); print(n, "frames", len(q), "info scale = %.4f  (raw fix err med %.3f m)" % (out[n], 0))
json.dump(out, open(CALF, "w"), indent=1)
