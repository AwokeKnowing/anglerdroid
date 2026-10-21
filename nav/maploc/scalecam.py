"""Pose fixed at GT; scale the map about the CAMERA by s (pure range/size bias test); robust cost grid -> s*. true 1."""
from harness import *
from maploc import lsd_segments, associate
out = []
grid = np.exp(np.linspace(-0.12, 0.12, 49))
ZSEL = os.environ.get("ML_ZSEL", "all")
for n in sys.argv[1:]:
    L = make_loc(n); hub = L.hub; z = load(n); yr = np.gradient(np.unwrap(z["odom"][:, 2]), z["t"])
    for i in np.linspace(5, len(z["t"]) - 5, 40).astype(int):
        if abs(yr[i]) > 0.5: continue
        g = z["gt"][i]; c2w = se2(*g, z=L.zb) @ L.Tbc; im = gray(n, z["files"][i]); X, th = L.ge.visible(c2w, K, W, H)
        if ZSEL == "nofloor": k = X[:, 2] > 0.05; X, th = X[k], th[k]
        if ZSEL == "noceil": k = X[:, 2] < 2.2; X, th = X[k], th[k]
        if ZSEL == "nohoriz": k = (X[:, 2] > 0.05) & (X[:, 2] < 2.2); X, th = X[k], th[k]
        Sg, Sth = lsd_segments(im); c = c2w[:3, 3].copy(); c[2] = 0.0; cost = []   # pivot: floor point below the camera (camera height known)
        for s in grid:
            u, v, ok = L.project(c + (X - c) * s, g); j, d = associate(u, v, th, Sg, Sth, 15); d = np.where(j >= 0, d, 15.0)
            cost.append(np.where(d <= hub, 0.5 * d ** 2, hub * (d - 0.5 * hub))[ok].sum() + 0.5 * 15 ** 2 * 0)  # Huber
        k = int(np.argmin(cost))
        if 0 < k < len(grid) - 1:
            a, b, cc = cost[k - 1], cost[k], cost[k + 1]; off = 0.5 * (a - cc) / (a - 2 * b + cc); out.append(math.exp(math.log(grid[k]) + off * (math.log(grid[1]) - math.log(grid[0]))))
o = np.array(out); print(ZSEL, "frames", len(o), "s* median %.4f  mean %.4f  sd %.4f  sem %.4f" % (np.median(o), o.mean(), o.std(), o.std() / math.sqrt(len(o))))
