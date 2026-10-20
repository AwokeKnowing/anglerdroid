"""At GT: signed residual (image edge - map edge, px, + = further from image centre) by edge class."""
from harness import *
from maploc import lsd_segments, associate
L = make_loc(); R = {}
for n in sys.argv[1:]:
    z = load(n); yr = np.gradient(np.unwrap(z["odom"][:, 2]), z["t"])
    for i in np.linspace(5, len(z["t"]) - 5, 40).astype(int):
        if abs(yr[i]) > 0.5: continue
        g = z["gt"][i]; im = gray(n, z["files"][i]); X, th = L.ge.visible(se2(*g) @ Tbc, K, W, H); u, v, ok = L.project(X, g)
        S, Sth = lsd_segments(im); j, d = associate(u, v, th, S, Sth, 6); m = ok & (j >= 0)
        a, b = S[j[m], :2], S[j[m], 2:]; t = b - a; nr = np.stack([-t[:, 1], t[:, 0]], -1); nr /= np.linalg.norm(nr, axis=1, keepdims=True)
        p = np.stack([u[m], v[m]], -1); r = ((a - p) * nr).sum(1)       # image line minus map point along normal
        out = np.sign(((p - [K[0, 2], K[1, 2]]) * nr).sum(1)); r *= out   # + = image edge further out from centre
        zz = X[m, 2]; cls = np.where(zz < 0.05, "floor", np.where(zz > 2.2, "ceil", np.where(np.abs(th[m] - np.pi / 2) < 0.3, "vert", "other")))
        for c, rr in zip(cls, r): R.setdefault(c, []).append(rr)
for c, rr in R.items(): rr = np.array(rr); print(c, len(rr), "median %.2f px  trimmed-mean(|r|<3) %.2f" % (np.median(rr), rr[np.abs(rr) < 3].mean()))
