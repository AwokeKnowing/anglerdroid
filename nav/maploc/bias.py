from harness import *
L = Localizer(smap, K, W, H, Tbc)
for n in sys.argv[1:]:
    z = load(n); idx = np.linspace(0, len(z["t"]) - 1, 40).astype(int); D = []
    for i in idx:
        p, P, info = L.update(gray(n, z["files"][i]), z["gt"][i], np.diag([0.1**2, 0.1**2, math.radians(5)**2]))
        if info.get("ok"): D.append(rel(z["gt"][i], p))
    D = np.array(D); print(n, len(D), "corr at GT: med dx %.3f dy %.3f dyaw %.2fdeg | |d| med %.3f m, |dyaw| med %.2f deg" % (*np.median(D[:, :2], 0), math.degrees(np.median(D[:, 2])), np.median(np.hypot(D[:, 0], D[:, 1])), math.degrees(np.median(np.abs(D[:, 2])))))
