"""How well do the map's structural edges match the sim render at the GT pose?  (optimism caveat)"""
from harness import *
res = {}
for n in sys.argv[1:]:
    z = load(n); idx = np.linspace(0, len(z["t"]) - 1, 40).astype(int); d_all = []; cov = []
    for i in idx:
        dt, e = image_dt(gray(n, z["files"][i])); X, E = smap.edge_points(se2(*z["gt"][i]) @ Tbc, K, W, H, max_pts=10 ** 6)
        L = Localizer(smap, K, W, H, Tbc); u, v, ok = L.project(X, z["gt"][i]); d = dt[v[ok].astype(int), u[ok].astype(int)]; d_all.append(d)
        dtm = cv2.distanceTransform((~E).astype(np.uint8), cv2.DIST_L2, 5); cov.append(float((dtm[e > 0] < 2).mean()))   # image edges explained by map
        if i == idx[10]:
            vis = cv2.cvtColor(gray(n, z["files"][i]), cv2.COLOR_GRAY2BGR); vis[e > 0] = (0, 255, 0); vis[E] = (0, 0, 255); cv2.imwrite(f"/workspace/maploc/gtcheck_{n}.png", vis)
    d = np.concatenate(d_all)
    res[n] = dict(map_edge_to_image_px_med=float(np.median(d)), frac_map_edges_within_2px=float((d < 2).mean()), frac_image_edges_explained_by_map=float(np.median(cov)))
    print(n, json.dumps(res[n]))
