"""Joint local refinement of ONE base position seen by several real mast looks (in-place turns): (x, y, yaw0) with each
look's heading = yaw0 + its odometry yaw offset; score = mean truncated chamfer of rendered model edges to the looks'
long LSD segments (poses.real_dt).  Reports the score at the start pose, the refined pose, and the per-look scores.
usage: local_pose.py <out.json> <x> <y> <yaw_deg> <look_dir>:<rel_yaw_rad> [...]   env LP_WIN (m, 0.35), LP_WINDEG (10)"""
import json, math, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poses as P  # noqa: E402

out = sys.argv[1]; c = (float(sys.argv[2]), float(sys.argv[3]), math.radians(float(sys.argv[4])))
items = [a.rsplit(":", 1) for a in sys.argv[5:]]
looks = [(P.real_dt(cv2.imread(os.path.join(d, "mast_rgb.png"))), float(o)) for d, o in items]
mesh = P.camgeo.load_casita()
def full(b): return [P.score(mesh, (b[0], b[1], b[2] + o), DT) for DT, o in looks]
s_start = full(c)
cur = c; span = (float(os.environ.get("LP_WIN", 0.35)),) * 2 + (math.radians(float(os.environ.get("LP_WINDEG", 10))),); n = 13
for it in range(4):
    Xs = [P.edge_points(mesh, P.mast_cam((cur[0], cur[1], cur[2] + o))) for _, o in looks]
    def sc(b): return float(np.mean([P.score_pts(X, (b[0], b[1], b[2] + o), DT) for X, (DT, o) in zip(Xs, looks)]))
    bs, bb = sc(cur), cur
    for dx in np.linspace(-span[0], span[0], n):
        for dy in np.linspace(-span[1], span[1], n):
            for dt in np.linspace(-span[2], span[2], n):
                b = (cur[0] + dx, cur[1] + dy, cur[2] + dt); v = sc(b)
                if v < bs: bs, bb = v, b
    cur = bb; span = tuple(v * 0.35 for v in span)
s_end = full(cur)
res = dict(start=[c[0], c[1], math.degrees(c[2])], base=[cur[0], cur[1], math.degrees(cur[2])], looks=[d for d, _ in items], rel_yaw=[float(o) for _, o in items],
           score_start_px=s_start, score_end_px=s_end, mean_start=float(np.mean(s_start)), mean_end=float(np.mean(s_end)),
           moved_cm=[round((cur[0] - c[0]) * 100, 1), round((cur[1] - c[1]) * 100, 1)], dyaw_deg=round(math.degrees(cur[2] - c[2]), 2))
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: (np.round(v, 2).tolist() if isinstance(v, list) and v and isinstance(v[0], float) else v) for k, v in res.items() if k != "looks"}))
