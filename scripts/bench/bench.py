"""Common SLAM benchmark harness on recorded scan logs (rec/<env>.pkl + GT track rec/<env>.json).
python bench.py ENV LEVEL SEED METHOD     METHOD: odom | slam2 | kiss
Odometry corruption = differential-drive wheel odometry model (no IMU): wheel-radius scale error (systematic),
effective-track error (systematic, rotation-proportional), random scrub per rotation (sqrt growth), random
distance noise.  Levels after Borenstein UMBmark / typical values (roboticsprojecthub diff-drive odometry):
  realistic   radius 0.5 %, track 2 %, random scrub 2 % per pivot, 0.5 % distance noise
  pessimistic radius 1.5 %, track 4 %, random scrub 5 %, 1.5 % distance noise (stress case)
Metrics: ATE RMSE (xy, no alignment: same start), RPE per ~1 m path, final error, max yaw error, ms/scan."""
import sys, pickle, json, math, time, numpy as np
sys.path[:0] = [__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')), __import__('os').path.join(__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..')), 'src')]
from nav.slam import wrap, compose, inverse
env, lvl, seed, meth = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
P = dict(ideal=(0, 0, 0, 0), realistic=(0.005, 0.02, 0.02, 0.005), pessimistic=(0.015, 0.04, 0.05, 0.015))[lvl]
rng = np.random.default_rng(seed); sg = lambda: rng.choice([-1, 1])
r_e, t_e = P[0] * sg(), P[1] * sg()
rec = pickle.load(open(__import__('os').path.expanduser('~/.kevin/emu/rec/%s.pkl') % env, 'rb')); tr = json.load(open(__import__('os').path.expanduser('~/.kevin/emu/rec/%s.json') % env))['track']
tg = np.array([r['t'] for r in tr]); G = np.array([r['gt'] for r in tr]); G[:, 2] = np.unwrap(G[:, 2])
def gti(t): return np.array([np.interp(t, tg, G[:, i]) for i in range(3)])
if meth == 'slam2':
    from nav import slam2 as S; s = S.Slam2(); s._run = False
elif meth == 'kiss':
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    cfg = KISSConfig(); cfg.data.max_range = 6.0; cfg.data.min_range = 0.0; cfg.data.deskew = False; cfg.mapping.voxel_size = cfg.data.max_range / 100.0   # KISS default rule
    k = KissICP(cfg)
est, gt, ms = [], [], []; prev = pc = tp = None
for job in rec:
    t, po, obs, free = job[:4]
    if prev is None: pc = tuple(po)
    else:
        d = compose(inverse(prev), po); ds = math.hypot(d[0], d[1])
        dth = d[2] * (1 + t_e) + P[2] * abs(d[2]) * rng.standard_normal()      # systematic track + random scrub
        sc = (1 + r_e) * (1 + P[3] * rng.standard_normal() / max(math.sqrt(ds / 0.05), 1e-3) if ds > 1e-4 else 1)
        pc = compose(pc, (d[0] * sc, d[1] * sc, dth))
    prev, tp = tuple(po), t
    t0 = time.perf_counter()
    if meth == 'odom': pm = pc
    elif meth == 'slam2': s._step(t, pc, obs, free); pm = compose(s.T, pc)
    elif meth == 'kiss':
        pts = np.asarray(obs, np.float64)
        if len(pts) < 20: pm = est[-1] if est else (0, 0, 0)
        else:
            p3 = np.c_[pts[:, :2], np.zeros(len(pts))]; k.register_frame(p3, np.zeros(len(p3)))
            T = k.last_pose; pm = (T[0, 3], T[1, 3], math.atan2(T[1, 0], T[0, 0]))
    ms.append((time.perf_counter() - t0) * 1e3)
    if tg[0] <= t <= tg[-1]: est.append(pm); gt.append(gti(t))
E, Gt = np.array(est), np.array(gt)
ex = np.hypot(E[:, 0] - Gt[:, 0], E[:, 1] - Gt[:, 1]); ey = np.abs([wrap(a - b) for a, b in zip(E[:, 2], Gt[:, 2])])
path = np.r_[0, np.cumsum(np.hypot(*np.diff(Gt[:, :2], axis=0).T))]
rpe = []; j = 0
for i in range(len(path)):
    while j < len(path) and path[j] - path[i] < 1.0: j += 1
    if j >= len(path): break
    dg = compose(inverse(tuple(Gt[i])), tuple(Gt[j])); de = compose(inverse(tuple(E[i])), tuple(E[j]))
    rpe.append(math.hypot(dg[0] - de[0], dg[1] - de[1]))
print(json.dumps(dict(env=env, lvl=lvl, seed=seed, m=meth, path=round(path[-1], 1), ate=round(float(np.sqrt((ex ** 2).mean())), 3),
      final=round(float(ex[-1]), 3), max=round(float(ex.max()), 3), yawmax=round(math.degrees(ey.max()), 1),
      rpe1m=round(float(np.mean(rpe)), 3) if rpe else None, ms=round(float(np.mean(ms)), 2), ms_p99=round(float(np.percentile(ms, 99)), 1))))
