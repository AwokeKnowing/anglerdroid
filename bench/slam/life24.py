"""24 h (-> 1 month) continuous autonomous wander of Kevin in the MuJoCo twin: slam2 (LIFE mode) vs RTAB-Map, online.

ONE process per seed steps emu.world IN-PROCESS (no shm/sockets: never touches a live emu.world/nav.run) and feeds the SAME
simulated sensor stream to both SLAM systems as it is produced:
  slam2   nav/slam2.py (KEVIN_SLAM2_LIFE=1: reduced pose graph, DCS loops, node refresh)   rs2_ir1 depth @5 Hz, synchronous
  RTAB    rtabmap 0.22 library (sim/rtabmap/rtabmap_stream, defaults + external wheel+gyro odometry)  RGB-D @1 Hz
Sensor timing is identical to real time (sim clock); the sim runs faster than real time and WAITS for each system's
update to finish before the next one is due (both are given their compute; the ms/frame metric says whether it would
keep up on the robot).
Noise: emu.world realistic model (wheel radius/track error with the real ~+6 % yaw over-report (KEVIN_EMU_YAW_EPS),
encoder noise, gyro scale/bias/random walk, D4xx depth noise), IMU stamped 42.6 ms late (NP imu_dt), lumpy carpet hfield,
dressed rooms (emu/realism.py), 1-2 chairs moved every 6 h while the robot is away.
Ground truth: MuJoCo base pose; GT occupancy updated when chairs move.

Every 10 min sim: ATE (window + cumulative), drift, map IoU vs twin GT, map stability (IoU vs 1 h ago), loop closures
accepted / false (vs GT), ms per frame, RSS, disk, nodes -> OUT/metrics.jsonl.  Every 2 min: side-by-side map PNG
(OUT/frames/) for the video.  Hourly: slam2 save() (persistence) + RTAB DB size.

python bench/slam/life24.py --seed 0 --hours 24 --out ~/slambench/res24h/seed0
"""
from __future__ import annotations
import argparse, collections, heapq, json, math, os, shutil, subprocess, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, HERE]
from record import astar

wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi
def compose(a, b): c, s = math.cos(a[2]), math.sin(a[2]); return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))
def inverse(a): c, s = math.cos(a[2]), math.sin(a[2]); return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])
def rss_mb(pid):
    try:
        for l in open("/proc/%d/status" % pid):
            if l.startswith("VmRSS"): return int(l.split()[1]) / 1024
    except OSError: pass
    return 0.0
def cpu_s(pid):
    try: f = open("/proc/%d/stat" % pid).read().rsplit(")", 1)[1].split(); return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
    except OSError: return 0.0
def du_mb(p):
    if not os.path.exists(p): return 0.0
    if os.path.isfile(p): return os.path.getsize(p) / 2**20
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) / 2**20


class Rtab:
    """rtabmap_stream subprocess; one frame in flight (send k+1 only after reply k)."""
    def __init__(self, exe, camfile, db, params, log):
        env = dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser("~/slambench/z/rtabmap_rt/lib"), OMP_NUM_THREADS="1")
        self.p = subprocess.Popen([exe, camfile, db] + ["%s=%s" % kv for kv in params.items()], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "a"), env=env, bufsize=0, cwd=os.path.dirname(log))
        self.pending = False; self.last = None; self.ms = []
    def send(self, t, odom, sd, bgr, dep):
        self.wait()
        self.p.stdin.write(("F %.4f %.6f %.6f %.7f %.6g %.6g %.6g\n" % (t, *odom, *sd)).encode() + bgr.tobytes() + dep.tobytes()); self.p.stdin.flush(); self.pending = True
    def wait(self):
        if not self.pending: return self.last
        l = self.p.stdout.readline().decode().split()
        if not l or l[0] != "R": raise RuntimeError("rtabmap_stream died: %r" % l)
        self.last = dict(t=float(l[1]), id=int(l[2]), wm=int(l[3]), loop=int(l[4]), prox=int(l[5]), ms=float(l[6]), corr=(float(l[7]), float(l[8]), float(l[9])), exc=int(l[10]) if len(l) > 10 else 0,
                         hyp=float(l[11]) if len(l) > 11 else -1, hyp_id=int(float(l[12])) if len(l) > 12 else -1, rej=int(float(l[13])) if len(l) > 13 else -1, inl=int(float(l[14])) if len(l) > 14 else -1)
        self.hyps = getattr(self, "hyps", []); self.hyps.append((self.last["hyp"], self.last["rej"], self.last["inl"]))
        self.exc = getattr(self, "exc", 0) + self.last["exc"]
        self.ms.append(self.last["ms"]); self.pending = False; return self.last
    def graph(self):
        self.wait(); self.p.stdin.write(b"Q\n"); self.p.stdin.flush()
        n_, l_ = map(int, self.p.stdout.readline().decode().split()[1:3]); N = {}; L = []
        for _ in range(n_):
            s = self.p.stdout.readline().decode().split(); N[int(s[1])] = (float(s[2]), float(s[3]), float(s[4]))
        for _ in range(l_):
            s = self.p.stdout.readline().decode().split(); L.append((int(s[1]), int(s[2]), int(s[3]), (float(s[4]), float(s[5]), float(s[6]))))
        assert self.p.stdout.readline().decode().strip() == "END"; return N, L
    def close(self):
        try: self.wait(); self.p.stdin.write(b"X\n"); self.p.stdin.flush(); self.p.stdout.readline(); self.p.wait(60)
        except Exception: self.p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="casita"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=24.0); ap.add_argument("--noise", default="realistic")
    ap.add_argument("--chair-every-h", type=float, default=6.0); ap.add_argument("--metric-s", type=float, default=600.0); ap.add_argument("--frame-s", type=float, default=120.0)
    ap.add_argument("--rtab", default="1"); ap.add_argument("--rtab-db", default="disk", help="disk (rtabmap_ros default ~/.ros-like: OUT/rtabmap.db) | ram")
    ap.add_argument("--rtab-param", action="append", default=[], help="k=v RTAB-Map parameter override (default: library defaults)")
    ap.add_argument("--rtab-rss-cap-gb", type=float, default=8.0); ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--imu-dt", type=float, default=-0.0426, help="IMU stamp offset vs wheel clock (Kevin measured 42.6 ms wheel-vs-IMU lag)")
    ap.add_argument("--v", type=float, default=0.30); ap.add_argument("--w", type=float, default=0.8)
    a = ap.parse_args()
    out = os.path.abspath(os.path.expanduser(a.out)); os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["KEVIN_EMU_NOISE"] = a.noise; os.environ["KEVIN_EMU_DOG"] = "0"; os.environ["KEVIN_MUTE"] = "1"
    os.environ.setdefault("KEVIN_SLAM2_LIFE", "1")
    if os.environ.get("KEVIN_TORCH_SITE"): sys.path.append(os.environ["KEVIN_TORCH_SITE"])   # sim venv has no torch: borrow the robot's
    import cv2, mujoco
    from scipy import ndimage
    from emu import world as W, envs, realism
    class Sim(W.World):
        def _setup_io(self): pass
    RP = W._robot_params()
    ns = argparse.Namespace(env=a.env, seed=a.seed, tire=RP.V["wheels"], ir=1, subpix=0.08, holes=0.004, min_z=0.175, band=True, rt=1.0)
    S = Sim(ns); m, d = S.m, S.d; NP = W.NP; NP["imu_dt"] = a.imu_dt
    sys.path.append(os.path.join(REPO, "src")); import twin_geometry as TG
    from nav.slam import Odom, GridMap
    from nav.slam2 import Slam2
    import nav.slam2 as S2
    from nav.perceive import Rs2Cloud
    import occmap as OM
    for n in S.nodes: n.state = n.CLOSED; n.wd_en = False; n.ax.reset()
    e = S.env; rng = np.random.default_rng(1000 + a.seed)
    log = open(os.path.join(out, "log.txt"), "a")
    def P(*s):
        msg = "[%s t=%.0fs] " % (time.strftime("%H:%M:%S"), d.time) + " ".join(str(x) for x in s); print(msg, flush=True); log.write(msg + "\n"); log.flush()
    # ---------------- ground truth occupancy (GT_RES 2 cm), with movable chairs
    GTR = envs.GT_RES; org = np.asarray(e.gt_origin, float); GT0 = e.gt.copy()
    chairs = []
    for cx, cy in ((4.86, 6.2), (6.34, 6.2), (5.6, 5.62), (5.6, 6.78), (2.7, 12.95)):
        ids = [i for i, g in enumerate(e.geoms) if g.kind == "box" and g.mat == "wood" and math.hypot(g.pos[0] - cx, g.pos[1] - cy) < 0.3 and g.pos[2] < 1.0]
        if len(ids) == 6: chairs.append(dict(c=np.array([cx, cy]), ids=ids, off=np.zeros(2), rot=0.0))
    chair_ids = {i for c in chairs for i in c["ids"]}
    gid = {i: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "env%d" % i) for i in chair_ids}
    base_geom = {i: (np.array(e.geoms[i].pos, float), float(e.geoms[i].yaw)) for i in chair_ids}
    def rast(ids, pose_of):
        occ = np.zeros_like(GT0)
        for i in ids:
            p, yw = pose_of(i); g = e.geoms[i]; envs._raster_box(envs.Geom("box", tuple(p), g.size, yw, g.mat), occ, org)
        return ndimage.binary_dilation(occ, iterations=2)
    def chair_pose(i):
        c = next(c for c in chairs if i in c["ids"]); p0, y0 = base_geom[i]; cc, ss = math.cos(c["rot"]), math.sin(c["rot"])
        r = p0[:2] - c["c"]; q = c["c"] + c["off"] + np.array([cc * r[0] - ss * r[1], ss * r[0] + cc * r[1]]); return np.array([q[0], q[1], p0[2]]), y0 + c["rot"]
    nonchair = [i for i, g in enumerate(e.geoms) if g.kind in ("box", "cyl") and i not in chair_ids]
    A_ = rast([i for i in nonchair if e.geoms[i].kind == "box"], lambda i: (np.array(e.geoms[i].pos), e.geoms[i].yaw))
    Cc0 = rast(chair_ids, lambda i: base_geom[i])
    walls = ndimage.binary_dilation(getattr(e, "wall_occ", np.zeros_like(GT0)), iterations=2)
    GT_static = A_ | (GT0 & ~Cc0) | (GT0 & Cc0 & walls)
    gt_now = [GT0.copy()]
    P("chairs found %d (%s); GT static recon diff %d cells" % (len(chairs), [tuple(c["c"]) for c in chairs], int(((GT_static | Cc0) ^ GT0).sum())))
    def pool(o):
        H, Wd = o.shape[0] // 2, o.shape[1] // 2; return o[:H * 2, :Wd * 2].reshape(H, 2, Wd, 2).any(axis=(1, 3))
    # ---------------- planning cost (4 cm) from GT clearance
    cost = [None]
    def rebuild_cost():
        # keep-in: a 0.7 m band at the map border counts as wall, so exterior doorways (open to the edge of the twin)
        # are not driven through (seed 0 smoke: left by the west door at t~445 s, wedged tilted on the hfield edge)
        occ = gt_now[0].copy(); bw = int(float(os.environ.get("KEVIN_KEEPIN_M", "0.7")) / GTR); occ[:bw] = occ[-bw:] = True; occ[:, :bw] = occ[:, -bw:] = True
        clr = ndimage.distance_transform_edt(~occ) * GTR; c = np.where(clr >= 0.30, 1.0 + 0.6 / np.maximum(clr, 0.05), np.inf)
        H, Wd = c.shape[0] // 2, c.shape[1] // 2; cost[:] = [c[:H * 2, :Wd * 2].reshape(H, 2, Wd, 2).max(axis=(1, 3)), clr]
    rebuild_cost()
    if os.environ.get("DUMPGT"):
        lab, _ = ndimage.label(np.isfinite(cost[0])); r0, c0 = int((e.start[1] - org[1]) / (2 * GTR)), int((e.start[0] - org[0]) / (2 * GTR))
        reach = lab == lab[r0, c0]; print("REACH m2", reach.sum() * (2 * GTR) ** 2, "free m2", (~GT0).sum() * GTR ** 2)
        ys, xs = np.nonzero(reach); print("x", org[0] + xs.min() * 2 * GTR, org[0] + xs.max() * 2 * GTR, "y", org[1] + ys.min() * 2 * GTR, org[1] + ys.max() * 2 * GTR); sys.exit(0)
    def rc4(x, y): return int((y - org[1]) / (2 * GTR)), int((x - org[0]) / (2 * GTR))
    def xy4(r, c): return (org[0] + (c + 0.5) * 2 * GTR, org[1] + (r + 0.5) * 2 * GTR)
    def snap(rc_):
        cst = cost[0]
        if 0 <= rc_[0] < cst.shape[0] and 0 <= rc_[1] < cst.shape[1] and np.isfinite(cst[rc_]): return rc_
        rr, cc = np.nonzero(np.isfinite(cst)); k = np.argmin((rr - rc_[0]) ** 2 + (cc - rc_[1]) ** 2); return (int(rr[k]), int(cc[k]))
    places = list(e.places.values())
    # ---------------- sensors
    DW, DH, CW, CH = W.DW, W.DH, W.CW, W.CH
    Rd = mujoco.Renderer(m, DH, DW); Rd.enable_depth_rendering(); Rc = mujoco.Renderer(m, CH, CW); Rcd = mujoco.Renderer(m, CH, CW); Rcd.enable_depth_rendering()
    nrng = np.random.default_rng(a.seed + 7); dr = mujoco.MjData(m)
    def depth(R, cam, w, h):
        fpx = W._intr(w, h, W.FOVY_D if cam == "rs2_ir1" else W.FOVY_C)["fx"]; cols = np.arange(w, dtype=np.float32)[None, :]
        R.update_scene(dr, cam); z = R.render()
        zz = z + (z * z) * (ns.subpix / (fpx * W.BASELINE)) * nrng.standard_normal(z.shape, dtype=np.float32)
        bad = (z < ns.min_z) | (z > 8.0) | (nrng.random(z.shape, dtype=np.float32) < ns.holes) | (cols < fpx * W.BASELINE / np.maximum(z, 1e-3))
        xr = cols - (fpx * W.BASELINE) / np.maximum(z, 1e-3); smin = np.minimum.accumulate(xr[:, ::-1], axis=1)[:, ::-1]
        bad[:, :-1] |= smin[:, 1:] <= xr[:, :-1] - 0.5
        zz = zz * 1000.0; zz[bad] = 0; return zz.astype(np.uint16)
    ci = W._intr(CW, CH, W.FOVY_C); di = W._intr(DW, DH, W.FOVY_D)
    camfx = realism.CamFX(CW, CH, [[ci["fx"], 0, ci["ppx"]], [0, ci["fy"], ci["ppy"]], [0, 0, 1]], exposure_s=0.010, seed=2) if realism.REAL else None
    def ext(name): t, R = TG.frame(name); return dict(t=np.asarray(t).tolist(), R=np.asarray(R).tolist())
    meta = dict(intr=dict(rs2_depth=di), extr_twin_base_from_optical=dict(rs2_depth=ext("rs2_depth")))
    DS = OM.DepthScan(meta); floor_z = -float(TG.AXLE_H_M)
    cloud = Rs2Cloud(dict(di, width=DW, height=DH))
    # ---------------- estimators
    slam = Slam2(GridMap()); assert slam.do_match, "slam2 needs torch (set KEVIN_TORCH_SITE)"; o_r = Odom()              # o_r: RTAB's external odometry (same wheel+gyro, = odom-only baseline)
    tC, RC = TG.frame("rs2_color"); camfile = os.path.join(out, "cam_rs2_color.txt")
    with open(camfile, "w") as f: f.write("%.9g %.9g %.9g %.9g %d %d\n" % (float(ci["fx"]), float(ci["fy"]), float(ci["ppx"]), float(ci["ppy"]), CW, CH) + " ".join("%.9g" % float(v) for v in np.c_[np.asarray(RC), np.asarray(tC)].ravel()) + "\n")
    rparams = {"Mem/GenerateIds": "false"}                 # bookkeeping only: node id = frame number (maps to sim time)
    for kv in a.rtab_param: k_, v_ = kv.split("=", 1); rparams[k_] = v_
    rdb = os.path.join(out, "rtabmap.db") if a.rtab_db == "disk" else ""
    if rdb and os.path.exists(rdb): os.remove(rdb)
    rt = Rtab(os.path.join(REPO, "sim/rtabmap/rtabmap_stream"), camfile, rdb, rparams, os.path.join(out, "rtabmap.log")) if a.rtab == "1" else None
    rtab_capped = False; rt_id_t = {}
    gx = Grid = OM.Grid; GS = (GT0.shape[0] // 2, GT0.shape[1] // 2)
    grids = {k: Grid(org, GS) for k in ("gt", "slam2", "rtab", "odom")}
    start = e.start; A0 = tuple(start)                     # estimator frames start at (0,0,0) = GT start pose
    json.dump(dict(args=vars(a), seed=a.seed, env=a.env, start=list(start), noise=a.noise, np_noise={k: (v if isinstance(v, (int, float)) else str(v)) for k, v in NP.items()},
                   drawn=dict(wheel_r_err=(S.rerr - 1).tolist(), track_err=float(S.trerr - 1), gyro_bias=S.gbias.tolist(), gyro_scale_err=(S.gscale - 1).tolist()),
                   yaw_eps=os.environ.get("KEVIN_EMU_YAW_EPS"), realism=realism.REAL, rtab_params=rparams, rtab_db=rdb, slam2_life=S2.LIFE,
                   chairs=[list(c["c"]) for c in chairs]), open(os.path.join(out, "config.json"), "w"), indent=1)
    P("start", start, "slam2 LIFE", S2.LIFE, "rtab", bool(rt), rparams, "imu_dt", NP["imu_dt"])
    # ---------------- control
    r_w = TG.WHEEL_R_M; track = TG.TRACK_M
    def cmd(v, w):
        vl, vr = v - w * track / 2, v + w * track / 2
        S.nodes[0].ax.input_vel = -vl / (2 * math.pi * r_w); S.nodes[1].ax.input_vel = vr / (2 * math.pi * r_w)
    def pose():
        q = d.qpos; return float(q[0]), float(q[1]), math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
    st = dict(mode="dwell", until=3.0); vc = wc = 0.0; ev = collections.Counter()
    def new_goal():
        x, y, _ = pose()
        for _ in range(20):
            if rng.random() < 0.6: gx_, gy_ = places[rng.integers(len(places))][:2]
            else:
                rr, cc = np.nonzero(cost[1] >= 0.45); k = rng.integers(len(rr)); gx_, gy_ = org[0] + (cc[k] + .5) * GTR, org[1] + (rr[k] + .5) * GTR
            if math.hypot(gx_ - x, gy_ - y) < 1.0: continue
            s0, g0 = snap(rc4(x, y)), snap(rc4(gx_, gy_)); path = astar(cost[0], s0, g0)
            if path and len(path) > 5: return np.array([xy4(r, c) for r, c in path[::2]] + [xy4(*path[-1])])
        return None
    # ---------------- IMU/wheel plumbing (causal gyro<->wheel offset estimate, as tools/prep_rtx imu_offset)
    imu_buf = collections.deque(maxlen=400); off = [0.0, False]; xc_w, xc_g = [], []
    t_prev_w = None
    def est_offset():
        tw = np.array([p[0] for p in xc_w]); ww = np.array([p[1] for p in xc_w]); tg = np.array([p[0] for p in xc_g]); gg = np.array([p[1] for p in xc_g])
        best = (-1, 0.0)
        for o in np.arange(-0.1, 0.1001, 0.0005):
            g_ = np.interp(tw + o, tg, gg); c = np.corrcoef(ww, g_)[0, 1]
            if c > best[0]: best = (c, o)
        return best
    # ---------------- metrics state
    gth = collections.deque(maxlen=int(3600 * 10 * 26))       # (t, x, y, yaw) @10 Hz, ~26 h ring (month: per-day chunks TODO)
    err = []                                                  # (t, e_slam2, e_rtab, e_odom, yaw errs...)
    snaps = collections.deque(maxlen=8); path_len = 0.0; last_xy = None
    s2_ms = []; s2_loops_seen = 0; t_wall0 = time.time(); cpu0 = cpu_s(os.getpid()); rcpu0 = 0.0
    rt_loops = {}                                             # (from,to) -> (t_from, t_to, z, type)
    sd_acc = np.zeros(3); last_cam_pose = None; rt_pose_w = None
    mfile = open(os.path.join(out, "metrics.jsonl"), "a"); efile = open(os.path.join(out, "err_1hz.csv"), "a")
    efile.write("t,gt_x,gt_y,gt_yaw,slam2_x,slam2_y,slam2_yaw,rtab_x,rtab_y,rtab_yaw,odom_x,odom_y,odom_yaw,gyro_int,wheel_int,upz\n")
    def gt_at(t):
        k = np.searchsorted(gth_t, t)
        if k <= 0 or k >= len(gth_t): return None
        return gth_a[k]
    def est_world(p): return compose(A0, p)
    def rel_err(z, ga, gb):
        g = compose(inverse(ga), gb); return math.hypot(z[0] - g[0], z[1] - g[1]), abs(wrap(z[2] - g[2]))
    dt = m.opt.timestep; n_imu = n_enc = n_cam = 0; n_rt = 0; next_metric = a.metric_s; next_frame = a.frame_s; next_save = 3600.0
    chair_times = [h * 3600 for h in np.arange(a.chair_every_h, a.hours + 1e-6, a.chair_every_h)]
    cam_hz = 15.0; T_END = a.hours * 3600.0
    gth_t = np.zeros(0); gth_a = None; gint = [math.pi / 2, math.pi / 2]
    while d.time < T_END:
        tnow = d.time
        # ---- wander controller @50 Hz
        if int(tnow / dt) % max(1, int(0.02 / dt)) == 0:
            x, y, th = pose(); v_t = w_t = 0.0; md = st["mode"]
            if md == "dwell":
                if tnow >= st["until"]:
                    Pth = new_goal()
                    if Pth is None: st = dict(mode="dwell", until=tnow + 5); ev["no_path"] += 1
                    else: st = dict(mode="path", P=Pth, k=0, prog=(tnow, x, y)); ev["goals"] += 1
            elif md == "spin":
                st["acc"] += wrap(th - st["last"]); st["last"] = th; rem = st["ang"] - st["acc"]
                if abs(rem) < 0.03 or tnow > st["tmax"]: st = dict(mode="dwell", until=tnow + 0.8)
                else: w_t = math.copysign(min(a.w, max(0.15, 1.5 * abs(rem))), rem)
            elif md == "back":
                if tnow >= st["until"]: st = dict(mode="dwell", until=tnow + 0.5)
                else: v_t = -0.10
            elif md == "path":
                Pp = st["P"]; k = st["k"]; dd = np.hypot(Pp[:, 0] - x, Pp[:, 1] - y); k = max(k, int(np.argmin(dd[k:k + 40])) + k); st["k"] = k
                j = k
                while j < len(Pp) - 1 and math.hypot(Pp[j, 0] - x, Pp[j, 1] - y) < 0.35: j += 1
                gxx, gyy = Pp[j]; dist_end = math.hypot(Pp[-1, 0] - x, Pp[-1, 1] - y)
                if tnow - st["prog"][0] > 20.0:
                    if math.hypot(x - st["prog"][1], y - st["prog"][2]) < 0.10: st = dict(mode="back", until=tnow + 2.0); ev["stuck"] += 1
                    else: st["prog"] = (tnow, x, y)
                if st["mode"] != "path": pass
                elif dist_end < 0.08 or (k >= len(Pp) - 2 and dist_end < 0.15):
                    u = rng.random()
                    if u < 0.25: st = dict(mode="spin", ang=float(rng.choice([-1, 1]) * rng.uniform(math.pi / 2, 2 * math.pi)), acc=0.0, last=th, tmax=tnow + 30); ev["spins"] += 1
                    else: st = dict(mode="dwell", until=tnow + (rng.uniform(5, 60) if u < 0.55 else 0.5))
                else:
                    er = wrap(math.atan2(gyy - y, gxx - x) - th)
                    if abs(er) > 0.7: w_t = math.copysign(min(a.w, 1.6 * abs(er)), er)
                    else: v_t = min(a.v, 0.6 * dist_end + 0.05) * max(0.2, math.cos(er)); w_t = max(-a.w, min(a.w, 2.2 * er))
            vc += max(-0.02, min(0.02, v_t - vc)); wc += max(-0.06, min(0.06, w_t - wc)); cmd(vc, wc)
            # chair moves (robot >2 m away from the chair)
            if chair_times and tnow >= chair_times[0] and chairs:
                moved = []
                for c in rng.permutation(len(chairs))[:int(rng.integers(1, 3))]:
                    c = chairs[c]
                    for _ in range(30):
                        off_ = c["off"] + rng.uniform(-0.6, 0.6, 2); rot = c["rot"] + rng.uniform(-0.8, 0.8)
                        if not (0.3 <= np.linalg.norm(off_ - c["off"]) <= 0.8): continue
                        if math.hypot(*(c["c"] + off_ - np.array([x, y]))) < 2.0: continue
                        old = (c["off"].copy(), c["rot"]); c["off"], c["rot"] = off_, rot
                        newocc = rast(c["ids"], chair_pose); others = rast([i for cc in chairs if cc is not c for i in cc["ids"]], chair_pose)
                        clash = (newocc & (GT_static | others)).sum() > 40 or ndimage.binary_dilation(newocc, iterations=12)[rc4(x, y)[0] * 2, rc4(x, y)[1] * 2]
                        if clash: c["off"], c["rot"] = old; continue
                        moved.append((tuple(np.round(c["c"], 2)), tuple(np.round(off_ - old[0], 2)), round(math.degrees(rot - old[1]))))
                        for i in c["ids"]:
                            p_, yw = chair_pose(i); m.geom_pos[gid[i]] = p_; m.geom_quat[gid[i]] = (math.cos(yw / 2), 0, 0, math.sin(yw / 2))
                        break
                gt_now[0] = GT_static | rast(chair_ids, chair_pose); rebuild_cost(); mujoco.mj_forward(m, d)
                chair_times.pop(0); ev["chair_moves"] += len(moved); P("chairs moved", moved)
        S.step(time.monotonic())
        tnow = d.time
        # ---- IMU 200 Hz (stamped late by NP imu_dt; harness only sees the stamp)
        if tnow >= n_imu * 0.005:
            n_imu += 1
            if NP["g_rw"]: S.grw += S.rng.normal(0, NP["g_rw"] * math.sqrt(0.005), 3).astype(np.float32)
            g = (d.sensordata[S.gyro:S.gyro + 3].astype(np.float32) * S.gscale + S.gbias + S.grw + np.float32(NP["g_drift"] * tnow) + S.rng.normal(0, NP["g_white"], 3).astype(np.float32))
            gz = float((TG.IMU_R @ ((g - TG.GYRO_BIAS) * TG.GYRO_SCALE))[2]); ts = tnow + NP["imu_dt"]
            imu_buf.append((ts, gz))
            if not off[1]: xc_g.append((ts, gz))
        # ---- wheels 100 Hz -> both odometries
        if tnow >= n_enc * 0.01:
            n_enc += 1; S.contact()
            f = [nd.dir * nd.vel * S.rerr[k] for k, nd in enumerate(S.nodes[:2])]
            mm, dd_ = 0.5 * (f[0] + f[1]), 0.5 * (f[1] - f[0]) / S.trerr; f = [mm - dd_, mm + dd_]
            vl, vr = [(fk + S.rng.normal(0, 0.002)) * 2 * math.pi * r_w for fk in f]
            lo = (t_prev_w if t_prev_w is not None else tnow - 0.01) + off[0]; hi = tnow + off[0]
            gs = [gq for tq, gq in imu_buf if lo < tq <= hi]
            if not gs and imu_buf:      # window not covered yet (stamps lag, offset not estimated): newest samples of the same span
                tl = imu_buf[-1][0]; gs = [gq for tq, gq in imu_buf if tl - (hi - lo) < tq <= tl]; ev["imu_window_fallback"] += 1
            gzm = float(np.mean(gs)) if gs else None
            if gzm is not None and t_prev_w is not None: gint[0] += gzm * (tnow - t_prev_w); gint[1] += (vr - vl) / track * (tnow - t_prev_w)
            t_prev_w = tnow
            cz = vc == 0 and wc == 0
            slam.odom.update(tnow, vl, vr, gzm, commanded_zero=cz); o_r.update(tnow, vl, vr, gzm, commanded_zero=cz)
            if not off[1]:
                xc_w.append((tnow, (vr - vl) / track))
                if tnow > 300.0 and np.std([p[1] for p in xc_w]) > 0.05:
                    c_, o_ = est_offset(); off[:] = [float(o_), True]; xc_w.clear(); xc_g.clear(); P("gyro-wheel offset estimate %.1f ms (corr %.3f; true imu_dt %.1f ms)" % (o_ * 1e3, c_, NP["imu_dt"] * 1e3))
            if n_enc % 10 == 0:
                x, y, th = pose(); gth.append((tnow, x, y, th))
                if last_xy is not None: path_len += math.hypot(x - last_xy[0], y - last_xy[1])
                last_xy = (x, y)
        # ---- camera clock 15 Hz: slam2 @5 Hz (rs2_ir1 depth), RTAB @1 Hz (rs2_color RGB + depth), grids @1 Hz
        if tnow >= n_cam / cam_hz:
            fi = n_cam; n_cam += 1
            po = o_r.pose
            if last_cam_pose is not None:     # per-frame odometry std devs (bench/slam/run_rtabmap.py), accumulated over 1 s
                dl = compose(inverse(last_cam_pose), po); ds_ = math.hypot(dl[0], dl[1]); dtf = 1 / cam_hz
                sd_acc += np.array([0.005 * ds_ + 1e-4, 0.001 * ds_ + 1e-4, 2.12e-4 * math.sqrt(dtf) + 0.0035 * dtf + 1e-5]) ** 2
            last_cam_pose = po
            do_s2 = fi % 3 == 0 and abs(slam.odom.w) < 0.6; do_grid = fi % 15 == 0; do_rt = fi % 15 == 0 and rt is not None and not rtab_capped
            if do_s2 or do_grid or do_rt:
                dr.qpos[:] = d.qpos; dr.mocap_pos[:] = d.mocap_pos; dr.mocap_quat[:] = d.mocap_quat; mujoco.mj_kinematics(m, dr); mujoco.mj_camlight(m, dr)
            if do_s2 or do_grid:
                dep = depth(Rd, "rs2_ir1", DW, DH)
                if do_s2:
                    obs, flo = cloud.points(dep); t0 = time.perf_counter()
                    slam._step(tnow, slam.odom.at(tnow), obs.astype(np.float32), flo[::3].astype(np.float32)); s2_ms.append((time.perf_counter() - t0) * 1e3)
            if do_rt:
                rt.wait()
                Rc.update_scene(dr, "rs2_color"); col = Rc.render()
                if camfx is not None: col = camfx(col, float(d.qvel[5]))
                cdep = depth(Rcd, "rs2_color", CW, CH)
                rt.send(tnow, po, np.sqrt(sd_acc), cv2.cvtColor(col, cv2.COLOR_RGB2BGR), cdep); n_rt += 1; rt_id_t[n_rt] = tnow; sd_acc[:] = 0
            if do_grid:
                gp = pose(); xs = compose(A0, slam.pose()); xo = compose(A0, o_r.pose)
                rlast = rt.last if rt is not None else None
                xr = compose(A0, compose(rlast["corr"], o_r.pose)) if rlast else xo
                err.append((tnow, math.hypot(xs[0] - gp[0], xs[1] - gp[1]), math.hypot(xr[0] - gp[0], xr[1] - gp[1]), math.hypot(xo[0] - gp[0], xo[1] - gp[1]),
                            abs(wrap(xs[2] - gp[2])), abs(wrap(xr[2] - gp[2])), abs(wrap(xo[2] - gp[2]))))
                efile.write("%.2f," % tnow + ",".join("%.4f,%.4f,%.5f" % q for q in (gp, xs, xr, xo)) + ",%.5f,%.5f,%.4f\n" % (gint[0], gint[1], 1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2)))
                if abs(float(d.qvel[5])) < 0.6:
                    Pb = DS.base_points(dep); ob, fl = DS.split(Pb, floor_z)
                    for k_, T_ in (("gt", gp), ("slam2", xs), ("rtab", xr), ("odom", xo)):
                        c_, s_ = math.cos(T_[2]), math.sin(T_[2]); Rm = np.array([[c_, -s_], [s_, c_]]); tt = np.array(T_[:2])
                        grids[k_].add(Rm @ DS.t[:2] + tt, ob[:, :2] @ Rm.T + tt, fl[::2, :2] @ Rm.T + tt)
        # ---- 2-min frame for the video
        if tnow >= next_frame:
            next_frame += a.frame_s
            go = pool(gt_now[0]); tiles = []
            E = np.array(err[-int(600):]) if err else None
            for k_, name in (("gt", "GT pose"), ("slam2", "slam2"), ("rtab", "RTAB-Map"), ("odom", "odom only")):
                L = grids[k_].L; img = np.full(L.shape + (3,), 128, np.uint8); img[L < -0.5] = 235; img[L > 0.5] = 20
                img[go & ~(L > 0.5)] = (img[go & ~(L > 0.5)] * 0.5 + np.array([60, 60, 230]) * 0.5).astype(np.uint8)
                img = np.ascontiguousarray(img[::-1]).repeat(2, 0).repeat(2, 1)
                gx_, gy_, _ = pose(); r_, c_ = rc4(gx_, gy_); cv2.circle(img, (c_ * 2, (L.shape[0] - 1 - r_) * 2), 5, (0, 160, 0), -1)
                lab = name if E is None or k_ == "gt" else "%s  err %.2f m" % (name, E[-1][{"slam2": 1, "rtab": 2, "odom": 3}[k_]])
                cv2.putText(img, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA); tiles.append(img)
            fr = np.concatenate(tiles, 1)
            cv2.putText(fr, "seed %d  sim %02d:%02d  path %.0f m  slam2 nodes %d  rtab nodes %s" % (a.seed, int(tnow // 3600), int(tnow % 3600 // 60), path_len, len(slam.kfs), rt.last["wm"] if rt and rt.last else "-"),
                        (6, fr.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(out, "frames", "%06d.png" % int(tnow)), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        # ---- hourly persistence (slam2 save + reload check)
        if tnow >= next_save:
            next_save += 3600.0; t0 = time.perf_counter(); slam.save(os.path.join(out, "slam2.pkl")); sv_ms = (time.perf_counter() - t0) * 1e3
            chk = Slam2(GridMap()); chk.load(os.path.join(out, "slam2.pkl")); P("slam2 saved %.1f MB in %.0f ms; reload nodes %d/%d edges %d/%d" % (du_mb(os.path.join(out, "slam2.pkl")), sv_ms, len(chk.kfs), len(slam.kfs), len(chk.edges), len(slam.edges))); del chk
        # ---- 10-min metrics
        if tnow >= next_metric:
            next_metric += a.metric_s
            gth_t = np.array([q[0] for q in gth]); gth_a = np.array([q[1:] for q in gth])
            E = np.array(err); Ew = E[E[:, 0] > tnow - a.metric_s]
            rm = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else None
            rec = dict(t=round(tnow), seed=a.seed, wall_s=round(time.time() - t_wall0), path_m=round(path_len, 1), events=dict(ev))
            for j_, k_ in ((1, "slam2"), (2, "rtab"), (3, "odom")):
                rec[k_] = dict(ate_win=round(rm(Ew[:, j_]), 3), ate_cum=round(rm(E[:, j_]), 3), err_now=round(float(E[-1, j_]), 3), err_max=round(float(E[:, j_].max()), 3),
                               drift_pct=round(100 * float(E[-1, j_]) / max(1.0, path_len), 3), yaw_err_deg=round(math.degrees(E[-1, j_ + 3]), 2))
            ref_known = grids["gt"].known(); go = pool(gt_now[0]); occ_now = {}
            for k_ in ("gt", "slam2", "rtab", "odom"):
                sc = OM.score(grids[k_], ref_known, go); occ_now[k_] = grids[k_].occ().copy()
                rec.setdefault(k_, {}).update(map_occ_iou=sc["occ_iou"], map_free_iou=sc["free_iou"], map_f1_t=sc["occ_f1_t"])
            if len(snaps) >= 6:
                old = snaps[-6]
                for k_ in occ_now:
                    kn = ref_known; a_, b_ = occ_now[k_] & kn, old[k_] & kn
                    rec[k_]["stab_iou_1h"] = round(float((a_ & b_).sum() / max(1, (a_ | b_).sum())), 3)
            snaps.append(occ_now)
            # slam2 loops vs GT
            ll = getattr(slam, "loop_log", []); n_f = n_t = 0
            for (ta, tb, z, ei) in ll:
                ga, gb = gt_at(ta), gt_at(tb)
                if ga is None or gb is None: continue
                ep, ea = rel_err(z, tuple(ga), tuple(gb)); n_t += 1; n_f += ep > 0.3 or ea > math.radians(5)
            imp_t = imp_f = 0
            for (i_, j_, z, _) in slam.edges:
                if abs(i_ - j_) <= 1 or i_ >= len(slam.kfs) or j_ >= len(slam.kfs): continue
                ka, kb = slam.kfs[i_], slam.kfs[j_]
                if abs(getattr(ka, "path", 0) - getattr(kb, "path", 0)) < 4.0: continue
                ga, gb = gt_at(ka.t), gt_at(kb.t)
                if ga is None or gb is None: continue
                ep, ea = rel_err(tuple(z), tuple(ga), tuple(gb)); imp_t += 1; imp_f += ep > 0.3 or ea > math.radians(5)
            sst = {k: v for k, v in slam.stats.items() if isinstance(v, (int, float))}
            fb = sum(sum(getattr(v, "nbytes", 0) for v in (kf.obs, kf.free, kf.carve)) for kf in slam.kfs) + slam.field.cnt.nbytes
            rec["slam2"].update(nodes=len(slam.kfs), edges=len(slam.edges), loops=n_t, loops_false=n_f, longedges=imp_t, longedges_false=imp_f,
                                ms_mean=round(float(np.mean(s2_ms[-3000:])), 2) if s2_ms else None, ms_p95=round(float(np.percentile(s2_ms[-3000:], 95)), 2) if s2_ms else None,
                                data_mb=round(fb / 2**20, 2), save_mb=round(du_mb(os.path.join(out, "slam2.pkl")), 2),
                                stats={k: sst.get(k) for k in ("reanchor", "refresh", "remaps", "loops_downweighted", "opt_ms", "loop_shift_m", "accepted", "matches", "rej_chi2", "gated")})
            if rt is not None and not rtab_capped:
                N, L = rt.graph(); lt = collections.Counter(l[2] for l in L)
                for (fa, fb_, ty, z) in L:
                    if ty in (1, 2, 4) and (fa, fb_) not in rt_loops and fa in rt_id_t and fb_ in rt_id_t: rt_loops[(fa, fb_)] = (rt_id_t[fa], rt_id_t[fb_], z, ty)
                r_t = r_f = 0
                for (ta, tb, z, ty) in rt_loops.values():
                    ga, gb = gt_at(ta), gt_at(tb)
                    if ga is None or gb is None: continue
                    ep, ea = rel_err(z, tuple(ga), tuple(gb)); r_t += 1; r_f += ep > 0.3 or ea > math.radians(5)
                rss = rss_mb(rt.p.pid); rc_ = cpu_s(rt.p.pid)
                rec["rtab"].update(nodes=len(N), links=len(L), link_types={int(k): v for k, v in lt.items()}, wm=rt.last["wm"] if rt.last else None, loops=r_t, loops_false=r_f,
                                   exceptions=getattr(rt, "exc", 0), hyp_max=round(max([h[0] for h in rt.hyps[-600:]] or [0]), 3), hyp_over_thr=sum(h[0] >= 0.11 for h in rt.hyps[-600:]), hyp_rejected=sum(h[1] > 0 for h in rt.hyps[-600:]), ms_mean=round(float(np.mean(rt.ms[-600:])), 1), ms_p95=round(float(np.percentile(rt.ms[-600:], 95)), 1), rss_mb=round(rss), db_mb=round(du_mb(rdb), 1) if rdb else None,
                                   cpu_s_per_simh=round((rc_ - rcpu0) * 3600 / a.metric_s, 1))
                rcpu0 = rc_
                if rss > a.rtab_rss_cap_gb * 1024: rtab_capped = True; P("RTAB RSS %.1f GB > cap: RTAB frozen from here (reported)" % (rss / 1024)); rec["rtab"]["capped"] = True
            rec["harness"] = dict(rss_mb=round(rss_mb(os.getpid())), cpu_s=round(cpu_s(os.getpid()) - cpu0), rt_factor=round(tnow / max(1, time.time() - t_wall0), 2),
                                  disk_free_gb=round(shutil.disk_usage(out).free / 2**30, 1), out_mb=round(du_mb(out), 1), gyro_offset_ms=round(off[0] * 1e3, 1) if off[1] else None)
            mfile.write(json.dumps(rec, default=lambda o: o.item()) + "\n"); mfile.flush(); efile.flush()
            P("M slam2 ate %.3f/%.3f iou %.3f n %d | rtab ate %.3f/%.3f iou %.3f n %s | odom ate %.3f | %.2fx rt" % (
                rec["slam2"]["ate_win"], rec["slam2"]["ate_cum"], rec["slam2"]["map_occ_iou"], len(slam.kfs), rec["rtab"]["ate_win"], rec["rtab"]["ate_cum"], rec["rtab"]["map_occ_iou"],
                rec["rtab"].get("nodes"), rec["odom"]["ate_win"], rec["harness"]["rt_factor"]))
            if rec["harness"]["disk_free_gb"] < a.min_free_gb: P("disk free < %g GB: stopping" % a.min_free_gb); break
    slam.save(os.path.join(out, "slam2.pkl"))
    if rt is not None: rt.close()
    json.dump(dict(done=True, t=d.time, path_m=path_len, events=dict(ev)), open(os.path.join(out, "done.json"), "w"))
    P("done"); os._exit(0)


if __name__ == "__main__":
    main()
