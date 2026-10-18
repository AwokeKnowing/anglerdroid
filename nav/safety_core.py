"""Safety core: dedicated deterministic 30 Hz loop, separate process from the planner/nav brain (2026-10-17 design,
James; PROPOSAL branch safety-core-loop-2026-10-17, sim only, unmerged).

Per RS1 frame (the 30 Hz clock), in its own process (no GIL shared with the planner), pinned to one CPU, SCHED_FIFO
when the OS allows it (see docs/safety_core.md):
  1. operability: RS1 and RS2 frame age (time.monotonic() at decision - capture) <= AGE_MAX_S (0.10 s) and no missed
     30 Hz cadence (frame interval <= CADENCE_MAX_S) and RS1 view not occluded -> else STOP.
  2. deterministic transform: RS1 top-down ego labels (nav.perceive.Ego, same code as the brain) + RS2 forward cone
     (nav.perceive.Rs2Cloud, all heights 3 cm .. 1.0 m) collapsed to 2-D, then moved into the robot frame of NOW
     using the measured wheel-odometry twist over each frame's age.
  3. hard bands: 5 cm in front of the masked footprint and 5 cm at the rear edge of what RS1 sees behind it.  Any
     obstacle in a band -> no motion toward it (v toward it = 0, exactly; no creep).
  4. last-resort brake: the commanded (or measured, if faster) motion is swept along its arc for the distance needed
     to still stop if this tick passes it (one tick + command latency + measured braking distance at MAX braking).  If
     that sweep enters the margin, command zero now, with the vel ramp raised just as far as needed (normal ramp 3 ..
     max 5 turns/s^2) to stop before the obstacle.  No gradual throttling: a command passes unchanged or is zeroed.
The core is the only writer to the wheels (it owns WheelBase); the brain sends desired (v, w) over nav.bus and reads
wheel state back.  A brain command older than CMD_STALE_S, or a dead parent process, -> zero.

Measured (sim, emu.tools.brake_test, twin ODrive model, carpet): stop distance from command-zero to rest
  ramp 3: 0.31 m/s 0.050 m, 0.444 0.089, 0.609 0.135  -> d = 0.0987 v + v^2 / (2 * 2.47)
  ramp 5: 0.31 m/s 0.040 m, 0.444 0.068, 0.609 0.100  -> d = 0.0926 v + v^2 / (2 * 4.26)   pitch <= 6.4 deg, no skid
  ramp 8: 0.609 m/s 0.083 m but wheel skid 0.24 m/s, pitch 8.1 deg -> ramp 5 taken as MAX (sim; real bot: verify).
"""
from __future__ import annotations
import gc, json, math, os, sys, time
import numpy as np

import nav  # noqa (src on path)
from twin_geometry import FOOTPRINT_SAFE, SAFETY_MARGIN, WHEEL_R_M, TRACK_M

HZ = 30.0; T_TICK = 1.0 / HZ
AGE_MAX_S = float(os.environ.get("KEVIN_SC_AGE_MAX", "0.10"))
CADENCE_MAX_S = float(os.environ.get("KEVIN_SC_CADENCE_MAX", str(1.5 / HZ)))    # a dropped frame = missed cadence
CMD_STALE_S = 0.3
T_CMD = 0.010                         # core decision -> CAN frame on the wire (assumed; measured: see log t_sent - t_dec)
RAMP_NORMAL, RAMP_MAX = float(os.environ.get("KEVIN_VEL_RAMP", "3.0")), 5.0
BRAKE = {RAMP_NORMAL: (0.0987, 2.47), RAMP_MAX: (0.0926, 4.26)}     # (lag s, decel m/s^2), fits above
BAND = 0.05; MARGIN = SAFETY_MARGIN; H_MAX = 1.0
STATE_FIELDS = ["t_dec", "period", "age1", "age2", "v_cmd", "w_cmd", "v_m", "w_m", "v_out", "w_out", "code", "ramp", "free",
                "need", "sd0", "band_f", "band_r", "n_pts", "rt", "cpu", "t_sent", "cap1", "cap2", "brakes"]
CODES = ["ok", "idle", "inop_age", "inop_cadence", "no_depth", "band_front", "band_rear", "brake", "cmd_stale", "escape"]


def d_stop(v, ramp):
    lag, a = BRAKE[ramp] if ramp in BRAKE else BRAKE[RAMP_MAX]
    v = abs(v); return v * lag + v * v / (2 * a)


def need_dist(v, ramp=RAMP_MAX):
    """distance travelled if this tick lets |v| through and the next tick has to stop it: one tick + command latency,
    then the measured stop at 'ramp'."""
    return abs(v) * (T_TICK + T_CMD) + d_stop(v, ramp)


def _edges(poly):
    P = poly; Q = np.roll(P, -1, 0); e = Q - P
    n = np.stack([e[:, 1], -e[:, 0]], 1); n /= np.linalg.norm(n, axis=1, keepdims=True)
    if (n @ P.mean(0) - (n * P).sum(1)).max() > 0: n = -n
    return n, (n * P).sum(1)


N_, C_ = _edges(FOOTPRINT_SAFE)


def sd(P):
    return (P @ N_.T - C_).max(1) if len(P) else np.zeros(0)


def pose_on_arc(s, sign, kappa):
    """pose after arclength s (>= 0) along an arc of curvature kappa, direction sign (+1 fwd / -1 back)."""
    if abs(kappa) < 1e-6: return sign * s, 0.0, 0.0
    th = sign * s * kappa
    return math.sin(th) / kappa, (1 - math.cos(th)) / kappa, th


def to_body(P, x, y, th):
    c, s = math.cos(th), math.sin(th); d = P - (x, y)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


R_FOOT = float(np.hypot(FOOTPRINT_SAFE[:, 0], FOOTPRINT_SAFE[:, 1]).max())


def _sweep(P, poses):
    """min signed distance of points P to the footprint at each pose (x, y, th) -> (len(poses),), vectorised."""
    x, y, th = poses[:, 0, None], poses[:, 1, None], poses[:, 2, None]
    c, s = np.cos(th), np.sin(th); dx, dy = P[None, :, 0] - x, P[None, :, 1] - y
    L = np.stack([c * dx + s * dy, -s * dx + c * dy], -1)                       # (S, N, 2)
    return (L @ N_.T - C_).max(-1).min(-1)


def _first_hit(m, steps):
    hit = np.nonzero((m < MARGIN) & (steps > 0))[0]
    return (float(steps[hit[0]]) if len(hit) else float(steps[-1])), float(m[:hit[0] + 1].min() if len(hit) else m.min())


def free_along(P, sign, kappa, max_s, step=0.01):
    """first arclength where the footprint comes within MARGIN of any point (max_s if none), and min sd up to it."""
    if not len(P): return max_s, np.inf
    P = P[np.hypot(P[:, 0], P[:, 1]) < R_FOOT + max_s + MARGIN + 0.02]
    if not len(P): return max_s, np.inf
    steps = np.arange(0.0, max_s + 1e-9, step); steps = steps if steps[-1] >= max_s - 1e-9 else np.r_[steps, max_s]
    poses = np.array([pose_on_arc(s, sign, kappa) for s in steps])
    return _first_hit(_sweep(P, poses), steps)


def free_rotation(P, sign, max_a, step=0.02):
    if not len(P): return max_a, np.inf
    P = P[np.hypot(P[:, 0], P[:, 1]) < R_FOOT + MARGIN + 0.02]
    if not len(P): return max_a, np.inf
    steps = np.arange(0.0, max_a + 1e-9, step); steps = steps if steps[-1] >= max_a - 1e-9 else np.r_[steps, max_a]
    poses = np.stack([np.zeros_like(steps), np.zeros_like(steps), sign * steps], 1)
    return _first_hit(_sweep(P, poses), steps)


def alpha_max(ramp=RAMP_MAX):
    return BRAKE[ramp][1] / (TRACK_M / 2)


def need_angle(w, ramp=RAMP_MAX):
    lag, _ = BRAKE[ramp]; w = abs(w); return w * (T_TICK + T_CMD + lag) + w * w / (2 * alpha_max(ramp))


def motion_comp(P, v, w, dt):
    """points observed dt ago -> current robot frame, robot having moved along (v, w) for dt (wheel odometry)."""
    if not len(P) or dt <= 0 or (abs(v) < 1e-4 and abs(w) < 1e-4): return P
    th = w * dt
    if abs(w) > 1e-6: x, y = v / w * math.sin(th), v / w * (1 - math.cos(th))
    else: x, y = v * dt, 0.0
    return to_body(P, x, y, th)


class Bands:
    """front band: BAND ahead of the masked footprint; rear band: BAND at the rear edge of what RS1 sees behind the
    masked footprint (learned from which ego pixels return depth).  Width = footprint half-width (wheels)."""
    def __init__(self, ego):
        X, Y, body = ego.X, ego.Y, ego.body
        self.hw = float(np.abs(FOOTPRINT_SAFE[:, 1]).max())
        cols = np.abs(Y) <= self.hw
        self.x_front = float(X[body & cols].max()) + 0.005           # mask front (+ half px)
        self.x_body_rear = float(X[body & cols].min()) - 0.005
        self.x_rear = self.x_body_rear                               # updated from the visibility statistics
        self._seen = np.zeros(X.shape, np.float32); self._n = 0; self.X, self.Y, self.cols = X, Y, cols
    def learn(self, lab):
        from perception.labels import CLEAR, OBSTACLE
        if self._n >= 60: return
        self._seen += ((lab == CLEAR) | (lab == OBSTACLE)); self._n += 1
        if self._n in (10, 30, 60):
            vis = (self._seen / self._n > 0.5) & self.cols & (self.X < self.x_body_rear)
            if vis.any(): self.x_rear = float(self.X[vis].max()) + 0.005
    def front(self, P):
        return bool(len(P)) and bool(((P[:, 0] >= self.x_front) & (P[:, 0] < self.x_front + BAND) & (np.abs(P[:, 1]) <= self.hw)).any())
    def rear(self, P):
        return bool(len(P)) and bool(((P[:, 0] <= self.x_rear) & (P[:, 0] > self.x_rear - BAND) & (np.abs(P[:, 1]) <= self.hw)).any())
    def rects(self):
        return [[self.x_front, -self.hw, self.x_front + BAND, self.hw], [self.x_rear - BAND, -self.hw, self.x_rear, self.hw]]


def decide(P, bands, v_c, w_c, v_m, w_m):
    """pure function of the tick's inputs -> (v_out, w_out, code, ramp, free, need, sd0, band_f, band_r)."""
    sd0 = float(sd(P).min()) if len(P) else np.inf
    bf, br = bands.front(P), bands.rear(P)
    moving = abs(v_m) > 0.02 or abs(w_m) > 0.05
    if abs(v_c) < 1e-3 and abs(w_c) < 1e-3:
        if not moving: return 0.0, 0.0, "idle", RAMP_NORMAL, np.nan, 0.0, sd0, bf, br
        # coasting to the zero command: is the normal ramp enough?
        if abs(v_m) > 0.02:
            fr, _ = free_along(P, 1 if v_m > 0 else -1, w_m / v_m if abs(v_m) > 1e-3 else 0.0, need_dist(v_m, RAMP_NORMAL) + 0.05)
            ramp = _ramp_for(fr, v_m)
            return 0.0, 0.0, "brake" if ramp > RAMP_NORMAL else "idle", ramp, fr, need_dist(v_m, RAMP_NORMAL), sd0, bf, br
        return 0.0, 0.0, "idle", RAMP_NORMAL, np.nan, 0.0, sd0, bf, br
    if abs(v_c) >= 1e-3:
        sign = 1 if v_c > 0 else -1
        if (sign > 0 and bf) or (sign < 0 and br):
            return 0.0, 0.0, "band_front" if sign > 0 else "band_rear", _brake_ramp(P, v_m, w_m), np.nan, 0.0, sd0, bf, br
        v_e = max(abs(v_c), abs(v_m) if v_m * v_c > 0 else 0.0); kappa = w_c / v_c
        need = need_dist(v_e)
        fr, mn = free_along(P, sign, kappa, need + 0.02)
        if fr >= need: return v_c, w_c, "ok", RAMP_NORMAL, fr, need, sd0, bf, br
        # already inside the margin: a motion that never gets closer anywhere is an escape, allowed
        if sd0 < MARGIN and mn >= sd0 - 1e-3: return v_c, w_c, "escape", RAMP_NORMAL, fr, need, sd0, bf, br
        return 0.0, 0.0, "brake", _brake_ramp(P, v_m, w_m), fr, need, sd0, bf, br
    sign = 1 if w_c > 0 else -1; w_e = max(abs(w_c), abs(w_m) if w_m * w_c > 0 else 0.0)
    need = need_angle(w_e); fa, mn = free_rotation(P, sign, need + 0.02)
    if fa >= need: return v_c, w_c, "ok", RAMP_NORMAL, fa, need, sd0, bf, br
    if sd0 < MARGIN and mn >= sd0 - 1e-3: return v_c, w_c, "escape", RAMP_NORMAL, fa, need, sd0, bf, br
    return 0.0, 0.0, "brake", RAMP_MAX, fa, need, sd0, bf, br


def _ramp_for(avail, v):
    """smallest ramp (normal .. max) whose measured stop distance (+ command latency) fits in 'avail'."""
    v = abs(v); dn, dm = v * T_CMD + d_stop(v, RAMP_NORMAL), v * T_CMD + d_stop(v, RAMP_MAX)
    if avail >= dn: return RAMP_NORMAL
    if avail <= dm or dn <= dm: return RAMP_MAX
    return RAMP_NORMAL + (RAMP_MAX - RAMP_NORMAL) * (dn - avail) / (dn - dm)


def _brake_ramp(P, v_m, w_m):
    if abs(v_m) <= 0.02: return RAMP_MAX if abs(w_m) > 0.05 else RAMP_NORMAL
    fr, _ = free_along(P, 1 if v_m > 0 else -1, w_m / v_m, need_dist(v_m, RAMP_NORMAL) + 0.05)
    return _ramp_for(fr, v_m)                    # fr already stops MARGIN short of the obstacle


def governor(P, v, w, v_meas=0.0, factor=2.0):
    """PLANNER side (brain): largest scale of (v, w) whose sweep keeps factor x the core's threshold (need + margin)
    clear, so the core rarely fires.  Returns (v, w, scale, need2)."""
    for s in (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.0):
        vs, ws = v * s, w * s
        if abs(vs) >= 1e-3:
            ve = max(abs(vs), abs(v_meas) if v_meas * vs > 0 else 0.0); need2 = factor * (need_dist(ve) + MARGIN)
            fr, _ = free_along(P, 1 if vs > 0 else -1, ws / vs, need2, step=0.02)
            if fr >= need2: return vs, ws, s, need2
        elif abs(ws) >= 1e-3:
            need2 = factor * need_angle(ws); fa, _ = free_rotation(P, 1 if ws > 0 else -1, need2, step=0.04)
            if fa >= need2: return vs, ws, s, need2
        else: return 0.0, 0.0, s, 0.0
    return 0.0, 0.0, 0.0, 0.0


class FrameClock:
    """camera stamp -> monotonic (sim stamps are monotonic; other domains: latency-floor offset)."""
    def __init__(self): self.off = None; self.t_off = 0.0
    def capture(self, stamp, t_recv):
        d = t_recv - stamp
        if 0.0 <= d < 2.0: return stamp
        if self.off is None or d < self.off or abs(d - self.off) > 1.0 or t_recv - self.t_off > 10.0: self.off, self.t_off = d, t_recv
        return min(stamp + self.off, t_recv)


def _realtime(cpu):
    got = []
    try: os.sched_setaffinity(0, {cpu}); got.append("cpu%d" % cpu)
    except Exception as e: got.append("affinity failed (%s)" % e.__class__.__name__)
    try: os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(os.environ.get("KEVIN_SC_RTPRIO", "50")))); got.append("SCHED_FIFO")
    except Exception as e:
        got.append("SCHED_FIFO denied (%s)" % e.__class__.__name__)
        try: os.nice(-10); got.append("nice -10")
        except Exception: got.append("nice unchanged")
    try:
        import resource; got.append("rtprio limit %s" % (resource.getrlimit(resource.RLIMIT_RTPRIO),))
    except Exception: pass
    return ", ".join(got)


def core_main(rs1_intr, rs2_intr, rs2_shape, parent_pid, log_path=None):
    """entry point of the safety core process (multiprocessing 'spawn')."""
    cpu = int(os.environ.get("KEVIN_SC_CPU", str((os.cpu_count() or 1) - 1)))
    rt = _realtime(cpu); print("safety_core: pid %d %s" % (os.getpid(), rt), flush=True)
    from nav import bus
    from nav.perceive import Ego, Rs2Cloud
    B = bus.open_all(rs2_shape, create=False)
    import wheelbase
    cwd = os.getcwd(); os.chdir(nav.SRC)
    try: wb = wheelbase.WheelBase()
    finally: os.chdir(cwd)
    ego = Ego(); ego.intr = rs1_intr; rs2 = Rs2Cloud(rs2_intr); bands = Bands(ego)
    clk1, clk2 = FrameClock(), FrameClock()
    LOG = open(log_path, "a", buffering=1 << 16) if log_path else None
    gc.collect(); gc.freeze(); gc.disable()                       # no GC pauses inside the loop (refcounting still frees)
    s1 = s2 = 0; cap1p = None; cap2 = None; cap2p = None; P2 = np.zeros((0, 2), np.float32); r1 = {}; r2 = {}
    ramp_now = RAMP_NORMAL; t_prev = None; brakes = 0; viz_t = 0.0; ps = 0
    while True:
        if os.getppid() != parent_pid: wb.twist(0.0, 0.0); wb.shutdown(); return     # brain gone -> stop, exit
        new1 = _wait(B["rs1"], s1, CADENCE_MAX_S)                  # next RS1 frame, or wake up and stop
        t_dec = time.monotonic()
        vl, vr = wb.get_wheel_velocities_mps(); v_m = 0.5 * (vl + vr); w_m = (vr - vl) / TRACK_M
        cur = wb.get_motor_currents()
        B["wheel"].write(vl=vl, vr=vr, t=t_dec, il=cur[0] if cur else np.nan, ir=cur[1] if cur else np.nan, iage=cur[2] if cur else np.nan, batt=float(getattr(wb, "_batt_pct", 0) or 0))
        _, c = B["cmd"].read(); v_c, w_c, t_c = float(c["v"][0]), float(c["w"][0]), float(c["t"][0])
        code = None
        if t_dec - t_c > CMD_STALE_S: v_c = w_c = 0.0; code_c = "cmd_stale"
        else: code_c = None
        if new1:
            s1, n, r1 = _read_rs1(B["rs1"], r1); tA = time.monotonic(); cap1 = clk1.capture(float(r1["stamp"][0]), float(r1["t_recv"][0]))
            per = None if cap1p is None else cap1 - cap1p; cap1p = cap1
            ego.update(r1["verts"][:n]); P1 = ego.obstacle_xy(max_n=1500); valid = ego.valid_fraction(); bands.learn(ego.lab); tB = time.monotonic()
        else:
            per = t_dec - (cap1p or t_dec); P1 = None; valid = 0.0; tA = tB = t_dec
        if B["rs2"].seq != s2:
            s2, r2 = B["rs2"].read(out=r2); cap2 = clk2.capture(float(r2["stamp"][0]), float(r2["t_recv"][0]))
            per2 = None if cap2p is None else cap2 - cap2p; cap2p = cap2
            sp = B["rs2pose"].read()[1]
            if int(sp["set"][0]): rs2.set_pose(sp["T"], sp["R"].reshape(3, 3))
            P2, _ = rs2.points(r2["depth"], h_max=H_MAX)
        else: per2 = None
        tC = time.monotonic()                                          # frame age AT DECISION (after the transform)
        age1 = (tC - cap1p) if cap1p is not None else np.inf; age2 = (tC - cap2) if cap2 is not None else np.inf
        if not new1 or age1 > AGE_MAX_S or age2 > AGE_MAX_S:
            code = "inop_age"
        elif (per is not None and per > CADENCE_MAX_S) or (per2 is not None and per2 > CADENCE_MAX_S):
            code = "inop_cadence"
        elif valid < 0.05:
            code = "no_depth"
        free = need = np.nan; sd0 = np.nan; bf = br = False; npts = 0; P = np.zeros((0, 2), np.float32)
        if code is not None:
            v_o = w_o = 0.0; ramp = RAMP_MAX if (abs(v_m) > 0.02 or abs(w_m) > 0.05) else RAMP_NORMAL
        else:
            P = np.concatenate([motion_comp(P1, v_m, w_m, age1), motion_comp(P2, v_m, w_m, age2)]) if len(P2) else motion_comp(P1, v_m, w_m, age1)
            npts = len(P)
            v_o, w_o, code, ramp, free, need, sd0, bf, br = decide(P, bands, v_c, w_c, v_m, w_m)
            if code_c and code == "idle": code = code_c
        if ramp != ramp_now:                                       # brake ramp first, then the zero command
            with wb.bus_lock:
                for ax in (wb.left, wb.right): ax.write_property_nowait('axis0.controller.config.vel_ramp_rate', float(ramp))
            ramp_now = ramp
        if code in ("brake", "band_front", "band_rear", "inop_age", "inop_cadence", "no_depth") and (abs(v_c) > 1e-3 or abs(w_c) > 1e-3): brakes += 1
        tD = time.monotonic(); wb.twist(v_o, w_o); t_sent = time.monotonic()
        period = None if t_prev is None else t_dec - t_prev; t_prev = t_dec
        f = [t_dec, period if period is not None else np.nan, age1, age2, v_c, w_c, v_m, w_m, v_o, w_o, CODES.index(code) if code in CODES else -1,
             ramp, free, need, sd0, bf, br, npts, 0, cpu, t_sent, cap1p or np.nan, cap2 or np.nan, brakes]
        B["state"].write(f=np.array(f, np.float64))
        if new1 and t_dec - viz_t > 0.2 and len(P):                                # 5 Hz snapshot for the viewer overlay
            k = min(len(P), 2000); B["viz"].write(pts=P[:k].astype(np.float32), n=np.array([k]), bands=np.array(bands.rects(), np.float64).ravel(), t=np.array([t_dec]))
            viz_t = t_dec
        if LOG is not None:
            LOG.write(json.dumps([round(x, 4) if isinstance(x, float) and math.isfinite(x) else (None if isinstance(x, float) else x) for x in
                                  [t_dec, period, age1, age2, v_c, w_c, v_m, w_m, v_o, w_o, code, ramp, free, need, sd0, bf, br, npts, t_sent, per, per2, valid, tA - t_dec, tB - tA, tC - tB, tD - tC, t_sent - tD]]) + "\n")
            ps += 1
            if ps % 30 == 0: LOG.flush()


def _read_rs1(blk, out):
    """seqlock read of only the n valid verts (the block holds MAX_VERTS)."""
    for _ in range(200):
        s0 = int(blk.hdr[0])
        if s0 & 1: time.sleep(0.0001); continue
        n = int(blk.arr["n"][0])
        if "verts" not in out or len(out["verts"]) != n: out["verts"] = np.empty((n, 3), np.float32)
        np.copyto(out["verts"], blk.arr["verts"][:n]); out["stamp"] = blk.arr["stamp"].copy(); out["t_recv"] = blk.arr["t_recv"].copy()
        if int(blk.hdr[0]) == s0: return s0, n, out
    raise TimeoutError("bus rs1: writer stuck")


def _wait(blk, last, timeout):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        s = int(blk.hdr[0])
        if s != last and not (s & 1): return True
        time.sleep(0.0003)
    return False
