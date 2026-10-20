"""Kevin navigation stack, one process:

  RS1 frame (30 Hz clock) -> odom(wheels+gyro) -> ego labels -> [MPPI graph replay] -> safety envelope -> wheels
                                   |  5 Hz scans               ^ path (1 Hz / on goal)
                                   v                           |
                      SLAM thread (CSM + log-odds map) --> planner thread (map window, A*, MapCPU)
  commands (web / CLI / VLN hook) -> goals.parse -> Brain.command

python -m nav.run [--port 8080] [--map NAME] [--no-match]     (KEVIN_HAL=sim for the emulator)"""
from __future__ import annotations
import argparse, collections, json, math, os, threading, time
import numpy as np
import nav  # noqa
from nav.hw import Hardware
from nav.perceive import Ego, Rs2Cloud
from nav.slam import Slam, GridMap, wrap, transform, carve_rays
from nav.safety import Envelope, FrameClock
import twin_geometry as _TG
_sl = os.environ.get("KEVIN_SAFETY_LOG"); _SLOG = open(_sl, "a", buffering=1) if _sl else None
L_BODY = float(_TG.FOOTPRINT_SAFE[:, 0].max() - _TG.FOOTPRINT_SAFE[:, 0].min())   # 0.40 m nose to tail
from nav import goals as G
from nav import goalio as GI
from emu import shm
_VMAX = float(os.environ.get("KEVIN_VMAX", "0.6")); _VREV = min(0.2, _VMAX); _WMAX = _TG.W_MAX_RADS        # pi/2 (twin_geometry)
# Free-space carving of the SLAM/planning map from current depth rays (KEVIN_CARVE=0: floor-point decay only, as before)
CARVE = os.environ.get("KEVIN_CARVE", "1") != "0"
STUCK_DUMP = os.environ.get("KEVIN_STUCK_DUMP")      # dir: planner state dumps at no-progress / give-up (diagnostics)
# control cycle (James 2026-10-18).  KEVIN_CMD_CYCLE=1 (opt-in): goals / global replans are latched and taken at the
# next cycle start (a new goal never interrupts a plan before its command is issued) and each plan starts from the
# latency-compensated predicted state (nav/predict.py) at the time its command takes effect.  Default: the previous
# loop (plan from the latest measured pose and the last commanded v, w).  Both run inline after perception.
CTL_MODE = "cmd" if os.environ.get("KEVIN_CMD_CYCLE") == "1" or os.environ.get("KEVIN_CTL") == "cmd" else "frame"   # opt-in: KEVIN_CMD_CYCLE=1
GOAL_SAME_M = 0.20     # a re-sent goal within the xy goal tolerance (0.20 m, below) is the same goal
GYRO_INC = os.environ.get("KEVIN_GYRO_INC", "1") != "0"     # 0: mean gyro rate x frame dt (previous behaviour)
from nav.hw import AXLE_H
RS1_T = _TG.RS1_T

HZ = 30.0



def _thread_cpu():
    """CPU seconds used by each live thread of this process, by thread name (profiling without ptrace)."""
    out = {}
    for t in threading.enumerate():
        try: c = time.clock_gettime(time.pthread_getcpuclockid(t.ident))
        except Exception: continue
        out[t.name] = round(out.get(t.name, 0.0) + c, 3)
    return out

class Stats:
    def __init__(self, n=900): self.d = collections.defaultdict(lambda: collections.deque(maxlen=n))
    def add(self, k, v): self.d[k].append(v)
    def summary(self):
        out = {}
        for k, q in list(self.d.items()):            # the 30 Hz loop may add a key meanwhile
            a = np.asarray(q, float)
            if len(a): out[k] = dict(mean=round(float(a.mean()), 3), p50=round(float(np.percentile(a, 50)), 3),
                                     p99=round(float(np.percentile(a, 99)), 3), max=round(float(a.max()), 3), n=len(a))
        return out


class Brain:
    def __init__(self, hw: Hardware, match=True, plan=True):
        self.hw = hw; self.ego = Ego(); self.ego.intr = hw.rs1.depth_intrinsics() if hasattr(hw.rs1, "depth_intrinsics") else None; self.rs2 = Rs2Cloud(hw.rs2_intr)
        # Option A (James 2026-10-20): with the safety core, read ITS ego labels (sc_ego) instead of a second Ego.update
        # here.  Default on in the sim only (KEVIN_HAL=sim); KEVIN_EGO_SHARED=0/1 overrides.
        self.ego_shared = (os.environ.get("KEVIN_EGO_SHARED", "1" if os.environ.get("KEVIN_HAL") == "sim" else "0") == "1"
                           and getattr(hw, "core", False) and "ego" in getattr(hw, "bus", {}))
        if self.ego_shared:
            from nav.ego_shared import EgoShared
            self.ego = EgoShared(hw.bus["ego"]); self.ego.intr = hw.rs1.depth_intrinsics(); print("[nav] ego labels from the safety core (sc_ego)", flush=True)
        self.rec = None; self.t0 = round(time.time(), 3)          # session id: the map frame is this run's start pose
        if os.environ.get("KEVIN_RECORD"):
            from nav.record import Recorder
            self.rec = Recorder(os.environ["KEVIN_RECORD"]); print("[nav] recording to %s" % self.rec.d, flush=True)
        self.autocal = None
        if os.environ.get("KEVIN_AUTOCAL", "0" if os.environ.get("KEVIN_HAL") == "sim" else "1") == "1":
            from nav.autocal import AutoCal, STILL_V
            self.autocal = AutoCal(hw.rs2_intr, log=self.log); self._still_v = STILL_V
        _sk = os.environ.get('KEVIN_SLAM', '2'); self.slam = (__import__('nav.vslam_cuv', fromlist=['CuvSlam']).CuvSlam(GridMap()) if _sk == 'cuv' else __import__('nav.slam2', fromlist=['Slam2']).Slam2(GridMap(), match=match) if _sk == '2' else Slam(GridMap(), match=match)); self.places = G.Places(); self.env = Envelope(); self._fclk = FrameClock()
        # KEVIN_NAV_DECOUPLE=1 (James 21:47 architecture): driving, LightNav waypoints, the nav envelope's memory and the
        # planners work in the ODOMETRY frame on a short-horizon local grid built from odometry only (live frames are the
        # truth; unseen evidence decays with TAU_LOCAL).  The SLAM map/pose is an optional layer: named places, map-frame
        # goals and labels go through T_map<-odom only while the SLAM pose agrees with the current view (map_check()).
        self.decouple = os.environ.get("KEVIN_NAV_DECOUPLE", "0") == "1"
        self.slam_off = os.environ.get("KEVIN_SLAM_OFF", "0") == "1"                     # test: no SLAM at all
        self.slam_lag = float(os.environ.get("KEVIN_SLAM_LAG_S", "0"))                     # test: SLAM correction lagged
        _e = [float(v) for v in os.environ.get("KEVIN_SLAM_ERR", "0,0,0").split(",")]      # test: injected error dx,dy,dth_deg
        self.slam_err = (_e[0], _e[1], math.radians(_e[2]))
        self.lgrid = GridMap() if self.decouple else None; self._lg_t = None; self._lg_job = None; self._lg_ev = threading.Event()
        if self.decouple: threading.Thread(target=self._lg_worker, daemon=True, name="localgrid").start()
        self._Thist = collections.deque(maxlen=2000); self.map_ok = False; self.map_agree = None; self._mc_t = 0.0
        self.goal_shm = os.environ.get("KEVIN_GOAL_SHM", "0") == "1"      # goal interface blocks (nav/goalio.py)
        self._ns = shm.Block("kevin_navstate", GI.L_NAVSTATE, create=True) if self.goal_shm else None
        self.planner = None
        if plan:
            from nav.plan import Planner
            self.planner = Planner(self.navgrid)
        from nav.obslayer import ObsLayer
        self.obs = ObsLayer()                       # current RS1 frame, authoritative over the map where it sees (planners)
        from nav.bump import BumpDetector                # contact detector (motor current + IMU), additive safety
        import twin_geometry as _TG
        self.bump = BumpDetector(_TG.TRACK_M); self.bump_latched = None; self._bump_hold = 0.0
        if self.planner is not None: self.planner.obs = self.obs
        self.mode = "idle"; self.msg = ""; self.v = 0.0; self.w = 0.0; self.turn_target = None
        self.core_alarm = None; self._core_ok_t = None; self._core_bad_t = 0.0
        self.goal = None; self.goal_heading = None; self.goal_label = None; self.goal_t0 = 0.0
        self.stats = Stats(); self.events = collections.deque(maxlen=50); self.tick = 0
        self._cmd_lock = threading.Lock(); self._pending = []
        from nav.predict import CmdPredictor
        self.pred = CmdPredictor(float(os.environ.get("KEVIN_VEL_RAMP", "3.0")))
        self._ctl_cv = threading.Condition(); self._t_iss = 0.0; self._snap = None; self._snap_n = 0; self._latched_g = None
        self._rs2_seq = -1; self._rs2_rays = None; self._rs2_obs_h = np.zeros(0, np.float32); self._rs2_obs = np.zeros((0, 2), np.float32); self._rs2_floor = np.zeros((0, 2), np.float32); self._rs2_t = 0.0
        self.vln = None
        self._goal_lock = threading.RLock()   # stop/cancel vs the async global replan committing a goal
        self.epoch = 0      # goal epoch: +1 on every stop/cancel; async producers (VLN) tag commands with the epoch of their request
        self.running = True
        if self.planner: threading.Thread(target=self._plan_loop, daemon=True, name="plan").start()
        if CTL_MODE == "cmd":                      # frame (default) = the previous loop; only the predictor's scoring runs
            threading.Thread(target=self._fit_loop, daemon=True, name="latfit").start()

    # -------------------------------------------------------------- commands (any thread)
    def command(self, text, frame=None, epoch=None):
        c = G.parse(text)     # frame="nav": coordinates are already in the nav (odom) frame (LightNav waypoints), not the map frame
        if frame: c["frame"] = frame
        if epoch is not None: c["epoch"] = epoch   # issued for a request made at this goal epoch; stale after any later stop
        with self._cmd_lock: self._pending.append(c)
        with self._ctl_cv: self._ctl_cv.notify_all()
        return c

    def log(self, s):
        self.events.append((time.time(), s)); self.msg = s; print("[nav]", s, flush=True)

    TAU_LOCAL = 30.0      # s: local-grid evidence half-life scale = the map lag the system must tolerate (James 21:47)
    AGREE_R, AGREE_MIN = 2, 0.5   # map check: live RS2 obstacle points within AGREE_R cells of a mapped obstacle; >= AGREE_MIN

    def _lg_worker(self):      # local odom-frame grid off the 30 Hz loop (carving costs ms); always takes the newest frame
        while True:
            self._lg_ev.wait(); self._lg_ev.clear(); j = self._lg_job
            if j is None: continue
            t, po, obs, free, obs_h, rays = j
            try:
                if self._lg_t is not None:
                    with self.lgrid.lock: self.lgrid.L *= np.float32(math.exp(-max(t - self._lg_t, 0.0) / self.TAU_LOCAL))
                self._lg_t = t; cv = carve_rays(rays, self.lgrid.res) if rays else None; lg = getattr(self, "_lg_last", None)
                if lg is None or math.hypot(po[0] - lg[0], po[1] - lg[1]) > 0.04 or abs(wrap(po[2] - lg[2])) > math.radians(2.5):
                    self.lgrid.integrate(po, obs, free, obs_h=obs_h, carve=cv); self._lg_last = po      # same rule as the SLAM grid
                elif cv is not None:      # parked: no new evidence weight (would saturate), carve what is seen through now
                    self.lgrid.integrate(po, obs, np.zeros((0, 2), np.float32), obs_h=obs_h, carve=cv, w_hit=0.0)
            except Exception as e: self.log("local grid: %s" % e)

    def navpose(self):
        return self.slam.odom.pose if self.decouple else self.slam.pose()

    @property
    def navgrid(self):
        return self.lgrid if self.decouple else self.slam.grid

    def map_T(self):
        """T map<-odom as the map layer sees it (lag / error injection for tests)."""
        from nav.slam import compose
        T = tuple(getattr(self.slam, "T", (0.0, 0.0, 0.0)))
        if self.slam_lag > 0:
            tl = time.monotonic() - self.slam_lag; T = (0.0, 0.0, 0.0)
            for tt, TT in self._Thist:
                if tt <= tl: T = TT
                else: break
        return compose(self.slam_err, T) if any(self.slam_err) else T

    def map_check(self):
        """Does the SLAM map agree with the current view?  Fraction of live RS2 obstacle points that land (with the map-layer
        pose) within AGREE_R cells of a mapped obstacle (L > 0.7, the grid's own occupied test)."""
        from nav.slam import compose, transform
        P = getattr(self, "_rs2_obs", None); g = self.slam.grid
        if self.slam_off or P is None or len(P) < 100: self.map_ok = False; self.slam.stats['map_ok'] = False; return
        pm = compose(self.map_T(), self.slam.odom.pose); W = transform(P[::4].astype(np.float32), pm)
        r, c = g.rc(np.asarray(W, np.float64)); occ = g.L > 0.7; hit = np.zeros(len(r), bool); known = np.zeros(len(r), bool)
        for dr in range(-self.AGREE_R, self.AGREE_R + 1):
            for dc in range(-self.AGREE_R, self.AGREE_R + 1):
                rr, cc = np.clip(r + dr, 0, g.n - 1), np.clip(c + dc, 0, g.n - 1); hit |= occ[rr, cc]; known |= g.seen[rr, cc]
        # only points that fall where the map has evidence can (dis)agree; new territory is no evidence either way:
        # then keep the last verdict for at most TAU_LOCAL (the map lag the system is built to tolerate)
        now = time.monotonic()
        if known.sum() >= 50:
            self.map_agree = float(hit[known].mean()); self.map_ok = self.map_agree >= self.AGREE_MIN; self._mc_ok_t = now
        elif now - getattr(self, "_mc_ok_t", -1e9) > self.TAU_LOCAL: self.map_ok = False
        self.slam.stats['map_agree'] = None if self.map_agree is None else round(self.map_agree, 3); self.slam.stats['map_ok'] = self.map_ok

    def to_nav(self, xy):
        from nav.slam import inverse, transform
        return tuple(transform(np.array([xy[:2]], np.float32), inverse(self.map_T()))[0]) if self.decouple else tuple(xy[:2])

    def to_map(self, xy):
        from nav.slam import transform
        return tuple(transform(np.array([xy[:2]], np.float32), self.map_T())[0]) if self.decouple else tuple(xy[:2])

    def _odom_to_nav(self, x, y, th):
        """odometry-frame pose -> the frame nav plans in (odom itself when decoupled, else the SLAM frame via T)."""
        if self.decouple: return x, y, th
        from nav.slam import compose
        return compose(tuple(getattr(self.slam, "T", (0.0, 0.0, 0.0))), (x, y, 0.0 if th is None else th))[:2] + (None if th is None else wrap(th + getattr(self.slam, "T", (0, 0, 0))[2]),)

    def _apply(self, c):
        if c.get("frame") == "odom":   # goals from the goal interface are in the odometry frame
            if c["kind"] == "goto_xy": c = dict(c, frame="nav"); c["x"], c["y"], c["heading"] = self._odom_to_nav(c["x"], c["y"], c["heading"])
            elif c["kind"] == "face": c = dict(c, frame="nav"); c["rad"] = self._odom_to_nav(0.0, 0.0, c["rad"])[2]
        k = c["kind"]; x, y, th = self.navpose()
        if c.get("epoch") is not None and c["epoch"] != self.epoch:      # requested before the latest stop: drop
            self.late_dropped = getattr(self, "late_dropped", 0) + 1
            self.log("dropped late '%s' (request epoch %d, now %d)" % (k, c["epoch"], self.epoch)); return
        if self.core_alarm is not None and k not in ("stop", "where", "places", "label", "forget"):
            self.log("refused '%s': %s" % (k, self.core_alarm)); return
        if k not in ("where", "places", "label", "forget"): self._goal_bumps = 0; self._bump_resume = None
        if self.bump_latched is not None and k not in ("where", "places", "label", "forget"):
            self.bump_latched = None                                       # a new command acknowledges the bump latch
        if k == "stop":
            with self._goal_lock:
                self.epoch += 1; self.vln and self.vln.cancel()
                self.mode = "idle"; self.goal = None; self.planner and self.planner.cancel()
            self.log("stopped")
        elif k == "where": self.log("pose %.2f %.2f %.0f°" % (x, y, math.degrees(th)))
        elif k == "places": self.log("places: " + ", ".join("%s(%.1f,%.1f)" % (n, p[0], p[1]) for n, p in self.places.d.items()) or "none")
        elif k == "label":
            px, py = (c["x"], c["y"]) if "x" in c else self.to_map((x, y))
            if self.decouple and "x" not in c and not self.map_ok: self.log("label here needs a verified map (places live in the map frame)"); return
            self.places.set(c["name"], px, py, th); self.log("labelled %s at %.2f,%.2f" % (c["name"], px, py))
        elif k == "forget": self.log(("forgot " if self.places.remove(c["name"]) else "no place ") + c["name"])
        elif k == "goto_xy":
            if c.get("frame") == "nav": self._goto((c["x"], c["y"]), c["heading"], None, cmd=True)
            elif self.decouple and not self.map_ok: self.log("no verified map: goal taken in the odometry frame"); self._goto((c["x"], c["y"]), c["heading"], None, cmd=True)
            else: self._goto(self.to_nav((c["x"], c["y"])), c["heading"], None, cmd=True)
        elif k == "goto_place":
            p = self.places.get(c["name"])
            if p is None: self.log("unknown place '%s' (known: %s)" % (c["name"], ", ".join(self.places.d) or "none"))
            elif self.decouple and not self.map_ok: self.log("named places need a map that agrees with the current view (agreement %s); use LightNav ('ask ...') or 'forward/turn'" % (None if self.map_agree is None else round(self.map_agree, 2)))
            else: self._goto(self.to_nav(p[:2]), None, c["name"], cmd=True)
        elif k == "relative": self._goto((x + c["d"] * math.cos(th), y + c["d"] * math.sin(th)), None, None, allow_back=c["d"] < 0)
        elif k == "turn":                       # relative turn: track the remaining angle (a 360 used to wrap to 0 -> 'arrived')
            self.turn_target = wrap(th + c["rad"]); self.turn_rem = c["rad"]; self._turn_th = th; self.mode = "turn"; self.log("turning %.0f°" % math.degrees(c["rad"]))
            # the arrival report is about THIS turn (it used to print the previous goal's label and time since that goal)
            self.goal_t0 = time.monotonic(); self.goal_label = "turn %+.0f°" % math.degrees(c["rad"]); self.goal_req = None; self.goal = None; self._resume_path = False
        elif k == "drive":
            self.mode = "drive"; self.drive_cmd = (c["v"], c["w"], time.monotonic() + c["secs"]); self.goal_t0 = time.monotonic()
            self.log("drive v=%.2f w=%.2f for %.1f s" % (c["v"], c["w"], c["secs"]))
        elif k == "face": self.turn_target = wrap(c["rad"]); self.turn_rem = None; self.mode = "turn"
        elif k == "save": self.log("saved map to " + G.save_map(c["name"], self.slam.grid, self.places, self.slam.pose()))
        elif k == "load":
            try:
                G.load_map(c["name"], self.slam.grid, self.places)
                if hasattr(self.slam, "reset_to_grid"): self.slam.reset_to_grid()
                self.log("loaded map %s (%d places); relocalizing..." % (c["name"], len(self.places.d)))
                self._reloc_pending = True; self.slam.frozen = True
            except Exception as e: self.log("load failed: %s" % e)
        elif k == "relocalize": self._reloc_pending = True
        elif k == "ask":
            if self.vln is None: self.log("no VLN model connected (start LightNav-0 and set KEVIN_VLN_URL)")
            else: self.vln.instruct(c["text"])
        else: self.log(c.get("msg", "?"))

    def _goto(self, xy, heading, label, allow_back=False, cmd=False):
        """Global coarse path on the whole map -> local subgoal for the 1 cm window/MPPI. Reports unreachable honestly."""
        if self.planner is None: self.log("planner disabled"); return
        from nav import globalplan as GP
        pose = self.navpose(); xy = np.asarray(xy, float)
        # the same goal again (LightNav re-sends its receding-horizon waypoint every prediction, ~1.7 s): keep the
        # running plan, the controller and the progress watchdog.  Re-planning reset the 'no progress' clock each time,
        # so a wedged robot never got to retreat (live 07:01: 90 x "going to point (-3.82, -0.61)", cmd 0 0).
        same = (cmd and self.mode in ("path", "turn", "backout", "escape") and getattr(self, "goal_req", None) is not None and label == self.goal_label
                and float(np.hypot(*(xy - np.asarray(self.goal_req, float)))) < GOAL_SAME_M)
        if same: self.goal_heading = heading; self.stats.add("goal_same", 1.0); return
        cont = cmd and self.mode in ("path", "turn", "backout", "escape") and getattr(self, "_prog", None) is not None    # new goal mid-route
        keep = (self._prog, self._stuck, self._best_d, self._best_t, self.goal_t0) if cont else None
        g = GP.plan(self.navgrid, pose[:2], xy, self.obs)
        if g is None:
            self.mode = "idle"; self.log("unreachable: no path to %s (%.2f, %.2f)" % (label or "point", xy[0], xy[1])); return
        self.gpath = g; self.goal_req = xy; self.goal = np.asarray(g["goal"], float); self.goal_last = self.goal.tolist()
        self.goal_heading = heading; self.goal_label = label; self.goal_t0 = time.monotonic()
        self._prog = (self.goal_t0, pose); self._stuck = 0; self._snap_logged = False; self._resume_path = False; self.mode = "path"; self._best_d = 1e9; self._best_t = self.goal_t0
        if keep is not None and not getattr(self, "_bo_retry", False):     # mid-route update: the watchdog keeps running
            self._prog, self._stuck, _, self._best_t, self.goal_t0 = keep
        ok = self._set_sub(pose)
        if not ok:
            if getattr(self, "_bo_retry", False): self._bo_fail = True; return                 # retry after a retreat: caller decides
            # wedged (e.g. in a doorway): leave along the way we came in (see _retreat), then plan again
            if self._retreat((xy, heading, label), "no local path from here"): return
            self.mode = "idle"; self.log("unreachable: local planner found no path toward %s (%s)" % (label or "point", getattr(self, "_sub_why", ""))); return
        self._backing_for = None
        note = "" if g["snap_m"] < 0.05 else " (closest reachable point, %.2f m from it)" % g["snap_m"]
        self.log("going to %s (%.2f, %.2f)%s global %.0f ms, local %.1f ms" % (label or "point", xy[0], xy[1], note, g["ms"], self.planner.timing["newgoal_ms"]))
        # only for a goal started from rest: a goal UPDATED mid-route (LightNav's receding-horizon waypoint every ~1 s,
        # computed from a 0.3-1 s old pose) must not stop the robot to spin -- MPPI blends it, and the no-progress
        # watchdog still turns to face the path if MPPI actually stalls (fix3 fridge runs: 4 'turning 97-166 deg' per
        # run on stale waypoints, 4 m covered vs 6-8 m before).
        if keep is None: self._face_path(pose)

    def _face_path(self, pose):
        """The path leaves BEHIND (carrot one body length ahead on the global path is > 90 deg off the heading, i.e.
        driving forward moves away from it) and Kevin has room to spin here (axle clearance >= R_ROTATE): turn in place
        to face it, then hand back to MPPI (Nav2 RPP rotate-to-heading).  MPPI cannot find that itself: a half turn
        fills its 2 s horizon without progress, reversing in the open is costed as a last resort, so the cheapest
        rollout is standing still (stuck dump 125134: goal 0.53 m behind, 0.52 m clearance, u = 0 for 30 s).
        Where Kevin cannot spin, MPPI keeps the path (multi-point turn / retreat along the trail)."""
        gp = getattr(self, "gpath", None); x, y, th = pose
        if gp is None or len(gp["xy"]) < 2 or self.planner is None or self.planner.clearance((x, y)) < _TG.R_ROTATE: return False
        P = np.asarray(gp["xy"], float); i = int(np.argmin(np.hypot(P[:, 0] - x, P[:, 1] - y)))
        s_ = np.r_[0, np.cumsum(np.hypot(*np.diff(P[i:], axis=0).T))]; j = i + min(int(np.searchsorted(s_, L_BODY)), len(s_) - 1)
        if math.hypot(P[j, 0] - x, P[j, 1] - y) < 0.05: return False
        hd = math.atan2(P[j, 1] - y, P[j, 0] - x); e = wrap(hd - th)
        if abs(e) <= math.pi / 2: return False
        self.turn_target = hd; self.turn_rem = None; self._resume_path = True; self._turn_t0 = time.monotonic(); self.mode = "turn"
        self.log("path leaves behind: turning %.0f deg to face it (room to turn here)" % math.degrees(e)); return True

    def _retreat(self, goal_for, why):
        """Leave a dead end along our own trail (where we actually drove: traversable).  The retreat point (the last
        trail pose with room to spin, below) is an ordinary MPPI goal, so the robot turns and drives forward when there is room and reverses only into space
        the RS1 currently sees as free (World.edt_rev).  Afterwards the original goal is planned again."""
        tr = list(getattr(self, "trail", [])); x, y, th = self.navpose()
        if len(tr) < 3 or self.planner is None: return False
        # target: the most recent trail pose where Kevin can spin in place (axle clearance >= R_ROTATE, current map +
        # observation).  From there every direction is open again.  The whole trail is swept-footprint 'known
        # traversable' (obslayer SWEPT_M = trail length), so MPPI may reverse along it.  No such pose on the trail ->
        # its far end.  At least one body length back (L_BODY): a retreat of a few cm changes neither the view nor
        # the approach (seen: 3x "retreating 0.17 m" from a pose that already had room to turn, then gave up).
        d = 0.0; tgt = None; rot = False
        for i in range(len(tr) - 1, 0, -1):
            d += math.hypot(tr[i][0] - tr[i - 1][0], tr[i][1] - tr[i - 1][1])
            if d >= L_BODY and self.planner.clearance(tr[i - 1]) >= _TG.R_ROTATE: tgt = tr[i - 1]; rot = True; break
        tgt = tgt or tr[0]
        if math.hypot(tgt[0] - x, tgt[1] - y) < 0.15: return False
        ok, _ = self.planner.set_goal((x, y, th, self.v, self.w), np.asarray(tgt, float), None)
        if not ok: return False
        self._backing_for = goal_for; self.back_tgt = np.asarray(tgt, float); self.back_t0 = time.monotonic(); self.back_T = 12.0 + 2.0 * d / abs(__import__('core').Limits.v_min);   # 2x the time at full reverse speed
        self.mode = "backout"
        self.log("%s: retreating along the way in (%.2f m) to %s" % (why, math.hypot(tgt[0] - x, tgt[1] - y),
                 "the last spot with room to turn around" if rot else "the start of the trail (no room to turn on it)"))
        return True

    # -------------------------------------------------------------- footprint escape (nav/escape.py)
    def _escape_start(self, goal_for, why):
        """Wedged where Kevin cannot spin (axle clearance < R_ROTATE) and MPPI makes no progress: search a short sequence
        of in-place turns / 4 cm moves that the governor (same rule as the speed limiter, 2x the core's) passes at creep
        speed and whose swept footprint keeps 2 cm clear, ending where Kevin can spin (nav/escape.py; L15 kitchen stall).
        The search runs off the control loop (robot held still); no plan -> the caller's retreat."""
        gi = getattr(self, "_gov_in", None)
        if gi is None or self.planner is None or os.environ.get("KEVIN_ESCAPE", "0") != "1": return False   # default OFF until validated live (n>=3)
        x, y, th = self.navpose()
        if self.planner.clearance((x, y)) >= _TG.R_ROTATE: return False
        pts, p0, bands = gi; Pw = transform(np.asarray(pts[:, :2], np.float32), p0)
        self._esc = dict(state="search", t0=time.monotonic(), goal_for=goal_for, why=why, ep=self.epoch, segs=None, info=None)
        self.mode = "escape"; self.v = self.w = 0.0
        def job(E=self._esc, pose=(x, y, th)):
            from nav import escape as ESC
            try: E["segs"], E["info"] = ESC.plan(Pw, pose, bands=bands)
            except Exception as e: E["segs"], E["info"] = None, dict(why="error %r" % e)
            E["state"] = "found"
        threading.Thread(target=job, daemon=True, name="escape").start(); return True

    def _escape_end(self, ok, msg):
        E = self._esc; self.v = self.w = 0.0; self._stuck_dump("escape_%s" % ("done" if ok else "fail")); xy, hd, lb = E["goal_for"]
        k = self._stuck; self.stats.add("escape_ok" if ok else "escape_fail", 1.0); self.log(msg)
        if ok or not self._retreat(E["goal_for"], E["why"]):
            self._bo_fail = False; self._bo_retry = True; self._goto(xy, hd, lb); self._bo_retry = False
            if self._bo_fail: self.mode = "idle"; self.log("unreachable: local planner found no path toward %s (after escaping)" % (lb or "point"))
        self._stuck = k

    def _escape_step(self, pose):
        """Execute the escape plan: closed-loop on the nav pose, one segment at a time, creep speeds (the governor and
        the core still filter every command).  A segment that does not complete in time (blocked) aborts to the retreat."""
        from nav import escape as ESC
        E = self._esc; x, y, th = pose; now = time.monotonic(); self.v = self.w = 0.0
        if E["state"] == "search":
            if now - E["t0"] > 6.0: E["state"] = "found"; E["segs"], E["info"] = None, dict(why="search timeout")
            return
        if E["state"] == "found":
            if E["ep"] != self.epoch: self.mode = "idle"; return
            if not E["segs"]: return self._escape_end(False, "%s: no footprint escape (%s)" % (E["why"], (E["info"] or {}).get("why")))
            E.update(state="run", i=0, t_seg=now, p_seg=pose, done=0.0, th_prev=th)
            self.log("%s: wedged (axle clearance %.2f m < %.2f to turn): footprint escape %s (%s ms search)" % (
                E["why"], E["info"].get("axle0", 0), _TG.R_ROTATE, " ".join("%s %+.0f%s" % (k, math.degrees(a) if k == "rot" else a * 100, "deg" if k == "rot" else "cm") for k, a in E["segs"]), E["info"].get("ms")))
        if E["state"] != "run": return
        kind, a = E["segs"][E["i"]]
        if kind == "rot":
            E["done"] += wrap(th - E["th_prev"]); E["th_prev"] = th; rem = a - E["done"]; fin = abs(rem) < math.radians(3)
            T = 3.0 + 2.0 * abs(a) / ESC.CREEP_W
            if not fin: self.w = float(math.copysign(min(ESC.CREEP_W, max(0.25, 2.0 * abs(rem))), rem))
        else:
            p0 = E["p_seg"]; prog = (x - p0[0]) * math.cos(p0[2]) + (y - p0[1]) * math.sin(p0[2]); rem = a - prog; fin = abs(rem) < 0.008 or rem * a < 0
            T = 3.0 + 2.0 * abs(a) / ESC.CREEP_V
            if not fin: self.v = float(math.copysign(ESC.CREEP_V, rem)); self.w = float(np.clip(1.5 * wrap(p0[2] - th), -0.3, 0.3))
        if fin:
            E["i"] += 1; self.v = self.w = 0.0
            if E["i"] >= len(E["segs"]): return self._escape_end(True, "escaped: room to turn here, planning again")
            E.update(t_seg=now, p_seg=pose, done=0.0, th_prev=th)
        elif now - E["t_seg"] > T: self._escape_end(False, "footprint escape blocked at step %d/%d (%s): retreating" % (E["i"] + 1, len(E["segs"]), kind))

    def _set_sub(self, pose):
        from nav import globalplan as GP
        sub, final = GP.subgoal(self.gpath["xy"], pose[:2], ahead=4.0)
        tgt = self.goal if final else sub
        ok, m = self.planner.set_goal((*pose, self.v, self.w), tgt, self.goal_heading if final else None)
        self._sub_why = None if ok else "%s; axle clearance %.2f m" % (m, self.planner.clearance(pose[:2]))
        return ok

    # -------------------------------------------------------------- background planner (map window refresh)
    def _plan_loop(self):
        from nav import globalplan as GP
        n = 0
        while self.running:
            time.sleep(0.8 if self.mode in ("path", "backout") else 2.0); n += 1
            try:
                pose = self.navpose()
                self.planner.refresh(pose[:2]); self.stats.add("map_refresh_ms", self.planner.timing["refresh_ms"])
                ep = self.epoch          # a replan computed across a stop must not resurrect the goal (commit below)
                if self.mode == "path" and self.goal is not None:
                    if n % 3 == 0:                                    # global replan ~every 2.4 s on the grown map
                        g = GP.plan(self.navgrid, pose[:2], self.goal_req, self.obs)
                        if self._stuck >= 2:          # repeated no-progress: the goal pocket is tighter than the
                            gc = GP._plan(self.navgrid, pose[:2], self.goal_req, 0.30, self.obs)   # plan claims; aim for a
                            if gc is not None:                                               # conservatively reachable point
                                if gc["snap_m"] > 0.05 and not getattr(self, "_snap_logged", False):
                                    self._snap_logged = True; self.log("goal pocket too tight: aiming for closest reachable point (%.2f m from %s)" % (gc["snap_m"], self.goal_label or "goal"))
                                g = gc
                        if g is not None: self.stats.add("global_ms", g["ms"])
                        if CTL_MODE == "cmd": self._latched_g = (g,); continue      # taken at the next cycle start
                    if CTL_MODE == "cmd":
                        if n % 3 != 0: self._latched_g = (None,)
                        continue
                    with self._goal_lock:
                        if self.epoch != ep or self.mode != "path" or self.goal is None:
                            self.stats.add("replan_dropped", 1.0); continue
                        if n % 3 == 0 and g is not None:
                            self.gpath = g; self.goal = np.asarray(g["goal"], float); self.goal_last = self.goal.tolist()
                        if not self._set_sub(pose): self.log("local replan failed")
            except Exception as e:
                self.log("planner error: %r" % e)

    # -------------------------------------------------------------- 30 Hz loop
    CORE_RESTART_S = 3.0      # supervised restart: dead/hung core -> restart after this long, then every this long

    def _core_watch(self):
        """safety core supervision, every step: dead or stale heartbeat -> stop all goals, refuse motion, alarm (state
        core_alarm + event -> viewer), supervised restart.  Alarm clears after the core has been healthy for 2 s."""
        ok, why = self.hw.core_health(); now = time.monotonic()
        if ok:
            if self.core_alarm is not None:
                if self._core_ok_t is None: self._core_ok_t = now
                elif now - self._core_ok_t > 2.0:
                    self.log("safety core OK again (restarts %d): alarm cleared, motion allowed - goals were cancelled, re-issue" % self.hw.core_restarts)
                    self.core_alarm = None
            return self.core_alarm is None
        self._core_ok_t = None
        if self.core_alarm is None:
            self.core_alarm = why; self._core_bad_t = now
            self._apply({"kind": "stop"}); self.log("SAFETY CORE ALARM: %s - all goals stopped, motion refused" % why)
        self.core_alarm = why; self.hw.drive(0.0, 0.0); self.v = self.w = 0.0
        if not self.hw.emergency_zero() and not getattr(self, "_ez_logged", False):
            self._ez_logged = True; self.log("safety core alarm: emergency zero over CAN FAILED (%s)" % getattr(self.hw, "ez_err", "?"))
        if now - self._core_bad_t > self.CORE_RESTART_S and now - getattr(self, "_core_rs_t", 0.0) > self.CORE_RESTART_S:
            self._core_rs_t = now; self.log("safety core: supervised restart #%d (%s)" % (self.hw.core_restarts + 1, why))
            try: self.hw.restart_core()
            except Exception as e: self.log("safety core restart failed: %s" % e)
        return False

    def step(self):
        if self.ego_shared:                        # the core's labels for its newest RS1 frame (fresh: age <= AGE_MAX_S)
            t_frame, verts = self.ego.wait(0.2)      # verts = core frame id (EgoShared.update ignores it); None = no fresh frame
            if t_frame is not None: self.stats.add("ego_age_ms", self.ego.age * 1e3)
        else:
            t_frame, verts = self.hw.wait_rs1(200)
        t0 = time.monotonic(); self._sec0 = (time.perf_counter(), time.thread_time())
        if getattr(self.hw, "core", False) and not self._core_watch():
            self.mode = "idle"; self.goal = None; self.stats.add("core_alarm", 1)
            with self._cmd_lock: pend, self._pending = self._pending, []
            for c in pend: self._apply(c)                                  # answered now (refused unless stop/where/...)
            return
        if verts is None:
            self.hw.drive(0.0, 0.0); self.pred.issued(time.monotonic(), 0.0, 0.0); self.stats.add("frame_timeout", 1); return
        self.tick += 1
        vl, vr = self.hw.wheel_vels(); t_w = self.hw.wheel_t() or time.monotonic(); gz = self.hw.yaw_rate()
        yi = getattr(self.hw, "yaw_inc", None)        # gyro rotation on its own clock -> rate over THIS odom step
        # odom time = the time of the data it integrates, not the RS1 frame stamp: the gyro increment runs to its last
        # sample (yi[1], ~now) and the wheel speeds were read at t_w, both ~one frame age (45-50 ms) after t_frame.
        # Stamping the step at t_frame made odom.at(t_cap) return a pose ~45 ms in the future (6 deg at 2.5 rad/s):
        # RS1/RS2 frames were mapped at the wrong pose in turns (L1-L3 fit vs GT: odom leads by 45-50 ms).
        t_od = t_frame
        if gz is not None and yi is not None and GYRO_INC:
            t_od = max(float(yi[1]), t_frame)
            self._yaw_carry = getattr(self, "_yaw_carry", 0.0) + yi[0]
            ot = self.slam.odom.t; dtf = 0.0 if ot is None else min(max(t_od - ot, 0.0), 0.6)
            if dtf > 0: gz = self._yaw_carry / dtf; self._yaw_carry = 0.0
        elif self.hw.wheel_t() is not None: t_od = max(float(t_w), t_frame)
        self._t_od = t_od
        self.slam.odom.update(t_od, vl, vr, gz, commanded_zero=(self.v == 0.0 and self.w == 0.0))
        self.slam.stats['gyro_bias'] = round(self.slam.odom.bias, 5); self.slam.stats['zupt'] = getattr(self.slam.odom, 'n_zupt', 0); self.slam.stats['zupt_rej'] = getattr(self.slam.odom, 'n_zupt_rej', 0)
        try:
            lc = getattr(self, "last", None) or {}
            ev = self.bump.update(t_frame, (lc.get("v", 0.0), lc.get("w", 0.0)), (vl, vr), gz, self.hw.motor_currents(), getattr(self.hw, "accel_batch", None))
            self.hw.accel_batch = None
            if ev: self._on_bump(ev)
        except Exception as e:
            self.stats.add("bump_err", 1)
        self._sec("odom_bump")
        lab = self.ego.update(verts); self._sec("ego")
        ego_obs = self.ego.obstacle_xy(max_n=1500); valid = self.ego.valid_fraction()
        if valid > 0: self.obs.update(self.navpose(), lab, self.ego.obs_mask, self.ego.body, t_frame)
        self._sec("obslayer")
        snap = None
        seq, st = self.hw.rs2_seq, self.hw.rs2_stamp
        if seq != self._rs2_seq:
            seq, st, depth, _c, ir1, ir2 = self.hw.rs2_snapshot(); self._rs2_seq = seq; self._rs2_t = st
            if hasattr(self.slam, 'feed'): self.slam.feed(st, ir1, ir2, self.hw.rs2_intr, getattr(self.hw, 'imu', None))
            if self.autocal is not None:
                still = self.v == 0.0 and self.w == 0.0 and abs(vl) < self._still_v and abs(vr) < self._still_v
                acc = getattr(getattr(self.hw, 'imu', None), 'accel', None)
                nw = self.autocal.n_win
                self.autocal.update(still, None if acc is None else np.array(acc, float), depth if still else None)
                if self.autocal.n_win != nw:
                    self.rs2.set_pose(*self.autocal.pose())
                    if hasattr(self.hw, "set_rs2_pose"): self.hw.set_rs2_pose(*self.autocal.pose())
            self._rs2_obs, self._rs2_obs_h, self._rs2_floor = self.rs2.points(depth, return_h=True)
            self._rs2_rays = self.rs2.rays(depth) if CARVE else None
            if seq % 6 == 0 and getattr(self.slam.odom, "ekf", False):       # 5 Hz Manhattan heading (odom EKF, as kslam)
                z4, nw = self.rs2.wall_z4(depth); self.slam.odom.observe_walls(st, z4, nw)
                self.slam.stats['gyro_scale'] = round(self.slam.odom.scale, 5); self.slam.stats['walls'] = (self.slam.odom.n_wall, self.slam.odom.n_wall_rej, nw, round(abs(z4), 3), self.slam.odom.psi0 is not None)
        self._sec("rs2")
        # 5 Hz map/match update by the frame CLOCK (was every 6th brain tick: at the measured ~15 Hz brain rate that was
        # 2.5 Hz -> slam2 processed 1.6 Hz in L14); skip fast spins: RS1/RS2/odom time skew
        if t_frame - getattr(self, "_t_sub", -1e9) >= 0.2 - 1e-3 and abs(self.slam.odom.w) < 0.6:
            self._t_sub = t_frame
            # RS2 points are ~1 frame old: express them at the RS1 stamp using odom (tiny correction, keeps scans rigid)
            po, pr = self.slam.odom.at(t_frame), self.slam.odom.at(self._rs2_t)
            from nav.slam import compose, inverse
            d = compose(inverse(po), pr)
            eo = getattr(self.ego, 'map_obs', ego_obs); obs = np.concatenate([eo, transform(self._rs2_obs, d)]) if len(self._rs2_obs) else eo
            eh = getattr(self.ego, 'map_obs_h', None)
            if eh is None or len(eh) != len(eo): eh = np.full(len(eo), np.nan, np.float32)
            obs_h = np.concatenate([eh, self._rs2_obs_h]) if len(self._rs2_obs) else eh
            rays = None
            if CARVE:      # free-space carving from THIS frame's depth rays (RS1 top-down + RS2 forward, RS2 moved to the RS1 stamp)
                rays = [self.ego.rays((RS1_T[0], RS1_T[1], RS1_T[2] + AXLE_H))]
                if self._rs2_rays is not None and t_frame - self._rs2_t < 0.3:
                    o2, E2, f2 = self._rs2_rays
                    rays.append((np.r_[transform(o2[None, :2], d)[0], o2[2]].astype(np.float32),
                                 np.c_[transform(E2[:, :2], d), E2[:, 2]].astype(np.float32), f2))
            free = np.concatenate([self.ego.clear_xy(), transform(self._rs2_floor[::3], d)]) if len(self._rs2_floor) else self.ego.clear_xy()
            rs2m = transform(self._rs2_obs, d) if len(self._rs2_obs) else None
            # matching uses RS2 only (long range, twin-calibrated mount); RS1 ego points still go into the map
            if self.decouple:
                self._lg_job = (t_frame, po, obs.astype(np.float32), free.astype(np.float32), obs_h.astype(np.float32), rays)
                self._lg_ev.set()
                self._Thist.append((time.monotonic(), tuple(getattr(self.slam, "T", (0.0, 0.0, 0.0)))))
                if time.monotonic() - self._mc_t > 1.0: self._mc_t = time.monotonic(); self.map_check()
            if not self.slam_off: self.slam.submit(t_frame, obs.astype(np.float32), free.astype(np.float32),
                             rs2m.astype(np.float32) if rs2m is not None and len(rs2m) > 100 else None,
                             obs_h=obs_h.astype(np.float32), rays=rays)
        if getattr(self, "_reloc_pending", False) and len(self._rs2_obs) > 200 and not getattr(self, "_reloc_busy", False):
            self._reloc_pending = False; self._reloc_busy = True
            P = self._rs2_obs[np.hypot(self._rs2_obs[:, 0], self._rs2_obs[:, 1]) < 3.5].astype(np.float32); po = self.slam.odom.pose
            def job():
                try:
                    from nav.slam import compose, inverse
                    est, sc, margin = self.slam.relocalize(P)
                    ok = sc > 0.5 and margin > 0.03
                    if ok: self.slam.T = compose(est, inverse(po)); self.slam.frozen = False
                    self.log("relocalized at %.2f %.2f %.0f deg (score %.2f, margin %.2f, %.0f ms)%s" % (est[0], est[1], math.degrees(est[2]), sc, margin,
                             self.slam.stats.get("reloc_ms", 0), "" if ok else " -- AMBIGUOUS, pose not changed; drive a bit and 'relocalize'"))
                finally: self._reloc_busy = False
            threading.Thread(target=job, daemon=True).start()
        self._sec("slam_submit")
        self._feed_core(); self._sec("core_feed")
        self.pred.measured(t_od, vl, vr, self.slam.odom.pose, t_w)  # odom pose at its data time, wheel speeds at their read time
        with self._ctl_cv:
            self._snap = (t_frame, ego_obs, valid, vl, vr, gz); self._snap_n += 1; self._snap_t = time.monotonic(); self._ctl_cv.notify_all()
        self.stats.add("rx_age_ms", (t0 - t_frame) * 1e3)            # frame stamp -> brain picks it up
        tp = getattr(self, "_tpl", None); self._tpl = t0
        self.stats.add("percep_ms", (time.monotonic() - t0) * 1e3)
        if tp is not None: self.stats.add("percep_period_ms", (t0 - tp) * 1e3)
        self._sec("snap"); self._control(); self._sec("control")       # inline in both modes (see _control: a separate plan thread added ~60 ms, measured)

    # -------------------------------------------------------------- control cycle
    def _fit_loop(self):
        while self.running:
            time.sleep(1.0)
            try: tf = time.perf_counter(); self.pred.fit_latency(); self.stats.add("latfit_ms", (time.perf_counter() - tf) * 1e3)
            except Exception as e: self.stats.add("latfit_err", 1)

    def _sec(self, name):
        """per-section wall and thread-CPU time of the brain step (stats sec_*/cpu_* ms): wall - cpu = waiting
        (GIL / CPU run queue / blocking calls)."""
        w, c = time.perf_counter(), time.thread_time(); w0, c0 = getattr(self, "_sec0", (w, c)); self._sec0 = (w, c)
        self.stats.add("sec_" + name + "_ms", (w - w0) * 1e3); self.stats.add("cpu_" + name + "_ms", (c - c0) * 1e3)

    def _feed_core(self):
        """the safety core's last tick (what it SENT the wheels) into the predictor's command history."""
        cs = self.hw.core_state() if getattr(self.hw, "core", False) else None
        if cs: self.pred.core(cs.get("t_sent"), cs.get("v_out"), cs.get("w_out"), cs.get("ramp"), cs.get("period"))
        return cs

    # The command-triggered cycle ran in its own thread in v1/v2 (2026-10-18).  Measured (lat_probe, same goals):
    # frame->command p50 226 -> 287 ms, RS1 handling 28 -> 42 ms, plan compute 4.5 -> 15 ms (two threads contending
    # for the GIL and the 4 shared CPUs), and the snapshot waited ~21 ms for the cycle.  The safety core forwards
    # brain commands only at its RS1-frame ticks, so a plan started at hand-over cannot act sooner than the next tick
    # anyway.  The cycle therefore runs inline right after perception: issuing its command ends the cycle and the next
    # one starts with the next observation; goals stay latched to the cycle start, and the start state is the
    # prediction at the core's forwarding tick + transport latency.  Added latency vs the old loop: the prediction only.

    def _control(self):
        pf = getattr(self, "_prof", None)
        if pf is None and os.environ.get("KEVIN_CTL_PROF"):
            import cProfile; pf = self._prof = cProfile.Profile(); self._prof_n = 0
        if pf is None: return self._control1()
        pf.enable()
        try: return self._control1()
        finally:
            pf.disable(); self._prof_n += 1
            if self._prof_n % 300 == 0: pf.dump_stats(os.environ["KEVIN_CTL_PROF"])

    def _control1(self):
        t_frame, ego_obs, valid, vl, vr, gz = self._snap
        t0 = time.monotonic(); self.stats.add("snap_age_ms", (t0 - getattr(self, "_snap_t", t0)) * 1e3)
        lg = self._latched_g
        if lg is not None and CTL_MODE == "cmd":                    # global replan latched by _plan_loop
            self._latched_g = None
            if self.mode == "path" and self.goal is not None:
                g = lg[0]
                if g is not None: self.gpath = g; self.goal = np.asarray(g["goal"], float); self.goal_last = self.goal.tolist()
                if not self._set_sub(self.navpose()): self.log("local replan failed")
        br = getattr(self, "_bump_resume", None)
        if br and time.monotonic() >= br[0]:
            self._bump_resume = None
            if self.mode == "idle": k = self._goal_bumps; self._goto(*br[1]); self._goal_bumps = k
        with self._cmd_lock: pend, self._pending = self._pending, []
        if self.goal_shm:              # goal interface (KEVIN_GOAL_SHM=1): publish nav state, take a producer's new goal
            od = self.slam.odom.pose
            self._ns.write(time.monotonic_ns(), odom=np.array(od, float), t_od=getattr(self, "_t_od", 0.0), mode=GI.MODES.get(self.mode, 5),
                           ack=self.vln.last_gseq if self.vln is not None and hasattr(self.vln, "last_gseq") else 0,
                           map_T=np.array(getattr(self.slam, "T", (0.0, 0.0, 0.0)), float), map_c=0.0)
            if self.vln is not None and hasattr(self.vln, "poll"): pend = pend + self.vln.poll()
        for c in pend: self._apply(c)
        x, y, th = self.navpose(); dt = 1.0 / HZ; vs, ws = self.v, self.w        # plan start state (frame mode)
        t_eff = self.pred.effect_time(t0)
        xm, ym = x, y; pr = None
        if CTL_MODE == "cmd":
            # latency-compensated start: measured state forward through the issued commands to when THIS command acts
            dt = self.pred.core_period                                # a command holds for one core tick
            self._feed_core(); fw = None
            if getattr(self.hw, "core", False): vl_f, vr_f = self.hw.wheel_vels(); fw = (self.hw.wheel_t(), vl_f, vr_f)
            pr = self.pred.predict(t_eff, fresh=fw)
            if pr is not None and self.mode in ("path", "backout") and os.environ.get("KEVIN_PRED_POSE", "1") == "1":   # =0: diagnosis
                from nav.slam import compose
                # pose = the prediction at effect time.  The control base stays the last ISSUED command (vs, ws =
                # self.v, self.w): MPPI's u is the command (its accel limits are on the command, which is what the
                # ODrive ramp chases).  Integrating u from the predicted ACTUAL speed (d5d6fe3 and before) made each
                # command = actual + a*dt: it could never lead the plant's lag -> 0.08 m/s crawl, 0/9 goals.
                (x, y, th) = compose(self.slam.T, pr[0]) if hasattr(self.slam, "T") else pr[0]
        tr = self.__dict__.setdefault("trail", [])                    # breadcrumbs of where Kevin actually drove
        if not tr or math.hypot(tr[-1][0] - x, tr[-1][1] - y) > 0.05: tr.append((x, y)); del tr[:-80]
        L = self.planner and __import__("core").Limits; integrate_v = __import__("core").integrate_v
        if self.mode == "path" and self.planner is not None:
            if math.hypot(self.goal[0] - x, self.goal[1] - y) < 0.20:          # xy goal tolerance (Nav2 SimpleGoalChecker default 0.25 m)
                if self.goal_heading is not None: self.turn_target = self.goal_heading; self.turn_rem = None; self.mode = "turn"
                else: self._arrive()
            else:
                tp = time.perf_counter(); r = self.planner.step((x, y, th, vs, ws)); self.stats.add("mppi_ms", (time.perf_counter() - tp) * 1e3)
                if r:
                    a, al = r
                    self.v = integrate_v(vs, a, dt, L.v_min, L.v_max); self.w = float(np.clip(ws + al * dt, -L.w_max, L.w_max))
                gp = getattr(self, "gpath", None)
                if gp is not None and len(gp["xy"]) > 1:               # path tracking error (measured pose vs global path)
                    P = np.asarray(gp["xy"], float); A, B = P[:-1], P[1:]; AB = B - A; q = np.array([xm, ym])
                    tt = np.clip(((q - A) * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-9), 0, 1)
                    self.stats.add("track_err_cm", float(np.min(np.hypot(*(A + tt[:, None] * AB - q).T))) * 100)
                tn = time.monotonic(); dg = math.hypot(self.goal[0] - x, self.goal[1] - y)
                if dg < getattr(self, "_best_d", 1e9) - 0.10: self._best_d = dg; self._best_t = tn
                if tn - self._prog[0] > 5.0:
                    p0 = self._prog[1]
                    # stuck = not moving for 5 s, OR moving without getting >=10 cm closer for 12 s (wandering/oscillating)
                    if (math.hypot(x - p0[0], y - p0[1]) < 0.08 and abs(wrap(th - p0[2])) < 0.2) or tn - getattr(self, "_best_t", tn) > 12.0:
                        self._best_t = tn
                        if self._stuck >= 3 and dg < 0.35:                # honest near-arrival instead of "stuck"
                            self.mode = "idle"; self.goal = None; self.planner.cancel(); self.v = self.w = 0.0
                            self.log("stopped %.2f m short of %s (closest reachable)" % (dg, self.goal_label or "goal")); return
                        self._stuck += 1; self._stuck_dump("noprog%d" % self._stuck)
                        if self._stuck >= 4: self.mode = "idle"; self.goal = None; self.planner.cancel(); self.log("gave up: stuck (unreachable from here)")
                        elif self._face_path((x, y, th)): pass          # room to turn here: face the path (no retreat needed)
                        elif self._escape_start((self.goal_req, self.goal_heading, self.goal_label), "no progress (%d)" % self._stuck): pass   # wedged: footprint escape
                        else:
                            self._bo_stuck = self._stuck
                            if not self._retreat((self.goal_req, self.goal_heading, self.goal_label), "no progress (%d)" % self._stuck):
                                k = self._stuck; self._bo_stuck = 0; self._goto(self.goal_req, self.goal_heading, self.goal_label); self._stuck = k   # replan here (fresh observation), keep the count
                    self._prog = (tn, (x, y, th))
        elif self.mode == "backout":                                  # MPPI to the retreat point (see _retreat)
            done = math.hypot(self.back_tgt[0] - x, self.back_tgt[1] - y) < 0.15 or time.monotonic() - self.back_t0 > getattr(self, 'back_T', 12.0)
            r = None if done else self.planner.step((x, y, th, vs, ws))
            if r is None:
                self._stuck_dump("backout_end_%s" % ("reached" if math.hypot(self.back_tgt[0] - x, self.back_tgt[1] - y) < 0.15 else "timeout"))
                self.v = self.w = 0.0; xy, hd, lb = self._backing_for; self._backing_for = None
                self._bo_fail = False; self._bo_retry = True; self._goto(xy, hd, lb); self._bo_retry = False
                if self._bo_fail: self.mode = "idle"; self.log("unreachable: local planner found no path toward %s (after retreating)" % (lb or "point"))
                elif getattr(self, "_bo_stuck", 0): self._stuck = self._bo_stuck; self._bo_stuck = 0   # keep the stuck count (gives up after 4)
            else:
                a, al = r; self.v = integrate_v(vs, a, dt, L.v_min, L.v_max); self.w = float(np.clip(ws + al * dt, -L.w_max, L.w_max))
        elif self.mode == "escape":
            self._escape_step((x, y, th))
        elif self.mode == "drive":
            v, w, tend = self.drive_cmd
            if time.monotonic() > tend: self.v = self.w = 0.0; self.mode = "idle"; self.log("drive done")
            else:
                L = __import__("core").Limits                      # same accel limits as MPPI
                self.v += float(np.clip(v - self.v, -L.a_max * dt, L.a_max * dt)); self.w += float(np.clip(w - self.w, -L.alpha_max * dt, L.alpha_max * dt))
        elif self.mode == "turn":
            if getattr(self, "turn_rem", None) is not None:
                self.turn_rem -= wrap(th - self._turn_th); self._turn_th = th; e = self.turn_rem
            else: e = wrap(self.turn_target - th)
            if getattr(self, "_resume_path", False) and time.monotonic() - getattr(self, "_turn_t0", 0) > 6.0:
                e = 0.0; self.w = 0.0                                   # blocked spin: give the path back to MPPI
            if abs(e) < math.radians(3) and abs(self.w) < 0.3:
                self.v = self.w = 0.0
                if getattr(self, "_resume_path", False) and self.goal is not None:  # noqa
                    self._resume_path = False; self.mode = "path"; self._prog = (time.monotonic(), (x, y, th)); self._set_sub((x, y, th))
                else: self._arrive()
            elif self.planner is not None and self.planner.clearance((x, y)) < _TG.R_ROTATE:
                # no room to spin in place (current observation + map): never back up blind to make room.  Mid-route the
                # path goes back to MPPI (it can make a multi-point turn, reversing only into observed-free space); a final
                # heading at the goal is skipped honestly.
                self.v = self.w = 0.0
                if getattr(self, "_resume_path", False) and self.goal is not None:
                    self._resume_path = False; self.mode = "path"; self._prog = (time.monotonic(), (x, y, th)); self._set_sub((x, y, th))
                else: self.log("no room to turn in place: final heading skipped"); self._arrive()
            else: self.v = 0.0; self.w = float(np.clip(2.2 * e, -_WMAX, _WMAX))
        else:
            self.v = self.w = 0.0
        # safety envelope: fresh RS1 ego + RS2 obstacles (robot frame), stamped
        pts = np.concatenate([ego_obs, self._rs2_obs]) if len(self._rs2_obs) and t_frame - self._rs2_t < 0.3 else ego_obs
        if (self.tick % 3 == 0 and getattr(self, "_mpts_tick", -1) != self.tick) or not hasattr(self, "_mpts"):
            self._mpts = self._map_points(xm, ym, self.navpose()[2]); self._mpts_tick = self.tick
        mp = self._mpts
        if len(self._mpts):
            # remembered map cells that land inside/at the footprint are pose-error artifacts (the robot cannot be
            # inside a wall it is not touching; fresh sensors cover contact): drop them, keep the rest as memory
            mp = self._mpts[self.env._sd(self._mpts) > 0.03]
            if getattr(self.hw, "core", False) and len(mp):
                # planner-side governor only (the legacy in-brain envelope and the safety core are unchanged): a memory
                # cell the RS1 sees as clear floor RIGHT NOW is a pose-error ghost -- the planners' observation layer
                # already drops it, so the governor must too, or it vetoes the planner's every move toward floor the
                # camera sees free (stuck dump 124920: map cell 7.7 cm off the nose, RS1 FREE there, governed 20 s).
                from nav.obslayer import FREE
                mp = mp[self.obs.observed(mp) != FREE]
            if len(mp): pts = np.concatenate([pts, mp])
        v_pre, w_pre = self.v, self.w; t_cap = None                  # t_cap: only the KEVIN_SAFETY_STALE_NOW variant maps it
        if getattr(self.hw, "core", False):
            # safety lives in the safety core process (nav/safety_core.py, last resort, owns the wheels).  Here: the
            # PLANNER keeps 2x the core's speed-relative threshold clear (governor), so the core rarely has to fire.
            from nav.governor import governor
            gw = "off"
            if valid < 0.05: v, w, sc, need2 = 0.0, 0.0, 0.0, 0.0
            elif os.environ.get("KEVIN_SC_GOVERNOR", "1") == "0": v, w, sc, need2 = self.v, self.w, 1.0, 0.0   # test: core brake alone
            else:
                # same rules as the core, 2x (nav/governor.py), with the core's own hard bands.  The brain's points
                # (RS1 ego + RS2 + map memory) are a superset of the core's sources (RS1 ego + RS2).
                cv = self.hw.core_viz(); bands = cv[1] if cv is not None else None
                v, w, sc, need2, gw = governor(pts, self.v, self.w, v_meas=self.slam.odom.v, w_meas=self.slam.odom.w, bands=bands,
                                              escape=self.mode in ("escape", "backout"))   # escapes may move along an obstacle already in the planner zone
                self._gov_in = (pts, (xm, ym, self.navpose()[2]), bands)     # nav/escape.py searches on this exact point set
            info = dict(state="ok" if gw in ("ok", "idle") else "governed", scale=sc, need2=need2, gov=gw, clearance=float(self.env._sd(pts).min()) if len(pts) else None)
            if len(pts):
                _d = self.env._sd(pts); _i = int(np.argmin(_d)); _near = [round(float(pts[_i, 0]), 3), round(float(pts[_i, 1]), 3), round(float(_d[_i]), 3)]
            else: _near = None
            self._gov = (self.v, self.w, v, w, need2, sc, _near, len(pts), len(self._mpts))
            if sc < 1.0: self.stats.add("governed", 1)
        else:
            if os.environ.get("KEVIN_SAFETY_STALE_NOW", "0") == "1":
                # superseded proposal (safety-staleness-now-2026-10-17), legacy in-brain envelope only: frame age against
                # the current time, age added to the reaction time, > 0.5 s immobilised.  Default OFF (the safety core owns this).
                t_cap = self._fclk.capture(t_frame, t0)
                v, w, info = self.env.filter(self.v, self.w, pts, t_cap, valid, now=time.monotonic(), v_meas=self.slam.odom.v, w_meas=self.slam.odom.w)
            elif self.env.measured:
                # superseded proposal (safety-measured-model-on-pr62-2026-10-17), legacy in-brain envelope only, behind
                # KEVIN_SAFETY_MEASURED=1 (default OFF): staleness against the wall clock, measured stop model (nav.safety)
                v, w, info = self.env.filter(self.v, self.w, pts, t_frame, valid, now=time.monotonic(), v_meas=self.slam.odom.v, w_meas=self.slam.odom.w)
            else:
                v, w, info = self.env.filter(self.v, self.w, pts, t_frame, valid, now=t_frame, v_meas=self.slam.odom.v, w_meas=self.slam.odom.w)
        v_saf, w_saf = v, w
        if info["state"] != "ok": self.v, self.w = v, w
        # sensor-rate speed bound: travel at most FRAME_STEP_M (heading FRAME_STEP_RAD) per depth frame, so a
        # degraded camera rate (e.g. GPU contention: 3.7 Hz seen) cannot outrun the obstacle view.  No effect at 30 Hz.
        pf = getattr(self, "_pf", None); self._pf = t_frame
        if pf is not None and t_frame > pf:
            fdt = self._fdt = 0.8 * getattr(self, "_fdt", 1 / 30) + 0.2 * min(t_frame - pf, 1.0)
            vc, wc = 0.05 / fdt, 0.35 / fdt
            if abs(v) > vc or abs(w) > wc:
                k = min(vc / max(abs(v), 1e-6), wc / max(abs(w), 1e-6)); v, w = v * k, w * k; self.stats.add("rate_capped", 1)
        v = max(-_VREV, min(_VMAX, v)); v, w = _TG.limit_vw(v, w)      # hard cap on EVERY command (incl. 'drive v w s'): pi/2, slow for 0.1 g lateral
        # angular-accel limit (twin_geometry.ALPHA_MAX) on spin-UP only, for the controllers that set w directly (turn,
        # follower, drive); slowing |w| / reversing through 0 is never delayed.  MPPI's w already obeys it in rollouts.
        tn = time.monotonic(); wp = getattr(self, "_w_iss", 0.0) if w * getattr(self, "_w_iss", 0.0) > 0 else 0.0
        wu = abs(wp) + _TG.ALPHA_MAX * min(max(tn - getattr(self, "_t_w_iss", tn), 0.0), 0.2)
        if abs(w) > wu: w = math.copysign(wu, w)
        if time.monotonic() < self._bump_hold: v = w = 0.0                  # contact detected: stop (additive; see _on_bump)
        self._w_iss, self._t_w_iss = w, tn
        self.hw.drive(v, w); t_iss = self._t_iss = time.monotonic(); self.pred.issued(t_iss, v, w)
        if self.mode in ("path", "backout"):             # score plan cycles: predicted start vs the raw measurement, both
            if pr is not None: self.pred.remember(t_eff, (pr[0], pr[1], pr[2]), pr[3])   # against the state measured at effect
            elif self.pred.meas: self.pred.remember(t_eff, None, self.pred.meas[-1])
        tc = getattr(self, "_tc0", None); self._tc0 = t0; self.pred.cycle(t_iss - t0, None if tc is None else t0 - tc)
        self.stats.add("ctl_compute_ms", (t_iss - t0) * 1e3); self.stats.add("frame_to_iss_ms", (t_iss - t_frame) * 1e3)
        if self.mode in ("path", "backout"): self.stats.add("sample_to_effect_ms", (t_eff - t_frame) * 1e3)
        if tc is not None: self.stats.add("ctl_period_ms", (t0 - tc) * 1e3)
        if _SLOG is not None:                                   # KEVIN_SAFETY_LOG: per-tick timing/clearance (debug only)
            def _near(P, n=4):                                  # closest n points: (x, y, signed dist to footprint), robot frame
                if P is None or not len(P): return None
                d = self.env._sd(P[:, :2]); k = np.argsort(d)[:n]
                return [[round(float(P[i, 0]), 3), round(float(P[i, 1]), 3), round(float(d[i]), 3)] for i in k]
            def _mind(P):
                if P is None or not len(P): return None
                F = P[(P[:, 0] > 0) & (np.abs(P[:, 1]) < 0.25)]
                return None if not len(F) else round(float(F[:, 0].min()), 3)
            _SLOG.write(json.dumps([round(t0, 4), round(t_frame, 4), round(time.monotonic(), 4), round(v_pre, 3), round(w_pre, 3), round(v_saf, 3), round(w_saf, 3),
                                    round(v, 3), round(w, 3), info.get("state"), info.get("scale"), info.get("clearance"), _mind(ego_obs),
                                    _mind(self._rs2_obs) if len(self._rs2_obs) and t_frame - self._rs2_t < 0.3 else None,
                                    round(vl, 3), round(vr, 3), round(self.slam.odom.v, 3), self.mode, _near(ego_obs),
                                    _near(self._rs2_obs) if len(self._rs2_obs) and t_frame - self._rs2_t < 0.3 else None,
                                    _near(mp) if len(self._mpts) and len(mp) else None, None if t_cap is None else round(t_cap, 4),
                                    None if info.get("age") is None else round(info["age"], 4)]) + "\n")
        if self.rec is not None:
            r1 = getattr(self.hw.rs1, "depth", None); r2 = getattr(self.hw.rs2, "depth", None); im = getattr(self.hw, "imu", None)
            self.rec.put(t=t_frame, t_od=getattr(self, "_t_od", t_frame), v=v, w=w, vl=vl, vr=vr, gz=gz, intr=self.hw.rs2_intr, pose=list(self.navpose()), odom=list(self.slam.odom.pose), mode=self.mode, safety=getattr(self, "_last_safety", None),
                         rs1=None if r1 is None else r1.copy(), rs2=None if r2 is None else r2.copy(),
                         accel=None if im is None else np.array(im.accel, np.float32), gyro=None if im is None else np.array(im.gyro, np.float32),
                         imu_t=None if im is None else getattr(im, "timestamp", None))
        t1 = time.monotonic()
        self.stats.add("loop_ms", (t1 - t0) * 1e3); self.stats.add("frame_to_cmd_ms", (t1 - t_frame) * 1e3)
        if hasattr(self, "_tl"): self.stats.add("period_ms", (t0 - self._tl) * 1e3)
        self._tl = t0; self.last = dict(v=v, w=w, safety=info, valid=valid)
        if STUCK_DUMP and self.mode in ("path", "backout", "turn", "escape"):     # last ~15 s of control decisions (for _stuck_dump)
            pt = self.planner.timing if self.planner is not None else {}
            self.__dict__.setdefault("_ring", collections.deque(maxlen=450)).append(
                (round(t0, 3), self.mode, round(x, 3), round(y, 3), round(th, 3), round(v_pre, 3), round(w_pre, 3), round(v_saf, 3), round(w_saf, 3),
                 round(v, 3), round(w, 3), info.get("state"), info.get("scale"), pt.get("mincost"), pt.get("u"), round(self.slam.odom.v, 3),
                 *(lambda c: (c.get("code"), c.get("v_out"), c.get("w_out"), c.get("free"), c.get("need"), c.get("sd0")) if c else (None,) * 6)(self.hw.core_state() if getattr(self.hw, "core", False) else None)))

    def _stuck_dump(self, why):
        """KEVIN_STUCK_DUMP=<dir>: everything the planners saw at a no-progress / give-up decision -> <dir>/<t>_<why>.npz
        (planning window occ + EDT + reverse mask, observation layer, global + local path, trail, MPPI proposals and
        min cost, governor, the last ~15 s of control decisions).  Rendered by emu/tools/stuck_render.py."""
        if not STUCK_DUMP or self.planner is None: return
        try:
            os.makedirs(STUCK_DUMP, exist_ok=True); p = self.planner
            occ, rev = p._occ_window()
            inf = getattr(p, "last_info", None) or {}
            def _np(a):
                try: return a.detach().float().cpu().numpy()
                except Exception: return np.asarray(a) if a is not None else np.zeros(0)
            with self.obs.lock: src, opose, otr = self.obs.src, self.obs.pose, list(self.obs.trail)
            gp = getattr(self, "gpath", None)
            meta = dict(why=why, t=time.time(), mode=self.mode, pose=list(self.navpose()), goal=None if self.goal is None else list(map(float, self.goal)),
                        goal_req=None if getattr(self, "goal_req", None) is None else list(map(float, self.goal_req)), label=self.goal_label,
                        stuck=self._stuck, best_d=getattr(self, "_best_d", None), prog=[self._prog[0], list(self._prog[1])] if getattr(self, "_prog", None) else None,
                        planner_status=p.status, planner_goal=None if p.goal is None else list(map(float, p.goal)), timing=dict(p.timing), org=list(map(float, p.org)),
                        gov=getattr(self, "_gov", None), clearance=p.clearance(self.navpose()[:2]), R_ROTATE=_TG.R_ROTATE,
                        gp_inflate=None if gp is None else gp.get("inflate"), gp_snap=None if gp is None else gp.get("snap_m"),
                        back_tgt=None if getattr(self, "back_tgt", None) is None else list(map(float, self.back_tgt)), events=list(self.events)[-12:],
                        ring=list(getattr(self, "_ring", [])), map_origin=list(map(float, self.navgrid.origin)), map_res=float(self.navgrid.res))
            fn = os.path.join(STUCK_DUMP, "%s_%s.npz" % (time.strftime("%H%M%S"), why.replace(" ", "_").replace("(", "").replace(")", "")))
            np.savez_compressed(fn, meta=json.dumps(meta, default=str), occ=occ, rev=rev, edt=np.asarray(p.m.edt, np.float32), edt_rev=_np(p.W.edt_rev),
                                obs=src if src is not None else np.zeros((0, 0), np.uint8), obs_pose=np.asarray(opose or (0, 0, 0)), obs_trail=np.asarray(otr),
                                trail=np.asarray(getattr(self, "trail", [])), gpath=np.asarray(gp["xy"]) if gp is not None else np.zeros((0, 2)),
                                lpath=np.asarray(p.path) if p.path is not None else np.zeros((0, 2)), props=_np(inf.get("props")), pcost=_np(inf.get("pcost")),
                                chosen=_np(inf.get("chosen")), mapL=self.navgrid.L)
            print("[nav] stuck dump", fn, flush=True)
        except Exception as e: print("[nav] stuck dump failed: %r" % e, flush=True)

    def _on_bump(self, ev):
        """Contact detected (nav/bump.py).  Additive safety: it can only stop the robot.
        1. Floor check: if the RS1 currently sees clear floor all along the contact edge (2-6 cm outside the footprint),
           there is no obstacle >= 3 cm there, so the motor load was the floor (tile/rug step, threshold) -- logged,
           not a stop.  An unobserved contact edge (the rear, mostly) is always treated as an obstacle.
        2. Otherwise: instant stop + 0.5 s hold, the contact edge is marked in the map (planners and the envelope's map
           memory keep away from it), the event is latched (viewer HUD, chat relay).  LightNav is cancelled; a goal is
           planned again around the marked obstacle (at most twice per goal, then it gives up honestly)."""
        from nav.bump import contact_points
        from nav.obslayer import FREE, OBST
        from twin_geometry import FOOTPRINT_SAFE
        P = contact_points(FOOTPRINT_SAFE, ev["angle"])
        if self.obs is not None:
            lab = np.concatenate([self.obs.observed(contact_points(FOOTPRINT_SAFE, ev["angle"], out=o)) for o in (0.02, 0.04, 0.06)])
            if len(lab) and (lab == FREE).all():
                self.bump.dismiss(ev); self.stats.add("bump_floor", 1)
                if time.monotonic() - getattr(self, "_floor_log_t", 0) > 3.0:
                    self._floor_log_t = time.monotonic(); self.log("floor step (camera sees clear floor %s): %s -- not a contact" % (ev["dir"], ev["why"]))
                return
        self.v = self.w = 0.0; self._bump_hold = time.monotonic() + 0.5; self.hw.drive(0.0, 0.0)
        pose = self.navpose()
        for _ in range(2): self.navgrid.integrate(pose, P, np.zeros((0, 2), np.float32))     # 2 hits = confident cell
        x, y, th = pose; ev = dict(ev, x=round(x, 2), y=round(y, 2)); self.bump_latched = ev; self.bump.last = ev
        resume = None
        vln_on = self.vln is not None and getattr(self.vln, "text", None) is not None
        if vln_on: self.vln.cancel()
        elif self.mode in ("path", "backout", "turn", "escape") and getattr(self, "goal_req", None) is not None:
            self._goal_bumps = getattr(self, "_goal_bumps", 0) + 1
            if self._goal_bumps <= 2: resume = (self.goal_req, self.goal_heading, self.goal_label)
        with self._goal_lock: self.epoch += 1; self.mode = "idle"; self.goal = None; self.planner and self.planner.cancel()
        self.log("BUMP %s: %s -- stopped, obstacle marked in the map%s" % (ev["dir"], ev["why"], "; planning around it" if resume else ""))
        if resume: self._bump_resume = (time.monotonic() + 0.6, resume)

    def _map_points(self, x, y, th, r=0.7):
        """Mapped obstacles within r m, robot frame: memory of what is beside/behind the robot (outside RS1/RS2 view)."""
        g = self.navgrid; c0 = np.floor((np.array([x - r, y - r]) - g.origin) / g.res).astype(int); n = int(2 * r / g.res)
        if (c0 < 0).any() or (c0 + n >= g.n).any(): return np.zeros((0, 2), np.float32)
        occ = g.L[c0[1]:c0[1] + n, c0[0]:c0[0] + n] > 1.5            # confident cells only (>= 2 hits)
        iy, ix = np.nonzero(occ)
        if not len(iy): return np.zeros((0, 2), np.float32)
        wx = g.origin[0] + (c0[0] + ix + 0.5) * g.res - x; wy = g.origin[1] + (c0[1] + iy + 0.5) * g.res - y
        c, s = math.cos(th), math.sin(th)
        return np.stack([c * wx + s * wy, -s * wx + c * wy], 1).astype(np.float32)

    def _arrive(self):
        el = time.monotonic() - self.goal_t0
        d = 0.0
        if getattr(self, "goal_req", None) is not None and self.goal is not None:
            x, y, _ = self.navpose(); d = math.hypot(self.goal_req[0] - x, self.goal_req[1] - y)
        self.log(("arrived at %s in %.1f s" % (self.goal_label or "goal", el)) + ("" if d < 0.2 else " (%.2f m from the requested point: closest reachable)" % d))
        self.mode = "idle"; self.goal = None
        self.planner and self.planner.cancel(); self.v = self.w = 0.0

    def state(self):
        x, y, th = self.navpose()
        p = self.planner
        return dict(t=time.time(), t0=self.t0, pose=[x, y, th], goal_last=getattr(self, "goal_last", None), odom=list(self.slam.odom.pose), mode=self.mode, msg=self.msg, core_alarm=self.core_alarm,
                    goal=None if self.goal is None else list(self.goal), label=self.goal_label,
                    path=None if (p is None or p.path is None) else p.path[::5].round(3).tolist(),
                    places=self.places.d, cmd=getattr(self, "last", {}), hb=getattr(self.hw, "hb_stats", None), vln=None if self.vln is None else self.vln.state(), slam=self.slam.stats,
                    safety=dict(last=getattr(self.env, "last", None), interventions=self.env.interventions, immobilised_ticks=self.env.immobilised, ticks=self.env.ticks,
                                core=self.hw.core_state() if hasattr(self.hw, "core_state") else None, governor=getattr(self, "_gov", None)),
                    timing=self.stats.summary(), threads_cpu_s=_thread_cpu(), ego_shared=dict(frame=self.ego.frame, stale=self.ego.n_stale, gap=self.ego.n_gap, torn=self.ego.n_torn) if self.ego_shared else None, plan_timing=None if p is None else p.timing, ctl=dict(mode=CTL_MODE, **self.pred.summary()),
                    bump=dict(self.bump.state(), latched=self.bump_latched),
                    events=[(round(a, 2), b) for a, b in list(self.events)[-12:]])

    def run(self):
        try:
            while self.running: self.step()
        finally:
            self.hw.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080); ap.add_argument("--map")
    ap.add_argument("--no-match", action="store_true"); ap.add_argument("--no-plan", action="store_true")
    ap.add_argument("--vln", default=os.environ.get("KEVIN_VLN_URL"))
    a = ap.parse_args()
    import signal, sys
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))      # clean stop: wheels zeroed in Brain.run's finally
    
    # CUDA enforcement: planner MUST run on GPU (Jetson Orin; CPU fallback forbidden)
    # Sim (KEVIN_HAL=sim) may run on CPU for dev convenience
    if os.environ.get("KEVIN_HAL", "real") == "real" and not a.no_plan:
        import torch
        if not torch.cuda.is_available():
            print("\n=== FATAL: torch.cuda.is_available() == False ===", file=sys.stderr)
            print("Kevin planner REQUIRES GPU on Orin. CPU fallback is FORBIDDEN.", file=sys.stderr)
            print("Fix: Install PyTorch built for Jetson/JetPack 7.2 with CUDA 13.x support", file=sys.stderr)
            print("See docs/bringup.md for setup instructions.", file=sys.stderr)
            print("Run scripts/build_orin_cuda.sh to verify CUDA.", file=sys.stderr)
            sys.exit(1)
        # Verify GPU is actually usable (not just "available")
        try:
            x = torch.zeros(10, device="cuda")
            _ = x + 1.0
        except Exception as e:
            print(f"\n=== FATAL: GPU device unusable: {e} ===", file=sys.stderr)
            print("torch.cuda.is_available() was True but device allocation failed.", file=sys.stderr)
            print("Check dmesg for CUDA driver errors, verify nvidia-smi works.", file=sys.stderr)
            sys.exit(1)
        print("[nav] CUDA verified: %s" % torch.cuda.get_device_name(0), flush=True)
    
    hw = Hardware()
    b = Brain(hw, match=not a.no_match, plan=not a.no_plan)
    if a.map: b.command("load " + a.map)
    if a.vln and b.goal_shm:      # LightNav in its own fixed-period process; nav only sees its goal block
        import subprocess
        b.vln = GI.GoalPort("lightnav", b.log)
        b._lnp = subprocess.Popen([sys.executable, "-m", "nav.lightnav_proc", "--url", a.vln], stdout=open(os.path.expanduser("~/.kevin/emu/lightnav_proc.log"), "w"), stderr=subprocess.STDOUT)
    elif a.vln:
        from nav.vln import VLN
        b.vln = VLN(a.vln, b)
    from nav.server import serve
    serve(b, a.port)
    code = 0
    try:
        b.run()                                                 # finally: hw.close() (wheels zeroed, CAN down)
    except SystemExit: pass
    except BaseException:
        import traceback; traceback.print_exc(); code = 1
    finally:
        # skip interpreter finalisation: tearing down still-joinable native threads (HAL/WheelBase, torch) there
        # raised std::terminate -> SIGABRT -> Ubuntu apport crash dialog on every eval shutdown (20:45 PT crash)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(code)


if __name__ == "__main__":
    main()
