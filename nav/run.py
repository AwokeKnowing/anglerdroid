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
from nav.slam import Slam, GridMap, wrap, transform
from nav.safety import Envelope, FrameClock
import twin_geometry as _TG
_sl = os.environ.get("KEVIN_SAFETY_LOG"); _SLOG = open(_sl, "a", buffering=1) if _sl else None
L_BODY = float(_TG.FOOTPRINT_SAFE[:, 0].max() - _TG.FOOTPRINT_SAFE[:, 0].min())   # 0.40 m nose to tail
from nav import goals as G
_VMAX = float(os.environ.get("KEVIN_VMAX", "0.6")); _VREV = min(0.2, _VMAX); _WMAX = float(os.environ.get("KEVIN_WMAX", "2.5"))

HZ = 30.0


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
        self.rec = None; self.t0 = round(time.time(), 3)          # session id: the map frame is this run's start pose
        if os.environ.get("KEVIN_RECORD"):
            from nav.record import Recorder
            self.rec = Recorder(os.environ["KEVIN_RECORD"]); print("[nav] recording to %s" % self.rec.d, flush=True)
        self.autocal = None
        if os.environ.get("KEVIN_AUTOCAL", "0" if os.environ.get("KEVIN_HAL") == "sim" else "1") == "1":
            from nav.autocal import AutoCal, STILL_V
            self.autocal = AutoCal(hw.rs2_intr, log=self.log); self._still_v = STILL_V
        _sk = os.environ.get('KEVIN_SLAM', '2'); self.slam = (__import__('nav.vslam_cuv', fromlist=['CuvSlam']).CuvSlam(GridMap()) if _sk == 'cuv' else __import__('nav.slam2', fromlist=['Slam2']).Slam2(GridMap(), match=match) if _sk == '2' else Slam(GridMap(), match=match)); self.places = G.Places(); self.env = Envelope(); self._fclk = FrameClock()
        self.planner = None
        if plan:
            from nav.plan import Planner
            self.planner = Planner(self.slam.grid)
        from nav.obslayer import ObsLayer
        self.obs = ObsLayer()                       # current RS1 frame, authoritative over the map where it sees (planners)
        from nav.bump import BumpDetector                # contact detector (motor current + IMU), additive safety
        import twin_geometry as _TG
        self.bump = BumpDetector(_TG.TRACK_M); self.bump_latched = None; self._bump_hold = 0.0
        if self.planner is not None: self.planner.obs = self.obs
        self.mode = "idle"; self.msg = ""; self.v = 0.0; self.w = 0.0; self.turn_target = None
        self.goal = None; self.goal_heading = None; self.goal_label = None; self.goal_t0 = 0.0
        self.stats = Stats(); self.events = collections.deque(maxlen=50); self.tick = 0
        self._cmd_lock = threading.Lock(); self._pending = []
        self._rs2_seq = -1; self._rs2_obs = np.zeros((0, 2), np.float32); self._rs2_floor = np.zeros((0, 2), np.float32); self._rs2_t = 0.0
        self.vln = None
        self.running = True
        if self.planner: threading.Thread(target=self._plan_loop, daemon=True, name="plan").start()

    # -------------------------------------------------------------- commands (any thread)
    def command(self, text):
        c = G.parse(text)
        with self._cmd_lock: self._pending.append(c)
        return c

    def log(self, s):
        self.events.append((time.time(), s)); self.msg = s; print("[nav]", s, flush=True)

    def _apply(self, c):
        k = c["kind"]; x, y, th = self.slam.pose()
        if k not in ("where", "places", "label", "forget"): self._goal_bumps = 0; self._bump_resume = None
        if self.bump_latched is not None and k not in ("where", "places", "label", "forget"):
            self.bump_latched = None                                       # a new command acknowledges the bump latch
        if k == "stop" and self.vln: self.vln.cancel()
        if k == "stop": self.mode = "idle"; self.goal = None; self.planner and self.planner.cancel(); self.log("stopped")
        elif k == "where": self.log("pose %.2f %.2f %.0f°" % (x, y, math.degrees(th)))
        elif k == "places": self.log("places: " + ", ".join("%s(%.1f,%.1f)" % (n, p[0], p[1]) for n, p in self.places.d.items()) or "none")
        elif k == "label":
            px, py = (c["x"], c["y"]) if "x" in c else (x, y)
            self.places.set(c["name"], px, py, th); self.log("labelled %s at %.2f,%.2f" % (c["name"], px, py))
        elif k == "forget": self.log(("forgot " if self.places.remove(c["name"]) else "no place ") + c["name"])
        elif k == "goto_xy": self._goto((c["x"], c["y"]), c["heading"], None)
        elif k == "goto_place":
            p = self.places.get(c["name"])
            if p is None: self.log("unknown place '%s' (known: %s)" % (c["name"], ", ".join(self.places.d) or "none"))
            else: self._goto(p[:2], None, c["name"])
        elif k == "relative": self._goto((x + c["d"] * math.cos(th), y + c["d"] * math.sin(th)), None, None, allow_back=c["d"] < 0)
        elif k == "turn":                       # relative turn: track the remaining angle (a 360 used to wrap to 0 -> 'arrived')
            self.turn_target = wrap(th + c["rad"]); self.turn_rem = c["rad"]; self._turn_th = th; self.mode = "turn"; self.log("turning %.0f°" % math.degrees(c["rad"]))
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

    def _goto(self, xy, heading, label, allow_back=False):
        """Global coarse path on the whole map -> local subgoal for the 1 cm window/MPPI. Reports unreachable honestly."""
        if self.planner is None: self.log("planner disabled"); return
        from nav import globalplan as GP
        pose = self.slam.pose(); xy = np.asarray(xy, float)
        g = GP.plan(self.slam.grid, pose[:2], xy, self.obs)
        if g is None:
            self.mode = "idle"; self.log("unreachable: no path to %s (%.2f, %.2f)" % (label or "point", xy[0], xy[1])); return
        self.gpath = g; self.goal_req = xy; self.goal = np.asarray(g["goal"], float); self.goal_last = self.goal.tolist()
        self.goal_heading = heading; self.goal_label = label; self.goal_t0 = time.monotonic()
        self._prog = (self.goal_t0, pose); self._stuck = 0; self._snap_logged = False; self._resume_path = False; self.mode = "path"; self._best_d = 1e9; self._best_t = self.goal_t0
        ok = self._set_sub(pose)
        if not ok:
            if getattr(self, "_bo_retry", False): self._bo_fail = True; return                 # retry after a retreat: caller decides
            # wedged (e.g. in a doorway): leave along the way we came in (see _retreat), then plan again
            if self._retreat((xy, heading, label), "no local path from here"): return
            self.mode = "idle"; self.log("unreachable: local planner found no path toward %s (%s)" % (label or "point", getattr(self, "_sub_why", ""))); return
        self._backing_for = None
        note = "" if g["snap_m"] < 0.05 else " (closest reachable point, %.2f m from it)" % g["snap_m"]
        self.log("going to %s (%.2f, %.2f)%s global %.0f ms, local %.1f ms" % (label or "point", xy[0], xy[1], note, g["ms"], self.planner.timing["newgoal_ms"]))

    def _retreat(self, goal_for, why):
        """Leave a dead end along our own trail (where we actually drove: traversable).  The retreat point (the last
        trail pose with room to spin, below) is an ordinary MPPI goal, so the robot turns and drives forward when there is room and reverses only into space
        the RS1 currently sees as free (World.edt_rev).  Afterwards the original goal is planned again."""
        tr = list(getattr(self, "trail", [])); x, y, th = self.slam.pose()
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
                pose = self.slam.pose()
                self.planner.refresh(pose[:2]); self.stats.add("map_refresh_ms", self.planner.timing["refresh_ms"])
                if self.mode == "path" and self.goal is not None:
                    if n % 3 == 0:                                    # global replan ~every 2.4 s on the grown map
                        g = GP.plan(self.slam.grid, pose[:2], self.goal_req, self.obs)
                        if self._stuck >= 2:          # repeated no-progress: the goal pocket is tighter than the
                            gc = GP._plan(self.slam.grid, pose[:2], self.goal_req, 0.30, self.obs)   # plan claims; aim for a
                            if gc is not None:                                               # conservatively reachable point
                                if gc["snap_m"] > 0.05 and not getattr(self, "_snap_logged", False):
                                    self._snap_logged = True; self.log("goal pocket too tight: aiming for closest reachable point (%.2f m from %s)" % (gc["snap_m"], self.goal_label or "goal"))
                                g = gc
                        if g is not None:
                            self.gpath = g; self.goal = np.asarray(g["goal"], float); self.goal_last = self.goal.tolist()
                            self.stats.add("global_ms", g["ms"])
                    if not self._set_sub(pose): self.log("local replan failed")
            except Exception as e:
                self.log("planner error: %r" % e)

    # -------------------------------------------------------------- 30 Hz loop
    def step(self):
        t_frame, verts = self.hw.wait_rs1(200)
        t0 = time.monotonic()
        if verts is None:
            self.hw.drive(0.0, 0.0); self.stats.add("frame_timeout", 1); return
        self.tick += 1
        vl, vr = self.hw.wheel_vels(); gz = self.hw.yaw_rate()
        self.slam.odom.update(t_frame, vl, vr, gz, commanded_zero=(self.v == 0.0 and self.w == 0.0))
        self.slam.stats['gyro_bias'] = round(self.slam.odom.bias, 5); self.slam.stats['zupt'] = getattr(self.slam.odom, 'n_zupt', 0)
        try:
            lc = getattr(self, "last", None) or {}
            ev = self.bump.update(t_frame, (lc.get("v", 0.0), lc.get("w", 0.0)), (vl, vr), gz, self.hw.motor_currents(), getattr(self.hw, "accel_batch", None))
            self.hw.accel_batch = None
            if ev: self._on_bump(ev)
        except Exception as e:
            self.stats.add("bump_err", 1)
        lab = self.ego.update(verts)
        ego_obs = self.ego.obstacle_xy(max_n=1500); valid = self.ego.valid_fraction()
        if valid > 0: self.obs.update(self.slam.pose(), lab, self.ego.obs_mask, self.ego.body, t_frame)
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
            self._rs2_obs, self._rs2_floor = self.rs2.points(depth)
        if self.tick % 6 == 0 and abs(self.slam.odom.w) < 0.6:   # 5 Hz map/match update (skip fast spins: RS1/RS2/odom time skew)
            # RS2 points are ~1 frame old: express them at the RS1 stamp using odom (tiny correction, keeps scans rigid)
            po, pr = self.slam.odom.at(t_frame), self.slam.odom.at(self._rs2_t)
            from nav.slam import compose, inverse
            d = compose(inverse(po), pr)
            eo = getattr(self.ego, 'map_obs', ego_obs); obs = np.concatenate([eo, transform(self._rs2_obs, d)]) if len(self._rs2_obs) else eo
            free = np.concatenate([self.ego.clear_xy(), transform(self._rs2_floor[::3], d)]) if len(self._rs2_floor) else self.ego.clear_xy()
            rs2m = transform(self._rs2_obs, d) if len(self._rs2_obs) else None
            # matching uses RS2 only (long range, twin-calibrated mount); RS1 ego points still go into the map
            self.slam.submit(t_frame, obs.astype(np.float32), free.astype(np.float32),
                             rs2m.astype(np.float32) if rs2m is not None and len(rs2m) > 100 else None)
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
        br = getattr(self, "_bump_resume", None)
        if br and time.monotonic() >= br[0]:
            self._bump_resume = None
            if self.mode == "idle": k = self._goal_bumps; self._goto(*br[1]); self._goal_bumps = k
        with self._cmd_lock: pend, self._pending = self._pending, []
        for c in pend: self._apply(c)
        x, y, th = self.slam.pose(); dt = 1.0 / HZ
        tr = self.__dict__.setdefault("trail", [])                    # breadcrumbs of where Kevin actually drove
        if not tr or math.hypot(tr[-1][0] - x, tr[-1][1] - y) > 0.05: tr.append((x, y)); del tr[:-80]
        L = self.planner and __import__("core").Limits; integrate_v = __import__("core").integrate_v
        if self.mode == "path" and self.planner is not None:
            if math.hypot(self.goal[0] - x, self.goal[1] - y) < 0.20:          # xy goal tolerance (Nav2 SimpleGoalChecker default 0.25 m)
                if self.goal_heading is not None: self.turn_target = self.goal_heading; self.turn_rem = None; self.mode = "turn"
                else: self._arrive()
            else:
                tp = time.perf_counter(); r = self.planner.step((x, y, th, self.v, self.w)); self.stats.add("mppi_ms", (time.perf_counter() - tp) * 1e3)
                if r:
                    a, al = r
                    self.v = integrate_v(self.v, a, dt, L.v_min, L.v_max); self.w = float(np.clip(self.w + al * dt, -L.w_max, L.w_max))
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
                        self._stuck += 1
                        if self._stuck >= 4: self.mode = "idle"; self.goal = None; self.planner.cancel(); self.log("gave up: stuck (unreachable from here)")
                        else:
                            self._bo_stuck = self._stuck
                            if not self._retreat((self.goal_req, self.goal_heading, self.goal_label), "no progress (%d)" % self._stuck):
                                k = self._stuck; self._bo_stuck = 0; self._goto(self.goal_req, self.goal_heading, self.goal_label); self._stuck = k   # replan here (fresh observation), keep the count
                    self._prog = (tn, (x, y, th))
        elif self.mode == "backout":                                  # MPPI to the retreat point (see _retreat)
            done = math.hypot(self.back_tgt[0] - x, self.back_tgt[1] - y) < 0.15 or time.monotonic() - self.back_t0 > getattr(self, 'back_T', 12.0)
            r = None if done else self.planner.step((x, y, th, self.v, self.w))
            if r is None:
                self.v = self.w = 0.0; xy, hd, lb = self._backing_for; self._backing_for = None
                self._bo_fail = False; self._bo_retry = True; self._goto(xy, hd, lb); self._bo_retry = False
                if self._bo_fail: self.mode = "idle"; self.log("unreachable: local planner found no path toward %s (after retreating)" % (lb or "point"))
                elif getattr(self, "_bo_stuck", 0): self._stuck = self._bo_stuck; self._bo_stuck = 0   # keep the stuck count (gives up after 4)
            else:
                a, al = r; self.v = integrate_v(self.v, a, dt, L.v_min, L.v_max); self.w = float(np.clip(self.w + al * dt, -L.w_max, L.w_max))
        elif self.mode == "drive":
            v, w, tend = self.drive_cmd
            if time.monotonic() > tend: self.v = self.w = 0.0; self.mode = "idle"; self.log("drive done")
            else:
                L = __import__("core").Limits                      # same accel limits as MPPI
                self.v += float(np.clip(v - self.v, -L.a_max / HZ, L.a_max / HZ)); self.w += float(np.clip(w - self.w, -L.alpha_max / HZ, L.alpha_max / HZ))
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
            else: self.v = 0.0; self.w = float(np.clip(2.2 * e, -1.6, 1.6))
        else:
            self.v = self.w = 0.0
        # safety envelope: fresh RS1 ego + RS2 obstacles (robot frame), stamped
        pts = np.concatenate([ego_obs, self._rs2_obs]) if len(self._rs2_obs) and t_frame - self._rs2_t < 0.3 else ego_obs
        if self.tick % 3 == 0 or not hasattr(self, "_mpts"): self._mpts = self._map_points(x, y, th)
        mp = self._mpts
        if len(self._mpts):
            # remembered map cells that land inside/at the footprint are pose-error artifacts (the robot cannot be
            # inside a wall it is not touching; fresh sensors cover contact): drop them, keep the rest as memory
            mp = self._mpts[self.env._sd(self._mpts) > 0.03]
            if len(mp): pts = np.concatenate([pts, mp])
        v_pre, w_pre = self.v, self.w; t_cap = None                  # t_cap: only the KEVIN_SAFETY_STALE_NOW variant maps it
        if getattr(self.hw, "core", False):
            # safety lives in the safety core process (nav/safety_core.py, last resort, owns the wheels).  Here: the
            # PLANNER keeps 2x the core's speed-relative threshold clear (governor), so the core rarely has to fire.
            from nav.safety_core import governor
            if valid < 0.05: v, w, sc, need2 = 0.0, 0.0, 0.0, 0.0
            elif os.environ.get("KEVIN_SC_GOVERNOR", "1") == "0": v, w, sc, need2 = self.v, self.w, 1.0, 0.0   # test: core brake alone
            else: v, w, sc, need2 = governor(pts, self.v, self.w, v_meas=self.slam.odom.v)
            info = dict(state="ok" if sc == 1.0 else "governed", scale=sc, need2=need2, clearance=float(self.env._sd(pts).min()) if len(pts) else None)
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
        v = max(-_VREV, min(_VMAX, v)); w = max(-_WMAX, min(_WMAX, w))      # hard cap on EVERY command (incl. 'drive v w s')
        if time.monotonic() < self._bump_hold: v = w = 0.0                  # contact detected: stop (additive; see _on_bump)
        self.hw.drive(v, w)
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
            self.rec.put(t=t_frame, v=v, w=w, vl=vl, vr=vr, gz=gz, intr=self.hw.rs2_intr, pose=list(self.slam.pose()), odom=list(self.slam.odom.pose), mode=self.mode, safety=getattr(self, "_last_safety", None),
                         rs1=None if r1 is None else r1.copy(), rs2=None if r2 is None else r2.copy(),
                         accel=None if im is None else np.array(im.accel, np.float32), gyro=None if im is None else np.array(im.gyro, np.float32),
                         imu_t=None if im is None else getattr(im, "timestamp", None))
        t1 = time.monotonic()
        self.stats.add("loop_ms", (t1 - t0) * 1e3); self.stats.add("frame_to_cmd_ms", (t1 - t_frame) * 1e3)
        if hasattr(self, "_tl"): self.stats.add("period_ms", (t0 - self._tl) * 1e3)
        self._tl = t0; self.last = dict(v=v, w=w, safety=info, valid=valid)

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
        pose = self.slam.pose()
        for _ in range(2): self.slam.grid.integrate(pose, P, np.zeros((0, 2), np.float32))     # 2 hits = confident cell
        x, y, th = pose; ev = dict(ev, x=round(x, 2), y=round(y, 2)); self.bump_latched = ev; self.bump.last = ev
        resume = None
        vln_on = self.vln is not None and getattr(self.vln, "text", None) is not None
        if vln_on: self.vln.cancel()
        elif self.mode in ("path", "backout", "turn") and getattr(self, "goal_req", None) is not None:
            self._goal_bumps = getattr(self, "_goal_bumps", 0) + 1
            if self._goal_bumps <= 2: resume = (self.goal_req, self.goal_heading, self.goal_label)
        self.mode = "idle"; self.goal = None; self.planner and self.planner.cancel()
        self.log("BUMP %s: %s -- stopped, obstacle marked in the map%s" % (ev["dir"], ev["why"], "; planning around it" if resume else ""))
        if resume: self._bump_resume = (time.monotonic() + 0.6, resume)

    def _map_points(self, x, y, th, r=0.7):
        """Mapped obstacles within r m, robot frame: memory of what is beside/behind the robot (outside RS1/RS2 view)."""
        g = self.slam.grid; c0 = np.floor((np.array([x - r, y - r]) - g.origin) / g.res).astype(int); n = int(2 * r / g.res)
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
            x, y, _ = self.slam.pose(); d = math.hypot(self.goal_req[0] - x, self.goal_req[1] - y)
        self.log(("arrived at %s in %.1f s" % (self.goal_label or "goal", el)) + ("" if d < 0.2 else " (%.2f m from the requested point: closest reachable)" % d))
        self.mode = "idle"; self.goal = None
        self.planner and self.planner.cancel(); self.v = self.w = 0.0

    def state(self):
        x, y, th = self.slam.pose()
        p = self.planner
        return dict(t=time.time(), t0=self.t0, pose=[x, y, th], goal_last=getattr(self, "goal_last", None), odom=list(self.slam.odom.pose), mode=self.mode, msg=self.msg,
                    goal=None if self.goal is None else list(self.goal), label=self.goal_label,
                    path=None if (p is None or p.path is None) else p.path[::5].round(3).tolist(),
                    places=self.places.d, cmd=getattr(self, "last", {}), hb=getattr(self.hw, "hb_stats", None), vln=None if self.vln is None else self.vln.state(), slam=self.slam.stats,
                    safety=dict(last=getattr(self.env, "last", None), interventions=self.env.interventions, immobilised_ticks=self.env.immobilised, ticks=self.env.ticks,
                                core=self.hw.core_state() if hasattr(self.hw, "core_state") else None, governor=getattr(self, "_gov", None)),
                    timing=self.stats.summary(), plan_timing=None if p is None else p.timing,
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
    if a.vln:
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
