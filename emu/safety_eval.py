"""Adversarial + nuisance tests for the collision-only safety envelope (nav/safety.py), in the emulator.

  normal   : straight 0.6 m/s runs parallel to a wall at 8-15 cm side clearance -> must NOT intervene
  popup    : at 0.6 m/s a 30x30 cm box appears D m ahead (D = 0.15..1.2) -> collision or not, gap, stop time
  crossing : a box crosses the path at 0.6 m/s, timed to arrive in front of the robot
  blind    : depth dies (RS1+RS2, then RS1 only) at 0.5 m/s -> time to standstill
Driving uses 'drive v w secs' (direct velocity, no planner) so only the envelope stands between robot and box.
python -m emu.safety_eval [--env room] [--out ~/.kevin/emu/safety]"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np
from emu.eval import start, stop, cmd, http
from emu import gt
from emu.shm import read_info

VCAP = float(os.environ.get("KEVIN_VMAX", "0.6"))   # tests trigger at 90 % of the (capped) drive speed
BOX = 0.15          # half size of the mocap box
FRONT = 0.126       # SIL12 front face (axle frame)


def lane(occ, org, res, need_len=3.5, side=None):
    """Pick a (x, y, yaw) in free space with >= need_len m straight ahead; side=d -> a wall at lateral distance d."""
    from scipy.ndimage import distance_transform_edt
    edt = distance_transform_edt(~occ) * res; H, W = occ.shape; best = None
    rng = np.random.default_rng(0)
    for _ in range(6000):
        iy, ix = rng.integers(0, H), rng.integers(0, W)
        if edt[iy, ix] < (0.30 if side is None else side + 0.21 - 0.01): continue
        if side is not None and edt[iy, ix] > side + 0.21 + 0.02: continue
        for yaw in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            ok = True
            for s in np.arange(0.05, need_len, 0.05):
                jx = int(ix + s * math.cos(yaw) / res); jy = int(iy + s * math.sin(yaw) / res)
                if not (0 <= jx < W and 0 <= jy < H) or edt[jy, jx] < (0.26 if side is None else side + 0.21 - 0.03): ok = False; break
            if ok:
                return (org[0] + (ix + 0.5) * res, org[1] + (iy + 0.5) * res, float(yaw))
    return best


_P = {}   # live (world, nav) procs + args; every drive starts from a fresh nav (see below)


def run_drive(port, pose, v, secs, during=None, sample_dt=0.02):
    # teleporting does not move odometry, so nav's obstacle memory from the previous test (wall, popup box) stays
    # glued to the robot and blocks the next drive; restart nav+world so each test is independent
    if _P:
        stop(_P["n"], _P["w"]); _P["w"], _P["n"] = start(_P["env"], 0, port, _P["out"])
    gt.ctl("clear_obstacles"); gt.ctl("teleport", x=pose[0], y=pose[1], yaw=pose[2]); time.sleep(1.2)
    s0 = gt.state(); c0 = s0["collisions"]; i0 = http(port, "/api/state")["safety"]["interventions"]
    cmd(port, "drive %.2f 0 %.1f" % (v, secs)); t0 = time.monotonic(); rows = []; ctx = {}
    while time.monotonic() - t0 < secs + 1.5:
        s = gt.state(); t = time.monotonic() - t0; rows.append((t, s["x"], s["y"], s["yaw"], s["v"], s["collisions"] - c0))
        if during: during(t, s, ctx)
        time.sleep(sample_dt)
    cmd(port, "stop"); time.sleep(0.3)
    st = http(port, "/api/state")
    return np.array(rows), st["safety"]["interventions"] - i0, ctx


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--env", default="room"); ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--out", default=os.path.expanduser("~/.kevin/emu/safety")); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    w, n = start(a.env, 0, a.port, a.out); _P.update(w=w, n=n, env=a.env, out=a.out)
    z = np.load("/dev/shm/kevin_gt_map.npz"); occ = z["occ"].astype(bool); org = z["origin"]; res = float(z["res"])
    R = dict(env=a.env)
    try:
        # ---------------- normal: wall-parallel runs
        R["normal"] = []
        for side in (0.08, 0.12, 0.15):
            p = lane(occ, org, res, 3.0, side=side)
            if p is None: R["normal"].append(dict(side=side, err="no lane")); continue
            rows, iv, _ = run_drive(a.port, p, 0.6, 3.5)
            R["normal"].append(dict(side_clear_m=side, interventions=iv, collisions=int(rows[-1, 5]), vmax=round(float(rows[:, 4].max()), 2),
                                    dist=round(float(math.hypot(rows[-1, 1] - p[0], rows[-1, 2] - p[1])), 2)))
            print("normal", R["normal"][-1], flush=True)
        # ---------------- popup
        # fresh nav: teleporting does not move odometry, so obstacle memory from the wall-side runs stays glued
        # to the robot and froze every later test (popup/crossing/blind never moved)
        p0 = lane(occ, org, res, 4.0)
        R["lane"] = p0; R["popup"] = []
        for D in (1.2, 0.8, 0.5, 0.35, 0.25, 0.15):
            def during(t, s, ctx, D=D):
                if "placed" not in ctx and s["v"] > 0.9 * min(0.6, VCAP):
                    c, sn = math.cos(s["yaw"]), math.sin(s["yaw"]); d = FRONT + D + BOX
                    gt.ctl("obstacle", i=0, x=s["x"] + c * d, y=s["y"] + sn * d); ctx["placed"] = (t, s["x"], s["y"], s["v"])
                    ctx["box"] = (s["x"] + c * d, s["y"] + sn * d)
            rows, iv, ctx = run_drive(a.port, p0, 0.6, 4.0, during)
            r = dict(D=D, interventions=iv, collisions=int(rows[-1, 5]))
            if "placed" in ctx:
                tp = ctx["placed"][0]; c, sn = math.cos(p0[2]), math.sin(p0[2]); bx, by = ctx["box"]
                along = (bx - rows[:, 1]) * c + (by - rows[:, 2]) * sn - BOX - FRONT    # gap front->box face along heading
                after = rows[:, 0] >= tp; stopped = after & (np.abs(rows[:, 4]) < 0.02)
                r.update(v_at_popup=round(ctx["placed"][3], 2), min_gap_m=round(float(along[after].min()), 3),
                         t_to_stop_s=round(float(rows[stopped][0, 0] - tp), 2) if stopped.any() else None)
            R["popup"].append(r); print("popup", r, flush=True)
        # ---------------- crossing: box moves across the lane at 0.6 m/s, arriving when the robot is ~0.6 m away
        R["crossing"] = []
        for lead in (0.9, 0.6, 0.4):
            def during(t, s, ctx, lead=lead):
                c, sn = math.cos(p0[2]), math.sin(p0[2])
                if "t0" not in ctx and s["v"] > 0.9 * min(0.6, VCAP):
                    ctx["t0"] = t; ctx["meet"] = (s["x"] + c * (FRONT + lead + BOX + 0.6 * 1.0), s["y"] + sn * (FRONT + lead + BOX + 0.6 * 1.0))
                if "t0" in ctx:
                    u = t - ctx["t0"]; lat = -0.6 + 0.6 * u                    # from 0.6 m to the side, crossing in 1.0 s
                    if lat < 0.8: gt.ctl("obstacle", i=0, x=ctx["meet"][0] - sn * lat, y=ctx["meet"][1] + c * lat)
            rows, iv, ctx = run_drive(a.port, p0, 0.6, 4.0, during)
            R["crossing"].append(dict(lead=lead, interventions=iv, collisions=int(rows[-1, 5]))); print("crossing", R["crossing"][-1], flush=True)
        # ---------------- blind
        R["blind"] = []
        for cams in (["rs1", "rs2"], ["rs1"], ["rs2"]):
            def during(t, s, ctx, cams=cams):
                ctx.setdefault("t0", t)
                if "t" not in ctx and (s["v"] > 0.9 * min(0.5, VCAP) or t - ctx["t0"] > 1.0): gt.ctl("blind", cams=cams); ctx["t"] = t; ctx["v_at"] = s["v"]
            rows, iv, ctx = run_drive(a.port, p0, 0.5, 3.0, during)
            gt.ctl("blind", cams=[])
            r = dict(cams=cams, triggered="t" in ctx, v_at_blind=round(float(ctx.get("v_at", 0.0)), 2), vmax=round(float(rows[:, 4].max()), 2),
                     interventions=iv, collisions=int(rows[-1, 5]))
            if "t" in ctx:
                after = rows[:, 0] >= ctx["t"]; st = after & (np.abs(rows[:, 4]) < 0.02)
                r.update(t_to_stop_s=round(float(rows[st][0, 0] - ctx["t"]), 2) if st.any() else None,
                         moved_after_m=round(float(math.hypot(rows[-1, 1] - rows[after][0, 1], rows[-1, 2] - rows[after][0, 2])), 3))
            R["blind"].append(r); print("blind", r, flush=True); time.sleep(1.0)
    finally:
        if _P: w, n = _P["w"], _P["n"]
        try: R["final"] = http(a.port, "/api/state")["safety"]
        except Exception: pass
        stop(n, w)
    json.dump(R, open(os.path.join(a.out, "safety_%s.json" % a.env), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
