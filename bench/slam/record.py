"""Offline sim recorder for the SLAM benchmark (bench/slam).

Steps the emu.world MuJoCo twin IN-PROCESS (no shm, no CAN/UDP sockets: never touches a running emu.world/nav.run),
drives Kevin along an A*-planned route through the house with a ground-truth pure-pursuit controller (data generation
only), and logs everything a SLAM method could use, with the emulator's own sensor/odometry error model:

  OUT/meta.json              env, seed, noise level + drawn error params, intrinsics, twin extrinsics (base<-optical),
                             TRUE mast HFOV and TRUE RS2 mount (withheld from methods that claim self-calibration)
  OUT/frames.csv             i, t   (camera frame clock, s)
  OUT/rs2_color/%06d.jpg     forward D435i RGB 640x480 (q95)
  OUT/rs2_depth/%06d.png     forward D435i depth u16 mm 848x480 (D4xx noise model: Z^2 subpix, min-Z, left band, occlusion, holes)
  OUT/rs2_ir1|ir2/%06d.png   forward D435i IR pair y8 848x480 (read noise)
  OUT/mast/%06d.jpg          mast RGB 640x480 through the TRUE (withheld) lens: DFOV 118, residual barrel (realism.mast_truth)
  OUT/rs1_depth/%06d.png     top-down D435 depth u16 mm 848x480
  OUT/gt.csv                 t, x, y, yaw, z, roll, pitch (base_link in world), v, w   @100 Hz
  OUT/wheel.csv              t, vl, vr  (m/s, as the robot reads them: radius/track error + encoder noise) @100 Hz
  OUT/imu.csv                t, gx, gy, gz, ax, ay, az  (D435i IMU optical frame: scale/bias/random-walk/white) @200 Hz
  OUT/cmd.csv                t, v, w    (commanded twist)

python bench/slam/record.py --route tour --out ~/slambench/data/casita_tour [--hz 15] [--rs2-pitch 27] [--rs1]
Realism (textures, decor, lights, CamFX, mast lens truth) comes from emu/realism.py via emu.world (KEVIN_EMU_REAL=1).
"""
from __future__ import annotations
import argparse, csv, heapq, json, math, os, shutil, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO]

ROUTES = {
    # place sequence; "@spin" = 360 deg in-place turn there, "@spinr" = -360 deg (opposite direction)
    "tour":  ["living room", "@spin", "kitchen", "@spin", "hallway", "bedroom", "@spinr", "hallway", "kitchen table", "living room", "@spin"],
    "spins": ["living room", "@spin", "@spinr", "@spin", "kitchen", "@spin", "@spin", "@spinr", "living room", "@spinr"],
    "long":  ["living room", "front door", "kitchen", "laundry", "hallway", "bedroom", "desk", "bedroom 2", "hallway", "kitchen table", "couch", "@spin", "living room"],
    # genuinely different routes: own start pose (START) and path through other rooms
    "bedrooms": ["hallway", "bedroom", "desk", "@spin", "bedroom", "hallway", "laundry", "kitchen", "@spinr", "kitchen table"],
    "kitchen_loop": ["kitchen", "front door", "living room", "couch", "@spinr", "living room", "kitchen", "laundry", "@spin", "kitchen table"],
    "door_long": ["living room", "kitchen table", "laundry", "hallway", "bedroom 2", "@spin", "hallway", "bedroom", "@spinr", "hallway"],
}
START = {"bedrooms": (5.0, 11.0, math.pi), "kitchen_loop": (5.6, 4.95, -math.pi / 2), "door_long": (0.9, 4.0, 0.0)}   # x, y, yaw (else env.start)


def astar(free_cost, start, goal):
    """8-connected A* on a cost grid (inf = blocked). start/goal (r, c). Returns list of (r, c)."""
    H, W = free_cost.shape; INF = np.inf
    g = np.full((H, W), INF); g[start] = 0; par = {}
    pq = [(0.0, start)]; nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    while pq:
        f, (r, c) = heapq.heappop(pq)
        if (r, c) == goal: break
        gc = g[r, c]
        if f - math.hypot(r - goal[0], c - goal[1]) > gc + 1e-9: continue
        for dr, dc, w in nb:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W): continue
            k = free_cost[rr, cc]
            if k == INF: continue
            ng = gc + w * k
            if ng < g[rr, cc]:
                g[rr, cc] = ng; par[(rr, cc)] = (r, c); heapq.heappush(pq, (ng + math.hypot(rr - goal[0], cc - goal[1]), (rr, cc)))
    if goal not in par and goal != start: return None
    p = [goal]
    while p[-1] != start: p.append(par[p[-1]])
    return p[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="casita"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--route", default="tour", choices=list(ROUTES)); ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=15.0, help="camera log rate")
    ap.add_argument("--noise", default="realistic"); ap.add_argument("--rs1", action="store_true", help="also log RS1 top-down depth")
    ap.add_argument("--rs2-pitch", type=float, default=27.0, help="TRUE forward D435i pitch (twin believes 27)")
    ap.add_argument("--v", type=float, default=0.30); ap.add_argument("--w", type=float, default=0.8)
    ap.add_argument("--max-s", type=float, default=600.0)
    ap.add_argument("--light", action="store_true", help="depth only (no IR pair, no MuJoCo RGB): cameras come from the RTX renderer (tools/realsim, bench/slam/render_rtx.py)")
    a = ap.parse_args()
    mf = float(os.environ.get("KEVIN_REC_MIN_FREE_GB", "20"))
    if shutil.disk_usage(os.path.expanduser("~")).free / 2**30 < mf: sys.exit("record: <%g GB free, refusing" % mf)
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["KEVIN_EMU_NOISE"] = a.noise; os.environ["KEVIN_EMU_DOG"] = "0"
    os.environ["KEVIN_EMU_RS2_PITCH"] = str(a.rs2_pitch)
    import cv2, mujoco
    from emu import world as W, envs, realism
    class Sim(W.World):
        def _setup_io(self): pass          # no shm / sockets: fully isolated from the live emulator
    RP = W._robot_params()                 # resolved robot config: sim wheels = the geometry the log records
    ns = argparse.Namespace(env=a.env, seed=a.seed, tire=RP.V["wheels"], ir=1, subpix=0.08, holes=0.004, min_z=0.175, band=True, rt=1.0)
    S = Sim(ns); m, d = S.m, S.d; NP = W.NP
    sys.path.append(os.path.join(REPO, "src")); import twin_geometry as TG   # after the twin (its tire_profile wins, as in emu.world)
    for n in S.nodes: n.state = n.CLOSED; n.wd_en = False; n.ax.reset()
    e = S.env
    if a.route in START:                   # new start pose: teleport (keeps ride height vs the ground) and plan from there
        S.teleport(*START[a.route]); mujoco.mj_forward(m, d); e.start = START[a.route]
    # ---- route on the GT clearance map
    clr = e.clearance; need = 0.30
    cost = np.where(clr >= need, 1.0 + 0.6 / np.maximum(clr, 0.05), np.inf)
    def rc(x, y): return e.gt_world_to_rc(x, y)
    def xy(r, c): return (e.gt_origin[0] + (c + 0.5) * W.envs.GT_RES, e.gt_origin[1] + (r + 0.5) * W.envs.GT_RES)
    segs = []; cur = tuple(e.start[:2])
    for p in ROUTES[a.route]:
        if p.startswith("@"): segs.append(("spin", 2 * math.pi * (-1 if p == "@spinr" else 1))); continue
        tgt = e.places[p]; s0, g0 = tuple(int(v) for v in rc(*cur)), tuple(int(v) for v in rc(*tgt))
        # snap goal to nearest cell with enough clearance (places sit next to furniture)
        if not np.isfinite(cost[g0]):
            rr, cc = np.nonzero(np.isfinite(cost)); k = np.argmin((rr - g0[0]) ** 2 + (cc - g0[1]) ** 2); g0 = (int(rr[k]), int(cc[k]))
        if not np.isfinite(cost[s0]):
            rr, cc = np.nonzero(np.isfinite(cost)); k = np.argmin((rr - s0[0]) ** 2 + (cc - s0[1]) ** 2); s0 = (int(rr[k]), int(cc[k]))
        path = astar(cost, s0, g0)
        if path is None: print("record: no path to", p, flush=True); continue
        P = np.array([xy(r, c) for r, c in path[::4]] + [xy(*path[-1])]); segs.append(("path", P, p)); cur = tuple(P[-1])
    print("record: %d segments" % len(segs), flush=True)
    # ---- output
    out = os.path.abspath(os.path.expanduser(a.out)); os.makedirs(out, exist_ok=True)
    for sd in (("rs2_depth",) + (("rs1_depth",) if a.rs1 else ()) + (() if a.light else ("rs2_color", "rs2_ir1", "rs2_ir2", "mast"))): os.makedirs(os.path.join(out, sd), exist_ok=True)
    fgt = open(os.path.join(out, "gt.csv"), "w"); fgt.write("t,x,y,yaw,z,roll,pitch,v,w\n")
    fwh = open(os.path.join(out, "wheel.csv"), "w"); fwh.write("t,vl,vr\n")
    fim = open(os.path.join(out, "imu.csv"), "w"); fim.write("t,gx,gy,gz,ax,ay,az\n")
    fcm = open(os.path.join(out, "cmd.csv"), "w"); fcm.write("t,v,w\n")
    ffr = open(os.path.join(out, "frames.csv"), "w"); ffr.write("i,t\n")
    # ---- renderers + D4xx noise model (same formulas as emu/world.py _render_loop)
    DW, DH, CW, CH, MW, MH = W.DW, W.DH, W.CW, W.CH, W.MW, W.MH
    Rd = mujoco.Renderer(m, DH, DW); Rd.enable_depth_rendering(); R8 = mujoco.Renderer(m, DH, DW); Rc = mujoco.Renderer(m, CH, CW); Rm = mujoco.Renderer(m, MH, MW)
    rng = np.random.default_rng(a.seed + 7); fpx = W._intr(DW, DH, W.FOVY_D)["fx"]; cols = np.arange(DW, dtype=np.float32)[None, :]
    W3 = np.array([0.299 * 0.9, 0.587 * 0.9, 0.114 * 0.9], np.float32); dr = mujoco.MjData(m)
    def depth(cam):
        Rd.update_scene(dr, cam); z = Rd.render()
        zz = z + (z * z) * (ns.subpix / (fpx * W.BASELINE)) * rng.standard_normal(z.shape, dtype=np.float32)
        bad = (z < ns.min_z) | (z > 8.0) | (rng.random(z.shape, dtype=np.float32) < ns.holes) | (cols < fpx * W.BASELINE / np.maximum(z, 1e-3))
        xr = cols - (fpx * W.BASELINE) / np.maximum(z, 1e-3); smin = np.minimum.accumulate(xr[:, ::-1], axis=1)[:, ::-1]
        bad[:, :-1] |= smin[:, 1:] <= xr[:, :-1] - 0.5
        zz = zz * 1000.0; zz[bad] = 0; return zz.astype(np.uint16)
    def rgb(R, cam): R.update_scene(dr, cam); return R.render()
    wz_now = [0.0]
    def ir(cam):
        g = cv2.cvtColor(rgb(R8, cam), cv2.COLOR_RGB2GRAY).astype(np.float32) * np.float32(0.9)
        if realism.REAL:
            L = abs(wz_now[0]) * 0.004 * fpx
            if L >= 1.0: g = cv2.blur(g, (int(round(L)) | 1, 1))
        return np.clip(g + rng.normal(0, 2.0, (DH, DW)).astype(np.float32), 0, 255).astype(np.uint8)
    fx = {}
    if realism.REAL:      # same camera pipeline as emu.world's render loop
        Kc = [[W._intr(CW, CH, W.FOVY_C)["fx"], 0, (CW - 1) / 2], [0, W._intr(CW, CH, W.FOVY_C)["fy"], (CH - 1) / 2], [0, 0, 1]]; mt = W.MAST_TRUE
        fx["c"] = realism.CamFX(CW, CH, Kc, exposure_s=0.010, seed=2)
        fx["m"] = realism.CamFX(MW, MH, [[mt["fx"], 0, mt["cx"]], [0, mt["fy"], mt["cy"]], [0, 0, 1]], dist=(mt["k1"], mt["k2"]), render_K=W.MAST_RK, exposure_s=0.015, vignette=0.35, seed=3)
    def fxa(k, img): return fx[k](img, wz_now[0]) if k in fx else img
    # ---- meta
    def ext(name): t, R = TG.frame(name); return dict(t=t.tolist(), R=R.tolist())
    di, ci, mi = W._intr(DW, DH, W.FOVY_D), W._intr(CW, CH, W.FOVY_C), W._intr(MW, MH, W.MAST_FOVY)
    meta = dict(env=a.env, seed=a.seed, route=a.route, route_places=ROUTES[a.route], noise=a.noise, cam_hz=a.hz, start_world=list(e.start),
                drawn_errors=dict(wheel_r_err=(S.rerr - 1).tolist(), track_err=float(S.trerr - 1), gyro_bias=S.gbias.tolist(), gyro_scale_err=(S.gscale - 1).tolist()),
                intr=dict(rs2_depth=di, rs2_ir=di, rs2_color=ci, rs1_depth=di, ir_baseline_m=W.BASELINE),
                extr_twin_base_from_optical=dict(rs2_ir1=ext("rs2_ir1"), rs2_ir2=ext("rs2_ir2"), rs2_color=ext("rs2_color"), rs2_depth=ext("rs2_depth"), rs2_imu=ext("rs2_imu"), rs1_depth=ext("rs1_depth"), mast=ext("rgb")),
                withheld_truth=dict(mast=W.MAST_TRUE if realism.REAL else dict(hfov_deg=W.MAST_HFOV, intr=mi), rs2_mount=W.RS2_TRUE),
                robot_belief=dict(mast_hfov_deg=W.MAST_HFOV, rs2_pitch_deg=27.0), realism=realism.REAL,
                twin=dict(wheel_r_m=TG.WHEEL_R_M, track_m=TG.TRACK_M, axle_h_m=TG.AXLE_H_M), wheels=RP.V["wheels"], robot_config_hash=RP.HASH,
                notes="gt.csv yaw/x/y = base_link in world; mast pinhole, no distortion; IR depth-aligned to ir1; color offset +15 mm y")
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1); RP.save_with(out)
    # ---- control (GT pure pursuit; data generation only)
    r_w = TG.WHEEL_R_M; track = TG.TRACK_M
    def cmd(v, w):
        vl, vr = v - w * track / 2, v + w * track / 2
        S.nodes[0].ax.input_vel = -vl / (2 * math.pi * r_w); S.nodes[1].ax.input_vel = vr / (2 * math.pi * r_w)
    def pose():
        q = d.qpos; yaw = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2)); return float(q[0]), float(q[1]), yaw
    wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi
    dt = m.opt.timestep; n_imu = n_enc = n_cam = 0; fi = 0; si = 0; seg_state = {}; vc = wc = 0.0; t_settle = 1.5; done_t = None
    t_wall = time.time(); enc = [0.0, 0.0]
    while d.time < a.max_s:
        # --- controller at 50 Hz
        if int(d.time / dt) % max(1, int(0.02 / dt)) == 0:
            x, y, th = pose(); v_t = w_t = 0.0
            if d.time < t_settle or si >= len(segs): pass
            else:
                sg = segs[si]
                if sg[0] == "spin":
                    if "acc" not in seg_state: seg_state.update(acc=0.0, last=th)
                    seg_state["acc"] += wrap(th - seg_state["last"]); seg_state["last"] = th
                    rem = sg[1] - seg_state["acc"]
                    if abs(rem) < 0.03: si += 1; seg_state = {}; t_settle = d.time + 0.8
                    else: w_t = math.copysign(min(a.w, max(0.15, 1.5 * abs(rem))), rem)
                else:
                    P = sg[1]; k = seg_state.get("k", 0)
                    dd = np.hypot(P[:, 0] - x, P[:, 1] - y); k = max(k, int(np.argmin(dd[k:k + 40])) + k); seg_state["k"] = k
                    j = k
                    while j < len(P) - 1 and math.hypot(P[j, 0] - x, P[j, 1] - y) < 0.35: j += 1
                    gx, gy = P[j]; dist_end = math.hypot(P[-1, 0] - x, P[-1, 1] - y)
                    if dist_end < 0.06 or (k >= len(P) - 2 and dist_end < 0.12): si += 1; seg_state = {}; t_settle = d.time + 0.5
                    else:
                        err = wrap(math.atan2(gy - y, gx - x) - th)
                        if abs(err) > 0.7: w_t = math.copysign(min(a.w, 1.6 * abs(err)), err)
                        else:
                            v_t = min(a.v, 0.6 * dist_end + 0.05) * max(0.2, math.cos(err)); w_t = max(-a.w, min(a.w, 2.2 * err))
            # accel-limit the command (like the stack's ramp)
            vc += max(-0.02, min(0.02, v_t - vc)); wc += max(-0.06, min(0.06, w_t - wc)); cmd(vc, wc)
            if si >= len(segs) and done_t is None: done_t = d.time
            if done_t is not None and d.time > done_t + 2.0: break
            if int(d.time * 50) % 5 == 0: fcm.write("%.4f,%.4f,%.4f\n" % (d.time, vc, wc))
        S.step(time.monotonic())
        if d.time >= n_imu * 0.005:
            n_imu += 1
            if NP["g_rw"]: S.grw += S.rng.normal(0, NP["g_rw"] * math.sqrt(0.005), 3).astype(np.float32)
            g = (d.sensordata[S.gyro:S.gyro + 3].astype(np.float32) * S.gscale + S.gbias + S.grw + np.float32(NP["g_drift"] * d.time) + S.rng.normal(0, NP["g_white"], 3).astype(np.float32))
            ac = d.sensordata[S.acc:S.acc + 3].astype(np.float32) + S.rng.normal(0, NP["a_white"], 3).astype(np.float32)
            fim.write("%.4f,%.6f,%.6f,%.6f,%.5f,%.5f,%.5f\n" % (d.time + NP["imu_dt"], *g, *ac))
        if d.time >= n_enc * 0.01:
            n_enc += 1; S.contact()
            f = [nd.dir * nd.vel * S.rerr[k] for k, nd in enumerate(S.nodes[:2])]
            mm, dd_ = 0.5 * (f[0] + f[1]), 0.5 * (f[1] - f[0]) / S.trerr; f = [mm - dd_, mm + dd_]
            vl, vr = [(fk + S.rng.normal(0, 0.002)) * 2 * math.pi * r_w for fk in f]
            fwh.write("%.4f,%.5f,%.5f\n" % (d.time, vl, vr))
            q = d.qpos; x, y, th = pose(); Rb = d.xmat[S.base].reshape(3, 3); vb = Rb.T @ d.qvel[:3]
            roll = math.atan2(Rb[2, 1], Rb[2, 2]); pitch = -math.asin(max(-1, min(1, Rb[2, 0])))
            fgt.write("%.4f,%.5f,%.5f,%.6f,%.5f,%.6f,%.6f,%.4f,%.4f\n" % (d.time, x, y, th, q[2], roll, pitch, vb[0], d.qvel[5]))
        if d.time >= n_cam / a.hz:
            n_cam += 1
            dr.qpos[:] = d.qpos; dr.mocap_pos[:] = d.mocap_pos; dr.mocap_quat[:] = d.mocap_quat; mujoco.mj_kinematics(m, dr); mujoco.mj_camlight(m, dr)
            nm = "%06d" % fi
            cv2.imwrite(os.path.join(out, "rs2_depth", nm + ".png"), depth("rs2_ir1"))
            wz_now[0] = float(d.qvel[5])
            if a.rs1: cv2.imwrite(os.path.join(out, "rs1_depth", nm + ".png"), depth("rs1_depth"))
            if not a.light:
                cv2.imwrite(os.path.join(out, "rs2_color", nm + ".jpg"), cv2.cvtColor(fxa("c", rgb(Rc, "rs2_color")), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(out, "mast", nm + ".jpg"), cv2.cvtColor(fxa("m", rgb(Rm, "mast_rgb")), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(out, "rs2_ir1", nm + ".png"), ir("rs2_ir1")); cv2.imwrite(os.path.join(out, "rs2_ir2", nm + ".png"), ir("rs2_ir2"))
            ffr.write("%d,%.4f\n" % (fi, d.time)); fi += 1
            if fi % 150 == 0:
                x, y, th = pose(); print("record: t=%.1f frames=%d seg %d/%d pose %.2f %.2f %.0f collisions %d (%.1fx rt)" % (d.time, fi, si, len(segs), x, y, math.degrees(th), S.collisions, d.time / (time.time() - t_wall)), flush=True)
    for f_ in (fgt, fwh, fim, fcm, ffr): f_.close()
    meta.update(duration_s=round(float(d.time), 2), frames=fi, collisions=int(S.collisions), finished=si >= len(segs))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("record: done %s t=%.1f frames=%d collisions=%d finished=%s wall %.0f s" % (out, d.time, fi, S.collisions, si >= len(segs), time.time() - t_wall), flush=True)


if __name__ == "__main__":
    main()
