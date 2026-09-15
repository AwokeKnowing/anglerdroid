#!/usr/bin/env python3
"""Dump one trained rollout and look for obs/reward/throttle design bugs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_SIM = Path(__file__).resolve().parent.parent
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

from rl_nav.env import RlNavVec  # noqa: E402
from rl_nav.net import ActorCritic  # noqa: E402
from rl_nav import kernels as K  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
GOAL_R = float(K.GOAL_R)


def _hist(ego):
    x = np.asarray(ego, dtype=np.float32).ravel()
    return {
        "unk": float((x < 0.15).mean()),
        "clear": float(((x >= 0.15) & (x < 0.50)).mean()),
        "goal": float(((x >= 0.50) & (x < 0.90)).mean()),
        "obs": float((x >= 0.90).mean()),
        "unique": [float(v) for v in np.unique(np.round(x, 3))],
    }


def _goal_body(snap):
    c, s = np.cos(snap["yaw"]), np.sin(snap["yaw"])
    alive = np.where(np.asarray(snap["galive"]) > 0)[0]
    if len(alive) == 0:
        return None
    i = int(alive[0])
    dx = float(snap["gx"][i] - snap["x"])
    dy = float(snap["gy"][i] - snap["y"])
    fx = c * dx + s * dy
    fy = -s * dx + c * dy
    dist = float(np.hypot(dx, dy))
    bear = float(np.arctan2(fy, fx))
    in_rs1 = (K.RS1_X0 <= fx <= K.RS1_X1) and (K.RS1_Y0 <= fy <= K.RS1_Y1)
    ang = abs(bear)
    in_rs2 = (dist <= K.RS2_RANGE) and (ang <= K.RS2_HALF)
    return {
        "gx": float(snap["gx"][i]),
        "gy": float(snap["gy"][i]),
        "fx": fx,
        "fy": fy,
        "dist": dist,
        "bear_deg": float(np.degrees(bear)),
        "in_fov": bool(in_rs1 or in_rs2),
        "in_rs1": bool(in_rs1),
        "in_rs2": bool(in_rs2),
    }


def _front_plus_x_furniture(snap):
    """Distance from nose to nearest box along heading, CPU copy of the kernel."""
    x, y, yaw = snap["x"], snap["y"], snap["yaw"]
    c, s = np.cos(yaw), np.sin(yaw)
    nx, ny = x + c * K.NOSE, y + s * K.NOSE
    best = 9e9
    for b in snap["boxes"]:
        hx = float(b[3])
        if hx <= 0:
            continue
        cx, cy, hy = float(b[0]), float(b[1]), float(b[4])
        dx, dy = nx - cx, ny - cy
        # axis-aligned (house boxes have yaw 0)
        if abs(dx) <= hx + 0.01 and abs(dy) <= hy + 0.01:
            return 0.0
        # slab on AABB
        tmin, tmax = -1e9, 1e9
        for orig, vel, h in ((dx, c, hx + 0.01), (dy, s, hy + 0.01)):
            if abs(vel) < 1e-12:
                if abs(orig) > h:
                    tmin = 1.0
                    tmax = 0.0
                    break
                continue
            t1, t2 = (-h - orig) / vel, (h - orig) / vel
            tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
        if tmax < tmin or tmax < 0:
            continue
        hit = 0.0 if tmin < 0 else tmin
        best = min(best, hit)
    return float(min(best, 8.0))


@torch.no_grad()
def main():
    device = torch.device("cuda:0")
    env = RlNavVec(n=64, seed=11, device="cuda:0")
    env.set_curriculum(0)
    net = ActorCritic().to(device)
    ckpt = OUT / "ckpt.pt"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    net.load_state_dict(blob["net"])
    net.eval()
    print("loaded", ckpt, "step", blob.get("step"), "stage", blob.get("stage"), flush=True)

    env.reset_all()
    rows = []
    # detailed env0
    e0 = []
    for t in range(K.EP_STEPS + 2):
        ego, vec = env.ego, env.vec
        prim, write, logp, value = net.act(ego, vec, deterministic=True)
        # ablations on the current batch (first 16)
        sl = slice(0, 16)
        p_real = net.act(ego[sl], vec[sl], deterministic=True)[0]
        p_zero = net.act(torch.zeros_like(ego[sl]), vec[sl], deterministic=True)[0]
        ego_obs = ego[sl].clone()
        ego_obs[ego_obs >= 0.90] = 0.33
        p_noobs = net.act(ego_obs, vec[sl], deterministic=True)[0]
        ego_g2o = ego[sl].clone()
        m = (ego_g2o >= 0.50) & (ego_g2o < 0.90)
        ego_g2o[m] = 1.0
        p_g2o = net.act(ego_g2o, vec[sl], deterministic=True)[0]
        ego_hide = ego[sl].clone()
        ego_hide[m] = 0.33
        p_hide = net.act(ego_hide, vec[sl], deterministic=True)[0]

        snap = env.env0_cpu()
        g = _goal_body(snap)
        h = _hist(snap["ego"])
        rec = {
            "t": t,
            "x": snap["x"],
            "y": snap["y"],
            "yaw": snap["yaw"],
            "v": float(env.v[0]),
            "w": float(env.w[0]),
            "plus_x": snap["plus_x"],
            "hits": snap["hits"],
            "cut": snap["cut"],
            "prim": int(prim[0]),
            "value": float(value[0]),
            "ego": h,
            "goal": g,
            "throt": snap["plus_x"] < K.THROT_M,
            "hardstop": snap["plus_x"] < K.STOP_M,
            "agree_zero": float((p_real == p_zero).float().mean()),
            "agree_noobs": float((p_real == p_noobs).float().mean()),
            "agree_g2o": float((p_real == p_g2o).float().mean()),
            "agree_hide": float((p_real == p_hide).float().mean()),
        }
        e0.append(rec)

        nego, nvec, rew, done = env.step(prim, write)
        rec["rew"] = float(rew[0])
        rec["done"] = int(done[0])
        if int(done[0]) and t > 0:
            rec["cut_end"] = int(env.cut[0])  # already reset; use previous
            break

    # full-batch episode stats (reset and run to timeout)
    env.reset_all()
    n = env.n
    min_d = np.full(n, 99.0)
    min_d_px = np.full(n, 99.0)
    min_d_v = np.zeros(n)
    saw_goal_px = np.zeros(n, dtype=np.int32)
    max_goal_px = np.zeros(n)
    first_fov = np.full(n, -1)
    hit = np.zeros(n, dtype=np.int32)
    cut = np.zeros(n, dtype=np.int32)
    path = np.zeros(n)
    start_d = np.zeros(n)
    start_fov = np.zeros(n, dtype=np.int32)
    start_gpx = np.zeros(n)
    agree_z = []
    agree_hide = []
    agree_g2o = []

    gal0 = env.galive.detach().cpu().numpy()
    gx0 = env.gx.detach().cpu().numpy()
    gy0 = env.gy.detach().cpu().numpy()
    x0 = env.x.detach().cpu().numpy()
    y0 = env.y.detach().cpu().numpy()
    yaw0 = env.yaw.detach().cpu().numpy()
    ego0 = env.ego.detach().cpu().numpy()
    for i in range(n):
        sx, sy = float(x0[i]), float(y0[i])
        alive = np.where(gal0[i] > 0)[0]
        if len(alive) == 0:
            start_d[i] = -1
            continue
        gix = int(alive[0])
        dx = float(gx0[i, gix] - sx)
        dy = float(gy0[i, gix] - sy)
        start_d[i] = float(np.hypot(dx, dy))
        yaw = float(yaw0[i])
        c, s = np.cos(yaw), np.sin(yaw)
        fx, fy = c * dx + s * dy, -s * dx + c * dy
        dist = start_d[i]
        in_rs1 = (K.RS1_X0 <= fx <= K.RS1_X1) and (K.RS1_Y0 <= fy <= K.RS1_Y1)
        in_rs2 = (dist <= K.RS2_RANGE) and (abs(np.arctan2(fy, fx)) <= K.RS2_HALF)
        start_fov[i] = int(in_rs1 or in_rs2)
        start_gpx[i] = float(((ego0[i] >= 0.50) & (ego0[i] < 0.90)).mean())

    for t in range(K.EP_STEPS):
        prim, write, _, _ = net.act(env.ego, env.vec, deterministic=True)
        sl = slice(0, 32)
        p_real = prim[sl]
        p_zero = net.act(torch.zeros_like(env.ego[sl]), env.vec[sl], deterministic=True)[0]
        hide = env.ego[sl].clone()
        mg = (hide >= 0.50) & (hide < 0.90)
        hide[mg] = 0.33
        p_hide = net.act(hide, env.vec[sl], deterministic=True)[0]
        g2o = env.ego[sl].clone()
        g2o[mg] = 1.0
        p_g2o = net.act(g2o, env.vec[sl], deterministic=True)[0]
        agree_z.append(float((p_real == p_zero).float().mean()))
        agree_hide.append(float((p_real == p_hide).float().mean()))
        agree_g2o.append(float((p_real == p_g2o).float().mean()))

        gpx = ((env.ego >= 0.50) & (env.ego < 0.90)).float().mean(dim=(1, 2)).cpu().numpy()
        max_goal_px = np.maximum(max_goal_px, gpx)
        saw_goal_px |= (gpx > 0).astype(np.int32)
        env.step(prim, write)

    # redo batch with pre-reset capture
    env.reset_all()
    hit[:] = 0
    cut[:] = 0
    path[:] = 0
    min_d[:] = 99
    ep_done = np.zeros(n, dtype=np.int32)
    near_nohit = 0
    closest_rows = []
    for t in range(K.EP_STEPS):
        prim, write, _, _ = net.act(env.ego, env.vec, deterministic=True)
        xs = env.x.detach().cpu().numpy()
        ys = env.y.detach().cpu().numpy()
        gxs = env.gx.detach().cpu().numpy()
        gys = env.gy.detach().cpu().numpy()
        gal = env.galive.detach().cpu().numpy()
        px = env.plus_x.detach().cpu().numpy()
        vv = env.v.detach().cpu().numpy()
        ego_np = env.ego.detach().cpu().numpy()
        for i in range(n):
            if ep_done[i]:
                continue
            alive = np.where(gal[i] > 0)[0]
            if len(alive) == 0:
                continue
            d = float(np.hypot(gxs[i, alive[0]] - xs[i], gys[i, alive[0]] - ys[i]))
            if d < min_d[i]:
                min_d[i] = d
                min_d_px[i] = float(px[i])
                min_d_v[i] = float(vv[i])
                closest_rows.append({
                    "i": i, "t": t, "d": d, "plus_x": float(px[i]), "v": float(vv[i]),
                    "gpx": float(((ego_np[i] >= 0.50) & (ego_np[i] < 0.90)).mean()),
                    "opx": float((ego_np[i] >= 0.90).mean()),
                })
        nego, nvec, rew, done = env.step(prim, write)
        rew_np = rew.cpu().numpy()
        done_np = done.cpu().numpy()
        cut_np = env.cut.detach().cpu().numpy()
        hits_now = env.hits.detach().cpu().numpy()
        # After reset, hits/cut on dead envs are 0. Infer from reward before we lost it.
        for i in range(n):
            if ep_done[i]:
                continue
            if rew_np[i] >= 1.5 or (rew_np[i] >= 0.8 and done_np[i]):
                hit[i] = 1
            if done_np[i]:
                # cut already reset. use reward sign
                if rew_np[i] <= -1.0:
                    cut[i] = 2
                elif hit[i]:
                    cut[i] = 1
                else:
                    cut[i] = 3
                path[i] = float(env.path_m[i])  # reset too...
                ep_done[i] = 1
        if ep_done.all():
            break

    print("\n=== START OF EPISODE (64 envs) ===")
    print("start_dist mean/min/max", start_d[start_d >= 0].mean(), start_d[start_d >= 0].min(), start_d[start_d >= 0].max())
    print("start in geometric FOV", start_fov.mean(), "ego goal-pixel frac mean", start_gpx.mean(), "any goal px", (start_gpx > 0).mean())
    print("goals placed at reset", float((gal0.sum(axis=1) > 0).mean()), "saw goal pixels sometime", float(saw_goal_px.mean()), "max gpx mean", float(max_goal_px.mean()))

    print("\n=== TRAINED DETERMINISTIC (64 envs, 1 episode) ===")
    print("collect_frac", hit.mean(), "crash_frac", (cut == 2).mean(), "timeout_frac", (cut == 3).mean())
    print("min_dist mean/median/min", min_d.mean(), np.median(min_d), min_d.min())
    print("reached <0.35", (min_d <= GOAL_R).mean(), "<0.50", (min_d <= 0.50).mean(), "<0.80", (min_d <= 0.80).mean())
    close = min_d <= 0.80
    if close.any():
        print("when min_d<=0.80: plus_x", min_d_px[close].mean(), "v", min_d_v[close].mean())
        print("  plus_x<THROT", (min_d_px[close] < K.THROT_M).mean(), "plus_x<STOP", (min_d_px[close] < K.STOP_M).mean())
        print("  collected among those", hit[close].mean())
    near_miss = (min_d <= 0.80) & (hit == 0)
    print("near-miss (d<=0.80, no collect)", near_miss.mean(), "n", int(near_miss.sum()))
    if near_miss.any():
        print("  near-miss min_d", min_d[near_miss][:8], "plus_x", min_d_px[near_miss][:8], "v", min_d_v[near_miss][:8])

    print("\n=== ABLATION (action agreement, 1=policy ignores that change) ===")
    print("ego=0 vs real", float(np.mean(agree_z)))
    print("hide goal 0.66→0.33 vs real", float(np.mean(agree_hide)))
    print("goal 0.66→1.0 (as obstacle) vs real", float(np.mean(agree_g2o)))

    print("\n=== ENV0 STEP TRACE ===")
    print("t  dist  bear  fov  gpx   opx  plus_x  v     prim  rew  throt stop")
    for r in e0:
        g = r["goal"]
        ds = "%5.2f" % g["dist"] if g else "  -- "
        br = "%5.0f" % g["bear_deg"] if g else "   --"
        fv = int(g["in_fov"]) if g else 0
        print(
            "%3d %s %s  %d  %5.3f %5.3f  %5.2f %6.3f  %2d %+5.2f  %d %d"
            % (
                r["t"], ds, br, fv, r["ego"]["goal"], r["ego"]["obs"],
                r["plus_x"], r["v"], r["prim"], r.get("rew", 0.0),
                int(r["throt"]), int(r["hardstop"]),
            )
        )

    print("\n=== ENV0 EGO UNIQUE AT t=0 and closest ===")
    print("t0 unique", e0[0]["ego"]["unique"], "hist", {k: e0[0]["ego"][k] for k in ("unk", "clear", "goal", "obs")})
    if e0[0]["goal"]:
        print("t0 goal body", e0[0]["goal"])
    dists = [(r["goal"]["dist"] if r["goal"] else 99, r) for r in e0]
    dists.sort(key=lambda z: z[0])
    print("closest env0", dists[0][0], "plus_x", dists[0][1]["plus_x"], "v", dists[0][1]["v"], "gpx", dists[0][1]["ego"]["goal"])
    print("env0 final hits/cut/done", e0[-1]["hits"], e0[-1]["cut"], e0[-1].get("done"), "rews", [r.get("rew") for r in e0 if r.get("rew")])

    print("\n=== ENV0 ABLATION OVER TIME (first 8) ===")
    for r in e0[:8]:
        print("t", r["t"], "agree_zero", "%.2f" % r["agree_zero"], "hide", "%.2f" % r["agree_hide"], "g2o", "%.2f" % r["agree_g2o"])

    dump = OUT / "probe_env0.json"
    slim = []
    for r in e0:
        g = r["goal"] or {}
        slim.append({
            "t": r["t"], "dist": g.get("dist"), "bear": g.get("bear_deg"),
            "fov": g.get("in_fov"), "gpx": r["ego"]["goal"], "opx": r["ego"]["obs"],
            "plus_x": r["plus_x"], "v": r["v"], "prim": r["prim"],
            "rew": r.get("rew"), "hits": r["hits"],
        })
    dump.write_text(json.dumps({"step": int(blob.get("step", 0)), "env0": slim}, indent=2))
    print("wrote", dump)


if __name__ == "__main__":
    main()
