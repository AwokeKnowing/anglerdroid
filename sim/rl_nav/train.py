#!/usr/bin/env python3
"""PPO on RlNavVec. Map-free 5 Hz primitive + 64-byte memory."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_SIM = Path(__file__).resolve().parent.parent
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

from rl_nav.env import RlNavVec  # noqa: E402
from rl_nav.net import ActorCritic  # noqa: E402
from rl_nav.dash import EpisodeRecorder  # noqa: E402
from rl_nav.eval_fail import apply_fix, maybe_crash_spiral, probe  # noqa: E402
from rl_nav import kernels as K  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
CHART_S = 60.0
LIVE_S = 30.0
EVAL_EVERY = 100_000_000


def ppo(net, opt, buf, clip=0.2, entropy=0.015, epochs=3, minibatch=2048, gamma=0.99, lam=0.95):
    ego, vec, prim, write, old_logp, old_v, rew, done = buf
    t, n = rew.shape
    with torch.no_grad():
        adv = torch.zeros_like(rew)
        last = torch.zeros(n, device=rew.device)
        for i in range(t - 1, -1, -1):
            nxt = 0.0 if i == t - 1 else old_v[i + 1]
            mask = 1.0 - done[i]
            delta = rew[i] + gamma * nxt * mask - old_v[i]
            last = delta + gamma * lam * mask * last
            adv[i] = last
        ret = adv + old_v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ego_b = ego.reshape(t * n, *ego.shape[2:])
    vec_b = vec.reshape(t * n, vec.shape[-1])
    prim_b = prim.reshape(t * n)
    write_b = write.reshape(t * n, write.shape[-1])
    old_logp_b = old_logp.reshape(t * n)
    adv_b = adv.reshape(t * n)
    ret_b = ret.reshape(t * n)
    idx = torch.randperm(t * n, device=rew.device)
    mb = min(int(minibatch), t * n)
    ent_acc = 0.0
    loss_acc = 0.0
    n_mb = 0
    for _ in range(int(epochs)):
        for s in range(0, t * n, mb):
            sl = idx[s:s + mb]
            logp, ent, value = net.evaluate(ego_b[sl], vec_b[sl], prim_b[sl], write_b[sl])
            ratio = (logp - old_logp_b[sl]).exp()
            surr = torch.min(ratio * adv_b[sl], ratio.clamp(1.0 - clip, 1.0 + clip) * adv_b[sl])
            loss = -surr.mean() + 0.5 * (ret_b[sl] - value).pow(2).mean() - entropy * ent.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ent_acc += float(ent.mean().item())
            loss_acc += float(loss.item())
            n_mb += 1
    return ent_acc / max(1, n_mb), loss_acc / max(1, n_mb)


def maybe_advance(env, st):
    cover = st.get("cover", 0.0)
    crash = st["crash_frac"]
    need = (0.32, 0.45, 0.55, 0.65, 0.72)
    if env.stage >= 5:
        return
    if crash < 0.25 and cover >= need[env.stage]:
        env.set_curriculum(env.stage + 1)
        print("curriculum → stage %d movers=%d"
              % (env.stage, env.n_movers_hp),
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=512)
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-dash", action="store_true")
    ap.add_argument("--horizon", type=int, default=64)
    args = ap.parse_args()
    os.environ.setdefault("DISPLAY", ":1")
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    env = RlNavVec(n=args.n, seed=7, device=args.device)
    env.set_curriculum(0)
    eval_env = RlNavVec(n=32, seed=99, device=args.device)
    eval_env.set_curriculum(0)
    eval_env.crash_coef = env.crash_coef
    eval_env.stall_coef = env.stall_coef
    eval_env.yaw_coef = env.yaw_coef
    net = ActorCritic().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    ent_coef = 0.04
    last_eval = 0
    dash = None
    if not args.no_dash:
        try:
            from rl_nav.dash import RlDash
            dash = RlDash()
        except Exception as e:
            print("dash skip", e, flush=True)
    ego, vec = env.ego, env.vec
    horizon = int(args.horizon)
    buf_ego, buf_vec, buf_prim, buf_write, buf_logp, buf_v, buf_rew, buf_done = (
        [], [], [], [], [], [], [], []
    )
    t0 = time.time()
    end = t0 + args.hours * 3600.0 if args.hours > 0 else t0 + 365 * 86400
    step = 0
    n_upd = 0
    last_log = t0
    last_chart = 0.0
    last_live = 0.0
    sps_ema = 0.0
    last_ent = 0.0
    last_loss = 0.0
    last_st = {}
    ep_rec = EpisodeRecorder(OUT)
    env._env0_cb = lambda snap, done: ep_rec.on_tick(snap, done, step)
    ckpt = OUT / "ckpt.pt"
    if ckpt.is_file():
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        try:
            net.load_compatible(blob["net"])
            step = int(blob.get("step", 0))
            env.set_curriculum(int(blob.get("stage", 0)))
            eval_env.set_curriculum(env.stage)
            if "ent_coef" in blob:
                ent_coef = float(blob["ent_coef"])
            if "crash_coef" in blob:
                env.crash_coef = eval_env.crash_coef = float(blob["crash_coef"])
            if "stall_coef" in blob:
                env.stall_coef = eval_env.stall_coef = float(blob["stall_coef"])
            if "yaw_coef" in blob:
                env.yaw_coef = eval_env.yaw_coef = float(blob["yaw_coef"])
            last_eval = (step // EVAL_EVERY) * EVAL_EVERY
            print("loaded", ckpt, "step", step, "stage", env.stage, flush=True)
        except Exception as e:
            parked = OUT / "ckpt_ego_prev.pt"
            try:
                if not parked.is_file():
                    os.replace(ckpt, parked)
                    print("ckpt parked", parked, e, flush=True)
                else:
                    print("ckpt skip", e, flush=True)
            except OSError:
                print("ckpt skip", e, flush=True)
    print(
        "rl_nav train N=%d horizon=%d ep=%.0fs tour=%.0fs crash=%.2f stall=%.3f "
        "yaw=%.3f ent=%.3f eval/%dM"
        % (env.n, horizon, K.EP_S, K.T_TOUR_S, env.crash_coef, env.stall_coef,
           env.yaw_coef, ent_coef, EVAL_EVERY // 1_000_000),
        flush=True,
    )
    while time.time() < end:
        with torch.no_grad():
            prim, write, logp, value = net.act(ego, vec)
        nego, nvec, rew, done = env.step(prim, write)
        buf_ego.append(ego.clone())
        buf_vec.append(vec.clone())
        buf_prim.append(prim.clone())
        buf_write.append(write.clone())
        buf_logp.append(logp.clone())
        buf_v.append(value.clone())
        buf_rew.append(rew.clone())
        buf_done.append(done.float().clone())
        ego, vec = nego, nvec
        step += env.n
        if dash is not None and not dash.closed:
            dash.present(env, sps=sps_ema, note="upd=%d step=%d stg=%d" % (n_upd, step, env.stage))
        if len(buf_rew) >= horizon:
            t_s = time.time()
            buf = (
                torch.stack(buf_ego), torch.stack(buf_vec), torch.stack(buf_prim),
                torch.stack(buf_write), torch.stack(buf_logp), torch.stack(buf_v),
                torch.stack(buf_rew), torch.stack(buf_done),
            )
            ent, loss = ppo(net, opt, buf, entropy=ent_coef, gamma=0.999)
            last_ent, last_loss = ent, loss
            n_upd += 1
            buf_ego, buf_vec, buf_prim, buf_write, buf_logp, buf_v, buf_rew, buf_done = (
                [], [], [], [], [], [], [], []
            )
            dt = time.time() - t_s
            sps = horizon * env.n / max(dt, 1e-3)
            sps_ema = 0.9 * sps_ema + 0.1 * sps if sps_ema else sps
            if time.time() - last_log > 8.0:
                st = env.stats()
                last_st = st
                used = torch.cuda.memory_allocated() / (1024 ** 2)
                print(
                    "upd %d step=%d sps=%.0f ent=%.3f gpu=%.0fMiB hits=%.2f path=%.2f "
                    "crash=%.2f clear=%.2f liveh=%.2f px=%.2f stg=%d ep=%d "
                    "loss=%.3f ret=%.3f ceil=%.2f cover=%.3f"
                    % (n_upd, step, sps_ema, ent, used, st["mean_hits"], st["mean_path"],
                       st["crash_frac"], st["clear_frac"], st.get("live_hits", 0.0),
                       st["mean_plus_x"], st["stage"], st["n_ep"],
                       last_loss, st.get("mean_ret", 0.0), st.get("ceil", 1.0),
                       st.get("cover", 0.0)),
                    flush=True,
                )
                rec = {
                    "upd": n_upd,
                    "step": step,
                    "sps": sps_ema,
                    "ent": ent,
                    "hits": st["mean_hits"],
                    "path": st["mean_path"],
                    "liveh": st.get("live_hits", 0.0),
                    "crash": st["crash_frac"],
                    "clear": st["clear_frac"],
                    "stg": st["stage"],
                    "loss": last_loss,
                    "ret": st.get("mean_ret", 0.0),
                    "ceil": st.get("ceil", 1.0),
                    "ret_ceil": st.get("ret_ceil", 2.0),
                    "cover": st.get("cover", 0.0),
                    "ent_coef": ent_coef,
                    "crash_coef": env.crash_coef,
                    "stall_coef": env.stall_coef,
                    "yaw_coef": env.yaw_coef,
                }
                with (OUT / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                maybe_advance(env, st)
                last_log = time.time()
                if step - last_eval >= EVAL_EVERY:
                    eval_env.set_curriculum(env.stage)
                    eval_env.crash_coef = env.crash_coef
                    eval_env.stall_coef = env.stall_coef
                    eval_env.yaw_coef = env.yaw_coef
                    diag = probe(eval_env, net)
                    diag = maybe_crash_spiral(st, diag)
                    print(
                        "EVAL step=%d hits=%.2f disp=%.2f v=%.3f w=%.3f spin=%.2f "
                        "mode=%d frac=%.2f pent=%.3f ok=%s fails=%s"
                        % (
                            step, diag["hits_mean"], diag["disp_mean"], diag["v_mean"],
                            diag["w_abs"], diag["spin_frac"], diag["mode_prim"],
                            diag["mode_frac"], diag["prim_ent"], diag["ok"],
                            ",".join(diag["fails"]) or "none",
                        ),
                        flush=True,
                    )
                    if not diag["ok"]:
                        ent_coef, reset_opt = apply_fix(eval_env, env, net, diag, ent_coef)
                        if reset_opt:
                            opt = torch.optim.Adam(net.parameters(), lr=3e-4)
                            print("  fix: reset Adam", flush=True)
                    last_eval = step
                torch.save(
                    {"net": net.state_dict(), "step": step, "stage": env.stage, "n": env.n,
                     "ent_coef": ent_coef, "crash_coef": env.crash_coef,
                     "stall_coef": env.stall_coef, "yaw_coef": env.yaw_coef},
                    ckpt,
                )
                if last_log - last_chart >= CHART_S:
                    try:
                        from rl_nav.chart import write_canvas, write_clip_canvas
                        write_canvas()
                        write_clip_canvas()
                    except Exception as e:
                        print("chart skip", e, flush=True)
                    last_chart = last_log
            now = time.time()
            if now - last_live >= LIVE_S:
                last_live = now
                try:
                    from rl_nav.dash import compose_hud
                    import cv2
                    p = OUT / "last_frame.png"
                    if p.is_symlink():
                        p.unlink()
                    hud = compose_hud(
                        env.env_cpu(env._watch), st=dict(last_st),
                        note="live step=%d stg=%d" % (step, env.stage), sps=sps_ema,
                    )
                    cv2.imwrite(str(p), hud)
                except Exception as e:
                    print("live skip", e, flush=True)
    if dash is not None:
        dash.close()
    torch.save({"net": net.state_dict(), "step": step, "stage": env.stage, "n": env.n}, ckpt)
    print("rl_nav train stopped step", step, flush=True)


if __name__ == "__main__":
    main()
