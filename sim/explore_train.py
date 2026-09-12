#!/usr/bin/env python3
"""24h PPO+RND insect explore on the 3090. Reloads hparams when the evaluator writes them."""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

_SIM = Path(__file__).resolve().parent
_REPO = _SIM.parent
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

from explore_env import InsectEnv
from explore_net import ActorCritic, RND, EpisodicMem, obs_to_torch
from explore_dash import ExploreDash

OUT = _SIM / "explore"
HP_PATH = OUT / "hparams.json"
PLAN_PATH = OUT / "PLAN.txt"
CKPT = OUT / "ckpt.pt"
DEADLINE = OUT / "deadline.txt"


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def load_hp(prev=None):
    d = {
        "lr": 3e-4, "rnd_lr": 1e-4, "entropy": 0.010, "gamma": 0.99,
        "gae_lambda": 0.95, "clip": 0.2, "horizon": 1024, "epochs": 4,
        "minibatch": 256, "ep_steps": 300, "dt": 0.10, "max_v": 0.28,
        "max_w": 0.80, "novelty_coef": 0.10, "collision_coef": 5.5,
        "replan_s": 0.20, "score_s": 5.0, "score_dt": 0.20, "n_paths": 64,
        "a_v": 0.70, "a_w": 2.40, "v_rev": -0.10,
        "prim_progress": 0.55, "prim_stuck": 2.60, "prim_step": 0.45,
        "forward_coef": 0.0, "audio_coef": 0.20, "still_coef": 0.0,
        "door_open_p": 0.55, "plus_x_stop": 0.28, "clutter": "random",
        "depth_noise": 0.0, "rebuild_every": 25, "device": "cuda:0",
        "dash_hz": 12.0,
        "smooth_coef": 0.02, "smooth_max": 0.20,
        "smooth_focus_cov": 12.0, "smooth_full_cov": 28.0,
        "action_ema": 0.0, "action_ema_max": 0.15,
        "progress_coef": 0.0, "circle_coef": 0.0, "circle_dampen": 8.0,
        "epi_mem": 0.40, "pi_window_s": 8.0,
        "curve_coef": 0.0, "yaw_coef": 0.0, "yaw_lpf": 0.18,
        "spin_coef": 0.0, "plant": 0.0,
        "reorient_m": 2.0, "cover_coef": 1.4,
        "turn_coef": 0.35, "slow_coef": 0.25, "stale_coef": 0.12,
        "stuck_coef": 0.70, "commit_coef": 0.20, "jerk_coef": 0.04,
        "look_coef": 0.55, "look_m": 1.50, "ang_coef": 0.10,
        "visit_coef": 1.20, "visit_fwd_coef": 0.28,
        "visit_hit_coef": 1.20, "visit_hot": 8.0,
        "cost_cut": 80.0, "cost_cut_min_s": 6.0, "coll_done": 22,
        "yaw_only_max": 40, "rand_house": 1.0, "yaw_decay": 0.99,
        "yaw_mu_max": 0.25,
        "curve_mode": 1.0, "curve_s": 10.0, "curve_r_min": 0.60,
        "curve_r_max": 4.00, "curve_lat": 1.10, "curve_bead_r": 0.32,
        "curve_look": 0.45, "curve_win": 1.0, "curve_lose": 1.0,
        "curve_bead": 0.08, "curve_clear": 0.42, "curve_wall_coef": 0.55,
        "curve_hit_coef": 0.80, "curve_hit_m": 0.14,
        "curve_cont_s": 3.0, "curve_cont_coef": 0.90,
    }
    if HP_PATH.is_file():
        try:
            d.update(json.loads(HP_PATH.read_text()))
        except Exception as e:
            print("hparams read skip:", e, flush=True)
    if prev:
        # keep device
        d["device"] = prev.get("device", d["device"])
    return d


def heartbeat(payload: dict):
    lines = ["%s=%s" % (k, payload[k]) for k in payload]
    _write(OUT / "heartbeat.txt", "\n".join(lines) + "\n")
    _write(OUT / "STATUS.txt", "\n".join(lines) + "\n")


def append_ledger(row: dict):
    with open(OUT / "ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def write_metrics(rows):
    """Rolling 25-ep smoothness + coverage. First coefs are guesses; retune from these."""
    if not rows:
        return {}
    def mean(k):
        vals = [float(r.get(k) or 0) for r in rows]
        return float(np.mean(vals)) if vals else 0.0
    n = len(rows)
    cuts = {}
    for r in rows:
        c = str(r.get("cut") or "time")
        cuts[c] = cuts.get(c, 0) + 1
    out = {
        "n": n,
        "mean_cov_cells": round(mean("cov_cells"), 2),
        "mean_n_rooms": round(mean("n_rooms"), 2),
        "mean_path_m": round(mean("path_m"), 2),
        "mean_cover_per_m": round(mean("cover_per_m"), 3),
        "mean_wide": round(mean("wide_score"), 3),
        "mean_jerk": round(mean("jerk"), 4),
        "mean_dw": round(mean("dw_mean"), 4),
        "mean_open_turn": round(mean("open_turn"), 4),
        "mean_open_speed": round(mean("open_speed"), 4),
        "mean_heading_err": round(mean("heading_err"), 3),
        "mean_smooth": round(mean("smooth_score"), 3),
        "mean_cost": round(mean("cost"), 2),
        "mean_coll": round(mean("collisions"), 2),
        "mean_chord": round(mean("mean_chord"), 3),
        "mean_beads": round(mean("mean_beads"), 3),
        "win_frac": round(mean("win_frac"), 3),
        "mean_curve_cont": round(mean("curve_cont"), 3),
        "mean_prim_ms": round(mean("prim_ms"), 2),
        "mean_prim_hot": round(mean("prim_hot"), 3),
        "cut_frac": {k: round(v / n, 3) for k, v in cuts.items()},
        "time_frac": round(cuts.get("time", 0) / n, 3),
        "cost_cut_frac": round(cuts.get("cost", 0) / n, 3),
        "crash_frac": round(cuts.get("crash", 0) / n, 3),
    }
    _write(OUT / "metrics.json", json.dumps(out, indent=2) + "\n")
    return out


def load_deadline():
    if DEADLINE.is_file():
        try:
            return float(DEADLINE.read_text().strip())
        except Exception:
            pass
    end = time.time() + 20.0 * 3600.0
    _write(DEADLINE, "%.3f\n" % end)
    return end


def gae(rews, vals, dones, gamma, lam):
    adv = np.zeros_like(rews)
    last = 0.0
    for t in range(len(rews) - 1, -1, -1):
        nxt = 0.0 if t == len(rews) - 1 else vals[t + 1]
        mask = 1.0 - float(dones[t])
        delta = rews[t] + gamma * nxt * mask - vals[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    ret = adv + vals
    return adv, ret


class Rollout:
    def __init__(self):
        self.img, self.vec, self.act = [], [], []
        self.logp, self.val, self.rew, self.done = [], [], [], []

    def add(self, img, vec, act, logp, val, rew, done):
        self.img.append(img.squeeze(0).cpu())
        self.vec.append(vec.squeeze(0).cpu())
        self.act.append(act.squeeze(0).cpu())
        self.logp.append(float(logp))
        self.val.append(float(val))
        self.rew.append(float(rew))
        self.done.append(bool(done))

    def __len__(self):
        return len(self.rew)


def ppo_update(net, opt, rnd, rnd_opt, buf: Rollout, hp, device):
    img = torch.stack(buf.img).to(device)
    vec = torch.stack(buf.vec).to(device)
    act = torch.stack(buf.act).to(device)
    old_logp = torch.tensor(buf.logp, device=device, dtype=torch.float32)
    vals = np.asarray(buf.val, dtype=np.float32)
    rews = np.asarray(buf.rew, dtype=np.float32)
    dones = np.asarray(buf.done, dtype=np.float32)
    adv, ret = gae(rews, vals, dones, float(hp["gamma"]), float(hp["gae_lambda"]))
    adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
    ret_t = torch.tensor(ret, device=device, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    n = len(buf)
    mb = min(int(hp["minibatch"]), n)
    idx = np.arange(n)
    last_ent = 0.0
    last_pi = 0.0
    last_rnd = 0.0
    for _ in range(int(hp["epochs"])):
        np.random.shuffle(idx)
        for s in range(0, n, mb):
            sl = idx[s:s + mb]
            slt = torch.tensor(sl, device=device, dtype=torch.long)
            logp, ent, value, _ = net.evaluate(img[slt], vec[slt], act[slt], hx=None)
            ratio = (logp - old_logp[slt]).exp()
            clip = float(hp["clip"])
            surr = torch.min(ratio * adv_t[slt], ratio.clamp(1.0 - clip, 1.0 + clip) * adv_t[slt])
            pi_loss = -surr.mean()
            v_loss = 0.5 * (value - ret_t[slt]).pow(2).mean()
            ent_loss = -float(hp["entropy"]) * ent.mean()
            loss = pi_loss + v_loss + ent_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            rnd_err = rnd.novelty(img[slt])
            rnd_loss = rnd_err.mean()
            rnd_opt.zero_grad(set_to_none=True)
            rnd_loss.backward()
            rnd_opt.step()
            last_ent = float(ent.mean().item())
            last_pi = float(pi_loss.item())
            last_rnd = float(rnd_loss.item())
    _decay_yaw_mean(net, opt, float(hp.get("yaw_decay", 1.0)))
    return {"ent": last_ent, "pi": last_pi, "rnd": last_rnd, "ret": float(ret.mean())}


def _decay_yaw_mean(net, opt, decay):
    """Shrink yaw mean and drop Adam momentum on that row.

    Multiplying weights by 0.99 is a no-op if Adam's exp_avg still points at
    a pirouette — the next step restores it. Zero those moments too.
    """
    if decay >= 0.9999:
        return
    if getattr(net, "act_dim", 2) != 2 or net.pi_mu.weight.shape[0] < 2:
        return
    with torch.no_grad():
        net.pi_mu.weight.data[1].mul_(decay)
        if net.pi_mu.bias is not None:
            net.pi_mu.bias.data[1].mul_(decay)
    for p, idx in ((net.pi_mu.weight, 1), (net.pi_mu.bias, 1)):
        if p is None:
            continue
        st = opt.state.get(p)
        if not st:
            continue
        for k in ("exp_avg", "exp_avg_sq"):
            t = st.get(k)
            if t is not None and t.shape[0] > idx:
                t[idx].zero_()


def save_ckpt(net, rnd, opt, rnd_opt, step, extra=None):
    prev = OUT / "ckpt.prev.pt"
    if CKPT.is_file():
        try:
            import shutil
            shutil.copy2(CKPT, prev)
        except Exception:
            pass
    torch.save({
        "net": net.state_dict(),
        "rnd": rnd.state_dict(),
        "opt": opt.state_dict(),
        "rnd_opt": rnd_opt.state_dict(),
        "step": step,
        "extra": extra or {},
    }, CKPT)


def _load_matching(module, state):
    cur = module.state_dict()
    ok = {}
    skipped = []
    for k, v in state.items():
        if k in cur and tuple(cur[k].shape) == tuple(v.shape):
            ok[k] = v
        else:
            skipped.append(k)
    module.load_state_dict(ok, strict=False)
    return skipped


def maybe_load(net, rnd, opt, rnd_opt, device, hp=None):
    hp = hp or {}
    if not CKPT.is_file():
        return 0, opt, rnd_opt
    try:
        blob = torch.load(CKPT, map_location=device, weights_only=False)
        skipped = _load_matching(net, blob["net"])
        _load_matching(rnd, blob["rnd"])
        if skipped:
            print("ckpt skipped", skipped, flush=True)
        opt_ok = False
        if not skipped:
            try:
                opt.load_state_dict(blob["opt"])
                rnd_opt.load_state_dict(blob["rnd_opt"])
                opt_ok = True
                for p in opt.param_groups[0]["params"]:
                    st = opt.state.get(p)
                    if not st:
                        continue
                    for k in ("exp_avg", "exp_avg_sq"):
                        t = st.get(k)
                        if t is not None and tuple(t.shape) != tuple(p.shape):
                            opt_ok = False
                            break
                    if not opt_ok:
                        break
            except Exception as e:
                print("opt skip:", type(e).__name__, e, flush=True)
                opt_ok = False
        if not opt_ok:
            print("opt skip: fresh Adam (shape mismatch)", flush=True)
            opt = torch.optim.Adam(net.parameters(), lr=float(hp.get("lr", 3e-4)))
            rnd_opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, rnd.parameters()),
                lr=float(hp.get("rnd_lr", 1e-4)),
            )
        print("loaded ckpt step", blob.get("step"), flush=True)
        return int(blob.get("step") or 0), opt, rnd_opt
    except Exception as e:
        print("ckpt skip:", type(e).__name__, e, flush=True)
        return 0, opt, rnd_opt


def detox_yaw_head(net, hp):
    """Zero a spin-addicted yaw mean. Fresh Adam so momentum cannot revive it."""
    if getattr(net, "act_dim", 2) != 2:
        return torch.optim.Adam(net.parameters(), lr=float(hp["lr"]))
    with torch.no_grad():
        net.pi_mu.weight.data[1].zero_()
        if net.pi_mu.bias is not None:
            net.pi_mu.bias.data[1] = 0.0
        net.pi_logstd.data[1] = -1.4
    opt = torch.optim.Adam(net.parameters(), lr=float(hp["lr"]))
    print("detox: zeroed actor yaw head + new Adam", flush=True)
    return opt


def gpu_mem():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    used = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    return used, reserved


def set_smooth_gate(hp, env, cov_hist):
    """Whisper-smooth early; ramp action-rate penalty once coverage is real.

    Same schedule as Isaac-style ||a_t-a_{t-1}||^2 costs: keep them tiny until
    the task reward (here novelty/coverage) is online, then let smoothness
    matter. Evaluator can still override smooth_coef / smooth_max.
    """
    lo = float(hp.get("smooth_focus_cov", 12.0))
    hi = float(hp.get("smooth_full_cov", 28.0))
    mean_cov = float(np.mean(cov_hist)) if cov_hist else 0.0
    if hi <= lo:
        gate = 1.0 if mean_cov >= lo else 0.0
    else:
        gate = float(np.clip((mean_cov - lo) / (hi - lo), 0.0, 1.0))
    s0 = float(hp.get("smooth_coef", 0.02))
    s1 = float(hp.get("smooth_max", 0.35))
    emax = float(hp.get("action_ema_max", 0.35))
    hp["smooth_now"] = s0 + gate * (s1 - s0)
    hp["action_ema_now"] = float(hp.get("action_ema", 0.0)) + gate * emax
    hp["smooth_gate"] = gate
    env.hp = hp
    return gate, mean_cov


def main():
    os.environ.setdefault("DISPLAY", ":1")
    OUT.mkdir(parents=True, exist_ok=True)
    end = load_deadline()
    _write(OUT / "train.pid", str(os.getpid()) + "\n")
    hp = load_hp()
    HP_PATH.write_text(json.dumps(hp, indent=2) + "\n")
    hp_mtime = HP_PATH.stat().st_mtime if HP_PATH.is_file() else 0.0
    device = torch.device(hp.get("device", "cuda:0"))
    print("explore_train pid=%d device=%s deadline_h=%.2f" % (
        os.getpid(), device, max(0.0, (end - time.time()) / 3600.0),
    ), flush=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("cuda", torch.cuda.get_device_name(device), flush=True)

    env = InsectEnv(hp, seed=int(time.time()) % 99991)
    dash = None
    try:
        dash = ExploreDash(env.wall_all, clutter=env.furniture, people=env.people.folk)
    except Exception as e:
        print("dash skip:", type(e).__name__, e, flush=True)

    net = ActorCritic(vec_dim=10, act_dim=4 if float(hp.get("curve_mode", 1.0)) >= 0.5 else 2).to(device)
    net.yaw_mu_max = float(hp.get("yaw_mu_max", 1.0))
    rnd = RND().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(hp["lr"]))
    rnd_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, rnd.parameters()), lr=float(hp["rnd_lr"]))
    global_step, opt, rnd_opt = maybe_load(net, rnd, opt, rnd_opt, device, hp)
    if (OUT / "DETOX_YAW").is_file():
        try:
            (OUT / "DETOX_YAW").unlink()
        except Exception:
            pass
        opt = detox_yaw_head(net, hp)
    hx = None
    obs, info = env.reset()
    buf = Rollout()
    held_act = held_logp = held_val = None
    t_wall0 = time.time()
    sim_acc = 0.0
    ep_ret = 0.0
    ep_nov = 0.0
    n_upd = 0
    last_ckpt = time.time()
    last_snap = 0.0
    cov_hist = deque(maxlen=20)
    ep_hist = deque(maxlen=25)
    set_smooth_gate(hp, env, cov_hist)
    last_eval_note = ""
    if (OUT / "last_eval.txt").is_file():
        last_eval_note = (OUT / "last_eval.txt").read_text().strip().splitlines()[0][:120]
    rnd_mean = 0.02
    rnd_var = 1e-6
    epi_mem = EpisodicMem(192)
    stop = {"flag": False}

    def _sig(_s, _f):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    while time.time() < end and not stop["flag"]:
        if (OUT / "RESTART").is_file():
            try:
                (OUT / "RESTART").unlink()
            except Exception:
                pass
            print("RESTART requested", flush=True)
            break
        if HP_PATH.is_file() and HP_PATH.stat().st_mtime > hp_mtime + 0.05:
            hp = load_hp(hp)
            hp_mtime = HP_PATH.stat().st_mtime
            env.hp = hp
            net.yaw_mu_max = float(hp.get("yaw_mu_max", 1.0))
            for g in opt.param_groups:
                g["lr"] = float(hp["lr"])
            for g in rnd_opt.param_groups:
                g["lr"] = float(hp["rnd_lr"])
            print("reloaded hparams lr=%.1e ent=%.3f nov=%.2f door_p=%.2f" % (
                float(hp["lr"]), float(hp["entropy"]), float(hp["novelty_coef"]),
                float(hp["door_open_p"]),
            ), flush=True)
            set_smooth_gate(hp, env, cov_hist)
        if (OUT / "last_eval.txt").is_file():
            last_eval_note = (OUT / "last_eval.txt").read_text().strip().splitlines()[0][:120]

        img, vec = obs_to_torch(obs, device)
        with torch.no_grad():
            if held_act is None or bool(info.get("need_curve", True)):
                act, logp, value, hx = net.act(img, vec, hx)
                held_act, held_logp, held_val = act, logp, value
            else:
                act, logp, value = held_act, held_logp, held_val
                _, hx = net.encode(img, vec, hx)
        a = act.squeeze(0).cpu().numpy()
        nobs, rew, done, info = env.step(a)
        img2, _ = obs_to_torch(nobs, device)
        with torch.no_grad():
            nov = rnd.novelty(img2)
            z = rnd.embed(img2)
        nov_raw = float(nov.mean().item())
        rnd_mean = 0.99 * rnd_mean + 0.01 * nov_raw
        rnd_var = 0.99 * rnd_var + 0.01 * (nov_raw - rnd_mean) ** 2
        rnd_z = (nov_raw - rnd_mean) / max(rnd_var ** 0.5, 1e-6)
        epi = epi_mem.novelty(z)
        epi_w = float(hp.get("epi_mem", 0.40))
        nov_n = (1.0 - epi_w) * rnd_z + epi_w * (2.0 * epi - 0.3)
        st = float(info.get("straight", 1.0))
        spin = float(info.get("spin", 0.0))
        nov_n = nov_n / (1.0 + float(hp.get("circle_dampen", 8.0)) * max(0.0, 0.55 - st))
        if st < 0.45 or spin > 2.5:
            nov_n = min(0.0, nov_n)
        nov_n = float(max(-0.4, min(3.0, nov_n)))
        if info.get("seg_done") or done:
            rew = rew + float(hp["novelty_coef"]) * nov_n
        info["novelty"] = nov_n
        info["reward"] = rew
        buf.add(img, vec, act, float(logp.item()), float(value.item()), rew, done)
        ep_ret += rew
        ep_nov += nov_n
        global_step += 1
        obs = nobs
        if info.get("seg_done"):
            held_act = held_logp = held_val = None
        sim_acc += float(hp.get("dt", 0.10))
        wall = time.time() - t_wall0
        rt = sim_acc / max(1e-3, wall) if wall > 1.0 else 0.0
        if dash is not None and not dash.closed:
            used, reserved = gpu_mem()
            dash.present(
                env,
                note="vw5  step=%d" % global_step,
                rt=rt,
                extras={
                    "novelty": nov_n,
                    "status": "upd=%d  gpu=%.0f/%.0fMiB  hleft=%.2f  cost=%.1f ms=%.1f cells=%s nrm=%s"
                    % (n_upd, used, reserved, max(0.0, (end - time.time()) / 3600.0),
                       float(info.get("cost", 0)), float(info.get("prim_ms", 0)),
                       info.get("cov_cells", 0), info.get("n_rooms", 0)),
                    "eval": last_eval_note,
                },
                min_dt=1.0 / max(4.0, float(hp.get("dash_hz", 12.0))),
            )

        if done:
            rooms = info.get("rooms", "")
            row = {
                "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "ep": env.ep,
                "step": global_step,
                "ret": round(ep_ret, 3),
                "nov": round(ep_nov / max(1, env.steps), 4),
                "coverage": round(float(info.get("coverage", 0)), 5),
                "cov_cells": int(info.get("cov_cells", 0)),
                "n_rooms": int(info.get("n_rooms", 0)),
                "rooms": rooms,
                "path_m": round(float(info.get("path_m", 0)), 2),
                "cover_per_m": round(float(info.get("cover_per_m", 0)), 3),
                "collisions": int(info.get("collisions", 0)),
                "door_open": bool(info.get("door_open")),
                "end_room": info.get("room", ""),
                "clutter": info.get("clutter", ""),
                "jerk": round(float(info.get("jerk", 0)), 4),
                "visit_here": round(float(info.get("visit_here", 0)), 2),
                "visit_ahead": round(float(info.get("visit_ahead", 0)), 2),
                "dw_mean": round(float(info.get("dw_mean", 0)), 4),
                "open_turn": round(float(info.get("open_turn", 0)), 4),
                "open_speed": round(float(info.get("open_speed", 0)), 4),
                "heading_err": round(float(info.get("heading_err", 0)), 3),
                "smooth_score": round(float(info.get("smooth_score", 0)), 3),
                "wide_score": round(float(info.get("wide_score", 0)), 3),
                "mean_chord": round(float(info.get("mean_chord", 0)), 3),
                "mean_beads": round(float(info.get("mean_beads", 0)), 3),
                "win_frac": round(float(info.get("win_frac", 0)), 3),
                "curve_wins": int(info.get("curve_wins", 0)),
                "curve_losses": int(info.get("curve_losses", 0)),
                "cost": round(float(info.get("cost", 0)), 2),
                "cut": info.get("cut", ""),
                "spin": round(float(info.get("spin", 0)), 3),
                "straight": round(float(info.get("straight", 0)), 3),
                "hours_left": round(max(0.0, (end - time.time()) / 3600.0), 3),
                "curve_cont": round(float(info.get("curve_cont", 0)), 3),
                "curve_wall": round(float(info.get("curve_wall", 0)), 3),
                "prim_ms": round(float(info.get("prim_ms", 0)), 2),
                "prim_hot": round(float(info.get("prim_hot", 0)), 3),
            }
            append_ledger(row)
            ep_hist.append(row)
            mets = write_metrics(ep_hist)
            print(
                "ep %d ret=%.2f chord=%.2f win=%d/%d cut=%s cov=%d nrm=%d path=%.1f"
                % (env.ep, ep_ret, row.get("mean_chord", 0),
                   row.get("curve_wins", 0),
                   row.get("curve_wins", 0) + row.get("curve_losses", 0),
                   row["cut"] or "-",
                   row["cov_cells"], row["n_rooms"], row["path_m"]),
                flush=True,
            )
            if mets:
                print(
                    "roll25 cells=%.1f nrm=%.1f path=%.1f win=%.2f chord=%.2f beads=%.2f cut=%s"
                    % (mets["mean_cov_cells"], mets["mean_n_rooms"], mets["mean_path_m"],
                       mets.get("win_frac", 0), mets.get("mean_chord", 0),
                       mets.get("mean_beads", 0), mets["cut_frac"]),
                    flush=True,
                )
            cov_hist.append(float(row["cov_cells"]))
            set_smooth_gate(hp, env, cov_hist)
            epi_mem.reset()
            ep_ret = 0.0
            ep_nov = 0.0
            hx = None
            held_act = held_logp = held_val = None
            rebuild = int(hp.get("rebuild_every", 25) or 0)
            if rebuild and env.ep % rebuild == 0:
                seed = int(time.time()) % 99991
                print("rebuild env seed", seed, flush=True)
                if dash is not None:
                    dash.close()
                env = InsectEnv(hp, seed=seed)
                set_smooth_gate(hp, env, cov_hist)
                try:
                    dash = ExploreDash(env.wall_all, clutter=env.furniture, people=env.people.folk)
                except Exception as e:
                    print("dash rebuild skip:", e, flush=True)
                    dash = None
            obs, info = env.reset()

        if len(buf) >= int(hp["horizon"]):
            stats = ppo_update(net, opt, rnd, rnd_opt, buf, hp, device)
            n_upd += 1
            buf = Rollout()
            used, reserved = gpu_mem()
            print(
                "update %d ret=%.3f ent=%.3f rnd=%.4f gpu=%.0fMiB step=%d"
                % (n_upd, stats["ret"], stats["ent"], stats["rnd"], used, global_step),
                flush=True,
            )

        now = time.time()
        if now - last_ckpt > 600.0:
            save_ckpt(net, rnd, opt, rnd_opt, global_step)
            last_ckpt = now
        if now - last_snap > 20.0:
            used, reserved = gpu_mem()
            last_m = {}
            if (OUT / "metrics.json").is_file():
                try:
                    last_m = json.loads((OUT / "metrics.json").read_text())
                except Exception:
                    last_m = {}
            heartbeat({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "unix": "%.0f" % now,
                "pid": os.getpid(),
                "status": "running",
                "step": global_step,
                "ep": env.ep,
                "upd": n_upd,
                "xy": "%.2f,%.2f" % (env.x, env.y),
                "room": env._room_name(env.x, env.y),
                "rooms": info.get("rooms", ""),
                "cov_cells": int(getattr(env, "cov", np.zeros(1)).sum()),
                "n_rooms": int(info.get("n_rooms", 0)),
                "path_m": "%.2f" % float(info.get("path_m", 0)),
                "cost": "%.1f" % float(info.get("cost", 0)),
                "cut": info.get("cut", ""),
                "jerk": "%.4f" % float(info.get("jerk", 0)),
                "smooth": "%.3f" % float(info.get("smooth_score", 0)),
                "wide": "%.3f" % float(info.get("wide_score", 0)),
                "open_turn": "%.3f" % float(info.get("open_turn", 0)),
                "win_frac": "%.3f" % float(info.get("win_frac", 0)),
                "mean_chord": "%.2f" % float(info.get("mean_chord", 0)),
                "curve_wins": int(info.get("curve_wins", 0)),
                "prim_ms": "%.1f" % float(info.get("prim_ms", 0)),
                "roll_cells": "%.1f" % float(last_m.get("mean_cov_cells", 0)),
                "roll_jerk": "%.4f" % float(last_m.get("mean_jerk", 0)),
                "roll_smooth": "%.3f" % float(last_m.get("mean_smooth", 0)),
                "roll_wide": "%.3f" % float(last_m.get("mean_wide", 0)),
                "cost_cut_frac": "%.3f" % float(last_m.get("cost_cut_frac", 0)),
                "time_frac": "%.3f" % float(last_m.get("time_frac", 0)),
                "door": "open" if env.door_open else "shut",
                "rt": "%.2f" % rt,
                "gpu_mib": "%.0f" % used,
                "hours_left": "%.2f" % ((end - now) / 3600.0),
                "deadline_unix": "%.0f" % end,
                "lr": hp["lr"],
                "entropy": hp["entropy"],
                "novelty_coef": hp["novelty_coef"],
                "turn_coef": hp.get("turn_coef", 0),
                "stale_coef": hp.get("stale_coef", 0),
                "cost_cut": hp.get("cost_cut", 0),
                "plant": hp.get("plant", 0),
                "eval": last_eval_note[:80],
            })
            snap = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "step": global_step,
                "ep": env.ep,
                "updates": n_upd,
                "hours_left": (end - now) / 3600.0,
                "gpu_mib": used,
                "metrics": last_m,
                "hparams": {k: hp[k] for k in (
                    "lr", "entropy", "novelty_coef", "collision_coef",
                    "cover_coef", "turn_coef", "slow_coef", "stale_coef",
                    "stuck_coef", "commit_coef", "jerk_coef", "cost_cut",
                    "audio_coef", "door_open_p", "ep_steps", "horizon",
                    "depth_noise", "clutter", "plant", "rand_house",
                    "rebuild_every", "max_w", "yaw_lpf", "curve_mode",
                    "replan_s", "score_s", "a_v", "a_w", "max_v",
                ) if k in hp},
            }
            _write(OUT / "snapshot.json", json.dumps(snap, indent=2) + "\n")
            last_snap = now

    save_ckpt(net, rnd, opt, rnd_opt, global_step)
    heartbeat({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unix": "%.0f" % time.time(),
        "pid": os.getpid(),
        "status": "done" if time.time() >= end else "restart",
        "step": global_step,
        "hours_left": "%.2f" % max(0.0, (end - time.time()) / 3600.0),
        "deadline_unix": "%.0f" % end,
    })
    if dash is not None:
        dash.close()
    print("explore_train exit", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
