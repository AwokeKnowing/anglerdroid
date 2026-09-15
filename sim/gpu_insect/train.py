#!/usr/bin/env python3
"""Batched PPO on GpuInsectVec. Separate from sim/explore (do not touch that run)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_SIM = Path(__file__).resolve().parent.parent
_REPO = _SIM.parent
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

from explore_net import ActorCritic, RND  # noqa: E402
from gpu_insect.env import GpuInsectVec  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def load_hp():
    src = _SIM / "explore" / "hparams.json"
    hp = {}
    if src.is_file():
        hp = json.loads(src.read_text())
    hp.setdefault("lr", 3e-4)
    hp.setdefault("rnd_lr", 1e-4)
    hp.setdefault("entropy", 0.01)
    hp.setdefault("gamma", 0.99)
    hp.setdefault("gae_lambda", 0.95)
    hp.setdefault("clip", 0.2)
    hp.setdefault("horizon", 128)
    hp.setdefault("epochs", 3)
    hp.setdefault("minibatch", 2048)
    hp.setdefault("novelty_coef", 0.10)
    hp.setdefault("device", "cuda:0")
    hp["horizon"] = 64
    hp["minibatch"] = 2048
    return hp


def ppo(net, opt, rnd, rnd_opt, buf, hp):
    img, vec, act, old_logp, old_v, rew, done = buf
    t, n = rew.shape
    gamma = float(hp["gamma"])
    lam = float(hp["gae_lambda"])
    clip = float(hp["clip"])
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
    img_b = img.reshape(t * n, *img.shape[2:])
    vec_b = vec.reshape(t * n, vec.shape[-1])
    act_b = act.reshape(t * n, act.shape[-1])
    old_logp_b = old_logp.reshape(t * n)
    adv_b = adv.reshape(t * n)
    ret_b = ret.reshape(t * n)
    idx = torch.randperm(t * n, device=rew.device)
    mb = min(int(hp["minibatch"]), t * n)
    ent_acc = 0.0
    n_mb = 0
    for _ in range(int(hp["epochs"])):
        for s in range(0, t * n, mb):
            sl = idx[s:s + mb]
            logp, ent, value, _ = net.evaluate(img_b[sl], vec_b[sl], act_b[sl], hx=None)
            ratio = (logp - old_logp_b[sl]).exp()
            surr = torch.min(
                ratio * adv_b[sl],
                ratio.clamp(1.0 - clip, 1.0 + clip) * adv_b[sl],
            )
            loss = -surr.mean() + 0.5 * (ret_b[sl] - value).pow(2).mean() - float(hp["entropy"]) * ent.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if rnd is not None:
                nov = rnd.novelty(img_b[sl])
                rnd_opt.zero_grad(set_to_none=True)
                nov.mean().backward()
                rnd_opt.step()
            ent_acc += float(ent.mean().item())
            n_mb += 1
    return ent_acc / max(1, n_mb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=128)
    ap.add_argument("--hours", type=float, default=0.0, help="0 = run until killed")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-dash", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("DISPLAY", ":1")
    OUT.mkdir(parents=True, exist_ok=True)
    hp = load_hp()
    hp["device"] = args.device
    device = torch.device(args.device)
    env = GpuInsectVec(n=args.n, hparams=hp, seed=3, device=args.device)
    net = ActorCritic(vec_dim=10, hidden=128, act_dim=4).to(device)
    rnd = RND().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(hp["lr"]))
    rnd_opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, rnd.parameters()), lr=float(hp["rnd_lr"])
    )
    dash = None
    if not args.no_dash:
        try:
            from gpu_insect.dash import GpuDash
            dash = GpuDash()
        except Exception as e:
            print("dash skip", e, flush=True)
    img, vec = env.img, env.vec
    hx = None
    held_act = held_logp = held_val = None
    horizon = int(hp["horizon"])
    n = env.n
    buf_img, buf_vec, buf_act, buf_logp, buf_v, buf_rew, buf_done = [], [], [], [], [], [], []
    t0 = time.time()
    end = t0 + args.hours * 3600.0 if args.hours > 0 else t0 + 365 * 86400
    step = 0
    n_upd = 0
    last_log = t0
    sps_ema = 0.0
    ckpt = OUT / "ckpt.pt"
    if ckpt.is_file():
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        net.load_state_dict(blob["net"])
        print("loaded", ckpt, "step", blob.get("step"), flush=True)
        step = int(blob.get("step", 0))
    print("gpu insect train N=%d horizon=%d out=%s" % (n, horizon, OUT), flush=True)
    while time.time() < end:
        need = env.need_curve > 0
        with torch.no_grad():
            act_new, logp_new, val_new, hx = net.act(img, vec, hx)
            if held_act is None:
                held_act, held_logp, held_val = act_new, logp_new, val_new
            need3 = need.unsqueeze(-1)
            act = torch.where(need3, act_new, held_act)
            logp = torch.where(need, logp_new, held_logp)
            value = torch.where(need, val_new, held_val)
            held_act, held_logp, held_val = act, logp, value
        nimg, nvec, rew, done, seg, need2 = env.step(act)
        with torch.no_grad():
            nov = rnd.novelty(nimg)
            nov = (nov - nov.mean()) / (nov.std() + 1e-6)
            extra = float(hp["novelty_coef"]) * nov
            rew = rew + extra * (seg.float() + done.float()).clamp(0.0, 1.0)
        buf_img.append(img.clone())
        buf_vec.append(vec.clone())
        buf_act.append(act.clone())
        buf_logp.append(logp.clone())
        buf_v.append(value.clone())
        buf_rew.append(rew.clone())
        buf_done.append(done.float().clone())
        img, vec = nimg, nvec
        dead = (done > 0) | (seg > 0)
        if bool(dead.any().item()):
            held_act = torch.where(dead.unsqueeze(-1), torch.zeros_like(held_act), held_act)
            if hx is not None:
                hx = hx * (~dead).float().view(1, n, 1)
        step += n
        if dash is not None and not dash.closed:
            dash.present(env, sps=sps_ema, note="upd=%d step=%d" % (n_upd, step))
        if len(buf_rew) >= horizon:
            t_s = time.time()
            buf = (
                torch.stack(buf_img), torch.stack(buf_vec), torch.stack(buf_act),
                torch.stack(buf_logp), torch.stack(buf_v), torch.stack(buf_rew),
                torch.stack(buf_done),
            )
            ent = ppo(net, opt, rnd, rnd_opt, buf, hp)
            n_upd += 1
            buf_img, buf_vec, buf_act, buf_logp, buf_v, buf_rew, buf_done = [], [], [], [], [], [], []
            dt = time.time() - t_s
            sps_ema = 0.9 * sps_ema + 0.1 * (horizon * n / max(dt, 1e-3)) if sps_ema else (horizon * n / max(dt, 1e-3))
            if time.time() - last_log > 8.0:
                st = env.stats()
                used = torch.cuda.memory_allocated() / (1024 ** 2)
                print(
                    "upd %d step=%d sps=%.0f ent=%.3f gpu=%.0fMiB win=%.3f chord=%.2f "
                    "beads=%.2f path=%.2f cells=%.1f"
                    % (n_upd, step, sps_ema, ent, used, st["win_frac"], st["mean_chord"],
                       st["mean_beads"], st["mean_path"], st["mean_cells"]),
                    flush=True,
                )
                last_log = time.time()
                torch.save({"net": net.state_dict(), "step": step, "n": n}, ckpt)
    if dash is not None:
        dash.close()
    torch.save({"net": net.state_dict(), "step": step, "n": n}, OUT / "ckpt.pt")
    print("gpu insect train stopped step", step, flush=True)


if __name__ == "__main__":
    main()
