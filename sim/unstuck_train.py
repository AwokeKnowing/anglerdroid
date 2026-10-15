#!/usr/bin/env python3
"""Overnight PPO for the compact unstuck net. CPU by default — do not steal the 20h GPU.

  DISPLAY=:1 setsid .venv-newton/bin/python -u sim/unstuck_train.py --hours 6 \\
      >> sim/unstuck/train.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SIM = Path(__file__).resolve().parent
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

from unstuck_env import UnstuckEnv  # noqa: E402
from unstuck_net import TinyUnstuck, obs_to_torch  # noqa: E402

OUT = _SIM / "unstuck"
CKPT = OUT / "ckpt.pt"
HP_PATH = OUT / "hparams.json"


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def gae(rews, vals, dones, gamma, lam):
    adv = np.zeros_like(rews)
    last = 0.0
    for t in range(len(rews) - 1, -1, -1):
        nxt = 0.0 if t == len(rews) - 1 else vals[t + 1]
        mask = 1.0 - float(dones[t])
        delta = rews[t] + gamma * nxt * mask - vals[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    return adv, adv + vals


class Buf:
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


def ppo_update(net, opt, buf: Buf, hp, device):
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
    last_ent = last_pi = 0.0
    idx = np.arange(n)
    for _ in range(int(hp["epochs"])):
        np.random.shuffle(idx)
        for s in range(0, n, mb):
            sl = torch.tensor(idx[s:s + mb], device=device, dtype=torch.long)
            logp, ent, value, _ = net.evaluate(img[sl], vec[sl], act[sl], hx=None)
            ratio = (logp - old_logp[sl]).exp()
            clip = float(hp["clip"])
            surr = torch.min(ratio * adv_t[sl],
                             ratio.clamp(1.0 - clip, 1.0 + clip) * adv_t[sl])
            loss = -surr.mean() + 0.5 * (value - ret_t[sl]).pow(2).mean() - float(hp["entropy"]) * ent.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            last_ent = float(ent.mean().item())
            last_pi = float((-surr.mean()).item())
    return {"ent": last_ent, "pi": last_pi, "ret": float(ret.mean())}


def save_ckpt(net, opt, step, extra=None):
    OUT.mkdir(parents=True, exist_ok=True)
    prev = OUT / "ckpt.prev.pt"
    if CKPT.is_file():
        try:
            import shutil
            shutil.copy2(CKPT, prev)
        except Exception:
            pass
    torch.save({
        "net": net.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "extra": extra or {},
        "arch": "TinyUnstuck",
        "act_dim": 2,
        "vec_dim": 8,
    }, CKPT)
    home = Path.home() / ".kevin"
    try:
        home.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(CKPT, home / "unstuck.pt")
    except Exception:
        pass


def heartbeat(row):
    lines = ["%s=%s" % (k, row[k]) for k in row]
    _write(OUT / "heartbeat.txt", "\n".join(lines) + "\n")
    _write(OUT / "STATUS.txt", "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
    device = torch.device(args.device)
    hp = {
        "lr": 3e-4, "entropy": 0.025, "gamma": 0.98, "gae_lambda": 0.92,
        "clip": 0.2, "horizon": 512, "epochs": 4, "minibatch": 128,
        "ep_steps": 80, "dt": 0.10, "max_v": 0.28, "max_w": 0.80,
        "v_rev": -0.10, "a_v": 0.70, "a_w": 2.40, "replan_s": 0.20,
        "plus_x_stop": 0.28, "unstuck_px_lo": 0.10, "unstuck_px_hi": 0.24,
        "unstuck_clear_px": 0.50, "unstuck_hold_s": 0.40, "coll_done": 40,
        "rebuild_every": 12, "goal_p": 0.55, "device": str(device),
    }
    if HP_PATH.is_file():
        try:
            hp.update(json.loads(HP_PATH.read_text()))
        except Exception as e:
            print("hparams skip", e, flush=True)
    hp["device"] = str(device)
    _write(HP_PATH, json.dumps(hp, indent=2) + "\n")
    env = UnstuckEnv(hp=hp, seed=args.seed)
    net = TinyUnstuck().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(hp["lr"]))
    step = 0
    if CKPT.is_file():
        try:
            ck = torch.load(str(CKPT), map_location=device, weights_only=False)
            net.load_state_dict(ck["net"], strict=False)
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
            step = int(ck.get("step") or 0)
            print("resume step", step, "params", net.n_params(), flush=True)
        except Exception as e:
            print("ckpt skip", e, flush=True)
    else:
        print("fresh TinyUnstuck params", net.n_params(), flush=True)
    stop = {"flag": False}

    def _sig(_a, _b):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    deadline = time.time() + float(args.hours) * 3600.0
    _write(OUT / "deadline.txt", "%.3f\n" % deadline)
    _write(OUT / "train.pid", "%d\n" % os.getpid())
    buf = Buf()
    hx = None
    obs, info = env.reset()
    wins = []
    t0 = time.time()
    upd = 0
    ep_ret = 0.0
    while time.time() < deadline and not stop["flag"]:
        img, vec = obs_to_torch(obs, device)
        with torch.no_grad():
            act, logp, val, hx = net.act(img, vec, hx, deterministic=False)
        a = act.squeeze(0).cpu().numpy()
        nxt, rew, done, info = env.step(a)
        ep_ret += float(rew)
        buf.add(img, vec, act, float(logp), float(val), rew, done)
        step += 1
        obs = nxt
        if done:
            wins.append(1.0 if info.get("win") else 0.0)
            if len(wins) > 200:
                wins = wins[-200:]
            hx = None
            obs, info = env.reset()
            ep_ret = 0.0
        if len(buf) >= int(hp["horizon"]):
            stats = ppo_update(net, opt, buf, hp, device)
            upd += 1
            buf = Buf()
            win = float(np.mean(wins)) if wins else 0.0
            elapsed = time.time() - t0
            sps = step / max(elapsed, 1e-3)
            row = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "unix": int(time.time()),
                "pid": os.getpid(),
                "status": "running",
                "step": step,
                "upd": upd,
                "win_frac": round(win, 3),
                "ent": round(stats["ent"], 4),
                "ret": round(stats["ret"], 3),
                "sps": round(sps, 1),
                "plus_x": round(float(info.get("plus_x") or 0), 3),
                "hours_left": round((deadline - time.time()) / 3600.0, 2),
                "params": net.n_params(),
                "device": str(device),
            }
            heartbeat(row)
            with open(OUT / "ledger.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if upd % 4 == 0:
                save_ckpt(net, opt, step, extra={"win_frac": win})
                print(
                    "unstuck upd=%d step=%d win=%.2f ent=%.3f sps=%.0f left=%.2fh"
                    % (upd, step, win, stats["ent"], sps, row["hours_left"]),
                    flush=True,
                )
    save_ckpt(net, opt, step, extra={"win_frac": float(np.mean(wins)) if wins else 0.0})
    heartbeat({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unix": int(time.time()),
        "status": "done",
        "step": step,
        "upd": upd,
        "win_frac": round(float(np.mean(wins)) if wins else 0.0, 3),
        "hours_left": 0.0,
    })
    print("unstuck train done step", step, flush=True)


if __name__ == "__main__":
    main()
