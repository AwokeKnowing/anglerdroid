#!/usr/bin/env python3
"""Greedy probe: detect spin-lock / nest / crash-spiral and apply a live fix."""
from __future__ import annotations

import numpy as np
import torch

from rl_nav import kernels as K


@torch.no_grad()
def probe(eval_env, net, n_steps: int | None = None) -> dict:
    n_steps = int(n_steps or K.EP_STEPS)
    eval_env.reset_all()
    prims = []
    vs, ws, hits_t = [], [], []
    x0 = eval_env.x.detach().clone()
    y0 = eval_env.y.detach().clone()
    for _ in range(n_steps):
        prim, write, _, _ = net.act(eval_env.ego, eval_env.vec, deterministic=True)
        prims.append(prim.detach().cpu().numpy())
        vs.append(eval_env.v.detach().cpu().numpy())
        ws.append(eval_env.w.detach().cpu().numpy())
        hits_t.append(eval_env.hits.detach().cpu().numpy())
        eval_env.step(prim, write)
    prims = np.stack(prims)
    vs = np.stack(vs)
    ws = np.stack(ws)
    hits = eval_env.hits.detach().cpu().float()
    disp = torch.hypot(eval_env.x - x0, eval_env.y - y0)
    mode = int(np.bincount(prims[:, 0], minlength=K.N_PRIM).argmax())
    mode_frac = float((prims[:, 0] == mode).mean())
    # categorical entropy of env0 first-step logits
    h = net.encode(eval_env.ego[:1], eval_env.vec[:1])
    logits = net.pi(h)[0]
    pent = float(torch.distributions.Categorical(logits=logits).entropy().item())
    out = {
        "hits_mean": float(hits.mean().item()),
        "hits_p50": float(hits.median().item()),
        "disp_mean": float(disp.mean().item()),
        "v_mean": float(vs.mean()),
        "v_abs": float(np.abs(vs).mean()),
        "w_abs": float(np.abs(ws).mean()),
        "spin_frac": float(((np.abs(ws) > 0.40) & (vs < 0.06)).mean()),
        "mode_prim": mode,
        "mode_frac": mode_frac,
        "prim_ent": pent,
        "n": eval_env.n,
        "crash_coef": float(eval_env.crash_coef),
        "stall_coef": float(eval_env.stall_coef),
    }
    fails = []
    why = []
    spawn_hits = float(K.PAINT_N) + 3.0
    if out["hits_mean"] <= spawn_hits and out["spin_frac"] > 0.50 and out["disp_mean"] < 0.40:
        fails.append("spin_lock")
        why.append(
            "greedy locked prim %d (frac %.2f): v=%.3f w=%.3f disp=%.2fm hits=%.2f"
            % (mode, mode_frac, out["v_mean"], out["w_abs"], out["disp_mean"], out["hits_mean"])
        )
    if out["hits_mean"] <= spawn_hits and out["disp_mean"] < 0.80:
        fails.append("nest")
        why.append(
            "never left spawn fog (hits=%.2f disp=%.2fm). staying put beats exploring."
            % (out["hits_mean"], out["disp_mean"])
        )
    if out["prim_ent"] < 0.35 and out["mode_frac"] > 0.85:
        fails.append("entropy_dead")
        why.append("primitive entropy %.3f mode_frac %.2f — action head collapsed" % (pent, mode_frac))
    out["fails"] = fails
    out["why"] = why
    out["ok"] = len(fails) == 0
    return out


def apply_fix(eval_env, train_env, net, diag: dict, entropy: float) -> tuple[float, bool]:
    """Mutate train_env / net. Returns (entropy, need_opt_reset)."""
    fails = set(diag.get("fails") or [])
    if not fails:
        return entropy, False
    print("FAILMODE %s" % ",".join(fails), flush=True)
    for line in diag.get("why") or []:
        print("  why:", line, flush=True)
    reset_opt = False
    if fails & {"spin_lock", "nest", "entropy_dead"}:
        torch.nn.init.orthogonal_(net.pi.weight, 0.01)
        if net.pi.bias is not None:
            torch.nn.init.zeros_(net.pi.bias)
        net.write_logstd.data.fill_(-2.2)
        entropy = min(0.08, max(entropy, 0.04) + 0.02)
        train_env.stall_coef = max(float(train_env.stall_coef), 0.03)
        eval_env.stall_coef = train_env.stall_coef
        reset_opt = True
        print(
            "  fix: reinit pi + write_logstd, entropy→%.3f stall→%.3f"
            % (entropy, train_env.stall_coef),
            flush=True,
        )
    if "crash_spiral" in fails:
        train_env.crash_coef = max(0.50, float(train_env.crash_coef) * 0.70)
        eval_env.crash_coef = train_env.crash_coef
        print("  fix: crash_coef→%.3f" % train_env.crash_coef, flush=True)
    return entropy, reset_opt


def maybe_crash_spiral(st: dict, diag: dict) -> dict:
    if float(st.get("crash_frac", 0.0)) > 0.65 and float(st.get("cover", 0.0)) < 0.20:
        diag = dict(diag)
        fails = list(diag.get("fails") or [])
        if "crash_spiral" not in fails:
            fails.append("crash_spiral")
            why = list(diag.get("why") or [])
            why.append(
                "train crash=%.2f cover=%.3f — exploring is punished"
                % (st["crash_frac"], st.get("cover", 0.0))
            )
            diag["fails"] = fails
            diag["why"] = why
            diag["ok"] = False
    return diag
