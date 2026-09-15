#!/usr/bin/env python3
"""Vectorized GPU explorer. Coverage objective; 64-byte memory is the recurrence."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
import warp as wp

_SIM = Path(__file__).resolve().parent.parent
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

try:
    from . import kernels as K
    from .house_gen import build_pool
    from .fov80 import load_stencil
except ImportError:
    import kernels as K  # noqa: E402
    from house_gen import build_pool  # noqa: E402
    from fov80 import load_stencil  # noqa: E402


class RlNavVec:
    def __init__(self, n: int = 512, seed: int = 1, device: str = "cuda:0"):
        self.n = int(n)
        self.device_str = device
        self.seed0 = int(seed)
        self.torch_dev = torch.device(device)
        self.stage = 0
        self.n_movers_hp = 0
        self.crash_coef = 1.0
        self.stall_coef = 0.02
        self.yaw_coef = 0.0
        self._watch = 0
        wp.init()
        wp.set_device(device)
        self._stream = None
        try:
            self._stream = wp.stream_from_torch(torch.cuda.current_stream(self.torch_dev))
        except Exception:
            self._stream = None
        N = self.n
        d = self.torch_dev

        def f(*shape):
            return torch.zeros(*shape, device=d, dtype=torch.float32)

        def i32(*shape):
            return torch.zeros(*shape, device=d, dtype=torch.int32)

        self.boxes = f(N, K.MAX_B, 8)
        self.nbox = i32(N)
        self.x = f(N)
        self.y = f(N)
        self.yaw = f(N)
        self.v = f(N)
        self.w = f(N)
        self.mx = f(N, K.MAX_M)
        self.my = f(N, K.MAX_M)
        self.mvx = f(N, K.MAX_M)
        self.mvy = f(N, K.MAX_M)
        self.n_movers = i32(N)
        self.mem = f(N, K.MEM_N)
        self.steps = i32(N)
        self.t_sim = f(N)
        self.path_m = f(N)
        self.hits = i32(N)
        self.done = i32(N)
        self.cut = i32(N)
        self.v_scale = f(N)
        self.a_scale = f(N)
        self.w_scale = f(N)
        self.delay_n = i32(N)
        self.held_v = f(N)
        self.held_w = f(N)
        self.plus_x = f(N)
        self.throt_f = f(N)
        self.throt_b = f(N)
        self.throt_l = f(N)
        self.throt_r = f(N)
        self.rew = f(N)
        self.ep = i32(N)
        self.seen = i32(N, K.SEC_N)
        self.valid = i32(N, K.SEC_N)
        self.n_free = i32(N)
        self.ego = f(N, K.EGO_H, K.EGO_W)
        self.vec = f(N, K.VEC_N + K.MEM_N)
        self.fov = torch.as_tensor(load_stencil(), device=d, dtype=torch.int32)
        self.look = torch.as_tensor(K.LOOK_KNOBS, device=d, dtype=torch.float32)
        self.reset_mask = torch.ones(N, device=d, dtype=torch.int32)
        pb, pn, pxy = build_pool(128, seed=self.seed0)
        self.pool_boxes = torch.as_tensor(pb, device=d)
        self.pool_nbox = torch.as_tensor(pn, device=d, dtype=torch.int32)
        self.pool_xy = torch.as_tensor(pxy, device=d)
        self.roll_crash = 0.0
        self.roll_hits = 0.0
        self.roll_clear = 0.0
        self.roll_path = 0.0
        self.roll_ret = 0.0
        self.roll_cover = 0.0
        self.roll_ceil = 80.0
        self.n_done = 0
        self.ep_ret = f(N)
        self._wp = {}
        for name in (
            "boxes", "nbox", "x", "y", "yaw", "v", "w", "mx", "my", "mvx", "mvy",
            "n_movers", "mem", "steps", "t_sim", "path_m", "hits", "done", "cut",
            "v_scale", "a_scale", "w_scale", "delay_n", "held_v", "held_w",
            "plus_x", "throt_f", "throt_b", "throt_l", "throt_r",
            "rew", "ep", "seen", "valid", "n_free",
            "ego", "vec", "reset_mask", "fov", "look",
        ):
            t = getattr(self, name)
            self._wp[name] = wp.from_torch(t.contiguous())
        print(
            "rl_nav N=%d ego=%dx%d mem=%d sec=%d device=%s"
            % (N, K.EGO_H, K.EGO_W, K.MEM_N, K.SEC_N, device),
            flush=True,
        )
        self.reset_all()

    def set_curriculum(self, stage: int):
        stage = int(max(0, min(5, stage)))
        self.stage = stage
        # movers only. No hint bearing / shape-to-unseen: coverage paint
        # is the whole objective (go everywhere as fast as the cap allows).
        table = (0, 0, 0, 1, 2, 3)
        self.n_movers_hp = table[stage]

    def set_look(self, knobs, reobserve=True):
        """Hot-swap look knobs. Does not recompile Warp."""
        t = torch.as_tensor(knobs, device=self.torch_dev, dtype=torch.float32)
        if t.numel() != K.LOOK_KNOB_N:
            raise ValueError("look knobs want %d, got %d" % (K.LOOK_KNOB_N, int(t.numel())))
        self.look.copy_(t)
        if reobserve:
            self._observe()
            if self._stream is None:
                wp.synchronize()

    def _launch(self, kernel, dim, inputs):
        kw = dict(kernel=kernel, dim=dim, inputs=inputs, device=self.device_str)
        if self._stream is not None:
            kw["stream"] = self._stream
        wp.launch(**kw)

    def reset_all(self):
        self.reset_mask.fill_(1)
        self._wp["reset_mask"] = wp.from_torch(self.reset_mask.contiguous())
        self._reset_masked()
        self._observe()
        if self._stream is None:
            wp.synchronize()
        return self.ego, self.vec

    def _load_houses(self):
        sel = self.reset_mask > 0
        if not bool(sel.any().item()):
            return
        e_idx = sel.nonzero(as_tuple=False).squeeze(-1)
        p = int(self.pool_boxes.shape[0])
        keys = (int(self.seed0) + e_idx * 10007 + self.ep[e_idx] * 9176 + self.stage * 41) % p
        self.boxes[e_idx] = self.pool_boxes[keys]
        self.nbox[e_idx] = self.pool_nbox[keys]
        self.x[e_idx] = self.pool_xy[keys, 0]
        self.y[e_idx] = self.pool_xy[keys, 1]

    def _reset_masked(self):
        self._load_houses()
        w = self._wp
        self._launch(
            K.k_reset,
            self.n,
            [
                w["reset_mask"], int(self.seed0), w["ep"], w["boxes"], w["nbox"],
                w["x"], w["y"], w["yaw"], w["v"], w["w"], w["mx"], w["my"],
                w["mvx"], w["mvy"], w["n_movers"], w["mem"], w["steps"],
                w["t_sim"], w["path_m"], w["hits"], w["done"], w["cut"],
                w["v_scale"], w["a_scale"], w["w_scale"], w["delay_n"],
                w["held_v"], w["held_w"], w["plus_x"], w["seen"], w["valid"],
                w["n_free"], int(self.n_movers_hp),
                int(self.stage),
            ],
        )

    def _observe(self):
        w = self._wp
        self._launch(
            K.k_obs,
            (self.n, K.EGO_H, K.EGO_W),
            [
                w["boxes"], w["x"], w["y"], w["yaw"], w["mx"], w["my"],
                w["n_movers"], w["mem"], w["v"], w["w"], w["plus_x"],
                w["throt_f"], w["throt_b"], w["throt_l"], w["throt_r"],
                w["t_sim"], w["hits"], w["n_free"], w["v_scale"], w["w_scale"],
                w["fov"], w["look"],
                w["ego"], w["vec"],
            ],
        )

    def step(self, prim: torch.Tensor, write: torch.Tensor):
        if prim.dtype != torch.int32:
            prim = prim.to(dtype=torch.int32)
        prim = prim.contiguous()
        write = write.to(device=self.torch_dev, dtype=torch.float32).contiguous()
        prim_wp = wp.from_torch(prim)
        write_wp = wp.from_torch(write)
        w = self._wp
        self._launch(
            K.k_step,
            self.n,
            [
                prim_wp, write_wp, w["boxes"], w["x"], w["y"], w["yaw"], w["v"],
                w["w"], w["mx"], w["my"], w["mvx"], w["mvy"], w["n_movers"],
                w["mem"], w["steps"], w["t_sim"], w["path_m"], w["hits"],
                w["done"], w["cut"], w["v_scale"], w["a_scale"], w["w_scale"],
                w["delay_n"], w["held_v"], w["held_w"], w["plus_x"],
                w["throt_f"], w["throt_b"], w["throt_l"], w["throt_r"], w["rew"],
                w["seen"], w["valid"], w["n_free"],
                float(self.crash_coef), float(self.stall_coef),
            ],
        )
        self._observe()
        rew = self.rew.clone()
        done = self.done.clone()
        self.ep_ret += rew
        cb = getattr(self, "_env0_cb", None)
        if cb is not None:
            e = int(self._watch) % self.n
            died = int(self.done[e].item()) != 0
            try:
                cb(self.env_cpu(e), int(died))
            except Exception as ex:
                print("env0 grab skip", ex, flush=True)
            if died:
                pick = torch.nonzero(self.done > 0, as_tuple=False).view(-1)
                if pick.numel() > 0:
                    self._watch = int(pick[int(torch.randint(0, pick.numel(), ()).item())].item())
                else:
                    self._watch = int(torch.randint(0, self.n, ()).item())
        if bool(done.any().item()):
            dead = done > 0
            n_d = int(dead.sum().item())
            if n_d > 0:
                self.n_done += n_d
                self.roll_crash = 0.92 * self.roll_crash + 0.08 * float(
                    (self.cut[dead] == 2).float().mean().item()
                )
                self.roll_clear = 0.92 * self.roll_clear + 0.08 * float(
                    (self.cut[dead] == 1).float().mean().item()
                )
                self.roll_hits = 0.92 * self.roll_hits + 0.08 * float(
                    self.hits[dead].float().mean().item()
                )
                self.roll_path = 0.92 * self.roll_path + 0.08 * float(
                    self.path_m[dead].mean().item()
                )
                self.roll_ret = 0.92 * self.roll_ret + 0.08 * float(
                    self.ep_ret[dead].mean().item()
                )
                ceil = float(self.n_free[dead].float().mean().item())
                self.roll_ceil = 0.92 * self.roll_ceil + 0.08 * max(ceil, 1.0)
                cover = float(
                    (self.hits[dead].float() / self.n_free[dead].float().clamp(min=1)).mean().item()
                )
                self.roll_cover = 0.92 * self.roll_cover + 0.08 * cover
                self.ep_ret[dead] = 0.0
            self.reset_mask.copy_(done)
            self._wp["reset_mask"] = wp.from_torch(self.reset_mask.contiguous())
            self._reset_masked()
            self._observe()
        if self._stream is None:
            wp.synchronize()
        return self.ego, self.vec, rew, done

    def stats(self) -> dict:
        live_cover = float(
            (self.hits.float() / self.n_free.float().clamp(min=1)).mean().item()
        )
        return {
            "mean_hits": float(self.roll_hits),
            "mean_path": float(self.roll_path),
            "crash_frac": float(self.roll_crash),
            "clear_frac": float(self.roll_clear),
            "live_hits": float(self.hits.float().mean().item()),
            "mean_plus_x": float(self.plus_x.mean().item()),
            "mean_ret": float(self.roll_ret),
            "ceil": float(self.roll_ceil),
            "ret_ceil": float(self.roll_ceil + 1.0),
            "cover": float(self.roll_cover),
            "live_cover": live_cover,
            "stage": int(self.stage),
            "n_ep": int(self.n_done),
            "n": self.n,
            "crash_coef": float(self.crash_coef),
            "stall_coef": float(self.stall_coef),
            "yaw_coef": float(self.yaw_coef),
        }

    def env0_cpu(self) -> dict:
        return self.env_cpu(0)

    def env_cpu(self, e: int) -> dict:
        e = int(e) % self.n
        nf = max(int(self.n_free[e]), 1)
        return {
            "x": float(self.x[e]),
            "y": float(self.y[e]),
            "yaw": float(self.yaw[e]),
            "boxes": self.boxes[e].detach().cpu().numpy().copy(),
            "ego": self.ego[e].detach().cpu().numpy().copy(),
            "seen": self.seen[e].detach().cpu().numpy().copy(),
            "valid": self.valid[e].detach().cpu().numpy().copy(),
            "mx": self.mx[e].detach().cpu().numpy().copy(),
            "my": self.my[e].detach().cpu().numpy().copy(),
            "n_movers": int(self.n_movers[e]),
            "plus_x": float(self.plus_x[e]),
            "throt_f": float(self.throt_f[e]),
            "throt_b": float(self.throt_b[e]),
            "throt_l": float(self.throt_l[e]),
            "throt_r": float(self.throt_r[e]),
            "hits": int(self.hits[e]),
            "n_free": nf,
            "cover": float(int(self.hits[e]) / nf),
            "path_m": float(self.path_m[e]),
            "t_sim": float(self.t_sim[e]),
            "steps": int(self.steps[e]),
            "cut": int(self.cut[e]),
            "nbox": int(self.nbox[e]),
        }
