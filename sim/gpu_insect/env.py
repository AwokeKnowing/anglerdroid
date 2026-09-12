#!/usr/bin/env python3
"""Vectorized GPU insect env. State and obs stay on CUDA as torch tensors."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import torch
import warp as wp

_SIM = Path(__file__).resolve().parent.parent
if str(_SIM) not in sys.path:
    sys.path.insert(0, str(_SIM))

try:
    from . import kernels as K
except ImportError:
    import kernels as K  # noqa: E402


def _cov_shape():
    return K.COV_H, K.COV_W, K.COV_CELL


class GpuInsectVec:
    """N parallel Kevins. step() is Warp launches; img/vec are GPU tensors."""

    def __init__(self, n: int = 256, hparams: dict | None = None, seed: int = 1, device: str = "cuda:0"):
        self.n = int(n)
        self.hp = dict(hparams or {})
        self.device_str = str(self.hp.get("device", device))
        self.seed0 = int(seed)
        self.torch_dev = torch.device(self.device_str)
        wp.init()
        wp.set_device(self.device_str)
        self._stream = None
        try:
            self._stream = wp.stream_from_torch(torch.cuda.current_stream(self.torch_dev))
        except Exception:
            self._stream = None
        cov_h, cov_w, self.cov_cell = _cov_shape()
        N = self.n
        d = self.torch_dev

        def f(*shape):
            return torch.zeros(*shape, device=d, dtype=torch.float32)

        def i32(*shape):
            return torch.zeros(*shape, device=d, dtype=torch.int32)

        def u8(*shape):
            return torch.zeros(*shape, device=d, dtype=torch.uint8)

        self.boxes = f(N, K.MAX_B, 8)
        self.nbox = i32(N)
        self.x = f(N)
        self.y = f(N)
        self.yaw = f(N)
        self.visit = f(N, K.VISIT_N, K.VISIT_N)
        self.visit_tmp = f(N, K.VISIT_N, K.VISIT_N)
        self.cov = u8(N, cov_h, cov_w)
        self.path_m = f(N)
        self.coll = i32(N)
        self.coll_streak = i32(N)
        self.yaw_only = i32(N)
        self.steps = i32(N)
        self.t_sim = f(N)
        self.need_curve = i32(N)
        self.curve_got = i32(N, 3)
        self.curve_next = i32(N)
        self.curve_chord = f(N)
        self.curve_t0 = f(N)
        self.curve_n = i32(N)
        self.curve_wins = i32(N)
        self.curve_losses = i32(N)
        self.beads_sum = f(N)
        self.chord_win_sum = f(N)
        self.last_act = f(N, 4)
        self.rooms_seen = i32(N)
        self.visit_gx = i32(N)
        self.visit_gy = i32(N)
        self.ctrl = f(N, 4, 2)
        self.poly = f(N, K.POLY_N, 2)
        self.beads = f(N, 3, 2)
        self.wall_c = f(N)
        self.cont_c = f(N)
        self.plus_x = f(N)
        self.rew = f(N)
        self.done = i32(N)
        self.seg_done = i32(N)
        self.need_out = i32(N)
        self.cut = i32(N)
        self.cov_cells = i32(N)
        self.ep = i32(N)
        self.v_out = f(N)
        self.w_out = f(N)
        self.depth = f(N, K.DEPTH_H, K.DEPTH_W)
        self.img = f(N, 3, K.DEPTH_H, K.DEPTH_W)
        self.vec = f(N, 10)
        dx, dy = K.pinhole_dxdy()
        self.dx = torch.from_numpy(dx).to(d)
        self.dy = torch.from_numpy(dy).to(d)
        self.reset_mask = torch.ones(N, device=d, dtype=torch.int32)

        self._wp = {}
        for name in (
            "boxes", "nbox", "x", "y", "yaw", "visit", "visit_tmp", "cov",
            "path_m", "coll", "coll_streak", "yaw_only", "steps", "t_sim",
            "need_curve", "curve_got", "curve_next", "curve_chord", "curve_t0",
            "curve_n", "curve_wins", "curve_losses", "beads_sum", "chord_win_sum",
            "last_act", "rooms_seen", "visit_gx", "visit_gy", "ctrl", "poly",
            "beads", "wall_c", "cont_c", "plus_x", "rew", "done", "seg_done",
            "need_out", "cut", "cov_cells", "ep", "v_out", "w_out", "depth",
            "img", "vec", "dx", "dy", "reset_mask",
        ):
            t = getattr(self, name)
            self._wp[name] = wp.from_torch(t.contiguous())

        print(
            "gpu insect vec N=%d boxes<=%d depth=%dx%d device=%s"
            % (N, K.MAX_B, K.DEPTH_H, K.DEPTH_W, self.device_str),
            flush=True,
        )
        self.reset_all()

    def _launch(self, kernel, dim, inputs):
        kw = dict(kernel=kernel, dim=dim, inputs=inputs, device=self.device_str)
        if self._stream is not None:
            kw["stream"] = self._stream
        wp.launch(**kw)

    def _hp(self, key, default):
        return self.hp.get(key, default)

    def reset_all(self):
        self.reset_mask.fill_(1)
        self._wp["reset_mask"] = wp.from_torch(self.reset_mask.contiguous())
        self._reset_masked()
        self._observe()
        if self._stream is None:
            wp.synchronize()
        return self.img, self.vec

    def _reset_masked(self):
        w = self._wp
        self._launch(
            K.k_reset,
            self.n,
            [
                w["reset_mask"], int(self.seed0), w["ep"], w["boxes"], w["nbox"],
                w["x"], w["y"], w["yaw"], w["visit"], w["cov"],
                w["path_m"], w["coll"], w["coll_streak"], w["yaw_only"],
                w["steps"], w["t_sim"], w["need_curve"], w["curve_got"],
                w["curve_next"], w["curve_chord"], w["curve_t0"], w["curve_n"],
                w["curve_wins"], w["curve_losses"], w["beads_sum"],
                w["chord_win_sum"], w["last_act"], w["rooms_seen"],
                w["visit_gx"], w["visit_gy"], w["ctrl"], w["poly"], w["beads"],
                w["wall_c"], w["cont_c"], w["done"], w["seg_done"], w["cut"],
                w["cov_cells"],
            ],
        )

    def _observe(self):
        w = self._wp
        self._launch(
            K.k_rs2,
            (self.n, K.DEPTH_H, K.DEPTH_W),
            [w["boxes"], w["x"], w["y"], w["yaw"], w["dx"], w["dy"], w["depth"]],
        )
        self._launch(
            K.k_obs,
            (self.n, K.DEPTH_H, K.DEPTH_W),
            [
                w["boxes"], w["x"], w["y"], w["yaw"], w["depth"], w["visit"],
                w["last_act"], w["plus_x"], w["t_sim"], w["curve_t0"],
                w["curve_got"], w["curve_chord"], w["img"], w["vec"],
                float(self._hp("curve_s", 10.0)),
            ],
        )

    def step(self, act: torch.Tensor):
        """act: [N,4] float32 on GPU. Auto-resets done envs after the step."""
        if act.dtype != torch.float32 or act.device != self.torch_dev:
            act = act.to(device=self.torch_dev, dtype=torch.float32)
        act = act.contiguous()
        act_wp = wp.from_torch(act)
        w = self._wp
        hp = self.hp
        self._launch(
            K.k_step,
            self.n,
            [
                act_wp, w["boxes"], w["x"], w["y"], w["yaw"], w["visit"],
                w["visit_tmp"], w["cov"], w["path_m"], w["coll"], w["coll_streak"],
                w["yaw_only"], w["steps"], w["t_sim"], w["need_curve"],
                w["curve_got"], w["curve_next"], w["curve_chord"], w["curve_t0"],
                w["curve_n"], w["curve_wins"], w["curve_losses"], w["beads_sum"],
                w["chord_win_sum"], w["last_act"], w["visit_gx"], w["visit_gy"],
                w["ctrl"], w["poly"], w["beads"], w["wall_c"], w["cont_c"],
                w["plus_x"], w["rew"], w["done"], w["seg_done"], w["need_out"],
                w["cut"], w["cov_cells"], w["rooms_seen"], w["v_out"], w["w_out"],
                float(hp.get("dt", 0.10)),
                float(hp.get("max_v", 0.28)),
                float(hp.get("max_w", 0.80)),
                float(hp.get("plus_x_stop", 0.28)),
                float(hp.get("curve_r_min", 0.60)),
                float(hp.get("curve_r_max", 2.80)),
                float(hp.get("curve_lat", 1.10)),
                float(hp.get("curve_bead_r", 0.32)),
                float(hp.get("curve_look", 0.45)),
                float(hp.get("curve_s", 10.0)),
                float(hp.get("curve_bead", 0.20)),
                float(hp.get("cover_coef", 1.6)),
                float(hp.get("curve_win", 1.0)),
                float(hp.get("curve_lose", 1.0)),
                float(hp.get("curve_clear", 0.42)),
                float(hp.get("curve_wall_coef", 0.95)),
                float(hp.get("curve_hit_coef", 0.80)),
                float(hp.get("curve_hit_m", 0.14)),
                float(hp.get("curve_cont_s", 3.0)),
                float(hp.get("curve_cont_coef", 0.90)),
                int(hp.get("coll_done", 120)),
                int(hp.get("ep_steps", 600)),
                int(hp.get("yaw_only_max", 40)),
                float(self.cov_cell),
            ],
        )
        self._observe()
        # snapshot reward/done before auto-reset
        rew = self.rew.clone()
        done = self.done.clone()
        seg = self.seg_done.clone()
        need = self.need_out.clone()
        if bool(done.any().item()):
            self.reset_mask.copy_(done)
            self._wp["reset_mask"] = wp.from_torch(self.reset_mask.contiguous())
            self._reset_masked()
            self._observe()
        if self._stream is None:
            wp.synchronize()
        return self.img, self.vec, rew, done, seg, need

    def stats(self) -> dict:
        n = self.curve_n.clamp(min=1).float()
        rooms = self.rooms_seen
        n_rooms = torch.zeros_like(rooms, dtype=torch.float32)
        for b in range(6):
            n_rooms = n_rooms + ((rooms >> b) & 1).float()
        return {
            "win_frac": float((self.curve_wins.float() / n).mean().item()),
            "mean_chord": float((self.chord_win_sum / n).mean().item()),
            "mean_beads": float((self.beads_sum / n).mean().item()),
            "mean_path": float(self.path_m.mean().item()),
            "mean_cells": float(self.cov_cells.float().mean().item()),
            "mean_rooms": float(n_rooms.mean().item()),
            "crash_frac": float((self.cut == 2).float().mean().item()),
            "n_done_eps": int(self.ep.min().item()),
        }

    def env0_cpu(self) -> dict:
        """Tiny D2H for a preview window. Not the hot path."""
        e = 0
        boxes = self.boxes[e].detach().cpu().numpy()
        return {
            "x": float(self.x[e]),
            "y": float(self.y[e]),
            "yaw": float(self.yaw[e]),
            "boxes": boxes,
            "poly": self.poly[e].detach().cpu().numpy(),
            "depth": self.depth[e].detach().cpu().numpy(),
            "img": self.img[e].detach().cpu().numpy(),
            "plus_x": float(self.plus_x[e]),
            "path_m": float(self.path_m[e]),
            "cov_cells": int(self.cov_cells[e]),
            "win": int(self.curve_wins[e]),
            "loss": int(self.curve_losses[e]),
            "need": int(self.need_curve[e]),
            "nbox": int(self.nbox[e]),
        }
