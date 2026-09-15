"""Live 5 Hz PPO from sim/rl_nav — 80×60 + 64 prims + 64-byte mem.

Loads a torch ckpt.pt when torch is present (i777). On Orin, load the
exported ~/.kevin/rl_nav.onnx with onnxruntime instead.
"""
from __future__ import annotations

import os
import time

import numpy as np

EGO_H, EGO_W = 60, 80
MEM_N = 64
N_PRIM = 64
N_V, N_W = 8, 8
VEC_N = 7
V_MAX, W_MAX, V_REV = 0.25, 0.80, -0.20
A_MAX, ALPHA_MAX = 1.61, 4.0
DT_POLICY = 0.20
EP_S = 1200.0
EGO80_PX = 0.04
EGO80_RCX = 81.0 / 4.0
EGO80_RCY = 119.0 / 4.0
EGO80_X0 = -EGO80_RCX * EGO80_PX
EGO80_Y1 = EGO80_RCY * EGO80_PX


def _clip_vw(v, w, vmax, wmax):
    vv = float(np.clip(v, V_REV, vmax))
    ww = float(np.clip(w, -wmax, wmax))
    denom = vmax if vmax > 1e-6 else 1e-6
    frac = min(1.0, abs(vv) / denom)
    w_allow = wmax * (1.0 - 0.15 * frac)
    ww = float(np.clip(ww, -w_allow, w_allow))
    return vv, ww


def _prim_vw(prim, v, w, vmax, wmax, a_scale=1.0):
    dv = max(A_MAX * a_scale * DT_POLICY, 0.05)
    dw = max(ALPHA_MAX * a_scale * DT_POLICY, 0.20)
    v_lo = float(np.clip(v - dv, V_REV, vmax))
    v_hi = float(np.clip(v + dv, V_REV, vmax))
    w_lo = float(np.clip(w - dw, -wmax, wmax))
    w_hi = float(np.clip(w + dw, -wmax, wmax))
    if v_hi < v_lo:
        v_lo, v_hi = v_hi, v_lo
    if w_hi < w_lo:
        w_lo, w_hi = w_hi, w_lo
    a = int(np.clip(prim, 0, N_PRIM - 1))
    iv = a // N_W
    iw = a - iv * N_W
    fv = float(iv) / float(N_V - 1)
    fw = float(iw) / float(N_W - 1)
    v_cmd = v_lo + fv * (v_hi - v_lo)
    w_cmd = w_lo + fw * (w_hi - w_lo)
    return _clip_vw(v_cmd, w_cmd, vmax, wmax)


def _plus_x(ego):
    xs = (np.arange(EGO_W, dtype=np.float32) + 0.5) * EGO80_PX + EGO80_X0
    ys = EGO80_Y1 - (np.arange(EGO_H, dtype=np.float32) + 0.5) * EGO80_PX
    xx, yy = np.meshgrid(xs, ys)
    hit = (ego >= 0.85) & (xx > 0.08) & (np.abs(yy) < 0.18)
    if not np.any(hit):
        return 2.0
    return float(np.min(xx[hit]))


class RlNavLive:
    def __init__(self, model_path):
        self.mem = np.zeros((MEM_N,), dtype=np.float32)
        self.v = 0.0
        self.w = 0.0
        self.t0 = time.monotonic()
        self._last_t = 0.0
        self._last_cmd = (0.0, 0.0)
        self._n = 0
        self._backend = None
        self._sess = None
        self._torch = None
        self.net = None
        path = os.path.expanduser(str(model_path))
        if path.endswith(".onnx"):
            self._init_onnx(path)
            meta = "onnx"
        else:
            try:
                meta = self._init_torch(path)
            except ImportError:
                alt = os.path.splitext(path)[0] + ".onnx"
                if not os.path.isfile(alt):
                    alt = os.path.expanduser("~/.kevin/rl_nav.onnx")
                if not os.path.isfile(alt):
                    raise
                self._init_onnx(alt)
                path = alt
                meta = "onnx-fallback"
        print("rl_nav_live: loaded %s backend=%s %s" % (
            path, self._backend, meta), flush=True)

    def _init_onnx(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            path, sess_options=so, providers=["CPUExecutionProvider"])
        self._backend = "onnx"

    def _init_torch(self, path):
        import torch
        self._torch = torch
        self.net = self._build(torch)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        sd = blob.get("net", blob)
        if hasattr(self.net, "load_compatible"):
            self.net.load_compatible(sd)
        else:
            self.net.load_state_dict(sd, strict=False)
        self.net.eval()
        self._backend = "torch"
        return "step=%s stage=%s" % (blob.get("step"), blob.get("stage"))

    @staticmethod
    def _build(torch):
        import torch.nn as nn
        from pathlib import Path
        import sys
        root = Path(__file__).resolve().parents[1]
        sim = root / "sim" / "rl_nav"
        if sim.is_dir() and str(sim) not in sys.path:
            sys.path.insert(0, str(sim))
        try:
            from net import ActorCritic
            return ActorCritic()
        except Exception:
            pass

        class ActorCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_prim = N_PRIM
                self.mem_n = MEM_N
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 16, 5, stride=2, padding=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
                with torch.no_grad():
                    n = self.cnn(torch.zeros(1, 1, EGO_H, EGO_W)).numel()
                self.fc = nn.Linear(n + VEC_N + MEM_N, 256)
                self.pi = nn.Linear(256, N_PRIM)
                self.write = nn.Linear(256, MEM_N)
                self.v = nn.Linear(256, 1)
                self.write_logstd = nn.Parameter(torch.full((MEM_N,), -2.2))

            def encode(self, ego, vec):
                if ego.dim() == 3:
                    ego = ego.unsqueeze(1)
                z = self.cnn(ego).flatten(1)
                return torch.relu(self.fc(torch.cat([z, vec], dim=-1)))

            def act(self, ego, vec, deterministic=True):
                h = self.encode(ego, vec)
                logits = self.pi(h)
                write = torch.tanh(self.write(h)).clamp(-1.0, 1.0)
                prim = logits.argmax(-1)
                return prim, write, None, None

            def load_compatible(self, state):
                self.load_state_dict(state, strict=False)

        return ActorCritic()

    def _infer(self, ego, vec):
        if self._backend == "onnx":
            logits, write = self._sess.run(
                ["logits", "write"],
                {
                    "ego": np.ascontiguousarray(ego[None, None], dtype=np.float32),
                    "vec": np.ascontiguousarray(vec[None], dtype=np.float32),
                },
            )
            return int(np.argmax(logits[0])), np.clip(write[0], -1.0, 1.0).astype(np.float32)
        torch = self._torch
        with torch.no_grad():
            te = torch.from_numpy(np.ascontiguousarray(ego[None, None]))
            tv = torch.from_numpy(np.ascontiguousarray(vec[None]))
            prim, write, _, _ = self.net.act(te, tv, deterministic=True)
            return int(prim.item()), write[0].cpu().numpy().astype(np.float32)

    def act(self, ego, *, v_scale=1.0, w_scale=1.0, plus_x=None):
        now = time.monotonic()
        if now - self._last_t < 0.16 and self._n:
            return self._last_cmd
        ego = np.array(ego, dtype=np.float32, copy=True)
        if ego.ndim == 3:
            ego = ego[0]
        if ego.shape != (EGO_H, EGO_W):
            raise ValueError("ego %s != 60x80" % (ego.shape,))
        if plus_x is None:
            plus_x = _plus_x(ego)
        t_sim = min(EP_S, now - self.t0)
        vec = np.zeros((VEC_N + MEM_N,), dtype=np.float32)
        vec[0] = self.v / V_MAX
        vec[1] = self.w / W_MAX
        vec[2] = plus_x / 2.0
        vec[3] = t_sim / EP_S
        vec[4] = 0.0
        vec[5] = float(np.clip(v_scale, 0.0, 1.0))
        vec[6] = float(np.clip(w_scale, 0.0, 1.0))
        vec[VEC_N:] = self.mem
        prim_i, wr = self._infer(ego, vec)
        self.mem = np.clip(wr, -1.0, 1.0)
        # Prim band is full V_MAX/W_MAX. Vision safety scales the twist once
        # in wheelbase — do not bake scale into vmax here or a pin idles the
        # ODrives (blue) after the 5 s zero-vel watcher.
        self.v, self.w = _prim_vw(prim_i, self.v, self.w, V_MAX, W_MAX)
        self._last_t = now
        self._last_cmd = (float(self.v), float(self.w))
        self._n += 1
        if self._n <= 3 or self._n % 25 == 0:
            print("rl_nav_live: n=%d prim=%d v=%.3f w=%.3f plus_x=%.2f" % (
                self._n, prim_i, self.v, self.w, plus_x), flush=True)
        return self._last_cmd
