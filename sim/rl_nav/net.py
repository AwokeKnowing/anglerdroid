#!/usr/bin/env python3
"""Tiny CNN + 64-way primitive head + 64-byte memory write. No GRU: memory is env-side."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from . import kernels as K
except ImportError:
    import kernels as K


def _orth(layer, gain=1.0):
    if getattr(layer, "weight", None) is not None:
        nn.init.orthogonal_(layer.weight, gain=gain)
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)


class ActorCritic(nn.Module):
    def __init__(self, ego_h=K.EGO_H, ego_w=K.EGO_W, mem_n=K.MEM_N, n_prim=K.N_PRIM):
        super().__init__()
        self.n_prim = int(n_prim)
        self.mem_n = int(mem_n)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, 1, ego_h, ego_w)).numel()
        self.fc = nn.Linear(n + K.VEC_N + mem_n, 256)
        self.pi = nn.Linear(256, self.n_prim)
        self.write = nn.Linear(256, mem_n)
        self.v = nn.Linear(256, 1)
        self.write_logstd = nn.Parameter(torch.full((mem_n,), -2.2))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                _orth(m, math.sqrt(2))
        _orth(self.pi, 0.01)
        _orth(self.write, 0.01)
        _orth(self.v, 1.0)

    def load_compatible(self, state: dict) -> None:
        """Resume when VEC_N grew. CNN / heads keep weights; extra vec cols start at 0."""
        mine = self.state_dict()
        old_w = state.get("fc.weight")
        new_w = mine["fc.weight"]
        if old_w is not None and tuple(old_w.shape) != tuple(new_w.shape):
            with torch.no_grad():
                n_cnn = int(new_w.shape[1]) - K.VEC_N - self.mem_n
                grown = new_w.new_zeros(new_w.shape)
                n_old = int(old_w.shape[1])
                n_new = int(new_w.shape[1])
                old_vec = n_old - n_cnn - self.mem_n
                new_vec = n_new - n_cnn - self.mem_n
                grown[:, :n_cnn] = old_w[:, :n_cnn]
                nv = min(old_vec, new_vec)
                if nv > 0:
                    grown[:, n_cnn : n_cnn + nv] = old_w[:, n_cnn : n_cnn + nv]
                grown[:, n_cnn + new_vec :] = old_w[:, n_cnn + old_vec :]
                state = dict(state)
                state["fc.weight"] = grown
        self.load_state_dict(state, strict=True)

    def encode(self, ego, vec):
        if ego.dim() == 3:
            ego = ego.unsqueeze(1)
        z = self.cnn(ego)
        z = z.flatten(1)
        h = torch.relu(self.fc(torch.cat([z, vec], dim=-1)))
        return h

    def act(self, ego, vec, deterministic=False):
        h = self.encode(ego, vec)
        logits = self.pi(h)
        dist = torch.distributions.Categorical(logits=logits)
        prim = logits.argmax(-1) if deterministic else dist.sample()
        mu = torch.tanh(self.write(h))
        std = self.write_logstd.exp().clamp(0.02, 0.25)
        wdist = torch.distributions.Normal(mu, std)
        write = mu if deterministic else wdist.rsample()
        write = write.clamp(-1.0, 1.0)
        logp = dist.log_prob(prim) + (1.0 / float(self.mem_n)) * wdist.log_prob(write).sum(-1)
        value = self.v(h).squeeze(-1)
        return prim, write, logp, value

    def evaluate(self, ego, vec, prim, write):
        h = self.encode(ego, vec)
        logits = self.pi(h)
        dist = torch.distributions.Categorical(logits=logits)
        mu = torch.tanh(self.write(h))
        std = self.write_logstd.exp().clamp(0.02, 0.25)
        wdist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(prim) + (1.0 / float(self.mem_n)) * wdist.log_prob(write.clamp(-1.0, 1.0)).sum(-1)
        ent = dist.entropy()
        value = self.v(h).squeeze(-1)
        return logp, ent, value
