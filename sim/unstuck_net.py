#!/usr/bin/env python3
"""Compact CNN+GRU (v,w) policy for plus-x unstuck. ~35k params."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

IMG_H, IMG_W = 48, 80
VEC_DIM = 8
ACT_DIM = 2
# Ego crop in metres relative to axle (matches Kevin 1 cm map crop).
BACK_M, FWD_M, LAT_M = 0.80, 2.40, 0.96


def orthogonal_(layer, gain=1.0):
    if getattr(layer, "weight", None) is not None:
        nn.init.orthogonal_(layer.weight, gain=gain)
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)


class TinyUnstuck(nn.Module):
    def __init__(self, vec_dim=VEC_DIM, hidden=48, act_dim=ACT_DIM):
        super().__init__()
        self.act_dim = int(act_dim)
        self.vec_dim = int(vec_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 6)),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, 1, IMG_H, IMG_W)).numel()
        self.cnn_dim = int(n)
        self.fc = nn.Linear(self.cnn_dim + self.vec_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.pi_mu = nn.Linear(hidden, self.act_dim)
        self.pi_logstd = nn.Parameter(torch.tensor([-0.35, -0.25]))
        self.v = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                orthogonal_(m, math.sqrt(2))
        orthogonal_(self.fc, math.sqrt(2))
        orthogonal_(self.pi_mu, 0.01)
        orthogonal_(self.v, 1.0)

    def n_params(self):
        return int(sum(p.numel() for p in self.parameters()))

    def encode(self, img, vec, hx=None):
        z = torch.relu(self.fc(torch.cat([self.cnn(img).flatten(1), vec], dim=-1)))
        out, hx = self.gru(z.unsqueeze(1), hx)
        return out.squeeze(1), hx

    def _std(self):
        std = self.pi_logstd.exp()
        return torch.stack([std[0].clamp(0.08, 0.55), std[1].clamp(0.10, 0.50)])

    def act(self, img, vec, hx=None, deterministic=False):
        h, hx = self.encode(img, vec, hx)
        mu = torch.tanh(self.pi_mu(h))
        dist = torch.distributions.Normal(mu, self._std())
        a = mu if deterministic else dist.sample()
        a = a.clamp(-1.0, 1.0)
        logp = dist.log_prob(a).sum(-1)
        value = self.v(h).squeeze(-1)
        return a, logp, value, hx

    def evaluate(self, img, vec, act, hx=None):
        h, hx = self.encode(img, vec, hx)
        mu = torch.tanh(self.pi_mu(h))
        dist = torch.distributions.Normal(mu, self._std())
        a = act.clamp(-1.0, 1.0)
        logp = dist.log_prob(a).sum(-1)
        ent = dist.entropy().sum(-1)
        value = self.v(h).squeeze(-1)
        return logp, ent, value, hx


def obs_to_torch(obs, device):
    img = torch.from_numpy(np.asarray(obs["img"], dtype=np.float32)).unsqueeze(0).to(device)
    vec = torch.from_numpy(np.asarray(obs["vec"], dtype=np.float32)).unsqueeze(0).to(device)
    return img, vec
