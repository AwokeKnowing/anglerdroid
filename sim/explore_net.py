#!/usr/bin/env python3
"""Tiny CNN + GRU actor-critic and RND novelty on CUDA."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_(layer, gain=1.0):
    if getattr(layer, "weight", None) is not None:
        nn.init.orthogonal_(layer.weight, gain=gain)
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)


class ConvTower(nn.Module):
    def __init__(self, ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, 16, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            n = self.net(torch.zeros(1, ch, 48, 80)).numel()
        self.out_dim = int(n)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                orthogonal_(m, gain=math.sqrt(2))

    def forward(self, x):
        return self.net(x).flatten(1)


class ActorCritic(nn.Module):
    def __init__(self, vec_dim=10, hidden=128, act_dim=4):
        super().__init__()
        self.act_dim = int(act_dim)
        self.cnn = ConvTower(3)
        self.fc = nn.Linear(self.cnn.out_dim + vec_dim, 256)
        self.gru = nn.GRU(256, hidden, batch_first=True)
        self.pi_mu = nn.Linear(hidden, self.act_dim)
        self.pi_logstd = nn.Parameter(torch.full((self.act_dim,), -0.45))
        self.v = nn.Linear(hidden, 1)
        orthogonal_(self.fc, math.sqrt(2))
        orthogonal_(self.pi_mu, 0.01)
        orthogonal_(self.v, 1.0)
        self.yaw_mu_max = 1.0

    def encode(self, img, vec, hx=None):
        z = torch.relu(self.fc(torch.cat([self.cnn(img), vec], dim=-1)))
        z = z.unsqueeze(1)
        out, hx = self.gru(z, hx)
        return out.squeeze(1), hx

    def _mu(self, h):
        mu = torch.tanh(self.pi_mu(h))
        cap = float(getattr(self, "yaw_mu_max", 1.0))
        if self.act_dim == 2 and cap < 0.999:
            mu = torch.cat([mu[..., :1], mu[..., 1:2].clamp(-cap, cap)], dim=-1)
        return mu

    def _std(self):
        std = self.pi_logstd.exp()
        if self.act_dim == 2:
            return torch.stack([std[0].clamp(0.05, 0.70), std[1].clamp(0.04, 0.22)])
        return std.clamp(0.05, 0.55)

    def act(self, img, vec, hx=None, deterministic=False):
        h, hx = self.encode(img, vec, hx)
        mu = self._mu(h)
        dist = torch.distributions.Normal(mu, self._std())
        act = mu if deterministic else dist.sample()
        act = act.clamp(-1.0, 1.0)
        logp = dist.log_prob(act).sum(-1)
        value = self.v(h).squeeze(-1)
        return act, logp, value, hx

    def evaluate(self, img, vec, act, hx=None):
        h, hx = self.encode(img, vec, hx)
        mu = self._mu(h)
        dist = torch.distributions.Normal(mu, self._std())
        logp = dist.log_prob(act.clamp(-1.0, 1.0)).sum(-1)
        ent = dist.entropy().sum(-1)
        value = self.v(h).squeeze(-1)
        return logp, ent, value, hx


class RND(nn.Module):
    """Frozen random target vs predictor on pooled depth (channel 0)."""

    def __init__(self):
        super().__init__()
        self.target = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.target(torch.zeros(1, 1, 48, 80)).numel()
        self.out_dim = int(n)
        self.predictor = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(self.out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.out_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False

    def novelty(self, img):
        depth = img[:, :1]
        with torch.no_grad():
            tgt = self.target(depth)
        pred = self.predictor(depth)
        err = F.mse_loss(pred, tgt, reduction="none").mean(dim=1)
        return err

    def embed(self, img):
        depth = img[:, :1]
        with torch.no_grad():
            z = self.target(depth)
        return torch.nn.functional.normalize(z, dim=-1)


class EpisodicMem:
    """NGU-style episode memory: spinning re-visits embeddings, novelty dies."""

    def __init__(self, cap=192):
        self.cap = int(cap)
        self.buf = None
        self.n = 0
        self.i = 0

    def reset(self):
        self.n = 0
        self.i = 0

    def novelty(self, z):
        z = z.detach().reshape(1, -1)
        z = torch.nn.functional.normalize(z, dim=-1)
        if self.buf is None or self.buf.shape[-1] != z.shape[-1]:
            self.buf = z.new_zeros(self.cap, z.shape[-1])
            self.n = 0
            self.i = 0
        if self.n == 0:
            self.buf[0] = z
            self.n = 1
            self.i = 1
            return 1.0
        sim = torch.mm(self.buf[:self.n], z.t()).max().clamp(0.0, 1.0)
        nov = float(1.0 - sim)
        self.buf[self.i % self.cap] = z
        self.i += 1
        self.n = min(self.cap, self.n + 1)
        return nov


def obs_to_torch(obs, device):
    img = torch.from_numpy(np.asarray(obs["img"], dtype=np.float32)).unsqueeze(0).to(device)
    vec = torch.from_numpy(np.asarray(obs["vec"], dtype=np.float32)).unsqueeze(0).to(device)
    return img, vec
