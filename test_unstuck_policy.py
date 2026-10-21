"""test_unstuck_policy.py — compact unstuck net + pin-spawn env.

Run: .venv-newton/bin/python test_unstuck_policy.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "sim"))

from unstuck_policy import UnstuckPolicy, crop_kevin_ego, _look_px  # noqa: E402
from robot_config import RCX, RCY, FRAME_H, FRAME_W  # noqa: E402


def test_scripted_turns_from_pin():
    p = UnstuckPolicy(ckpt="/no/such.pt")
    v, w, src = p.act(0.14, 0.90, 0.20, 0.0, 0.0)
    assert src == "scripted"
    assert v <= 0.02, v
    assert w > 0.3, "more open left → +w, got %s" % w
    v2, w2, _ = p.act(0.14, 0.20, 0.90, 0.0, 0.0)
    assert w2 < -0.3, w2
    print("scripted L w=%.2f R w=%.2f" % (w, w2))


def test_crop_sees_forward_wall():
    h = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    col = RCX + 20
    h[:, col:col + 4] = 40
    crop = crop_kevin_ego(h)
    assert crop.shape == (48, 80)
    # axle col in crop is BACK_M / ((BACK+FWD)/80) = 20
    assert crop[:, 25:].mean() > crop[:, :15].mean()
    print("crop fwd mean=%.2f rear=%.2f" % (crop[:, 25:].mean(), crop[:, :15].mean()))


def test_look_px_wall():
    h = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    h[:, RCX + 18:RCX + 22] = 40
    d = _look_px(h, None, RCX, RCY, 0.0, 1.6)
    assert 0.12 < d < 0.28, d
    print("look_px %.3f" % d)


def test_tiny_net_forward():
    import torch
    from unstuck_net import TinyUnstuck
    net = TinyUnstuck()
    n = net.n_params()
    assert n < 40_000, n
    img = torch.zeros(2, 1, 48, 80)
    vec = torch.zeros(2, 8)
    act, logp, val, hx = net.act(img, vec)
    assert act.shape == (2, 2)
    print("TinyUnstuck params=%d act=%s" % (n, tuple(act.shape)))


def test_pin_env_scripted_can_clear():
    from unstuck_env import UnstuckEnv
    env = UnstuckEnv(hp={"goal_p": 0.0, "rebuild_every": 99}, seed=3)
    pol = UnstuckPolicy(ckpt="/no/such.pt")
    wins = 0
    for ep in range(8):
        obs, info = env.reset()
        assert 0.05 <= float(info["plus_x"]) <= 0.35, info
        hx_ok = False
        for _ in range(80):
            v, w, _ = pol.act(
                float(obs["plus_x"]), float(obs["left"]), float(obs["right"]),
                env.exe_v, env.exe_w)
            max_v = 0.28
            max_w = 0.80
            a = [v / max_v, w / max_w]
            obs, rew, done, info = env.step(a)
            if done:
                wins += int(bool(info.get("win")))
                hx_ok = True
                break
        assert hx_ok
    print("scripted pin wins %d/8 start_px=%.2f" % (wins, env.plus_x))
    assert wins >= 2, "task should be solvable by the tiny fallback"


def test_pixel_to_floor_center():
    from house_view import pixel_to_floor_xy
    x, y = pixel_to_floor_xy(400, 260, 800, 520)
    assert abs(x - 0.12) < 1.5, (x, y)
    assert abs(y - 0.40) < 2.5, (x, y)
    print("click center → (%.2f, %.2f)" % (x, y))


if __name__ == "__main__":
    test_scripted_turns_from_pin()
    test_crop_sees_forward_wall()
    test_look_px_wall()
    test_tiny_net_forward()
    test_pin_env_scripted_can_clear()
    test_pixel_to_floor_center()
    print("ok")
