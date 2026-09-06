"""Offline unit test for KevinRerunLogger — no cameras, no drive."""

from __future__ import annotations

import os
import tempfile

import numpy as np


def test_rerun_logger_writes_rgb_depth_obs():
    from rerun_log import KevinRerunLogger, available

    assert available(), "rerun-sdk must be installed on Kevin"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "kevin_test.rrd")
        log = KevinRerunLogger(enabled=True, save_path=path, every_n=1, spawn=False)
        assert log.enabled
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        rgb[40:80, 60:100] = (20, 180, 40)
        obs = np.zeros((240, 320), dtype=np.uint8)
        obs[100:140, 150:200] = 255
        height = np.zeros((240, 320), dtype=np.uint8)
        height[100:140, 150:200] = 45
        atlas = np.zeros((480, 640, 3), dtype=np.uint8)
        ok = log.maybe_log(
            ts=1_700_000_000.0,
            atlas=atlas,
            rgb=rgb,
            rs1=rgb,
            rs2=rgb,
            obs=obs,
            height_cm=height,
            safety={"fwd": 0.4, "bwd": 1.0, "ang": 0.7},
            force=True,
        )
        assert ok, "expected a log write"
        # Flush by dropping logger / process end — file should exist and grow.
        assert os.path.isfile(path), path
        assert os.path.getsize(path) > 500, "rrd too small: %d" % os.path.getsize(path)


def test_rerun_logger_throttle():
    from rerun_log import KevinRerunLogger

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "throttle.rrd")
        log = KevinRerunLogger(enabled=True, save_path=path, every_n=3, spawn=False)
        img = np.zeros((24, 32, 3), dtype=np.uint8)
        wrote = [
            log.maybe_log(ts=1.0 + i, rgb=img, obs=np.zeros((24, 32), dtype=np.uint8))
            for i in range(6)
        ]
        # ticks 3 and 6 (1-indexed) → indices where _n % 3 == 0
        assert wrote == [False, False, True, False, False, True], wrote


if __name__ == "__main__":
    test_rerun_logger_writes_rgb_depth_obs()
    test_rerun_logger_throttle()
    print("test_rerun_log: OK")
