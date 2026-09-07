"""Offline unit test for KevinRerunLogger — no cameras, no drive."""

from __future__ import annotations

import os
import time
import tempfile
import threading

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
        # Give background thread time to process
        time.sleep(0.3)
        log.shutdown()
        # File should exist and grow.
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
        log.shutdown()


def test_rerun_logger_nonblocking_under_backpressure():
    """Verify maybe_log returns quickly even when background thread is slow/blocked.
    
    Simulates backpressure by rapidly calling maybe_log many times (filling queue)
    while background thread is processing. If maybe_log blocked on rr.log calls,
    this would take >>100ms. With async queue, should complete in <50ms.
    """
    from rerun_log import KevinRerunLogger

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "nonblock.rrd")
        log = KevinRerunLogger(enabled=True, save_path=path, every_n=1, spawn=False)
        
        # Create realistic-sized frames (similar to actual usage)
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        obs = np.zeros((240, 320), dtype=np.uint8)
        atlas = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Rapidly call maybe_log 20 times (should fill queue and trigger drops)
        start = time.monotonic()
        for i in range(20):
            log.maybe_log(
                ts=1_700_000_000.0 + i,
                atlas=atlas,
                rgb=rgb,
                obs=obs,
                force=True,
            )
        elapsed_ms = (time.monotonic() - start) * 1000.0
        
        # All calls should return immediately - even with drops, <50ms total
        # (If it blocked on gRPC, would be >>100ms for 20 frames)
        assert elapsed_ms < 50.0, f"maybe_log blocked: {elapsed_ms:.1f}ms for 20 calls"
        
        # Verify drops occurred (queue maxsize=2, so >2 rapid calls should drop)
        assert log._drop_count > 0, "Expected frame drops when filling queue rapidly"
        
        # Give background thread time to drain queue
        time.sleep(0.5)
        log.shutdown()
        
        # File should exist and have some data (not all frames, due to drops)
        assert os.path.isfile(path), path
        assert os.path.getsize(path) > 500, "rrd too small: %d" % os.path.getsize(path)
        
        print(f"test_rerun_logger_nonblocking: OK ({elapsed_ms:.1f}ms for 20 calls, {log._drop_count} drops)")


if __name__ == "__main__":
    test_rerun_logger_writes_rgb_depth_obs()
    test_rerun_logger_throttle()
    test_rerun_logger_nonblocking_under_backpressure()
    print("test_rerun_log: ALL TESTS PASSED")
