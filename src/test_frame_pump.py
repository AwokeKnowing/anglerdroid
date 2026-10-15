"""FramePump + wait_new: camera clock, newest seq, no busy-spin."""
from __future__ import annotations

import time

from cameras import FramePump


class FakeCam:
    def __init__(self, period=0.01):
        self.period = period
        self.pub_seq = 0
        self.pub_t = 0.0
        self.n = 0
        self._stop = False

    def grab_latest(self, timeout_ms=None):
        if self._stop:
            return False
        time.sleep(self.period)
        self.n += 1
        self.pub_seq += 1
        self.pub_t = time.monotonic()
        return True


def test_pump_publishes_new_seq():
    cam = FakeCam(0.008)
    p = FramePump(cam, "fake").start()
    assert p.wait_new(0, 0.2)
    s1 = p.seq
    assert s1 >= 1
    assert p.wait_new(s1, 0.2)
    assert p.seq > s1
    p.stop()
    cam._stop = True


def test_wait_new_times_out():
    cam = FakeCam(2.0)
    p = FramePump(cam, "slow").start()
    t0 = time.monotonic()
    assert p.wait_new(0, 0.04) is False
    dt = time.monotonic() - t0
    assert 0.03 <= dt < 0.20
    p.stop()
    cam._stop = True


def test_process_overlaps_next_wait():
    """While the consumer 'processes', the pump keeps grabbing."""
    cam = FakeCam(0.01)
    p = FramePump(cam, "ov").start()
    assert p.wait_new(0, 0.2)
    s0 = p.seq
    time.sleep(0.035)
    assert p.seq > s0
    p.stop()
    cam._stop = True


if __name__ == "__main__":
    test_pump_publishes_new_seq()
    test_wait_new_times_out()
    test_process_overlaps_next_wait()
    print("ok")
