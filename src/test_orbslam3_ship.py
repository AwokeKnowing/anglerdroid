"""Protocol tests for ORB-SLAM3 IR ship (no camera, no C++)."""
import numpy as np

from orbslam3_ship import pack_stereo, unpack_stereo


def test_pack_roundtrip():
    left = np.arange(12, dtype=np.uint8).reshape(3, 4)
    right = (left * 2).astype(np.uint8)
    blob = pack_stereo(123456789, left, right, 424.0, 425.0, 200.0, 100.0, 0.05)
    ts, l2, r2, fx, fy, cx, cy, b = unpack_stereo(blob)
    assert ts == 123456789
    assert l2.shape == (3, 4)
    assert np.array_equal(l2, left)
    assert np.array_equal(r2, right)
    assert abs(fx - 424.0) < 1e-5
    assert abs(b - 0.05) < 1e-6


if __name__ == "__main__":
    test_pack_roundtrip()
    print("test_orbslam3_ship: OK")
