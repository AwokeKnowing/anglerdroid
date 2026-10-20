"""nav/lslam unit tests (no sim):  PYTHONPATH=src:. python tests/test_lslam.py   (or pytest)"""
import math, numpy as np
from nav.lslam import model as M
from nav.lslam.journal import Journal
from nav.lslam.submap import Submap
from nav.lslam.core import LifelongSlam, TRACKING, DEAD_RECKONING


def _room(n=400, rng=np.random.default_rng(0)):
    s = rng.uniform(-2, 2, n); return np.concatenate([np.stack([s, np.full(n, 2.0)], 1), np.stack([np.full(n, 2.0), s], 1),
                                                      np.stack([s, np.full(n, -2.0)], 1), np.stack([np.full(n, -1.5), s], 1)]).astype(np.float32)


def _scan(pose, W):   # world points -> robot frame
    c, s = math.cos(pose[2]), math.sin(pose[2]); d = W - pose[:2]; return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1).astype(np.float32)


def test_gate_and_compose():
    assert abs(M.gate(3) - 11.345) < 1e-2 and abs(M.gate(1) - 6.635) < 1e-2
    a = np.array([1.0, 2.0, 0.7]); b = np.array([-0.3, 0.4, -1.2])
    assert np.allclose(M.between(a, M.compose(a, b)), b)


def test_rebuild_is_exact():
    j = Journal(); sm = Submap(0, (0, 0, 0)); W = _room(); rng = np.random.default_rng(1)
    for k in range(30):
        p = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-1, 1)]); pts = _scan(p, W)
        j.append(k, 0, p, pts); sm.integrate(p, pts)
    L = sm.g.L.copy(); j.truncate(20); sm.rebuild(j.of(0))
    ref = Submap(0, (0, 0, 0))
    for e in j.e: ref.integrate(e.pose_sub, e.pts)
    assert np.array_equal(sm.g.L, ref.g.L) and not np.array_equal(sm.g.L, L) and sm.n == 20


def test_rejects_inconsistent_scan_and_recovers():
    W = _room(); om = M.OdomModel(0.02, 1e-3, 0.02, 1e-3); R = np.diag([0.01 ** 2, 0.01 ** 2, math.radians(0.5) ** 2])
    s = LifelongSlam(om, R); x = np.zeros(3)
    for k in range(40):                                  # straight drive, perfect odom
        s.step(k * 0.1, x, _scan(x, W)); x = x + [0.02, 0, 0]
    assert s.state == TRACKING and s.stats["rej"] == 0
    bad = _scan(x + [0.4, 0.3, 0.3], W)                  # scan from a wrong place
    x = x - [0.02, 0, 0]; s.step(4.0, x, bad); assert s.state == DEAD_RECKONING and s.stats["rej"] == 1
    for k in range(5): x = x + [0.02, 0, 0]; s.step(4.1 + k * 0.1, x, _scan(x, W))
    assert s.state == TRACKING and np.linalg.norm(s.x[:2] - x[:2]) < 0.03


if __name__ == "__main__":
    import traceback
    for n in [k for k in sorted(globals()) if k.startswith("test_")]:
        try: globals()[n](); print("PASS", n)
        except Exception: print("FAIL", n); traceback.print_exc()
