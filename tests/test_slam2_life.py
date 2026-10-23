"""nav/slam2 lifelong-mode back end tests (no sim, no GPU):  PYTHONPATH=src:. python tests/test_slam2_life.py  (or pytest)
Pose graph GN + Dynamic Covariance Scaling (Agarwal 2013, Phi=1): a correct loop fixes drift, a false loop is
down-weighted instead of bending the map, and the solve stays bounded on a revisit-heavy graph."""
import math, time, numpy as np
from nav.slam import compose, inverse, wrap
from nav import slam2 as S

ODO_INF = np.array([1 / 0.02 ** 2, 1 / 0.02 ** 2, 1 / math.radians(1.0) ** 2])
LOOP_INF = np.array([1 / 0.03 ** 2, 1 / 0.03 ** 2, 1 / math.radians(1.0) ** 2])


def _square(laps=1, side=4, step=0.5, yaw_bias=math.radians(0.05)):
    """GT square path + odometry with a per-step yaw bias (wheel-yaw scale error)."""
    gt = [(0.0, 0.0, 0.0)]
    for _ in range(laps * 4):
        for k in range(int(side / step)):
            x, y, th = gt[-1]; gt.append((x + step * math.cos(th), y + step * math.sin(th), th))
        x, y, th = gt[-1]; gt[-1] = (x, y, wrap(th + math.pi / 2))
    od = [compose(inverse(gt[i]), gt[i + 1]) for i in range(len(gt) - 1)]
    od = [(z[0], z[1], z[2] + yaw_bias) for z in od]
    est = [gt[0]]
    for z in od: est.append(compose(est[-1], z))
    edges = [(i, i + 1, z, ODO_INF) for i, z in enumerate(od)]
    return gt, est, edges


def _ate(a, b): return float(np.sqrt(np.mean(np.sum((np.array(a)[:, :2] - np.array(b)[:, :2]) ** 2, 1))))


# NOTE (measured 2026-10-22): DCS weighs a loop by its chi2 against the loop edge's own information, not against the
# chain's propagated covariance, so with Phi=1 a TRUE loop closing > ~10 cm / ~2 deg of drift is suppressed
# (16 m square, 0.15 deg/step yaw bias: scale 0.03, ATE 0.126 -> 0.113 m; plain GN 0.004 m).  slam2 LIFE closes small
# drifts continuously (re-anchor edges) so this has not bitten in the bench (loops_downweighted 0), but the candidate
# fix is GNC (Yang et al. 2020) or a PCM gate on the marginal covariance.  The tests pin the regime slam2 runs in.
def test_true_loop_fixes_drift():
    gt, est, edges = _square()
    n = len(gt) - 1; z = compose(inverse(gt[0]), gt[n])
    out = S.optimize(est, edges + [(0, n, z, LOOP_INF)], robust=[False] * len(edges) + [True])
    assert _ate(out, gt) < 0.5 * _ate(est, gt), (_ate(out, gt), _ate(est, gt))
    assert S.optimize.last_scale[-1] > 0.5          # the true loop is kept at (near) full weight


def test_false_loop_downweighted():
    gt, est, edges = _square()
    n = len(gt) - 1; zt = compose(inverse(gt[0]), gt[n])
    j = len(gt) // 2                                 # wrong place recognition: node j "is" node 0, 4 m away
    zf = (0.1, 0.0, 0.0)
    E = edges + [(0, n, zt, LOOP_INF), (0, j, zf, LOOP_INF)]
    rb = [False] * len(edges) + [True, True]
    out = S.optimize(est, E, robust=rb, iters=30)
    sc = S.optimize.last_scale
    assert sc[-1] < 0.1, sc[-1]                      # false loop switched off
    assert _ate(out, gt) < 0.5 * _ate(est, gt)       # map still corrected by the true loop
    plain = S.optimize(est, E, robust=None, iters=30)   # without DCS the false loop bends the map
    assert _ate(plain, gt) > 2 * _ate(out, gt), (_ate(plain, gt), _ate(out, gt))


def test_bounded_solve_many_revisits():
    """Reduced pose graph: 20 laps over the same nodes add edges, not nodes; the solve stays fast."""
    gt, est, edges = _square()
    n = len(gt) - 1
    E = list(edges); rb = [False] * len(E)
    rng = np.random.default_rng(0)
    for _ in range(20 * n):                          # re-anchor / loop edges between existing nodes
        i, j = sorted(rng.choice(n + 1, 2, replace=False))
        z = compose(inverse(gt[i]), gt[j]); E.append((int(i), int(j), z, LOOP_INF)); rb.append(True)
    t0 = time.perf_counter(); out = S.optimize(est, E, robust=rb); dt = time.perf_counter() - t0
    assert len(out) == n + 1 and _ate(out, gt) < 0.05 and dt < 2.0, (_ate(out, gt), dt)


if __name__ == "__main__":
    for f in (test_true_loop_fixes_drift, test_false_loop_downweighted, test_bounded_solve_many_revisits): f(); print("ok", f.__name__)
