"""nav/slam2 asynchronous loop closure (KEVIN_SLAM2_ASYNC_LC) tests:  PYTHONPATH=src:. python tests/test_slam2_async.py
The loop search is stubbed (it returns the true relative pose), so this checks the back-end plumbing, not the matcher:
  * the async result equals the synchronous one on the snapshot nodes, and nodes added while the solve ran are carried
    rigidly by the correction of the node they grew from;
  * the matching field ends exactly as if built from scratch at the final poses (no stale / doubled scan counts);
  * the deterministic lag mode (KEVIN_SLAM2_ASYNC_LAG) is reproducible and equals the free-running thread's result;
  * a reset while a solve is in flight drops its result; the front end keeps running while the worker searches."""
import math, threading, time, collections, numpy as np
from nav.slam import compose, inverse, wrap, GridMap
from nav import slam2 as S


def _graph(s, n_nodes=34, side=4.0, step=0.5, yaw_bias=math.radians(0.1), seed=0):
    """A drifting square lap as a LIFE-mode graph: node i at the drifted pose, chain edges from (drifted) odometry."""
    rng = np.random.default_rng(seed); gt = [(0.0, 0.0, 0.0)]
    while len(gt) < n_nodes:
        x, y, th = gt[-1]
        if len(gt) % int(side / step) == 0: th = wrap(th + math.pi / 2)
        gt.append((x + step * math.cos(th), y + step * math.sin(th), th))
    est = [gt[0]]
    for a, b in zip(gt, gt[1:]):
        z = compose(inverse(a), b); est.append(compose(est[-1], (z[0], z[1], z[2] + yaw_bias)))
    s.kfs, s.edges, s._robust, s._pairs, s.field = [], [], [], set(), S.Field()
    s._recent = collections.deque(maxlen=8)
    for i, p in enumerate(est):
        obs = (rng.uniform(-2, 2, (300, 2))).astype(np.float32)
        s.kfs.append(S.KF(p, p, obs, np.zeros((0, 2), np.float32), i * step, None, float(i))); s.field.add(p, obs)
        if i:
            s.edges.append((i - 1, i, compose(inverse(est[i - 1]), p), s._odom_info(step, 0.0))); s._robust.append(False); s._pairs.add((i - 1, i))
        s._recent.append((p, obs))
    n = len(est) - 1; s._anchor = (n, (0.0, 0.0, 0.0), est[n], n * step); s._xy = None
    s.T = compose(est[n], inverse(est[n]))
    return gt, est


def _new(lag=0, async_lc=True):
    s = S.Slam2(GridMap(size_m=24.0, res=0.04, origin=(-12.0, -12.0)), match=False); s._run = False
    s.async_lc, s.lc_lag = async_lc, lag
    return s


def _stub_search(gt, delay=0.0):
    """Loop search stub: the current node really is at gt relative to the loop node (j = 0 here)."""
    def f(fields, org, P, guess, *a, **k):
        if delay: time.sleep(delay)
        f.calls += 1; s, n, j = f.ctx
        return compose(s.kfs[j].pose if f.snap_pose_j is None else f.snap_pose_j, compose(inverse(gt[j]), gt[n])), 0.9, 0.5
    f.calls = 0; f.snap_pose_j = None; return f


def _close(s, gt, delay=0.0):
    n = len(s.kfs) - 1; st = _stub_search(gt, delay); st.ctx = (s, n, 0); st.snap_pose_j = s.kfs[0].pose
    S.search, old = st, S.search
    return n, st, old


def _grow(s, k, step=0.5):
    """k more nodes chained from the current anchor (what the front end does while the solve runs)."""
    for _ in range(k):
        a = len(s.kfs) - 1; p = compose(s.kfs[a].pose, (step, 0.0, 0.0)); obs = s.kfs[a].obs
        s.kfs.append(S.KF(p, p, obs, np.zeros((0, 2), np.float32), s.kfs[a].path + step, None, 0.0)); s.field.add(p, obs)
        s.edges.append((a, a + 1, (step, 0.0, 0.0), s._odom_info(step, 0.0))); s._robust.append(False); s._pairs.add((a, a + 1))
        s._anchor = (a + 1, (0.0, 0.0, 0.0), p, 0.0); s._xy = None


def _field_ok(s):
    """Field vs one built from scratch at the final poses: same total count (+-0.1 %: a moved scan can fall into one cell more or less; each moved scan removed at its old pose
    and re-added, none doubled or lost) and the same occupied cells up to the < 1 cm / 0.3 deg moves slam2 does not
    re-add by design."""
    f = S.Field()
    for k in s.kfs: f.add(k.pose, k.obs)
    a, b = f.cnt > 0, s.field.cnt > 0
    return abs(int(f.cnt.sum()) - int(s.field.cnt.sum())) <= 1e-3 * f.cnt.sum() and (a & b).sum() / (a | b).sum() > 0.95


def test_async_equals_sync_and_carries_new_nodes():
    a = _new(async_lc=False); gt, est = _graph(a); n, st, old = _close(a, gt)
    try:
        a._try_loop_life(n)
        assert a.stats["loops"] == 1 and a._lc is None and _field_ok(a)
        b = _new(lag=3); _graph(b); _close(b, gt)[1]; S.search = _stub_search(gt); S.search.ctx = (b, n, 0); S.search.snap_pose_j = b.kfs[0].pose
        b._try_loop_life(n)
        assert b._lc is not None and b.stats["loops"] == 0            # submitted, not applied
        pre = [k.pose for k in b.kfs]; _grow(b, 3); rel = [compose(inverse(b.kfs[n].pose), k.pose) for k in b.kfs[n + 1:]]
        for _ in range(2): b._nstep += 1; b._lc_poll()
        assert b._lc is not None                                        # lag 3: not yet
        b._nstep += 1; b._lc_poll()
        assert b._lc is None and b.stats["loops"] == 1 and b.stats["lc_lag"] == 3
        for i in range(n + 1): assert np.allclose(a.kfs[i].pose, b.kfs[i].pose, atol=1e-12), i   # = synchronous solve
        for r, k in zip(rel, b.kfs[n + 1:]): assert np.allclose(compose(inverse(b.kfs[n].pose), k.pose), r, atol=1e-9)
        assert _field_ok(b)                                             # field patched exactly (old counts removed)
        ae, ee = [np.hypot(*(np.array(x)[:n + 1, :2] - np.array(gt)[:n + 1, :2]).T).max() for x in ([k.pose for k in b.kfs], pre)]
        assert ae < 0.5 * ee, (ae, ee)                                  # the closure fixed the drift
        an, da, oa, _ = b._anchor
        assert np.allclose(b.T, compose(compose(b.kfs[an].pose, da), inverse(oa)))   # robot pose follows the anchor
        assert b.edges[-1][1] == n and b._robust[-1] is True and b.loop_log[-1][3] == len(b.edges) - 1
    finally: S.search = old


def test_deterministic_lag_and_free_thread_agree():
    out = []
    for lag, delay in ((2, 0.0), (2, 0.0), (0, 0.05)):
        s = _new(lag=lag); gt, _ = _graph(s); n, st, old = _close(s, gt, delay)
        try:
            s._try_loop_life(n); t0 = time.perf_counter(); polls = 0
            while s._lc is not None:
                if polls < 3: _grow(s, 1)
                s._nstep += 1; s._lc_poll(); polls += 1
                if s._lc is not None and lag == 0: time.sleep(0.01)      # front-end updates while the worker runs
                assert time.perf_counter() - t0 < 20
            if lag == 0: assert st.calls == 1 and polls >= 1             # the front end kept running during the search
            out.append((polls, np.array([k.pose for k in s.kfs[:n + 1]])))
        finally: S.search = old
    assert out[0][0] == out[1][0] == 2 and np.array_equal(out[0][1], out[1][1])   # lag mode reproducible
    assert np.allclose(out[0][1], out[2][1], atol=1e-12)                          # = free-running thread on the solved nodes


def test_reset_drops_inflight_result():
    s = _new(lag=0); gt, _ = _graph(s); n, st, old = _close(s, gt, delay=0.1)
    try:
        s._try_loop_life(n); th = s._lc[0]
        s.reset_to_grid(); th.join(); s._nstep += 1; s._lc_poll()
        assert s._lc is None and s.stats["loops"] == 0 and len(s.edges) == 0
    finally: S.search = old


def test_one_search_in_flight():
    s = _new(lag=5); gt, _ = _graph(s); n, st, old = _close(s, gt)
    try:
        s._try_loop_life(n); s._try_loop_life(n)
        assert s.stats.get("lc_busy") == 1
        s._lc[0].join(); assert st.calls == 1
    finally: S.search = old


if __name__ == "__main__":
    for f in (test_async_equals_sync_and_carries_new_nodes, test_deterministic_lag_and_free_thread_agree, test_reset_drops_inflight_result,
              test_one_search_in_flight): f(); print("ok", f.__name__)
