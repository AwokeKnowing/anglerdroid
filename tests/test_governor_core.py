"""Property test: the planner-side governor (nav/governor.py) never passes a command the safety core
(nav/safety_core.decide, unchanged) would zero on the same points and bands.  The old rule (safety_core.governor:
'gets no closer' at any distance, no bands) is counted for comparison.   python tests/test_governor_core.py [n]"""
import os, sys
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path[:0] = [R, os.path.join(R, "src")]
import nav  # noqa
from nav.safety_core import decide, governor as gov_old, sd, MARGIN
from nav.governor import governor as gov_new
from twin_geometry import FOOTPRINT_SAFE

HW = float(np.abs(FOOTPRINT_SAFE[:, 1]).max()); XF = float(FOOTPRINT_SAFE[:, 0].max()) + 0.005; XR = float(FOOTPRINT_SAFE[:, 0].min()) - 0.005
RECTS = np.array([[XF, -HW, XF + 0.05, HW], [XR - 0.05, -HW, XR, HW]])


class _Bands:                       # same boxes as safety_core.Bands with x_front/x_rear from the footprint
    def front(self, P): return bool(len(P)) and bool(((P[:, 0] >= XF) & (P[:, 0] < XF + 0.05) & (np.abs(P[:, 1]) <= HW)).any())
    def rear(self, P): return bool(len(P)) and bool(((P[:, 0] <= XR) & (P[:, 0] > XR - 0.05) & (np.abs(P[:, 1]) <= HW)).any())


def run(n=20000, seed=0):
    rng = np.random.default_rng(seed); B = _Bands(); st = dict(n=0, old_moves=0, old_disagree=0, new_moves=0, new_disagree=0, inside_margin=0)
    for _ in range(n):
        P = []
        while len(P) < rng.integers(1, 8):
            q = rng.uniform(-0.6, 0.6, 2)
            if sd(q[None])[0] > 0.0: P.append(q)
        P = np.array(P); st["inside_margin"] += int(sd(P).min() < MARGIN)
        v = rng.choice([0.0, rng.uniform(-0.25, 0.6)]); w = rng.uniform(-2.5, 2.5)
        vm = v * rng.uniform(0.0, 1.0); wm = w * rng.uniform(0.0, 1.0)
        out = dict(old=gov_old(P, v, w, v_meas=vm)[:2], new=gov_new(P, v, w, v_meas=vm, w_meas=wm, bands=RECTS)[:2])
        for k, (a, b) in out.items():
            if abs(a) > 1e-3 or abs(b) > 1e-3:
                st[k + "_moves"] += 1; r = decide(P, B, a, b, vm, wm)
                st[k + "_disagree"] += int(not (r[0] == a and r[1] == b))
        st["n"] += 1
    return st


def test_governor_never_less_strict_than_core():
    st = run(5000); assert st["new_disagree"] == 0, st


if __name__ == "__main__":
    print(run(int(sys.argv[1]) if len(sys.argv) > 1 else 20000))
