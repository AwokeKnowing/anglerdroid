"""governor: the block-pruned clearance sweep (nav/governor._sd_matrix_pruned) gives exactly the decisions of the full
(cells x poses) matrix, on random wall / clutter scenes and commands (SC_DECEL path, KEVIN_HAL=sim default)."""
import os, sys, time
os.environ.setdefault("KEVIN_HAL", "sim")
sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.join(os.path.dirname(__file__), "..", "src")]
import numpy as np
import nav.governor as G, nav.safety_core as SC


def scene(rng):
    P = []
    for _ in range(rng.integers(1, 5)):                     # walls: dense noisy lines near the robot
        a = rng.uniform(0, 2 * np.pi); d = rng.uniform(0.2, 0.6); L = rng.uniform(0.3, 1.5); s = np.linspace(-L / 2, L / 2, int(L / 0.003))
        c = np.array([np.cos(a), np.sin(a)]) * d; t = np.array([-np.sin(a), np.cos(a)])
        P.append(c + s[:, None] * t + rng.normal(0, 0.004, (len(s), 2)))
    for _ in range(rng.integers(0, 4)):                     # clutter blobs (chair legs)
        P.append(rng.uniform(-0.7, 0.7, 2) + rng.normal(0, 0.015, (rng.integers(20, 200), 2)))
    return np.concatenate(P).astype(np.float32)


def main(n=int(os.environ.get("N", 600))):
    full = lambda prep, poses, thr: (SC._sd_matrix_T(prep[0], poses), prep[1]); pruned = G._sd_matrix_pruned
    rng = np.random.default_rng(7); bad = 0; tf = tp = 0.0; dec = {}
    for i in range(n):
        P = scene(rng); v = float(rng.uniform(-0.6, 0.7)); w = float(rng.uniform(-1.6, 1.6))
        kw = dict(v_meas=v * rng.uniform(0.5, 1.3), w_meas=w * rng.uniform(0.5, 1.3), t_extra=float(rng.uniform(0, 0.2)), escape=bool(rng.integers(2)))
        G._sd_matrix_pruned = full; t = time.perf_counter(); a = G.governor(P, v, w, **kw); tf += time.perf_counter() - t
        G._sd_matrix_pruned = pruned; t = time.perf_counter(); b = G.governor(P, v, w, **kw); tp += time.perf_counter() - t
        bad += a != b; dec[a[4]] = dec.get(a[4], 0) + 1
    r = dict(n=n, different=bad, decisions=dec, full_ms=round(tf / n * 1e3, 2), pruned_ms=round(tp / n * 1e3, 2)); print(r)
    assert SC.SC_DECEL and bad == 0, r
    print("PASS")


if __name__ == "__main__": main()
