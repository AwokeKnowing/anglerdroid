"""Odom EKF: a long standstill must not move the gyro scale (live 2026-10-18: 1.6 h idle -> scale -0.50, next spins at
half rate, 89 deg heading error).  Sim gyro model = emu.world 'realistic' (scale +1 %, bias 3.5 mrad/s, white 3 mrad/s
per sample, bias random walk 3e-5 rad/s/sqrt(s)); wheel-speed jitter 1.3 mm/s per wheel (measured at standstill in the
realistic sim, log L1 17:01: wheel yaw rate std 5.2 mrad/s over 1423 still ticks).  python tests/test_odom_idle_scale.py [path/to/slam.py]"""
import importlib.util, math, os, sys
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path[:0] = [R, os.path.join(R, "src")]


def run(slam_path=None, idle_s=5760.0, seed=0, sv=0.0013):
    if slam_path:
        s = importlib.util.spec_from_file_location("slam_t", slam_path); S = importlib.util.module_from_spec(s); s.loader.exec_module(S)
    else:
        import nav.slam as S
    rng = np.random.default_rng(seed); o = S.Odom(); dt = 1 / 30; b = 0.0035; k = 0.01; t = 0.0; yaw = 0.0
    def step(w_true, commanded_zero):
        nonlocal t, b, yaw
        b += 3e-5 * math.sqrt(dt) * rng.standard_normal(); t += dt; yaw += w_true * dt
        g = (1 + k) * w_true + b + 0.003 / math.sqrt(6.7) * rng.standard_normal()     # ~6.7 gyro samples per 30 Hz tick
        vr = w_true * o.track / 2; o.update(t, -vr + sv * rng.standard_normal(), vr + sv * rng.standard_normal(), g, commanded_zero=commanded_zero)
    for _ in range(int(idle_s / dt)): step(0.0, True)
    s_idle = o.scale
    for _ in range(int(10 * 2 * math.pi / 1.5 / dt)): step(1.5, False)                # 10 spins at 1.5 rad/s
    err = math.degrees((o.pose[2] - yaw + math.pi) % (2 * math.pi) - math.pi)
    return dict(scale_after_idle=round(s_idle, 4), yaw_err_after_10_spins_deg=round(err, 1))


def test_idle_does_not_move_scale():
    r = run(idle_s=600.0); assert abs(r["scale_after_idle"]) < 0.02 and abs(r["yaw_err_after_10_spins_deg"]) < 10, r


if __name__ == "__main__":
    for seed in range(3): print(seed, run(sys.argv[1] if len(sys.argv) > 1 else None, seed=seed))
