"""Transfer tests: MPPI in sim + fidelity mode smoke."""
import math
from sim.world import create_scenario
from sim.robot import Robot
from sim.policy import create_policy
from sim.box3d import house_boxes, mast_collision


def test_mppi_policy_respects_fwd0():
    obs, height = create_scenario("couch_pinch")
    robot = Robot(0.81, 1.19, 0.0)
    policy = create_policy("mppi")
    policy.reset()
    robot.update_ego_maps(obs, height)
    # Force nose pin
    robot.safety.fwd_scale = 0.0
    scales = {"fwd": 0.0, "bwd": 1.0, "ang": 0.5}
    pose = {"x": robot.x, "y": robot.y, "theta": robot.theta}
    v, w = policy.act(robot.ego_obs, robot.ego_height, scales, pose)
    assert v <= 0.0, v


def test_mppi_empty_moves():
    obs, height = create_scenario("empty")
    robot = Robot(0.81, 1.19, 0.0)
    policy = create_policy("mppi")
    policy.reset()
    moved = False
    for step in range(90):
        robot.update_ego_maps(obs, height)
        scales = {
            "fwd": robot.safety.fwd_scale,
            "bwd": robot.safety.bwd_scale,
            "ang": robot.safety.ang_scale,
        }
        pose = {"x": robot.x, "y": robot.y, "theta": robot.theta}
        v, w = policy.act(robot.ego_obs, robot.ego_height, scales, pose)
        robot.step(v, w, 0.033)
        if abs(robot.x - 0.81) + abs(robot.y - 1.19) > 0.05:
            moved = True
            break
    assert moved, "MPPI did not move in empty"


def test_fidelity_smoke_no_crash_empty():
    obs, height = create_scenario("empty")
    robot = Robot(0.81, 1.19, 0.0, fidelity=True)
    policy = create_policy("housebot")
    policy.reset()
    for step in range(200):
        robot.update_ego_maps(obs, height)
        assert not robot.check_collision()
        scales = {
            "fwd": robot.safety.fwd_scale,
            "bwd": robot.safety.bwd_scale,
            "ang": robot.safety.ang_scale,
        }
        pose = {"x": robot.x, "y": robot.y, "theta": robot.theta}
        v, w = policy.act(robot.ego_obs, robot.ego_height, scales, pose)
        robot.step(v, w, 0.033)


def test_box3d_table_scenario_loads():
    obs, height = create_scenario("box3d_table")
    assert obs.shape[0] > 100
    hit, name = mast_collision(house_boxes(), 0.75, 1.9, 0.0)
    assert hit


def test_mppi_couch_escapes():
    """Pinned at couch: escape back/spin must produce path, 0 collisions."""
    obs, height = create_scenario("couch_pinch")
    robot = Robot(0.81, 1.19, 0.0)
    policy = create_policy("mppi")
    policy.reset()
    path = 0.0
    px, py = robot.x, robot.y
    escape_seen = False
    for step in range(300):
        robot.update_ego_maps(obs, height)
        assert not robot.check_collision(), step
        scales = {
            "fwd": robot.safety.fwd_scale,
            "bwd": robot.safety.bwd_scale,
            "ang": robot.safety.ang_scale,
        }
        pose = {"x": robot.x, "y": robot.y, "theta": robot.theta}
        v, w = policy.act(robot.ego_obs, robot.ego_height, scales, pose)
        if "escape" in getattr(policy, "last_decision", ""):
            escape_seen = True
        robot.step(v, w, 0.033)
        path += abs(robot.x - px) + abs(robot.y - py)
        px, py = robot.x, robot.y
    assert escape_seen or path > 0.25, (escape_seen, path)
    assert path > 0.20, path


def test_box3d_start_clear():
    from sim.world import scenario_start
    obs, height = create_scenario("box3d_table")
    sx, sy, st = scenario_start("box3d_table")
    robot = Robot(sx, sy, st)
    robot.update_ego_maps(obs, height)
    assert not robot.check_collision()
    assert robot.safety.fwd_scale > 0.3



def test_fidelity_mppi_hard_no_collision():
    """Fidelity + MPPI stress on tight layouts; 0 collisions (P0 transfer)."""
    from sim.world import scenario_start

    # cul_de_sac included when start pose is clear (default FOOT is).
    scenarios = ("couch_pinch", "hallway", "doorway", "cul_de_sac")
    steps = 400
    dt = 0.033
    for name in scenarios:
        obs, height = create_scenario(name)
        sx, sy, st = scenario_start(name)
        robot = Robot(sx, sy, st, fidelity=True)
        policy = create_policy("mppi")
        policy.reset()
        robot.update_ego_maps(obs, height)
        assert not robot.check_collision(), f"{name} start colliding"
        for step in range(steps):
            robot.update_ego_maps(obs, height)
            assert not robot.check_collision(), f"{name} collision at step {step}"
            scales = {
                "fwd": robot.safety.fwd_scale,
                "bwd": robot.safety.bwd_scale,
                "ang": robot.safety.ang_scale,
            }
            pose = {"x": robot.x, "y": robot.y, "theta": robot.theta}
            v, w = policy.act(robot.ego_obs, robot.ego_height, scales, pose)
            robot.step(v, w, dt)


def test_multi_rate_policy_hz_smoke():
    """C1 multi-rate: dynamics ~30 Hz, policy 10 Hz, fidelity on, 0 collisions."""
    from sim.run import run_simulation

    metrics = run_simulation(
        scenario_name="hallway",
        policy_name="mppi",
        steps=240,
        dt=0.033,
        fidelity=True,
        policy_hz=10,
    )
    assert metrics["collisions"] == 0, metrics



def test_corridor_centering_metric():
    """Hallway seed sweep: soft prefer raises mean lateral clearance vs hard-mask-only.

    Quantitative follow-up to soft A/B enjoy GIFs (corridor bias → centering metric).
    """
    from sim.run import run_simulation

    seeds = (7, 11, 42, 99, 123)
    soft_means = []
    hard_means = []
    for seed in seeds:
        m_on = run_simulation(
            "hallway", "mppi", steps=400, dt=0.033,
            fidelity=False, use_soft_cost=True, seed=seed,
        )
        m_off = run_simulation(
            "hallway", "mppi", steps=400, dt=0.033,
            fidelity=False, use_soft_cost=False, seed=seed,
        )
        assert m_on["collisions"] == 0, (seed, "soft", m_on)
        assert m_off["collisions"] == 0, (seed, "hard", m_off)
        soft_means.append(float(m_on["mean_lat_clear_m"]))
        hard_means.append(float(m_off["mean_lat_clear_m"]))
    soft_avg = sum(soft_means) / len(soft_means)
    hard_avg = sum(hard_means) / len(hard_means)
    # Soft prefer should not crush lateral margin; expect >= hard-only on average
    # (small epsilon for RNG / dither noise).
    assert soft_avg + 1e-4 >= hard_avg, (soft_avg, hard_avg, soft_means, hard_means)
    assert soft_avg > 0.05, soft_avg


def test_fidelity_hard_mask_only_no_collision():
    """P0 residual: fidelity=True + soft_cost OFF (hard DualScales mask only).

    Soft prefer is a sim bias; transfer residual must stay clean under latency/noise
    with hard envelope alone.
    """
    from sim.run import run_simulation

    scenarios = ("couch_pinch", "hallway", "doorway", "cul_de_sac")
    for name in scenarios:
        metrics = run_simulation(
            name,
            "mppi",
            steps=400,
            dt=0.033,
            fidelity=True,
            use_soft_cost=False,
            seed=42,
        )
        assert metrics["collisions"] == 0, (name, metrics)
        assert metrics["mean_lat_clear_m"] >= 0.0

if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_") and callable(fn):
            fn()
            print("✅", k)
    print("transfer tests passed")
