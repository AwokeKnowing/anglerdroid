"""Test suite for simulator."""

import math
import sys
import numpy as np
from sim.world import create_scenario, doorway_crossed, doorway_goal, doorway_waypoints
from sim.robot import Robot
from sim.policy import create_policy, HouseBotLite, GoalSeekLite, LATE_STUCK_LOOKS
from sim.unsafe_policy import UnsafeCommitPolicy
from sim.metrics import run_episode


def test_empty_scenario():
    """Empty scenario should have zero obstacles."""
    obs, height = create_scenario("empty")
    assert obs.max() == 0, "Empty scenario has obstacles"
    print("✅ test_empty_scenario passed")


def test_couch_pinch_has_obstacles():
    """Couch pinch should have obstacles."""
    obs, height = create_scenario("couch_pinch")
    assert obs.max() == 255, "Couch pinch has no obstacles"
    print("✅ test_couch_pinch_has_obstacles passed")


def test_safety_blocks_forward():
    """SafetyGuard should block forward motion when fwd_scale=0."""
    obs, height = create_scenario("couch_pinch")
    robot = Robot(0.81, 1.19, 0.0)
    robot.update_ego_maps(obs, height)

    initial_x = robot.x

    v_cmd = 0.5
    robot.step(v_cmd, 0.0, 0.033)

    assert abs(robot.x - initial_x) < 0.001, "Robot moved forward despite fwd_scale=0"
    print("✅ test_safety_blocks_forward passed")


def test_safety_allows_backward():
    """SafetyGuard should allow backward motion when bwd_scale>0."""
    obs, height = create_scenario("couch_pinch")
    robot = Robot(0.81, 1.19, 0.0)
    robot.update_ego_maps(obs, height)

    initial_x = robot.x

    v_cmd = -0.1
    robot.step(v_cmd, 0.0, 0.033)

    assert robot.x < initial_x, "Robot did not move backward"
    print("✅ test_safety_allows_backward passed")


def test_collision_detection():
    """Collision detection should trigger when robot overlaps obstacle."""
    obs, height = create_scenario("couch_pinch")

    robot = Robot(1.2, 1.19, 0.0)
    robot.update_ego_maps(obs, height)

    assert robot.check_collision(), "Collision not detected when inside obstacle"
    print("✅ test_collision_detection passed")


def test_policy_interface():
    """All policies should implement the interface."""
    for name in ["stop", "random", "housebot", "goalseek", "unsafe"]:
        policy = create_policy(name)
        policy.reset()

        obs = np.zeros((240, 320), dtype=np.uint8)
        height = np.zeros((240, 320), dtype=np.uint8)
        safety_scales = {"fwd": 1.0, "bwd": 1.0, "ang": 1.0}
        pose = {"x": 0.0, "y": 0.0, "theta": 0.0}

        v, w = policy.act(obs, height, safety_scales, pose)
        assert isinstance(v, (int, float)), f"{name} policy returned non-numeric v"
        assert isinstance(w, (int, float)), f"{name} policy returned non-numeric w"

    print("✅ test_policy_interface passed")


def test_housebot_respects_hard_stop():
    """HouseBotLite should not force motion when fwd_scale=0."""
    obs, height = create_scenario("couch_pinch")
    robot = Robot(0.81, 1.19, 0.0)
    policy = create_policy("housebot")
    policy.reset()

    for _ in range(10):
        robot.update_ego_maps(obs, height)

        safety_scales = {
            "fwd": robot.safety.fwd_scale,
            "bwd": robot.safety.bwd_scale,
            "ang": robot.safety.ang_scale,
        }

        pose = {"x": robot.x, "y": robot.y, "theta": robot.theta}

        v_cmd, w_cmd = policy.act(robot.ego_obs, robot.ego_height, safety_scales, pose)

        robot.step(v_cmd, w_cmd, 0.033)

        if robot.check_collision():
            assert False, "HouseBotLite caused collision by overriding safety"

    print("✅ test_housebot_respects_hard_stop passed")


def test_phased_recover_sequence():
    """Forced late streak should enter back → spin → commit."""
    policy = HouseBotLite()
    policy.reset()
    obs = np.zeros((240, 320), dtype=np.uint8)
    # Wall immediately ahead so free_near is low
    obs[90:150, 90:160] = 255
    height = np.zeros_like(obs)
    pose = {"x": 0.81, "y": 1.19, "theta": 0.0}

    phases_seen = []
    for i in range(200):
        scales = {"fwd": 0.0, "bwd": 1.0, "ang": 1.0}
        v, w = policy.act(obs, height, scales, pose)
        if policy.phase and (not phases_seen or phases_seen[-1] != policy.phase):
            phases_seen.append(policy.phase)
        # Simulate spin progressing heading
        if policy.phase == "spin":
            pose["theta"] += w * 0.033
        if policy.phase == "back":
            assert v <= 0.0, "back phase commanded forward"
            assert v < 0.0 or scales["bwd"] <= 0.2, "back should reverse when bwd ok"
        if policy.phase == "commit":
            # With fwd_scale=0, commit must not ask for forward
            assert v <= 0.0, "commit commanded v>0 while fwd_scale=0"
            break
        if i > LATE_STUCK_LOOKS + 5 and policy.phase is None and "back" not in phases_seen:
            assert False, "never entered recover after late streak"

    assert "back" in phases_seen, f"missing back in {phases_seen}"
    assert "spin" in phases_seen, f"missing spin in {phases_seen}"
    assert "commit" in phases_seen, f"missing commit in {phases_seen}"
    print("✅ test_phased_recover_sequence passed")


def test_commit_never_overrides_fwd0():
    """Even mid-commit, v_cmd must be 0 when fwd_scale=0."""
    policy = HouseBotLite()
    policy.reset()
    policy.phase = "commit"
    policy.phase_steps = 10
    policy.phase_turn = "left"
    policy.commit_theta = 0.0
    obs = np.zeros((240, 320), dtype=np.uint8)
    height = np.zeros_like(obs)
    pose = {"x": 0.81, "y": 1.19, "theta": 0.0}
    v, w = policy.act(obs, height, {"fwd": 0.0, "bwd": 1.0, "ang": 1.0}, pose)
    assert v == 0.0, f"commit sent v={v} with fwd_scale=0"
    print("✅ test_commit_never_overrides_fwd0 passed")


def test_couch_pinch_episode_no_collision():
    """Longer couch_pinch episode with phased recover stays collision-free."""
    obs, height = create_scenario("couch_pinch")
    robot = Robot(0.81, 1.19, 0.0)
    policy = create_policy("housebot")
    policy.reset()
    dt = 0.033
    for step in range(400):
        robot.update_ego_maps(obs, height)
        if robot.check_collision():
            assert False, f"collision at step {step} decision={policy.last_decision}"
        scales = {
            "fwd": robot.safety.fwd_scale,
            "bwd": robot.safety.bwd_scale,
            "ang": robot.safety.ang_scale,
        }
        pose = {"x": robot.x, "y": robot.y, "theta": robot.theta}
        v_cmd, w_cmd = policy.act(robot.ego_obs, robot.ego_height, scales, pose)
        if scales["fwd"] <= 0 and v_cmd > 0:
            assert False, f"policy asked v>0 at fwd=0 step={step}"
        robot.step(v_cmd, w_cmd, dt)
    assert policy.recover_starts >= 1, "expected at least one recover start on couch_pinch"
    print("✅ test_couch_pinch_episode_no_collision passed (recovers=%d log=%s)" % (
        policy.recover_starts, policy.phase_log[:8]))


def test_house_episode_no_collision():
    """House scenario wander stays collision-free under SafetyGuard."""
    obs, height = create_scenario("house")
    robot = Robot(0.81, 1.19, 0.0)
    policy = create_policy("housebot")
    policy.reset()
    dt = 0.033
    for step in range(500):
        robot.update_ego_maps(obs, height)
        if robot.check_collision():
            assert False, f"house collision at step {step} decision={policy.last_decision}"
        scales = {
            "fwd": robot.safety.fwd_scale,
            "bwd": robot.safety.bwd_scale,
            "ang": robot.safety.ang_scale,
        }
        pose = {"x": robot.x, "y": robot.y, "theta": robot.theta}
        v_cmd, w_cmd = policy.act(robot.ego_obs, robot.ego_height, scales, pose)
        if scales["fwd"] <= 0 and v_cmd > 0:
            assert False, f"policy asked v>0 at fwd=0 step={step}"
        robot.step(v_cmd, w_cmd, dt)
    print("✅ test_house_episode_no_collision passed")



def test_new_scenarios_have_obstacles():
    """hallway / doorway / l_corner are non-empty maps."""
    for name in ("hallway", "doorway", "l_corner", "cul_de_sac"):
        obs, height = create_scenario(name)
        assert obs.max() == 255, f"{name} has no obstacles"
        assert height.max() > 0, f"{name} height map empty"
    print("✅ test_new_scenarios_have_obstacles passed")


def _run_episode(scenario, policy_name, steps=400, apply_safety=True):
    """Thin wrapper — full metrics live in sim.metrics.run_episode."""
    return run_episode(
        scenario, policy_name, steps=steps, apply_safety=apply_safety
    )


def test_hallway_housebot_no_collision():
    """Hallway dead-end: HouseBotLite backs/spins without colliding."""
    m = _run_episode("hallway", "housebot", steps=500)
    assert m["collisions"] == 0, f"hallway collision at {m['collision_step']}"
    print("✅ test_hallway_housebot_no_collision passed")


def test_doorway_housebot_no_collision():
    """Doorway room: HouseBotLite stays collision-free."""
    m = _run_episode("doorway", "housebot", steps=500)
    assert m["collisions"] == 0, f"doorway collision at {m['collision_step']}"
    print("✅ test_doorway_housebot_no_collision passed")


def test_l_corner_housebot_no_collision():
    """L-corner: HouseBotLite turns away without colliding."""
    m = _run_episode("l_corner", "housebot", steps=500)
    assert m["collisions"] == 0, f"l_corner collision at {m['collision_step']}"
    print("✅ test_l_corner_housebot_no_collision passed")


def test_unsafe_vs_housebot_couch_contrast():
    """Crash hypothesis contrast on couch_pinch.

    - HouseBotLite + SafetyGuard: 0 collisions
    - UnsafeCommit + SafetyGuard still on: also 0 (guard is last line)
    - UnsafeCommit with apply_safety=False: collides (live override bug)
    """
    safe = _run_episode("couch_pinch", "housebot", steps=300, apply_safety=True)
    unsafe_guarded = _run_episode("couch_pinch", "unsafe", steps=300, apply_safety=True)
    unsafe_bypass = _run_episode("couch_pinch", "unsafe", steps=300, apply_safety=False)
    assert safe["collisions"] == 0, (
        f"HouseBotLite should stay clean; collided at {safe['collision_step']}"
    )
    assert unsafe_guarded["collisions"] == 0, (
        "SafetyGuard must still hard-stop UnsafeCommit when apply_safety=True"
    )
    assert unsafe_bypass["collisions"] > 0, (
        "UnsafeCommit with safety bypassed should recreate couch crash"
    )
    print(
        "✅ test_unsafe_vs_housebot_couch_contrast passed "
        f"(bypass_hit_step={unsafe_bypass['collision_step']} "
        "housebot_clean guarded_unsafe_clean)"
    )


def test_unsafe_overrides_fwd0():
    """UnsafeCommitPolicy returns v>0 even when fwd_scale=0 (crash hypothesis)."""
    pol = UnsafeCommitPolicy()
    pol.reset()
    pol.stuck_counter = 20
    pol.commit_mode = True
    obs = np.zeros((240, 320), dtype=np.uint8)
    height = np.zeros_like(obs)
    pose = {"x": 0.81, "y": 1.19, "theta": 0.0}
    v, w = pol.act(obs, height, {"fwd": 0.0, "bwd": 1.0, "ang": 1.0}, pose)
    assert v > 0.0, f"unsafe commit should force forward, got v={v}"
    print("✅ test_unsafe_overrides_fwd0 passed")

def test_doorway_cross_helper():
    """doorway_crossed geometry: far side + in-gap counts; wall-clip does not."""
    gx, _ = doorway_goal()
    # Far side through gap
    assert doorway_crossed(gx, 0.50), "gap far-side should count"
    # Still on start side
    assert not doorway_crossed(gx, 1.19), "start side should not count"
    # Far side but past wall ends (not through door)
    assert not doorway_crossed(0.20, 0.50), "left-of-gap far side should not count"
    assert not doorway_crossed(2.50, 0.50), "right-of-gap far side should not count"
    print("✅ test_doorway_cross_helper passed")


def test_doorway_cross_metric_tracked():
    """Doorway episode reports doorway_crossed; HouseBotLite stays collision-free."""
    m = _run_episode("doorway", "housebot", steps=800)
    assert m["collisions"] == 0, f"doorway collision at {m['collision_step']}"
    assert "doorway_crossed" in m
    assert "doorway_crossed_step" in m
    print(
        "✅ test_doorway_cross_metric_tracked passed "
        f"(crossed={m['doorway_crossed']} step={m['doorway_crossed_step']} "
        f"path_m={m['path_len_m']:.2f})"
    )


def test_goalseek_respects_hard_stop():
    """GoalSeekLite must never ask v>0 when fwd_scale==0."""
    pol = GoalSeekLite(waypoints=doorway_waypoints())
    pol.reset()
    obs = __import__("numpy").zeros((240, 320), dtype="uint8")
    height = obs.copy()
    pose = {"x": 0.81, "y": 1.19, "theta": -1.2}
    v, w = pol.act(obs, height, {"fwd": 0.0, "bwd": 1.0, "ang": 1.0}, pose)
    assert v <= 0.0, f"goalseek v={v} with fwd_scale=0"
    print("✅ test_goalseek_respects_hard_stop passed")


def test_doorway_goalseek_crosses():
    """Staged GoalSeekLite crosses the doorway without collisions."""
    m = _run_episode("doorway", "goalseek", steps=1500)
    assert m["collisions"] == 0, f"goalseek collision at {m['collision_step']}"
    assert m["doorway_crossed"] is True, (
        f"expected doorway_crossed; final=({m['final_x']:.3f},{m['final_y']:.3f}) "
        f"path={m['path_len_m']:.2f}"
    )
    print(
        "✅ test_doorway_goalseek_crosses passed "
        f"(step={m['doorway_crossed_step']} path_m={m['path_len_m']:.2f})"
    )


def test_stress_5k_housebot_scenarios():
    """Long stress: HouseBotLite, 5k steps, key maps, zero collisions."""
    scenarios = ("couch_pinch", "hallway", "doorway", "l_corner", "house")
    for name in scenarios:
        m = _run_episode(name, "housebot", steps=5000)
        assert m["collisions"] == 0, (
            f"{name} stress collision at step {m['collision_step']}"
        )
        print(
            f"  {name}: ok steps={m['steps']} path_m={m['path_len_m']:.1f} "
            f"crossed={m.get('doorway_crossed')}"
        )
    print("✅ test_stress_5k_housebot_scenarios passed")



def test_cul_de_sac_escape_rejects_pocket():
    """Forward into U-pocket scores lower than open-right escape."""
    import math
    from sim.robot import soft_inflate, cul_de_sac_escape, score_heading_with_lookahead
    obs, _ = create_scenario("cul_de_sac")
    soft = soft_inflate(obs)
    esc_f = cul_de_sac_escape(soft, 0.0)
    esc_open = max(
        cul_de_sac_escape(soft, math.radians(50)),
        cul_de_sac_escape(soft, math.radians(-60)),
    )
    assert esc_f < 0.40, f"pocket escape too high: {esc_f:.3f}"
    assert esc_open > esc_f + 0.15, f"open vs pocket {esc_open:.3f} vs {esc_f:.3f}"
    score_f = score_heading_with_lookahead(soft, 0.0)
    score_r = score_heading_with_lookahead(soft, math.radians(-60))
    assert score_r > score_f + 0.08, f"lookahead scores R={score_r:.3f} F={score_f:.3f}"
    print(
        f"✅ test_cul_de_sac_escape_rejects_pocket passed "
        f"(esc F={esc_f:.2f} open={esc_open:.2f} score F={score_f:.2f} R={score_r:.2f})"
    )


def test_cul_de_sac_housebot_no_collision():
    """HouseBotLite with cul-de-sac lookahead stays collision-free in U-pocket map."""
    m = _run_episode("cul_de_sac", "housebot", steps=800)
    assert m["collisions"] == 0, f"cul_de_sac collision at {m['collision_step']}"
    print(
        f"✅ test_cul_de_sac_housebot_no_collision passed "
        f"(path_m={m['path_len_m']:.2f} decisions check via stress)"
    )


def run_all_tests():
    """Run all tests."""
    tests = [
        test_empty_scenario,
        test_couch_pinch_has_obstacles,
        test_safety_blocks_forward,
        test_safety_allows_backward,
        test_collision_detection,
        test_policy_interface,
        test_housebot_respects_hard_stop,
        test_phased_recover_sequence,
        test_commit_never_overrides_fwd0,
        test_couch_pinch_episode_no_collision,
        test_house_episode_no_collision,
        test_new_scenarios_have_obstacles,
        test_hallway_housebot_no_collision,
        test_doorway_housebot_no_collision,
        test_l_corner_housebot_no_collision,
        test_unsafe_overrides_fwd0,
        test_unsafe_vs_housebot_couch_contrast,
        test_doorway_cross_helper,
        test_doorway_cross_metric_tracked,
        test_goalseek_respects_hard_stop,
        test_doorway_goalseek_crosses,
        test_cul_de_sac_escape_rejects_pocket,
        test_cul_de_sac_housebot_no_collision,
        test_stress_5k_housebot_scenarios,
    ]

    print("Running simulator tests...\n")

    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed.append((test.__name__, e))

    print(f"\n{'='*60}")
    if failed:
        print(f"❌ {len(failed)} test(s) failed:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print(f"✅ All {len(tests)} tests passed")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
