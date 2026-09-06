"""Test suite for simulator."""

import math
import sys
import numpy as np
from sim.world import create_scenario
from sim.robot import Robot
from sim.policy import create_policy, HouseBotLite, LATE_STUCK_LOOKS


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
    for name in ["stop", "random", "housebot"]:
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
