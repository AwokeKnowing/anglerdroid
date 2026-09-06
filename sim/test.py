"""Test suite for simulator."""

import sys
import numpy as np
from sim.world import create_scenario
from sim.robot import Robot
from sim.policy import create_policy


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
        safety_scales = {'fwd': 1.0, 'bwd': 1.0, 'ang': 1.0}
        pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        
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
            'fwd': robot.safety.fwd_scale,
            'bwd': robot.safety.bwd_scale,
            'ang': robot.safety.ang_scale,
        }
        
        pose = {'x': robot.x, 'y': robot.y, 'theta': robot.theta}
        
        v_cmd, w_cmd = policy.act(robot.ego_obs, robot.ego_height, safety_scales, pose)
        
        robot.step(v_cmd, w_cmd, 0.033)
        
        if robot.check_collision():
            assert False, "HouseBotLite caused collision by overriding safety"
    
    print("✅ test_housebot_respects_hard_stop passed")


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


if __name__ == '__main__':
    run_all_tests()
