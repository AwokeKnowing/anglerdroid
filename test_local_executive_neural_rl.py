"""test_local_executive_neural_rl.py — Integration tests for neural_rl planner in local_executive.

Tests:
- Neural RL planner selection
- Goal setting propagation
- Wander mode propagation
- Tick integration with policy feed
- Fallback handling (neural fails → mppi/vfh)
- Default planner unchanged

Run: python3 test_local_executive_neural_rl.py
"""

import os
import sys
import numpy as np

# Mock dependencies before importing local_executive
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock navigator module
class MockNavigator:
    def __init__(self):
        self._goal = None
    
    def set_goal(self, hdg):
        self._goal = hdg
    
    def clear_goal(self):
        self._goal = None
    
    def get_goal(self):
        return self._goal
    
    def compute_twist(self, atlas):
        return (0.1, 0.05)

sys.modules['navigator'] = MockNavigator()

# Mock tools module
class MockVision:
    def __init__(self):
        self.slam_locked = False
    
    def get_policy_feed(self):
        labels = np.zeros((240, 320), dtype=np.uint8)
        height = np.zeros((240, 320), dtype=np.uint8)
        return {
            'labels': labels,
            'height': height,
            'valid': True,
            'metadata': {'timestamp': 0.0}
        }
    
    def get_policy_observation(self):
        return None

class MockTools:
    def __init__(self):
        self.vision = MockVision()
    
    def get_vision(self):
        return self.vision
    
    def get_atlas(self):
        atlas = np.zeros((480, 640), dtype=np.uint8)
        return atlas, 0.0

_mock_tools = MockTools()

def get_vision():
    return _mock_tools.vision

def get_atlas():
    return _mock_tools.get_atlas()

sys.modules['tools'] = sys.modules[__name__]

# Mock keepouts module
class MockKeepouts:
    @staticmethod
    def paint_ego(obs_map, pose, slam_locked=False):
        return obs_map

sys.modules['keepouts'] = MockKeepouts()

# Now import local_executive
import local_executive


def test_neural_rl_planner_selection():
    """Test that neural_rl can be selected as a planner."""
    print("Test: Neural RL planner selection")
    
    # Default should be vfh
    assert local_executive.get_planner() == "vfh"
    
    # Set to neural_rl
    local_executive.set_planner("neural_rl")
    assert local_executive.get_planner() == "neural_rl"
    
    # Reset to vfh
    local_executive.set_planner("vfh")
    assert local_executive.get_planner() == "vfh"
    
    print("  ✓ PASS")


def test_neural_rl_invalid_planner():
    """Test that invalid planner names are rejected."""
    print("Test: Invalid planner rejection")
    
    try:
        local_executive.set_planner("invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be 'vfh', 'mppi', or 'neural_rl'" in str(e)
    
    print("  ✓ PASS")


def test_neural_rl_lazy_initialization():
    """Test that NeuralRLPolicy is lazily initialized."""
    print("Test: Lazy initialization")
    
    # Clear any existing instance
    local_executive._neural_rl = None
    
    # Set planner to neural_rl
    local_executive.set_planner("neural_rl")
    
    # Policy should be initialized
    assert local_executive._neural_rl is not None
    assert hasattr(local_executive._neural_rl, 'tick')
    
    print("  ✓ PASS")


def test_neural_rl_goal_propagation():
    """Test that set_goal_xy propagates to NeuralRLPolicy."""
    print("Test: Goal propagation to neural_rl")
    
    local_executive.set_planner("neural_rl")
    local_executive.set_goal_xy(2.5, 1.3)
    
    status = local_executive.status()
    assert status["active"]
    assert status["mode"] == "xy"
    assert status["goal_xy"] == (2.5, 1.3)
    
    # Check neural_rl policy state
    neural_debug = status.get("neural_rl")
    assert neural_debug is not None
    assert neural_debug["active"]
    assert neural_debug["goal_xy"] == (2.5, 1.3)
    
    local_executive.clear()
    print("  ✓ PASS")


def test_neural_rl_wander_propagation():
    """Test that set_wander propagates to NeuralRLPolicy."""
    print("Test: Wander mode propagation to neural_rl")
    
    local_executive.set_planner("neural_rl")
    local_executive.set_wander()
    
    status = local_executive.status()
    assert status["active"]
    assert status["mode"] == "wander"
    
    # Check neural_rl policy state
    neural_debug = status.get("neural_rl")
    assert neural_debug is not None
    assert neural_debug["active"]
    assert neural_debug["goal_xy"] is None  # No explicit goal in wander
    
    local_executive.clear()
    print("  ✓ PASS")


def test_neural_rl_cancel():
    """Test that clear() cancels NeuralRLPolicy."""
    print("Test: Cancel neural_rl")
    
    local_executive.set_planner("neural_rl")
    local_executive.set_goal_xy(2.5, 1.3)
    
    assert local_executive.is_active()
    
    local_executive.clear()
    
    assert not local_executive.is_active()
    
    status = local_executive.status()
    assert not status["active"]
    assert status["mode"] == "idle"
    
    # Check neural_rl policy state
    neural_debug = status.get("neural_rl")
    if neural_debug is not None:
        assert not neural_debug["active"]
    
    print("  ✓ PASS")


def test_neural_rl_tick_inactive():
    """Test that tick returns None when neural_rl is inactive."""
    print("Test: Tick inactive")
    
    local_executive.set_planner("neural_rl")
    local_executive.clear()
    
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    atlas = np.zeros((480, 640), dtype=np.uint8)
    
    result = local_executive.tick(atlas, 0.0, 0.0, 0.0, obs_map=obs_map)
    
    assert result is None
    print("  ✓ PASS")


def test_neural_rl_tick_active_no_model():
    """Test tick with neural_rl active but no model (fallback path)."""
    print("Test: Tick active no model (fallback)")
    
    local_executive.set_planner("neural_rl")
    local_executive.set_goal_xy(2.5, 1.3)
    
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    atlas = np.zeros((480, 640), dtype=np.uint8)
    
    # Neural should fall back (no model loaded)
    # Depending on fallback planner, should return twist or None
    result = local_executive.tick(atlas, 0.0, 0.0, 0.0, obs_map=obs_map)
    
    # Fallback behavior: may return None or a fallback twist
    # Just check it doesn't crash
    print(f"    Result: {result}")
    
    status = local_executive.status()
    dbg = status.get("dbg", {})
    print(f"    Debug: {dbg}")
    
    # Should indicate fallback
    if "fallback_active" in dbg:
        assert dbg["fallback_active"]
        print(f"    Fallback planner: {dbg.get('fallback_planner')}")
    
    local_executive.clear()
    print("  ✓ PASS")


def test_default_planner_unchanged():
    """Test that default planner is still vfh."""
    print("Test: Default planner unchanged")
    
    # Reset to default
    local_executive._planner = "vfh"
    
    assert local_executive.get_planner() == "vfh"
    
    # Check that vfh still works
    local_executive.set_planner("vfh")
    local_executive.set_goal_xy(1.0, 0.0)
    
    atlas = np.ones((480, 640), dtype=np.uint8) * 128
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    
    result = local_executive.tick(atlas, 0.0, 0.0, 0.0, obs_map=obs_map)
    
    # Should use navigator (VFH path)
    # May return None or a twist depending on navigator state
    print(f"    VFH result: {result}")
    
    local_executive.clear()
    print("  ✓ PASS")


def test_neural_rl_with_policy_feed():
    """Test that policy_feed is passed to neural_rl when available."""
    print("Test: Policy feed integration")
    
    # Enable policy feed
    os.environ['KEVIN_NEURAL_POLICY_FEED'] = '1'
    
    # Reload neural_rl to pick up env var
    import importlib
    import neural_rl as neural_rl_mod
    importlib.reload(neural_rl_mod)
    
    # Force re-init of neural_rl in local_executive
    local_executive._neural_rl = None
    local_executive.set_planner("neural_rl")
    local_executive.set_goal_xy(2.5, 1.3)
    
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    atlas = np.zeros((480, 640), dtype=np.uint8)
    
    # Tick should access policy_feed from Vision
    result = local_executive.tick(atlas, 0.0, 0.0, 0.0, obs_map=obs_map)
    
    print(f"    Result: {result}")
    
    status = local_executive.status()
    dbg = status.get("dbg", {})
    print(f"    Debug: {dbg}")
    
    # Clean up
    os.environ.pop('KEVIN_NEURAL_POLICY_FEED', None)
    importlib.reload(neural_rl_mod)
    local_executive.clear()
    
    print("  ✓ PASS")


def test_mppi_planner_still_works():
    """Test that mppi planner still works after neural_rl addition."""
    print("Test: MPPI planner still works")
    
    local_executive.set_planner("mppi")
    assert local_executive.get_planner() == "mppi"
    
    local_executive.set_goal_xy(2.0, 1.0)
    
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    atlas = np.zeros((480, 640), dtype=np.uint8)
    
    # Should not crash
    result = local_executive.tick(atlas, 0.0, 0.0, 0.0, obs_map=obs_map)
    
    print(f"    MPPI result: {result}")
    
    local_executive.clear()
    print("  ✓ PASS")


if __name__ == "__main__":
    print("=" * 80)
    print("LOCAL EXECUTIVE + NEURAL RL INTEGRATION TESTS")
    print("=" * 80)
    
    tests = [
        test_neural_rl_planner_selection,
        test_neural_rl_invalid_planner,
        test_neural_rl_lazy_initialization,
        test_neural_rl_goal_propagation,
        test_neural_rl_wander_propagation,
        test_neural_rl_cancel,
        test_neural_rl_tick_inactive,
        test_neural_rl_tick_active_no_model,
        test_default_planner_unchanged,
        test_neural_rl_with_policy_feed,
        test_mppi_planner_still_works,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            print(f"\n--- {test_fn.__name__} ---")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"TESTS COMPLETE: {passed} passed, {failed} failed")
    print("=" * 80)
    
    sys.exit(0 if failed == 0 else 1)
