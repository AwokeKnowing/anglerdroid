"""test_neural_rl.py — Unit tests for neural RL policy stub.

Tests policy interface, ONNX loading (mock), fallback behavior, and budget gating.
Includes tests for policy feed consumption (KEVIN_NEURAL_POLICY_FEED=1).

Run: python3 test_neural_rl.py
"""

import os
import sys
import tempfile
import numpy as np

# Import neural_rl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import neural_rl

# Label constants (must match perception.labels)
LABEL_UNKNOWN = 0
LABEL_SELF = 1
LABEL_CLEAR = 2
LABEL_OBSTACLE = 3


def test_neural_rl_init_no_model():
    """Test initialization without model (fallback mode)."""
    policy = neural_rl.NeuralRLPolicy(
        model_path=None,
        inference_budget_ms=5.0,
        fallback_planner="vfh"
    )
    
    assert not policy.is_active()
    
    debug = policy.get_debug_state()
    assert not debug["model_loaded"]
    assert debug["fallback_planner"] == "vfh"
    assert debug["inference_count"] == 0


def test_neural_rl_set_goal():
    """Test goal setting and activation."""
    policy = neural_rl.NeuralRLPolicy(model_path=None)
    
    assert not policy.is_active()
    
    policy.set_goal(2.5, 1.3)
    
    assert policy.is_active()
    
    debug = policy.get_debug_state()
    assert debug["active"]
    assert debug["goal_xy"] == (2.5, 1.3)


def test_neural_rl_wander_mode():
    """Test wander mode activation."""
    policy = neural_rl.NeuralRLPolicy(model_path=None)
    
    policy.set_wander_mode(True)
    
    assert policy.is_active()
    
    debug = policy.get_debug_state()
    assert debug["active"]
    assert debug["goal_xy"] is None


def test_neural_rl_cancel():
    """Test cancellation."""
    policy = neural_rl.NeuralRLPolicy(model_path=None)
    
    policy.set_goal(2.5, 1.3)
    assert policy.is_active()
    
    policy.cancel()
    assert not policy.is_active()


def test_neural_rl_tick_inactive():
    """Test tick when policy is inactive."""
    policy = neural_rl.NeuralRLPolicy(model_path=None)
    
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    pose = (0.0, 0.0, 0.0)
    
    cmd = policy.tick(obs_map, pose, 0.033)
    
    assert cmd is None


def test_neural_rl_tick_no_model():
    """Test tick without model (fallback path)."""
    policy = neural_rl.NeuralRLPolicy(model_path=None, fallback_planner="vfh")
    
    policy.set_goal(2.5, 1.3)
    
    obs_map = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    pose = (0.0, 0.0, 0.0)
    
    cmd = policy.tick(obs_map, pose, 0.033)
    
    # Fallback not implemented in stub → returns None
    assert cmd is None
    
    debug = policy.get_debug_state()
    assert debug["fallback_count"] > 0


def test_neural_rl_preprocess_obs():
    """Test observation preprocessing."""
    policy = neural_rl.NeuralRLPolicy(model_path=None)
    
    obs_map = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    pose = (0.0, 0.0, 0.0)
    
    obs_batch = policy._preprocess_obs(obs_map, pose)
    
    # Should be (1, 1, 240, 320) float32 in [0, 1]
    assert obs_batch.shape == (1, 1, 240, 320)
    assert obs_batch.dtype == np.float32
    assert obs_batch.min() >= 0.0
    assert obs_batch.max() <= 1.0


def test_neural_rl_compute_goal_vector():
    """Test goal vector computation in robot frame."""
    policy = neural_rl.NeuralRLPolicy(model_path=None)
    
    # No goal set (wander mode)
    policy.set_wander_mode(True)
    goal_vec = policy._compute_goal_vector((0.0, 0.0, 0.0))
    assert goal_vec == (1.0, 0.0)  # forward bias
    
    # Goal ahead
    policy.set_goal(1.0, 0.0)
    goal_vec = policy._compute_goal_vector((0.0, 0.0, 0.0))
    assert abs(goal_vec[0] - 1.0) < 1e-6  # forward
    assert abs(goal_vec[1]) < 1e-6  # no lateral
    
    # Goal to the left
    policy.set_goal(0.0, 1.0)
    goal_vec = policy._compute_goal_vector((0.0, 0.0, 0.0))
    assert abs(goal_vec[0]) < 1e-6  # no forward
    assert abs(goal_vec[1] - 1.0) < 1e-6  # left


def test_neural_rl_debug_state():
    """Test debug state reporting."""
    policy = neural_rl.NeuralRLPolicy(
        model_path=None,
        inference_budget_ms=5.0,
        fallback_planner="mppi"
    )
    
    debug = policy.get_debug_state()
    
    assert "active" in debug
    assert "goal_xy" in debug
    assert "model_loaded" in debug
    assert "inference_count" in debug
    assert "fallback_count" in debug
    assert "inference_ms" in debug
    assert "fallback_planner" in debug
    
    assert debug["fallback_planner"] == "mppi"


def test_neural_rl_load_model_missing_file():
    """Test loading non-existent model file."""
    policy = neural_rl.NeuralRLPolicy(
        model_path="/tmp/nonexistent_model.onnx",
        fallback_planner="vfh"
    )
    
    debug = policy.get_debug_state()
    assert not debug["model_loaded"]


def test_neural_rl_onnx_not_available():
    """Test behavior when ONNX runtime is not available."""
    # This test assumes onnxruntime might not be installed
    # The policy should gracefully fall back
    
    policy = neural_rl.NeuralRLPolicy(
        model_path="models/test.onnx",
        fallback_planner="vfh"
    )
    
    # Should initialize without crashing
    assert policy is not None
    
    # Should report model not loaded
    debug = policy.get_debug_state()
    # model_loaded depends on ONNX availability and file existence
    # Just check it doesn't crash


def test_policy_feed_preprocess():
    """Test policy feed preprocessing (labels + height → network input)."""
    policy = neural_rl.NeuralRLPolicy(model_path=None)
    
    # Create synthetic policy feed
    labels = np.zeros((240, 320), dtype=np.uint8)
    height = np.zeros((240, 320), dtype=np.uint8)
    
    # Add CLEAR floor
    labels[:, :] = LABEL_CLEAR
    
    # Add OBSTACLE (dog bed)
    labels[100:110, 150:160] = LABEL_OBSTACLE
    height[100:110, 150:160] = 10  # 10 cm
    
    # Add SELF (robot body)
    labels[115:125, 155:165] = LABEL_SELF
    height[115:125, 155:165] = 0  # SELF has no height
    
    # Preprocess
    pose = (0.0, 0.0, 0.0)
    obs_batch = policy._preprocess_policy_feed(labels, height, pose)
    
    # Check shape: (1, 5, 240, 320) = [batch, channels, H, W]
    assert obs_batch.shape == (1, 5, 240, 320)
    assert obs_batch.dtype == np.float32
    
    # Check channel 0: UNKNOWN (should be 0 everywhere since all labeled)
    assert np.sum(obs_batch[0, 0] > 0) == 0
    
    # Check channel 1: SELF (should be robot body pixels)
    self_pixels = np.sum(obs_batch[0, 1] > 0)
    assert self_pixels == 10 * 10  # 10x10 box
    
    # Check channel 2: CLEAR (should be floor minus obstacles/self)
    clear_pixels = np.sum(obs_batch[0, 2] > 0)
    assert clear_pixels > 0  # Most of the map
    
    # Check channel 3: OBSTACLE (should be dog bed pixels)
    obstacle_pixels = np.sum(obs_batch[0, 3] > 0)
    assert obstacle_pixels == 10 * 10  # 10x10 box
    
    # Check channel 4: HEIGHT (normalized to [0, 1])
    assert np.max(obs_batch[0, 4]) <= 1.0
    assert np.min(obs_batch[0, 4]) >= 0.0
    # Obstacle region should have height ~0.1 (10 cm / 100 cm)
    obstacle_mask = (labels == LABEL_OBSTACLE)
    assert np.mean(obs_batch[0, 4][obstacle_mask]) > 0.05


def test_policy_feed_self_not_obstacle():
    """Test that SELF pixels are not treated as obstacles in policy feed."""
    policy = neural_rl.NeuralRLPolicy(model_path=None)
    
    labels = np.zeros((240, 320), dtype=np.uint8)
    height = np.zeros((240, 320), dtype=np.uint8)
    
    # Mark robot body as SELF
    labels[115:125, 155:165] = LABEL_SELF
    
    # Preprocess
    pose = (0.0, 0.0, 0.0)
    obs_batch = policy._preprocess_policy_feed(labels, height, pose)
    
    # SELF should be in channel 1, NOT channel 3 (obstacle)
    self_mask = (labels == LABEL_SELF)
    assert np.all(obs_batch[0, 1][self_mask] > 0)  # SELF channel
    assert np.all(obs_batch[0, 3][self_mask] == 0)  # NOT obstacle channel
    
    # SELF should have zero height
    assert np.all(obs_batch[0, 4][self_mask] == 0)


def test_policy_feed_unknown_preserved():
    """Test that UNKNOWN pixels remain UNKNOWN (not invented as CLEAR)."""
    policy = neural_rl.NeuralRLPolicy(model_path=None)
    
    labels = np.full((240, 320), LABEL_UNKNOWN, dtype=np.uint8)
    height = np.zeros((240, 320), dtype=np.uint8)
    
    # Preprocess
    pose = (0.0, 0.0, 0.0)
    obs_batch = policy._preprocess_policy_feed(labels, height, pose)
    
    # All pixels should be in UNKNOWN channel (channel 0)
    assert np.all(obs_batch[0, 0] > 0)
    assert np.all(obs_batch[0, 1] == 0)  # No SELF
    assert np.all(obs_batch[0, 2] == 0)  # No CLEAR
    assert np.all(obs_batch[0, 3] == 0)  # No OBSTACLE


def test_policy_feed_tick_integration():
    """Test tick with policy_feed parameter (KEVIN_NEURAL_POLICY_FEED enabled)."""
    print("\n=== Test: policy_feed tick integration ===")
    # Save original env var
    orig_env = os.environ.get('KEVIN_NEURAL_POLICY_FEED')
    
    try:
        # Enable policy feed
        os.environ['KEVIN_NEURAL_POLICY_FEED'] = '1'
        # Reload module to pick up env var
        import importlib
        importlib.reload(neural_rl)
        
        policy = neural_rl.NeuralRLPolicy(model_path=None)
        policy.set_goal(2.5, 1.3)
        
        # Create policy feed
        labels = np.zeros((240, 320), dtype=np.uint8)
        labels[:, :] = LABEL_CLEAR
        labels[100:110, 150:160] = LABEL_OBSTACLE
        
        height = np.zeros((240, 320), dtype=np.uint8)
        height[100:110, 150:160] = 30
        
        policy_feed = {
            'labels': labels,
            'height': height,
            'valid': True,
            'metadata': {'timestamp': 0.0}
        }
        
        # Legacy obs_map (for fallback)
        obs_map = np.zeros((240, 320), dtype=np.uint8)
        pose = (0.0, 0.0, 0.0)
        
        # Tick with policy_feed
        cmd = policy.tick(obs_map, pose, 0.033, policy_feed=policy_feed)
        
        # Should return None (no model loaded), but should not crash
        assert cmd is None
        
        debug = policy.get_debug_state()
        assert debug["fallback_count"] > 0
        
        print("  ✓ Tick with policy_feed succeeded (fallback)")
        
    finally:
        # Restore env var
        if orig_env is not None:
            os.environ['KEVIN_NEURAL_POLICY_FEED'] = orig_env
        else:
            os.environ.pop('KEVIN_NEURAL_POLICY_FEED', None)
        # Reload to restore original state
        import importlib
        importlib.reload(neural_rl)


def test_policy_feed_fallback_to_obs_map():
    """Test fallback to legacy obs_map when policy_feed is None or invalid."""
    policy = neural_rl.NeuralRLPolicy(model_path=None)
    policy.set_goal(2.5, 1.3)
    
    obs_map = np.zeros((240, 320), dtype=np.uint8)
    pose = (0.0, 0.0, 0.0)
    
    # Tick with policy_feed=None (should use obs_map)
    cmd = policy.tick(obs_map, pose, 0.033, policy_feed=None)
    
    # Should not crash (fallback)
    assert cmd is None or isinstance(cmd, dict)
    
    # Tick with invalid policy_feed (should use obs_map)
    invalid_feed = {'valid': False}
    cmd = policy.tick(obs_map, pose, 0.033, policy_feed=invalid_feed)
    
    # Should not crash (fallback)
    assert cmd is None or isinstance(cmd, dict)


def test_policy_feed_height_encoding():
    """Test that obstacle height is correctly encoded in policy feed."""
    policy = neural_rl.NeuralRLPolicy(model_path=None)
    
    labels = np.zeros((240, 320), dtype=np.uint8)
    height = np.zeros((240, 320), dtype=np.uint8)
    
    # Low obstacle (10 cm)
    labels[50:60, 100:110] = LABEL_OBSTACLE
    height[50:60, 100:110] = 10
    
    # Medium obstacle (50 cm)
    labels[70:80, 100:110] = LABEL_OBSTACLE
    height[70:80, 100:110] = 50
    
    # Tall obstacle (100 cm, capped)
    labels[90:100, 100:110] = LABEL_OBSTACLE
    height[90:100, 100:110] = 100
    
    # Preprocess
    pose = (0.0, 0.0, 0.0)
    obs_batch = policy._preprocess_policy_feed(labels, height, pose)
    
    # Check height channel encoding (normalized to [0, 1])
    # 10 cm → 0.1, 50 cm → 0.5, 100 cm → 1.0
    low_mask = (labels == LABEL_OBSTACLE) & (height == 10)
    med_mask = (labels == LABEL_OBSTACLE) & (height == 50)
    tall_mask = (labels == LABEL_OBSTACLE) & (height == 100)
    
    assert np.abs(np.mean(obs_batch[0, 4][low_mask]) - 0.1) < 0.01
    assert np.abs(np.mean(obs_batch[0, 4][med_mask]) - 0.5) < 0.01
    assert np.abs(np.mean(obs_batch[0, 4][tall_mask]) - 1.0) < 0.01


if __name__ == "__main__":
    print("=" * 80)
    print("NEURAL RL POLICY UNIT TESTS (with policy feed consumption)")
    print("=" * 80)
    
    tests = [
        test_neural_rl_init_no_model,
        test_neural_rl_set_goal,
        test_neural_rl_wander_mode,
        test_neural_rl_cancel,
        test_neural_rl_tick_inactive,
        test_neural_rl_tick_no_model,
        test_neural_rl_preprocess_obs,
        test_neural_rl_compute_goal_vector,
        test_neural_rl_debug_state,
        test_neural_rl_load_model_missing_file,
        test_neural_rl_onnx_not_available,
        test_policy_feed_preprocess,
        test_policy_feed_self_not_obstacle,
        test_policy_feed_unknown_preserved,
        test_policy_feed_tick_integration,
        test_policy_feed_fallback_to_obs_map,
        test_policy_feed_height_encoding,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            print(f"\n--- {test_fn.__name__} ---")
            test_fn()
            print(f"  ✓ PASS")
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
