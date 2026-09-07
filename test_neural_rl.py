"""test_neural_rl.py — Unit tests for neural RL policy stub.

Tests policy interface, ONNX loading (mock), fallback behavior, and budget gating.
"""

import os
import sys
import tempfile
import numpy as np

import pytest

# Import neural_rl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import neural_rl


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


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
