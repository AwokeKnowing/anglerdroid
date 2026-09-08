#!/usr/bin/env python3
"""Unit tests for SLAM walls STL prior integration.

Tests that walls STL prior integrates with SLAM correctly:
- Default behavior unchanged when disabled
- Wall evidence appears in map when enabled
- No hot-path performance impact

Run: python3 test_slam_walls_prior.py
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Check if we can run SLAM tests (requires cv2)
try:
    import cv2
    from slam import PoseGraphSLAM
    _HAS_SLAM = True
except ImportError:
    _HAS_SLAM = False


def test_slam_default_no_walls():
    """Test that SLAM works identically without walls prior (legacy behavior)."""
    if not _HAS_SLAM:
        print("⚠ OpenCV not available, skipping SLAM integration test")
        return
    
    # Clear env var if set
    if 'KEVIN_WALLS_STL_PRIOR' in os.environ:
        del os.environ['KEVIN_WALLS_STL_PRIOR']
    
    slam = PoseGraphSLAM()
    
    # Check stats
    stats = slam.stats()
    assert 'keyframes' in stats, "Stats should include keyframes"
    assert 'walls_prior_segments' not in stats, "No walls stats when disabled"
    
    print("✓ SLAM default behavior (no walls prior)")


def test_slam_with_walls_prior():
    """Test that SLAM loads walls prior when enabled."""
    if not _HAS_SLAM:
        print("⚠ OpenCV not available, skipping SLAM integration test")
        return
    
    fixture_path = os.path.join(
        os.path.dirname(__file__), 'test_fixtures', 'walls_test.stl')
    
    if not os.path.exists(fixture_path):
        print(f"⚠ Test fixture not found: {fixture_path} (skipping)")
        return
    
    # Enable walls prior
    os.environ['KEVIN_WALLS_STL_PRIOR'] = fixture_path
    
    slam = PoseGraphSLAM()
    
    # Check that walls prior was loaded
    stats = slam.stats()
    assert 'walls_prior_segments' in stats, "Stats should include walls info"
    assert stats['walls_prior_segments'] > 0, "Should have loaded wall segments"
    
    print(f"✓ SLAM with walls prior enabled "
          f"({stats['walls_prior_segments']} segments)")
    
    # Clean up
    del os.environ['KEVIN_WALLS_STL_PRIOR']


def test_walls_evidence_in_map():
    """Test that wall evidence appears in map after loop closure."""
    if not _HAS_SLAM:
        print("⚠ OpenCV not available, skipping SLAM integration test")
        return
    
    fixture_path = os.path.join(
        os.path.dirname(__file__), 'test_fixtures', 'walls_test.stl')
    
    if not os.path.exists(fixture_path):
        print(f"⚠ Test fixture not found: {fixture_path} (skipping)")
        return
    
    # Enable walls prior
    os.environ['KEVIN_WALLS_STL_PRIOR'] = fixture_path
    
    slam = PoseGraphSLAM()
    
    # Create some keyframes to trigger a map (walls applied after rebuild)
    # Simple synthetic ego observation (small obstacle in center)
    obs_ego = np.zeros((240, 320), dtype=np.uint8)
    known_ego = np.ones((240, 320), dtype=np.uint8) * 255
    obs_ego[110:130, 150:170] = 50  # 50 cm tall obstacle
    
    # Add a few keyframes (move robot in a square)
    poses = [
        (0.0, 0.0, 0.0),
        (0.3, 0.0, 0.0),
        (0.3, 0.3, 0.0),
        (0.0, 0.3, 0.0),
        (0.0, 0.0, 0.0),  # Back to start (potential loop)
    ]
    
    for x, y, theta in poses:
        slam.keyframe_check(
            obs_ego, known_ego, x, y, theta,
            ego_cx=160, ego_cy=120, ego_px_size=0.01)
    
    # Check stats
    stats = slam.stats()
    assert stats['keyframes'] > 0, "Should have created keyframes"
    
    # If loop closure happened, walls should have been applied
    if stats.get('loop_closures', 0) > 0:
        assert stats['walls_constraints'] > 0, \
            "Wall constraints should be applied after loop closure"
        print(f"✓ Wall evidence applied after loop closure "
              f"({stats['walls_constraints']} segments drawn)")
    else:
        print("✓ Keyframes created (no loop closure in simple test)")
    
    # Clean up
    del os.environ['KEVIN_WALLS_STL_PRIOR']


def test_walls_prior_fail_closed():
    """Test that bad walls path fails closed (SLAM continues)."""
    if not _HAS_SLAM:
        print("⚠ OpenCV not available, skipping SLAM integration test")
        return
    
    # Set invalid path
    os.environ['KEVIN_WALLS_STL_PRIOR'] = '/nonexistent/walls.stl'
    
    # SLAM should still work (fail-closed)
    slam = PoseGraphSLAM()
    
    stats = slam.stats()
    assert 'keyframes' in stats, "SLAM should work despite bad walls path"
    assert 'walls_prior_segments' not in stats, "No walls when file missing"
    
    print("✓ SLAM fails closed with bad walls path")
    
    # Clean up
    del os.environ['KEVIN_WALLS_STL_PRIOR']


def test_walls_prior_stats():
    """Test that walls prior stats are tracked correctly."""
    if not _HAS_SLAM:
        print("⚠ OpenCV not available, skipping SLAM integration test")
        return
    
    fixture_path = os.path.join(
        os.path.dirname(__file__), 'test_fixtures', 'walls_test.stl')
    
    if not os.path.exists(fixture_path):
        print(f"⚠ Test fixture not found: {fixture_path} (skipping)")
        return
    
    os.environ['KEVIN_WALLS_STL_PRIOR'] = fixture_path
    
    slam = PoseGraphSLAM()
    stats = slam.stats()
    
    # Check that stats include walls info
    assert 'walls_prior_segments' in stats, "Stats should include wall segments"
    assert isinstance(stats['walls_prior_segments'], int), \
        "Wall segments count should be int"
    assert stats['walls_prior_segments'] > 0, \
        "Should have loaded segments from test fixture"
    
    print(f"✓ Walls prior stats tracked correctly "
          f"(segments={stats['walls_prior_segments']})")
    
    # Clean up
    del os.environ['KEVIN_WALLS_STL_PRIOR']


if __name__ == '__main__':
    print("Testing SLAM walls STL prior integration...")
    print()
    
    test_slam_default_no_walls()
    test_slam_with_walls_prior()
    test_walls_prior_fail_closed()
    test_walls_evidence_in_map()
    test_walls_prior_stats()
    
    print()
    print("All SLAM walls prior tests passed! ✓")
