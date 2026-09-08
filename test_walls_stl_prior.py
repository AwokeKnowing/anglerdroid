#!/usr/bin/env python3
"""Unit tests for walls STL prior assist (CONTRACT step 4 - walls prior).

Tests that walls STL prior loads correctly, provides wall segments, and
integrates with SLAM without breaking legacy behavior when disabled.

Run: python3 test_walls_stl_prior.py
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from perception.walls_stl_prior import (
    WallsSTLPrior,
    _load_stl_triangles,
    _extract_wall_segments_2d,
)


def test_default_disabled():
    """Test that WallsSTLPrior is disabled by default (no env var)."""
    # Clear env var if set
    if 'KEVIN_WALLS_STL_PRIOR' in os.environ:
        del os.environ['KEVIN_WALLS_STL_PRIOR']
    
    prior = WallsSTLPrior()
    
    assert not prior.is_loaded(), "Prior should be disabled by default"
    assert len(prior.get_wall_segments()) == 0, "No segments when disabled"
    
    # Query should return empty
    nearby = prior.query_walls_near_pose(0.0, 0.0, radius=10.0)
    assert len(nearby) == 0, "Query should return empty when disabled"
    
    print("✓ WallsSTLPrior disabled by default")


def test_load_test_fixture():
    """Test loading the synthetic test walls fixture."""
    fixture_path = os.path.join(
        os.path.dirname(__file__), 'test_fixtures', 'walls_test.stl')
    
    if not os.path.exists(fixture_path):
        print(f"⚠ Test fixture not found: {fixture_path} (skipping)")
        return
    
    # Load triangles directly
    triangles = _load_stl_triangles(fixture_path)
    
    assert len(triangles) > 0, "Should load triangles from test fixture"
    assert triangles.shape[1:] == (3, 3), "Triangles should be Nx3x3"
    
    print(f"✓ Loaded {len(triangles)} triangles from test fixture")
    
    # Extract wall segments
    segments = _extract_wall_segments_2d(triangles)
    
    assert len(segments) > 0, "Should extract wall segments"
    assert segments.shape[1] == 4, "Segments should be Mx4 [x0,y0,x1,y1]"
    
    print(f"✓ Extracted {len(segments)} wall segments")


def test_load_via_env_var():
    """Test loading walls via KEVIN_WALLS_STL_PRIOR env var."""
    fixture_path = os.path.join(
        os.path.dirname(__file__), 'test_fixtures', 'walls_test.stl')
    
    if not os.path.exists(fixture_path):
        print(f"⚠ Test fixture not found: {fixture_path} (skipping)")
        return
    
    # Set env var
    os.environ['KEVIN_WALLS_STL_PRIOR'] = fixture_path
    
    prior = WallsSTLPrior()
    
    assert prior.is_loaded(), "Prior should be loaded via env var"
    segments = prior.get_wall_segments()
    assert len(segments) > 0, "Should have wall segments"
    
    print(f"✓ Loaded {len(segments)} wall segments via env var")
    
    # Clean up
    del os.environ['KEVIN_WALLS_STL_PRIOR']


def test_missing_file_fails_closed():
    """Test that missing STL file fails closed (disabled, no crash)."""
    # Set env var to nonexistent path
    os.environ['KEVIN_WALLS_STL_PRIOR'] = '/nonexistent/walls.stl'
    
    prior = WallsSTLPrior()
    
    assert not prior.is_loaded(), "Prior should be disabled for missing file"
    assert len(prior.get_wall_segments()) == 0, "No segments when file missing"
    
    print("✓ Missing file fails closed (disabled)")
    
    # Clean up
    del os.environ['KEVIN_WALLS_STL_PRIOR']


def test_query_walls_near_pose():
    """Test querying walls near a pose."""
    fixture_path = os.path.join(
        os.path.dirname(__file__), 'test_fixtures', 'walls_test.stl')
    
    if not os.path.exists(fixture_path):
        print(f"⚠ Test fixture not found: {fixture_path} (skipping)")
        return
    
    os.environ['KEVIN_WALLS_STL_PRIOR'] = fixture_path
    prior = WallsSTLPrior()
    
    # Test fixture is a 3m × 2m room centered around origin
    # Query at center should find walls
    nearby_center = prior.query_walls_near_pose(1.5, 1.0, radius=2.0)
    assert len(nearby_center) > 0, "Should find walls near room center"
    
    # Query far away should find nothing
    nearby_far = prior.query_walls_near_pose(100.0, 100.0, radius=2.0)
    assert len(nearby_far) == 0, "Should find no walls far from room"
    
    print(f"✓ Query near center: {len(nearby_center)} segments")
    print(f"✓ Query far away: {len(nearby_far)} segments")
    
    # Clean up
    del os.environ['KEVIN_WALLS_STL_PRIOR']


def test_project_walls_to_map():
    """Test projecting wall segments to map pixel coordinates."""
    fixture_path = os.path.join(
        os.path.dirname(__file__), 'test_fixtures', 'walls_test.stl')
    
    if not os.path.exists(fixture_path):
        print(f"⚠ Test fixture not found: {fixture_path} (skipping)")
        return
    
    os.environ['KEVIN_WALLS_STL_PRIOR'] = fixture_path
    prior = WallsSTLPrior()
    
    # Project to a simple map (origin at 0,0, 1cm/px)
    segments_px = prior.project_walls_to_map(
        map_origin_x=0.0, map_origin_y=0.0, map_px_size=0.01)
    
    assert len(segments_px) > 0, "Should project wall segments to map"
    
    # Each segment should be (r0, c0, r1, c1) tuple
    for seg in segments_px:
        assert len(seg) == 4, "Segment should be (r0, c0, r1, c1)"
        r0, c0, r1, c1 = seg
        assert isinstance(r0, int), "Coordinates should be integers"
    
    print(f"✓ Projected {len(segments_px)} wall segments to map pixels")
    
    # Clean up
    del os.environ['KEVIN_WALLS_STL_PRIOR']


def test_segment_extraction_filters_noise():
    """Test that segment extraction filters out tiny edges."""
    # Create a degenerate triangle (very small)
    tiny_tri = np.array([
        [[0.0, 0.0, 1.0], [0.001, 0.0, 1.0], [0.0, 0.001, 1.0]]
    ], dtype=np.float32)
    
    segments = _extract_wall_segments_2d(tiny_tri)
    
    # Should filter out tiny edges (< 1cm threshold)
    assert len(segments) == 0, "Should filter out tiny edges"
    
    print("✓ Segment extraction filters noise (tiny edges)")


def test_wall_segments_format():
    """Test that extracted wall segments have correct format."""
    fixture_path = os.path.join(
        os.path.dirname(__file__), 'test_fixtures', 'walls_test.stl')
    
    if not os.path.exists(fixture_path):
        print(f"⚠ Test fixture not found: {fixture_path} (skipping)")
        return
    
    triangles = _load_stl_triangles(fixture_path)
    segments = _extract_wall_segments_2d(triangles)
    
    # Check format
    assert segments.dtype == np.float32, "Segments should be float32"
    assert segments.ndim == 2, "Segments should be 2D array"
    assert segments.shape[1] == 4, "Segments should be Mx4 [x0,y0,x1,y1]"
    
    # Check that segments have reasonable lengths (> 1cm)
    for seg in segments:
        x0, y0, x1, y1 = seg
        length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        assert length > 0.01, "Segment length should be > 1cm"
    
    print(f"✓ All {len(segments)} segments have valid format and lengths")


if __name__ == '__main__':
    print("Testing walls STL prior...")
    print()
    
    test_default_disabled()
    test_missing_file_fails_closed()
    test_load_test_fixture()
    test_load_via_env_var()
    test_query_walls_near_pose()
    test_project_walls_to_map()
    test_segment_extraction_filters_noise()
    test_wall_segments_format()
    
    print()
    print("All walls STL prior tests passed! ✓")
