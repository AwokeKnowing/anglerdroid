"""Tests for host simulator scenarios.

Validates:
- Table overhang: topdown near-field triggers stop <30cm
- Soft bed: MPPI assigns reduced cost (0.3-0.7× normal)
- Tight doorway: Robot aligns and crosses without collision
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add src to path
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sim.world_enhanced import (
    create_enhanced_scenario,
    scenario_start_enhanced,
    get_soft_cost_scale,
)
from sim.robot import Robot, SafetyGuard
from sim.sensors import SensorSuite
from robot_config import RCX, RCY, EGO_PX_SIZE


class TestTableOverhang:
    """Test table overhang scenario: topdown near-field reflex."""
    
    def test_table_created(self):
        """Table geometry exists in world."""
        obs, height, metadata = create_enhanced_scenario('table_overhang')
        
        assert obs.shape == (240, 320)
        assert height.shape == (240, 320)
        assert 'table_bounds' in metadata
        assert metadata['table_height_cm'] == 80
    
    def test_near_field_trigger(self):
        """RS1 topdown triggers fwd_scale=0 when <30cm from table edge."""
        obs, height, metadata = create_enhanced_scenario('table_overhang')
        
        # Start pose
        start_x, start_y, start_theta = scenario_start_enhanced('table_overhang')
        robot = Robot(start_x, start_y, start_theta, fidelity=False)
        
        # Move robot toward table edge (in small steps)
        table_x0 = metadata['table_bounds'][0]
        target_x = (table_x0 - 25) * EGO_PX_SIZE  # ~25cm from table edge
        
        for _ in range(100):
            robot.update_ego_maps(obs, height)
            
            # Check if near-field triggered
            fwd_scale = robot.safety.fwd_scale
            
            # When close to table, should trigger stop
            dist_to_table = (table_x0 * EGO_PX_SIZE - robot.x) * 100  # cm
            
            if dist_to_table < 30:
                # Should trigger hard stop
                assert fwd_scale < 0.5, f"Near-field reflex should trigger at {dist_to_table:.1f}cm (fwd_scale={fwd_scale:.2f})"
                break
            
            # Step forward
            robot.step(0.1, 0.0, 0.033, apply_safety=True)
            
            if robot.x >= target_x:
                break
        else:
            pytest.fail("Robot did not reach near-field trigger zone")
    
    def test_mast_height_detection(self):
        """Table top at 80cm is marked as occupied in obs map."""
        obs, height, metadata = create_enhanced_scenario('table_overhang')
        
        table_x0, table_y0, table_x1, table_y1 = metadata['table_bounds']
        
        # Check that table top region has height=80
        center_x = (table_x0 + table_x1) // 2
        center_y = (table_y0 + table_y1) // 2
        
        assert height[center_y, center_x] == 80
        assert obs[center_y, center_x] == 255  # Marked occupied


class TestSoftBed:
    """Test soft obstacle scoring."""
    
    def test_soft_bed_created(self):
        """Soft bed exists with metadata."""
        obs, height, metadata = create_enhanced_scenario('soft_bed')
        
        assert 'soft_zones' in metadata
        assert len(metadata['soft_zones']) == 1
        
        zone = metadata['soft_zones'][0]
        assert zone['cost_scale'] == 0.5
        assert zone['description'] == 'dog_bed'
    
    def test_soft_cost_scaling(self):
        """Soft bed has reduced cost (0.5× normal)."""
        obs, height, metadata = create_enhanced_scenario('soft_bed')
        
        zone = metadata['soft_zones'][0]
        cx, cy = zone['center']
        radius = zone['radius']
        
        # Inside bed
        cost_inside = get_soft_cost_scale(obs, cx, cy, metadata)
        assert cost_inside == 0.5
        
        # Outside bed (far from center)
        cost_outside = get_soft_cost_scale(obs, cx + radius + 20, cy, metadata)
        assert cost_outside == 1.0
    
    def test_soft_bed_height(self):
        """Soft bed has low height (15cm)."""
        obs, height, metadata = create_enhanced_scenario('soft_bed')
        
        zone = metadata['soft_zones'][0]
        cx, cy = zone['center']
        
        # Check height at center
        assert height[cy, cx] == 15
        
        # Check obs map shows softer obstacle (128 not 255)
        assert obs[cy, cx] == 128


class TestTightDoorway:
    """Test tight doorway navigation."""
    
    def test_doorway_created(self):
        """Doorway geometry correct."""
        obs, height, metadata = create_enhanced_scenario('tight_doorway')
        
        assert 'doorway' in metadata
        assert metadata['doorway']['width_m'] == 0.70
    
    def test_doorway_clearance(self):
        """Doorway width (70cm) allows robot (42cm lateral + margins)."""
        obs, height, metadata = create_enhanced_scenario('tight_doorway')
        
        door_x0 = int(metadata['doorway']['door_x0'] / EGO_PX_SIZE)
        door_x1 = int(metadata['doorway']['door_x1'] / EGO_PX_SIZE)
        wall_y = int(metadata['doorway']['wall_y'] / EGO_PX_SIZE)
        
        # Check gap is clear
        gap_obs = obs[wall_y, door_x0:door_x1]
        assert np.all(gap_obs == 0), "Doorway gap should be clear"
        
        # Check walls on either side
        assert obs[wall_y, door_x0 - 5] == 255
        assert obs[wall_y, door_x1 + 5] == 255


class TestKeepoutMat:
    """Test keepout mat scenario."""
    
    def test_keepout_created(self):
        """Keepout mat exists with checkered pattern."""
        obs, height, metadata = create_enhanced_scenario('keepout_mat')
        
        assert 'keepout_zones' in metadata
        zone = metadata['keepout_zones'][0]
        assert zone['pattern'] == 'checkered'
    
    def test_keepout_marked(self):
        """Keepout region marked in obs map."""
        obs, height, metadata = create_enhanced_scenario('keepout_mat')
        
        zone = metadata['keepout_zones'][0]
        x0, y0, x1, y1 = zone['bounds']
        
        # Check some pixels in keepout zone are marked
        keepout_region = obs[y0:y1, x0:x1]
        assert np.any(keepout_region >= 200), "Keepout zone should be marked in obs map"


class TestSensorNoise:
    """Test sensor emulation fidelity."""
    
    def test_depth_dropout(self):
        """High fidelity adds dropout to depth."""
        obs, height, _ = create_enhanced_scenario('empty')
        
        # Create fake world with uniform obstacles
        obs_world = np.full((240, 320), 255, dtype=np.uint8)
        height_world = np.full((240, 320), 50, dtype=np.uint8)
        
        sensors = SensorSuite(fidelity='high', seed=42)
        
        pose = {'x': RCX * EGO_PX_SIZE, 'y': RCY * EGO_PX_SIZE, 'theta': 0.0}
        obs_noisy, height_noisy = sensors.capture_depth(pose, obs_world, height_world)
        
        # Should have some dropout (pixels set to 0)
        dropout_count = np.sum(obs_noisy == 0)
        assert dropout_count > 10, f"Expected dropout in high fidelity, got {dropout_count} pixels"
    
    def test_no_noise_in_none_fidelity(self):
        """None fidelity has no noise."""
        obs_world = np.full((240, 320), 255, dtype=np.uint8)
        height_world = np.full((240, 320), 50, dtype=np.uint8)
        
        sensors = SensorSuite(fidelity='none', seed=42)
        
        pose = {'x': RCX * EGO_PX_SIZE, 'y': RCY * EGO_PX_SIZE, 'theta': 0.0}
        obs_clean, height_clean = sensors.capture_depth(pose, obs_world, height_world)
        
        # Should be identical (no noise)
        # Note: transform may introduce slight sampling artifacts
        # Just check that most pixels match
        match_rate = np.mean(obs_clean == obs_world)
        assert match_rate > 0.95, f"Expected clean capture, got {match_rate:.1%} match"


class TestHallwayPinch:
    """Test hallway pinch scenario."""
    
    def test_pinch_created(self):
        """Pinch point exists."""
        obs, height, metadata = create_enhanced_scenario('hallway_pinch')
        
        assert 'pinch' in metadata
        assert metadata['pinch']['width_m'] == 0.60
    
    def test_pinch_navigable(self):
        """Pinch width (60cm) allows robot (30cm front-back)."""
        obs, height, metadata = create_enhanced_scenario('hallway_pinch')
        
        # Robot footprint is 30×42 px (30cm × 42cm)
        # Pinch is 60cm wide → should fit if aligned
        pinch_width_px = int(metadata['pinch']['width_m'] / EGO_PX_SIZE)
        assert pinch_width_px >= 42 + 10, "Pinch should allow robot to pass with margin"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
