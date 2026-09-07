"""Enhanced world scenarios for high-fidelity host simulator.

Extends sim/world.py with:
- Table overhangs (mast crash case)
- Soft obstacles (dog bed) with cost tuning
- Checkered keepout mats
- Tight doorways + hallway pinches
"""

import numpy as np
from sim.world import (
    FRAME_W, FRAME_H, RCX, RCY, EGO_PX_SIZE,
    FOOT_X0, FOOT_Y0, FOOT_X1, FOOT_Y1,
    _add_rect, create_scenario as create_base_scenario
)

# Soft obstacle cost scaling (for MPPI cost model)
SOFT_OBSTACLE_COST_SCALE = 0.5  # 50% of normal obstacle cost


def create_enhanced_scenario(name: str):
    """Create enhanced scenario with high-fidelity features.
    
    Returns:
        (obs_map, height_map, metadata) - both (FRAME_H, FRAME_W) uint8
        metadata: dict with scenario-specific info (soft zones, keepouts, etc.)
    """
    obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    metadata = {}
    
    if name == "table_overhang":
        _add_table_overhang(obs, height, metadata)
    elif name == "table_overhang_tight":
        _add_table_overhang_tight(obs, height, metadata)
    elif name == "soft_bed":
        _add_soft_bed(obs, height, metadata)
    elif name == "keepout_mat":
        _add_keepout_mat(obs, height, metadata)
    elif name == "tight_doorway":
        _add_tight_doorway(obs, height, metadata)
    elif name == "hallway_pinch":
        _add_hallway_pinch(obs, height, metadata)
    elif name == "combined_house":
        _add_combined_house(obs, height, metadata)
    else:
        # Fall back to base scenarios
        obs, height = create_base_scenario(name)
        metadata = {}
    
    return obs, height, metadata


def _add_table_overhang(obs, height, metadata):
    """Dining table with overhang: mast crash test case.
    
    Floor clear under table (RS1 sees free), but mast hits top at 45cm+.
    Topdown near-field reflex must trigger <30cm from edge.
    """
    # Perimeter walls
    _add_rect(obs, height, 0, 0, 8, FRAME_H, h_cm=200)
    _add_rect(obs, height, FRAME_W - 8, 0, FRAME_W, FRAME_H, h_cm=200)
    _add_rect(obs, height, 0, 0, FRAME_W, 8, h_cm=200)
    _add_rect(obs, height, 0, FRAME_H - 8, FRAME_W, FRAME_H, h_cm=200)
    
    # Table: 80cm tall, 120cm wide, ~60cm ahead
    table_x0 = RCX + 60
    table_y0 = RCY - 60
    table_x1 = table_x0 + 120
    table_y1 = table_y0 + 120
    
    # Table legs (only obstacles at floor level)
    leg_w = 8
    _add_rect(obs, height, table_x0, table_y0, table_x0 + leg_w, table_y0 + leg_w, h_cm=80)
    _add_rect(obs, height, table_x1 - leg_w, table_y0, table_x1, table_y0 + leg_w, h_cm=80)
    _add_rect(obs, height, table_x0, table_y1 - leg_w, table_x0 + leg_w, table_y1, h_cm=80)
    _add_rect(obs, height, table_x1 - leg_w, table_y1 - leg_w, table_x1, table_y1, h_cm=80)
    
    # Table top (overhang zone: height=80cm, should be detected by topdown RS1)
    # Mark as height only (not floor obstacle) to simulate overhang
    table_top_region = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    table_top_region[table_y0:table_y1, table_x0:table_x1] = True
    
    # Remove leg footprints from overhang region
    table_top_region[table_y0:table_y0 + leg_w, table_x0:table_x0 + leg_w] = False
    table_top_region[table_y0:table_y0 + leg_w, table_x1 - leg_w:table_x1] = False
    table_top_region[table_y1 - leg_w:table_y1, table_x0:table_x0 + leg_w] = False
    table_top_region[table_y1 - leg_w:table_y1, table_x1 - leg_w:table_x1] = False
    
    # Height map for table top (80cm)
    height[table_top_region] = 80
    
    # For SafetyGuard: treat mast-height obstacles as blocking
    # (RS1 topdown should see this and trigger stop)
    obs[table_top_region] = 255  # Mark as occupied for safety
    
    metadata['table_bounds'] = (table_x0, table_y0, table_x1, table_y1)
    metadata['table_height_cm'] = 80
    metadata['near_field_trigger_distance_px'] = 30  # Should trigger <30cm


def _add_table_overhang_tight(obs, height, metadata):
    """Table overhang with narrow clearance on sides (must navigate around)."""
    # Similar to table_overhang but with side walls closer
    _add_rect(obs, height, 0, 0, 8, FRAME_H, h_cm=200)
    _add_rect(obs, height, FRAME_W - 8, 0, FRAME_W, FRAME_H, h_cm=200)
    _add_rect(obs, height, 0, 0, FRAME_W, 8, h_cm=200)
    _add_rect(obs, height, 0, FRAME_H - 8, FRAME_W, FRAME_H, h_cm=200)
    
    # Narrow side passages (only 50cm clearance)
    _add_rect(obs, height, 0, 0, RCX - 50, FRAME_H, h_cm=200)
    _add_rect(obs, height, RCX + 50, 0, FRAME_W, FRAME_H, h_cm=200)
    
    # Table ahead
    table_x0 = RCX + 40
    table_y0 = RCY - 40
    table_x1 = table_x0 + 80
    table_y1 = table_y0 + 80
    
    leg_w = 6
    _add_rect(obs, height, table_x0, table_y0, table_x0 + leg_w, table_y0 + leg_w, h_cm=80)
    _add_rect(obs, height, table_x1 - leg_w, table_y0, table_x1, table_y0 + leg_w, h_cm=80)
    _add_rect(obs, height, table_x0, table_y1 - leg_w, table_x0 + leg_w, table_y1, h_cm=80)
    _add_rect(obs, height, table_x1 - leg_w, table_y1 - leg_w, table_x1, table_y1, h_cm=80)
    
    table_top_region = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    table_top_region[table_y0:table_y1, table_x0:table_x1] = True
    table_top_region[table_y0:table_y0 + leg_w, table_x0:table_x0 + leg_w] = False
    table_top_region[table_y0:table_y0 + leg_w, table_x1 - leg_w:table_x1] = False
    table_top_region[table_y1 - leg_w:table_y1, table_x0:table_x0 + leg_w] = False
    table_top_region[table_y1 - leg_w:table_y1, table_x1 - leg_w:table_x1] = False
    
    height[table_top_region] = 80
    obs[table_top_region] = 255
    
    metadata['table_bounds'] = (table_x0, table_y0, table_x1, table_y1)


def _add_soft_bed(obs, height, metadata):
    """Soft dog bed: compressible obstacle with reduced MPPI cost.
    
    Policy should prefer detours but tolerate edge contact if trapped.
    """
    # Perimeter walls
    _add_rect(obs, height, 0, 0, 8, FRAME_H, h_cm=200)
    _add_rect(obs, height, FRAME_W - 8, 0, FRAME_W, FRAME_H, h_cm=200)
    _add_rect(obs, height, 0, 0, FRAME_W, 8, h_cm=200)
    _add_rect(obs, height, 0, FRAME_H - 8, FRAME_W, FRAME_H, h_cm=200)
    
    # Dog bed: circular-ish (~80cm diameter, 15cm tall)
    bed_cx = RCX + 80
    bed_cy = RCY
    bed_radius = 40  # px
    
    for y in range(FRAME_H):
        for x in range(FRAME_W):
            dx = x - bed_cx
            dy = y - bed_cy
            if dx * dx + dy * dy <= bed_radius * bed_radius:
                obs[y, x] = 128  # Soft obstacle (not 255)
                height[y, x] = 15
    
    # Metadata for MPPI cost scaling
    metadata['soft_zones'] = [
        {
            'center': (bed_cx, bed_cy),
            'radius': bed_radius,
            'cost_scale': SOFT_OBSTACLE_COST_SCALE,
            'description': 'dog_bed',
        }
    ]


def _add_keepout_mat(obs, height, metadata):
    """Checkered keepout mat: visual-only boundary (no physical obstacle).
    
    Detected by RGB classifier in house_bot; injected as virtual obstacle in obs map.
    For sim: pre-mark keepout pixels (skip RGB processing).
    """
    # Perimeter walls
    _add_rect(obs, height, 0, 0, 8, FRAME_H, h_cm=200)
    _add_rect(obs, height, FRAME_W - 8, 0, FRAME_W, FRAME_H, h_cm=200)
    _add_rect(obs, height, 0, 0, FRAME_W, 8, h_cm=200)
    _add_rect(obs, height, 0, FRAME_H - 8, FRAME_W, FRAME_H, h_cm=200)
    
    # Keepout mat: horizontal stripe ahead
    mat_y0 = RCY - 60
    mat_y1 = mat_y0 + 40
    mat_x0 = RCX + 50
    mat_x1 = mat_x0 + 100
    
    # Checkered pattern (8×8 px squares)
    for y in range(mat_y0, mat_y1, 8):
        for x in range(mat_x0, mat_x1, 8):
            if ((y - mat_y0) // 8 + (x - mat_x0) // 8) % 2 == 0:
                _add_rect(obs, height, x, y, min(x + 8, mat_x1), min(y + 8, mat_y1), h_cm=0)
                obs[y:min(y + 8, mat_y1), x:min(x + 8, mat_x1)] = 200  # Virtual obstacle
    
    metadata['keepout_zones'] = [
        {
            'bounds': (mat_x0, mat_y0, mat_x1, mat_y1),
            'pattern': 'checkered',
        }
    ]


def _add_tight_doorway(obs, height, metadata):
    """Residential doorway: 70cm clear width (tighter than base doorway).
    
    FOOT lateral span ~42cm + safety margins → requires precise alignment.
    """
    # Outer walls
    _add_rect(obs, height, 0, 0, 8, FRAME_H, h_cm=200)
    _add_rect(obs, height, FRAME_W - 8, 0, FRAME_W, FRAME_H, h_cm=200)
    _add_rect(obs, height, 0, 0, FRAME_W, 8, h_cm=200)
    _add_rect(obs, height, 0, FRAME_H - 8, FRAME_W, FRAME_H, h_cm=200)
    
    # Dividing wall with tight doorway
    wall_y0 = RCY - 50
    wall_y1 = wall_y0 + 10
    door_width = 70  # px (~70cm)
    door_x0 = RCX + 10
    door_x1 = door_x0 + door_width
    
    _add_rect(obs, height, 8, wall_y0, door_x0, wall_y1, h_cm=200)
    _add_rect(obs, height, door_x1, wall_y0, FRAME_W - 8, wall_y1, h_cm=200)
    
    metadata['doorway'] = {
        'wall_y': wall_y0 * EGO_PX_SIZE,
        'door_x0': door_x0 * EGO_PX_SIZE,
        'door_x1': door_x1 * EGO_PX_SIZE,
        'width_m': door_width * EGO_PX_SIZE,
    }


def _add_hallway_pinch(obs, height, metadata):
    """Long hallway with pinch point (narrows to 60cm in middle)."""
    # Side walls
    _add_rect(obs, height, 0, 0, RCX - 50, FRAME_H, h_cm=200)
    _add_rect(obs, height, RCX + 50, 0, FRAME_W, FRAME_H, h_cm=200)
    
    # Pinch point: protrusions narrow corridor to 60cm
    pinch_y0 = RCY - 80
    pinch_y1 = RCY - 60
    _add_rect(obs, height, RCX - 50, pinch_y0, RCX - 30, pinch_y1, h_cm=200)
    _add_rect(obs, height, RCX + 30, pinch_y0, RCX + 50, pinch_y1, h_cm=200)
    
    # Dead end ahead
    _add_rect(obs, height, RCX - 50, RCY - 120, RCX + 50, RCY - 110, h_cm=200)
    
    metadata['pinch'] = {
        'location_y': pinch_y0 * EGO_PX_SIZE,
        'width_m': 0.60,
    }


def _add_combined_house(obs, height, metadata):
    """Full house with table, soft bed, keepout mat, and tight doorway.
    
    Stress test: navigate through multiple challenge types.
    """
    # Start with perimeter
    _add_rect(obs, height, 0, 0, 8, FRAME_H, h_cm=200)
    _add_rect(obs, height, FRAME_W - 8, 0, FRAME_W, FRAME_H, h_cm=200)
    _add_rect(obs, height, 0, 0, FRAME_W, 8, h_cm=200)
    _add_rect(obs, height, 0, FRAME_H - 8, FRAME_W, FRAME_H, h_cm=200)
    
    # Couch (left side)
    _add_rect(obs, height, 20, 80, 80, 140, h_cm=40)
    
    # Table with overhang (front-right)
    table_x0 = RCX + 50
    table_y0 = RCY - 50
    table_x1 = table_x0 + 80
    table_y1 = table_y0 + 80
    
    leg_w = 6
    _add_rect(obs, height, table_x0, table_y0, table_x0 + leg_w, table_y0 + leg_w, h_cm=75)
    _add_rect(obs, height, table_x1 - leg_w, table_y0, table_x1, table_y0 + leg_w, h_cm=75)
    _add_rect(obs, height, table_x0, table_y1 - leg_w, table_x0 + leg_w, table_y1, h_cm=75)
    _add_rect(obs, height, table_x1 - leg_w, table_y1 - leg_w, table_x1, table_y1, h_cm=75)
    
    table_top = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    table_top[table_y0:table_y1, table_x0:table_x1] = True
    table_top[table_y0:table_y0 + leg_w, table_x0:table_x0 + leg_w] = False
    table_top[table_y0:table_y0 + leg_w, table_x1 - leg_w:table_x1] = False
    table_top[table_y1 - leg_w:table_y1, table_x0:table_x0 + leg_w] = False
    table_top[table_y1 - leg_w:table_y1, table_x1 - leg_w:table_x1] = False
    height[table_top] = 75
    obs[table_top] = 255
    
    # Dog bed (back-left)
    bed_cx = 50
    bed_cy = RCY + 60
    bed_radius = 30
    for y in range(FRAME_H):
        for x in range(FRAME_W):
            dx = x - bed_cx
            dy = y - bed_cy
            if dx * dx + dy * dy <= bed_radius * bed_radius:
                obs[y, x] = 128
                height[y, x] = 12
    
    metadata['soft_zones'] = [{
        'center': (bed_cx, bed_cy),
        'radius': bed_radius,
        'cost_scale': SOFT_OBSTACLE_COST_SCALE,
    }]
    
    # Chairs (scattered)
    _add_rect(obs, height, 180, 60, 200, 80, h_cm=45)
    _add_rect(obs, height, 220, 120, 240, 140, h_cm=45)


def scenario_start_enhanced(name: str):
    """Return start pose for enhanced scenarios.
    
    Returns:
        (x_m, y_m, theta_rad)
    """
    default_x = RCX * EGO_PX_SIZE
    default_y = RCY * EGO_PX_SIZE
    default_theta = 0.0
    
    if name in ("table_overhang", "table_overhang_tight", "soft_bed", "tight_doorway"):
        return (default_x, default_y, default_theta)
    elif name == "hallway_pinch":
        # Start further back to give room
        return (default_x, (RCY + 40) * EGO_PX_SIZE, default_theta)
    elif name == "combined_house":
        return (default_x, default_y, default_theta)
    else:
        return (default_x, default_y, default_theta)


def get_soft_cost_scale(obs_map, x_px, y_px, metadata):
    """Return cost scale for position (for MPPI soft obstacle scoring).
    
    Args:
        obs_map: (H, W) uint8 obstacle map
        x_px, y_px: Pixel coordinates
        metadata: Scenario metadata with 'soft_zones'
        
    Returns:
        cost_scale: float [0, 1] (1.0 = normal, <1.0 = soft)
    """
    if 'soft_zones' not in metadata:
        return 1.0
    
    for zone in metadata['soft_zones']:
        cx, cy = zone['center']
        radius = zone['radius']
        dx = x_px - cx
        dy = y_px - cy
        if dx * dx + dy * dy <= radius * radius:
            return zone['cost_scale']
    
    return 1.0
