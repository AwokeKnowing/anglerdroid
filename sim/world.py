"""Simple house floorplan with walls, couch, table, chairs.

Binary occupancy + optional height_cm map matching real ego map geometry.
"""

import numpy as np

FRAME_W = 320
FRAME_H = 240
RCX = 81
RCY = 119
EGO_PX_SIZE = 0.01

FOOT_X0 = 56
FOOT_Y0 = 97
FOOT_X1 = 101
FOOT_Y1 = 142
FOOT_PAD_FWD = 5
FOOT_PAD_BWD = 10
FOOT_PAD_LAT = 2


def create_scenario(name: str):
    """Create a scenario with occupancy and height maps.
    
    Returns:
        (obs_map, height_map) - both (FRAME_H, FRAME_W) uint8
        obs_map: 0=clear, 255=occupied
        height_map: cm above floor (0-255)
    """
    obs = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    height = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    
    if name == "empty":
        pass

    elif name == "couch_pinch":
        _add_couch_pinch(obs, height)

    elif name == "house":
        _add_house(obs, height)

    elif name == "hallway":
        _add_hallway(obs, height)

    elif name == "doorway":
        _add_doorway(obs, height)

    elif name == "l_corner":
        _add_l_corner(obs, height)

    else:
        raise ValueError(f"Unknown scenario: {name}")
    
    return obs, height


def _add_rect(obs, height, x0, y0, x1, y1, h_cm):
    """Add a rectangular obstacle."""
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(FRAME_W, x1)
    y1 = min(FRAME_H, y1)
    obs[y0:y1, x0:x1] = 255
    height[y0:y1, x0:x1] = min(255, h_cm)


def _add_couch_pinch(obs, height):
    """Scenario: couch directly ahead, robot can escape backward or spin."""
    couch_x0 = RCX + 20
    couch_y0 = RCY - 60
    couch_x1 = couch_x0 + 80
    couch_y1 = couch_y0 + 120
    _add_rect(obs, height, couch_x0, couch_y0, couch_x1, couch_y1, h_cm=40)
    
    wall_left_x0 = 0
    wall_left_x1 = 10
    _add_rect(obs, height, wall_left_x0, 0, wall_left_x1, FRAME_H, h_cm=200)
    
    wall_right_x0 = FRAME_W - 10
    wall_right_x1 = FRAME_W
    _add_rect(obs, height, wall_right_x0, 0, wall_right_x1, FRAME_H, h_cm=200)


def _add_house(obs, height):
    """Scenario: full house with walls, couch, table, chairs."""
    _add_rect(obs, height, 0, 0, 10, FRAME_H, h_cm=200)
    _add_rect(obs, height, FRAME_W-10, 0, FRAME_W, FRAME_H, h_cm=200)
    _add_rect(obs, height, 0, 0, FRAME_W, 10, h_cm=200)
    _add_rect(obs, height, 0, FRAME_H-10, FRAME_W, FRAME_H, h_cm=200)
    
    _add_rect(obs, height, 140, 80, 200, 140, h_cm=40)
    
    _add_rect(obs, height, 60, 180, 120, 220, h_cm=80)
    
    _add_rect(obs, height, 200, 180, 220, 200, h_cm=45)
    _add_rect(obs, height, 240, 180, 260, 200, h_cm=45)



def _add_hallway(obs, height):
    """Long corridor with walls on both sides; clear ahead then a dead-end."""
    # Side walls leave ~0.9 m corridor centered on robot
    _add_rect(obs, height, 0, 0, RCX - 45, FRAME_H, h_cm=200)
    _add_rect(obs, height, RCX + 45, 0, FRAME_W, FRAME_H, h_cm=200)
    # Dead-end wall ahead
    _add_rect(obs, height, RCX - 45, RCY - 100, RCX + 45, RCY - 85, h_cm=200)


def _add_doorway(obs, height):
    """Room walls with a narrow doorway offset to the right of heading."""
    # Outer walls
    _add_rect(obs, height, 0, 0, 10, FRAME_H, h_cm=200)
    _add_rect(obs, height, FRAME_W - 10, 0, FRAME_W, FRAME_H, h_cm=200)
    _add_rect(obs, height, 0, 0, FRAME_W, 10, h_cm=200)
    _add_rect(obs, height, 0, FRAME_H - 10, FRAME_W, FRAME_H, h_cm=200)
    # Dividing wall ahead with doorway gap (~35 px / ~35 cm)
    wall_y0 = RCY - 55
    wall_y1 = wall_y0 + 12
    door_x0 = RCX + 10
    door_x1 = door_x0 + 35
    _add_rect(obs, height, 10, wall_y0, door_x0, wall_y1, h_cm=200)
    _add_rect(obs, height, door_x1, wall_y0, FRAME_W - 10, wall_y1, h_cm=200)


def _add_l_corner(obs, height):
    """L-shaped obstacle forcing a left turn; right side blocked.

    Geometry stays clear of the start footprint (FOOT ~56..101, 97..142).
    """
    # Outer walls
    _add_rect(obs, height, 0, 0, 10, FRAME_H, h_cm=200)
    _add_rect(obs, height, FRAME_W - 10, 0, FRAME_W, FRAME_H, h_cm=200)
    _add_rect(obs, height, 0, 0, FRAME_W, 10, h_cm=200)
    _add_rect(obs, height, 0, FRAME_H - 10, FRAME_W, FRAME_H, h_cm=200)
    # Block ahead+right, leave left open; keep >=15 px clear of FOOT_X1/FOOT_Y0
    _add_rect(obs, height, FOOT_X1 + 20, 10, FRAME_W - 10, RCY + 50, h_cm=50)
    _add_rect(obs, height, RCX - 10, 10, FOOT_X1 + 20, FOOT_Y0 - 25, h_cm=50)

def world_to_pixel(x_m, y_m):
    """Convert world coords (meters) to pixel coords."""
    px = int(round(x_m / EGO_PX_SIZE))
    py = int(round(y_m / EGO_PX_SIZE))
    return px, py


def pixel_to_world(px, py):
    """Convert pixel coords to world coords (meters)."""
    x_m = px * EGO_PX_SIZE
    y_m = py * EGO_PX_SIZE
    return x_m, y_m
