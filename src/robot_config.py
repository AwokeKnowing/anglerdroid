"""Shared robot geometry constants used by vision, safety, cameras, globalmap."""

# Frame geometry
FRAME_W, FRAME_H = 320, 240
CROSSHAIR_CX, CROSSHAIR_CY = 159, 119

# Ego-space pixel size (metres)
EGO_PX_SIZE = 0.010           # 1 px = 10 mm

# Physical robot dimensions (metres)
WHEEL_DIAMETER_CM = 17.13
WHEELBASE_CM = 34.0
WHEEL_RADIUS_M = WHEEL_DIAMETER_CM / 200.0   # ~0.08565
WHEELBASE_M = WHEELBASE_CM / 100.0            # 0.34

# Robot footprint on costmap (pixels). Robot faces RIGHT in ego frame.
ROBOT_W = 30                  # front-back (x direction)
ROBOT_H = 42                  # side-to-side (y direction)
ROBOT_CX_OFF = -78            # x offset from crosshair center

FOOT_PAD_FWD = 5              # extra clear pixels forward (self-reflection margin)
FOOT_PAD_BWD = 10             # extra clear pixels backward (self-observation margin)
FOOT_PAD_LAT = 2              # extra clear pixels lateral

# Derived: robot center in ego frame
RCX = CROSSHAIR_CX + ROBOT_CX_OFF   # 81
RCY = CROSSHAIR_CY                   # 119

# Derived: footprint clear bounds
FOOT_X0 = max(0, RCX - ROBOT_W // 2 - FOOT_PAD_BWD)
FOOT_Y0 = max(0, RCY - ROBOT_H // 2 - FOOT_PAD_LAT)
FOOT_X1 = min(FRAME_W, RCX + ROBOT_W // 2 + FOOT_PAD_FWD)
FOOT_Y1 = min(FRAME_H, RCY + ROBOT_H // 2 + FOOT_PAD_LAT)


# Mast / tall payload: floor can look free under a table while the mast hits the top.
# Obstacles at or above MAST_CLEAR_CM are treated as mast-colliders (overhangs).
MAST_CLEAR_CM = 45          # cm above floor → dangerous for mast
MAST_RADIUS_PX = 8           # ego-map half-width of mast column around centerline
MAST_INFLATE_PX = 12         # extra inflation for tall obstacles in safety/planning
