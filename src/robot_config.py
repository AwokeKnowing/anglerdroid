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

FOOT_PAD_FWD = 0              # no extra forward self-clear; geometric front only (tune vs box in overlay)
FOOT_PAD_BWD = 10             # extra clear pixels backward (self-observation margin)
FOOT_PAD_LAT = 4              # extra clear pixels lateral (mast-inflate ghosts on spin)

# Derived: robot center in ego frame
RCX = CROSSHAIR_CX + ROBOT_CX_OFF   # 81
RCY = CROSSHAIR_CY                   # 119

# Multi-box footprint approximating robot 3D hull top-down projection.
# Body box (~30 wide) + 4 wheel boxes (corners) cover the true physical footprint.
# All boxes cleared to known-free in ego topdown map (self-mask for depth).

# Body box (central chassis, excluding wheels)
BODY_X0 = max(0, RCX - ROBOT_W // 2 - FOOT_PAD_BWD)
BODY_Y0 = max(0, RCY - ROBOT_H // 2)
BODY_X1 = min(FRAME_W, RCX + ROBOT_W // 2 + FOOT_PAD_FWD)
BODY_Y1 = min(FRAME_H, RCY + ROBOT_H // 2)

# Wheel boxes (4 corners, ~17 cm diameter wheels + margin)
WHEEL_HALF = 10  # pixels at 1 cm/px (~10 cm radius including margin)

# Front-left wheel
WHEEL_FL_X0 = max(0, RCX + ROBOT_W // 2 - 5 - WHEEL_HALF)
WHEEL_FL_Y0 = max(0, RCY - ROBOT_H // 2 - FOOT_PAD_LAT - WHEEL_HALF)
WHEEL_FL_X1 = min(FRAME_W, WHEEL_FL_X0 + 2 * WHEEL_HALF)
WHEEL_FL_Y1 = min(FRAME_H, WHEEL_FL_Y0 + 2 * WHEEL_HALF)

# Front-right wheel
WHEEL_FR_X0 = max(0, RCX + ROBOT_W // 2 - 5 - WHEEL_HALF)
WHEEL_FR_Y0 = max(0, RCY + ROBOT_H // 2 + FOOT_PAD_LAT - WHEEL_HALF)
WHEEL_FR_X1 = min(FRAME_W, WHEEL_FR_X0 + 2 * WHEEL_HALF)
WHEEL_FR_Y1 = min(FRAME_H, WHEEL_FR_Y0 + 2 * WHEEL_HALF)

# Back-left wheel
WHEEL_BL_X0 = max(0, RCX - ROBOT_W // 2 + 5 - WHEEL_HALF)
WHEEL_BL_Y0 = max(0, RCY - ROBOT_H // 2 - FOOT_PAD_LAT - WHEEL_HALF)
WHEEL_BL_X1 = min(FRAME_W, WHEEL_BL_X0 + 2 * WHEEL_HALF)
WHEEL_BL_Y1 = min(FRAME_H, WHEEL_BL_Y0 + 2 * WHEEL_HALF)

# Back-right wheel
WHEEL_BR_X0 = max(0, RCX - ROBOT_W // 2 + 5 - WHEEL_HALF)
WHEEL_BR_Y0 = max(0, RCY + ROBOT_H // 2 + FOOT_PAD_LAT - WHEEL_HALF)
WHEEL_BR_X1 = min(FRAME_W, WHEEL_BR_X0 + 2 * WHEEL_HALF)
WHEEL_BR_Y1 = min(FRAME_H, WHEEL_BR_Y0 + 2 * WHEEL_HALF)

# Legacy single-rect footprint bounds (for backward compat; prefer multi-box above)
# Encompasses body + wheels but as single rect (doesn't match actual hull shape)
FOOT_X0 = max(0, RCX - ROBOT_W // 2 - FOOT_PAD_BWD)
FOOT_Y0 = max(0, RCY - ROBOT_H // 2 - FOOT_PAD_LAT)
FOOT_X1 = min(FRAME_W, RCX + ROBOT_W // 2 + FOOT_PAD_FWD)
FOOT_Y1 = min(FRAME_H, RCY + ROBOT_H // 2 + FOOT_PAD_LAT)

# Mast / tall payload: floor can look free under a table while the mast hits the top.
# Obstacles at or above MAST_CLEAR_CM are treated as mast-colliders (overhangs).
MAST_CLEAR_CM = 45          # cm above floor → dangerous for mast
MAST_RADIUS_PX = 8           # ego-map half-width of mast column around centerline
MAST_INFLATE_PX = 12         # extra inflation for tall obstacles in safety/planning

# Self-mask semantics (James map metric):
# 1. UNDER_ROBOT_BOXES: floor under chassis/wheels footprint → mark clear+known (obs=0, known=255)
# 2. SELF_IGNORE_BOXES: robot body/mast self-hits → strip from obstacles but do NOT force known-clear
#    (obs=0, leave known unchanged — we can't see through ourselves)

# Under-robot clear zones: wheels + body floor (excluding mast column to avoid clearing real obstacles)
UNDER_ROBOT_BOXES = [
    (BODY_X0, BODY_Y0, BODY_X1, BODY_Y1),        # body floor
    (WHEEL_FL_X0, WHEEL_FL_Y0, WHEEL_FL_X1, WHEEL_FL_Y1),  # front-left wheel
    (WHEEL_FR_X0, WHEEL_FR_Y0, WHEEL_FR_X1, WHEEL_FR_Y1),  # front-right wheel
    (WHEEL_BL_X0, WHEEL_BL_Y0, WHEEL_BL_X1, WHEEL_BL_Y1),  # back-left wheel
    (WHEEL_BR_X0, WHEEL_BR_Y0, WHEEL_BR_X1, WHEEL_BR_Y1),  # back-right wheel
]

# Self-ignore zones: mast column (vertical strip around robot centerline)
# Mast self-reflection shows as fake obstacles → remove from obs but do NOT mark known-clear
MAST_SELF_X0 = max(0, RCX - ROBOT_W // 2 - FOOT_PAD_BWD)
MAST_SELF_Y0 = max(0, RCY - MAST_RADIUS_PX)
MAST_SELF_X1 = min(FRAME_W, RCX + ROBOT_W // 2 + FOOT_PAD_FWD)
MAST_SELF_Y1 = min(FRAME_H, RCY + MAST_RADIUS_PX)

SELF_IGNORE_BOXES = [
    (MAST_SELF_X0, MAST_SELF_Y0, MAST_SELF_X1, MAST_SELF_Y1),  # mast column
]

# Legacy: combined footprint (all boxes) for backward compat with viz code
FOOTPRINT_BOXES = UNDER_ROBOT_BOXES + SELF_IGNORE_BOXES

# RS1 camera-space visualization scale
# Ego-map boxes (1 cm/px orthographic) need scaling when drawn on RS1 color (perspective camera).
# RS1 color shows ~wider FOV with varying GSD; scale ~3–4× makes boxes cover robot hull + wheels.
# Tune these constants live in Rerun to align green/blue boxes with robot body in vision/rs1_mask_overlay.
# Map metric (UNDER_ROBOT_BOXES, SELF_IGNORE_BOXES) stays unchanged; this is viz-only.
RS1_VIZ_SCALE = 3.5           # scale ego boxes for RS1 color overlay (3.5 from mask_tune dumps)
RS1_VIZ_CX_SHIFT = 5          # forward shift in ego +x (pixels) to recenter scaled boxes on hull

# Self-mask overlay visualization opacity
# Fill alpha for under-robot / self-ignore boxes in both RS1 and ego footprint overlays.
# Low opacity (~25%) lets underlay/map show through; outlines stay crisp for edges.
SELF_MASK_VIZ_ALPHA = 0.20    # blend alpha for green/blue fills (0.20 = 20% opacity); override via ~/.kevin/mask_viz.json
