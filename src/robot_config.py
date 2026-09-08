"""Shared robot geometry constants used by vision, safety, cameras, globalmap."""

# Frame geometry
FRAME_W, FRAME_H = 320, 240
CROSSHAIR_CX, CROSSHAIR_CY = 159, 119

# Ego-space pixel size (metres)
EGO_PX_SIZE = 0.010           # 1 px = 10 mm

# Physical robot dimensions (metres / cm)
WHEEL_DIAMETER_CM = 17.13
WHEELBASE_CM = 34.0            # distance between left/right wheel centers
WHEEL_RADIUS_M = WHEEL_DIAMETER_CM / 200.0   # ~0.08565
WHEELBASE_M = WHEELBASE_CM / 100.0            # 0.34

# ---------------------------------------------------------------------------
# Self-mask / footprint geometry — ALL offsets in cm relative to the axle
# midpoint (point halfway between the two wheel centers).
#
# Ego frame: robot faces +X (right on the image). +Y is robot-left
# (up on the image → decreasing row). EGO_PX_SIZE = 1 cm/px so cm == px.
# Axle origin in ego pixels: (RCX, RCY).
# ---------------------------------------------------------------------------

# Body: 30 cm front-back × 33 cm left-right, centered on axle
BODY_LEN_CM = 30.0            # along +X (forward)
BODY_WID_CM = 33.0            # along +Y (left)
BODY_CX_CM = 0.0              # body center vs axle
BODY_CY_CM = 0.0

# Wheels: one box each, 18×6 cm (diameter × tire width), centers on axle line
WHEEL_LEN_CM = 18.0           # along +X
WHEEL_WID_CM = 6.0            # along +Y
WHEEL_L_CX_CM = 0.0
WHEEL_L_CY_CM = WHEELBASE_CM / 2.0    # left (+Y)
WHEEL_R_CX_CM = 0.0
WHEEL_R_CY_CM = -WHEELBASE_CM / 2.0   # right (−Y)

# Mast column (self-hits, not floor-clear). Size/offset from URDF (~3 cm post
# near back wall); slightly fattened so depth self-returns get stripped.
MAST_LEN_CM = 8.0
MAST_WID_CM = 8.0
MAST_CX_CM = -13.0            # behind axle (URDF mast_cx ≈ −0.135 m)
MAST_CY_CM = 0.0

# Legacy aliases (consumers still import these names)
ROBOT_W = int(round(BODY_LEN_CM))   # front-back
ROBOT_H = int(round(BODY_WID_CM))   # side-to-side
ROBOT_CX_OFF = -78            # x offset of axle from crosshair center
WHEEL_HALF = int(round(WHEEL_LEN_CM / 2.0))  # compat import only

# Safety scan pads (NOT baked into self-mask boxes)
FOOT_PAD_FWD = 0
FOOT_PAD_BWD = 10
FOOT_PAD_LAT = 4

# Axle origin in ego pixels
RCX = CROSSHAIR_CX + ROBOT_CX_OFF   # 81
RCY = CROSSHAIR_CY                   # 119


def _ego_box_from_axle(cx_cm, cy_cm, sx_cm, sy_cm):
    """Axis-aligned ego pixel box from axle-relative cm (center + size).

    +x_cm forward → +col; +y_cm left → −row.
    Returns (x0, y0, x1, y1) with exclusive x1/y1 so (x1-x0, y1-y0) == size.
    """
    sx = int(round(sx_cm))
    sy = int(round(sy_cm))
    pcx = RCX + float(cx_cm)
    pcy = RCY - float(cy_cm)  # left → up on image
    x0 = int(round(pcx - sx / 2.0))
    y0 = int(round(pcy - sy / 2.0))
    x1 = x0 + sx
    y1 = y0 + sy
    return (max(0, x0), max(0, y0), min(FRAME_W, x1), min(FRAME_H, y1))


BODY_BOX = _ego_box_from_axle(BODY_CX_CM, BODY_CY_CM, BODY_LEN_CM, BODY_WID_CM)
WHEEL_L_BOX = _ego_box_from_axle(WHEEL_L_CX_CM, WHEEL_L_CY_CM, WHEEL_LEN_CM, WHEEL_WID_CM)
WHEEL_R_BOX = _ego_box_from_axle(WHEEL_R_CX_CM, WHEEL_R_CY_CM, WHEEL_LEN_CM, WHEEL_WID_CM)
MAST_BOX = _ego_box_from_axle(MAST_CX_CM, MAST_CY_CM, MAST_LEN_CM, MAST_WID_CM)

# Named pixel aliases for body (compat)
BODY_X0, BODY_Y0, BODY_X1, BODY_Y1 = BODY_BOX

# Legacy FOOT_* AABB around hull + safety pads (safety.py forward/back scans)
_hull = [BODY_BOX, WHEEL_L_BOX, WHEEL_R_BOX, MAST_BOX]
FOOT_X0 = max(0, min(b[0] for b in _hull) - FOOT_PAD_BWD)
FOOT_Y0 = max(0, min(b[1] for b in _hull) - FOOT_PAD_LAT)
FOOT_X1 = min(FRAME_W, max(b[2] for b in _hull) + FOOT_PAD_FWD)
FOOT_Y1 = min(FRAME_H, max(b[3] for b in _hull) + FOOT_PAD_LAT)

# Mast / tall payload planning constants
MAST_CLEAR_CM = 45          # cm above floor → dangerous for mast
MAST_RADIUS_PX = int(round(MAST_WID_CM / 2.0))
MAST_INFLATE_PX = 12

# Self-mask semantics (James map metric / perception CONTRACT):
# 1. UNDER_ROBOT_BOXES: body+wheels footprint → strip obs only; do NOT force known=255
#    (invented under-chassis CLEAR is forbidden; leave known as sensed / unknown)
# 2. SELF_IGNORE_BOXES: mast (and similar) self-hits → strip obs, do NOT force known-clear
# Honest SELF is geometric paint on the ego-label path (perception.ego_rs1), not known-clear.
UNDER_ROBOT_BOXES = [
    BODY_BOX,       # body floor
    WHEEL_L_BOX,    # left wheel
    WHEEL_R_BOX,    # right wheel
]
SELF_IGNORE_BOXES = [
    MAST_BOX,       # mast column
]
FOOTPRINT_BOXES = UNDER_ROBOT_BOXES + SELF_IGNORE_BOXES

# Compat aliases for old mast strip names
MAST_SELF_X0, MAST_SELF_Y0, MAST_SELF_X1, MAST_SELF_Y1 = MAST_BOX

# RS1 camera-space visualization scale (viz-only; map metric unchanged)
RS1_VIZ_SCALE = 3.5
RS1_VIZ_CX_SHIFT = 5

# Self-mask overlay visualization opacity (override via ~/.kevin/mask_viz.json)
SELF_MASK_VIZ_ALPHA = 0.20
