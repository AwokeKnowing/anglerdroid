"""Shared robot geometry constants used by vision, safety, cameras, globalmap."""

# Frame geometry
FRAME_W, FRAME_H = 320, 240
CROSSHAIR_CX, CROSSHAIR_CY = 159, 119

# Ego-space pixel size (metres)
EGO_PX_SIZE = 0.010           # 1 px = 10 mm

# Physical robot dimensions — James tape 2026-09-17 (axle origin).
# Half-track 177 mm (axle → hub). Hub-to-hub 354 mm. Wheel radius 85 mm.
# Body ±180 mm; caster sticks out to 200 mm aft. Clean turn disk 200 mm;
# wander keep-clear 250 mm from axle (do not enter a gap you cannot turn in).
WHEEL_DIAMETER_CM = 17.0
# v310 tire profile switch (src/tire_profile.py; env KEVIN_TIRE=stock|soft_v3_2, default stock = the line above unchanged).
# soft_v3_2: 20.0 cm loaded (NOMINAL - re-measure by rolling 10 turns once fitted). RIDE_DZ_M = chassis/sensor rise vs stock.
from tire_profile import NAME as TIRE_PROFILE, RIDE_DZ_M, WHEEL_DIAMETER_CM as _TIRE_WHEEL_D_CM  # noqa: E402
if _TIRE_WHEEL_D_CM is not None:
    WHEEL_DIAMETER_CM = float(_TIRE_WHEEL_D_CM)
# 2026-10-13 (James): the digital twin is the source of truth. Geometry comes from src/twin_geometry.json
# (scripts/gen_twin_geometry.py, per KEVIN_TIRE profile) + src/calibration.json (reality->model hook).
# Was: WHEEL_DIAMETER_CM 17.0 (tape), WHEEL_HALFTRACK_CM 17.7 -> track 0.354.
import twin_geometry as _TG  # noqa: E402
WHEEL_DIAMETER_CM = 200.0 * _TG.WHEEL_R_M          # stock 17.13 (R 0.08565); soft_v3_2 20.13 (R 0.10065, 2026-10-21)
WHEEL_HALFTRACK_CM = 50.0 * _TG.TRACK_M            # 18.1 (tire contact centre; twin track 0.362)
WHEELBASE_CM = 2.0 * WHEEL_HALFTRACK_CM            # 36.2
WHEEL_RADIUS_M = WHEEL_DIAMETER_CM / 200.0         # 0.08565
WHEELBASE_M = WHEELBASE_CM / 100.0                 # 0.362
TURN_R_M = 0.20
WANDER_CLEAR_M = 0.25
WANDER_CLEAR_CM = WANDER_CLEAR_M * 100.0
# 2026-10-13: body outline from the twin (was tape 18 / 18 / 20 cm, width 33)
BODY_FRONT_CM = 100.0 * _TG.HULL_FRONT_M      # 12.6
BODY_AFT_CM = 100.0 * _TG.HULL_AFT_M          # 23.1
CASTER_AFT_CM = 100.0 * _TG.CASTER_AFT_M      # 27.4

# ---------------------------------------------------------------------------
# Self-mask / footprint geometry — ALL offsets in cm relative to the axle
# midpoint (point halfway between the two wheel centers).
#
# Ego frame: robot faces +X (right on the image). +Y is robot-left
# (up on the image → decreasing row). EGO_PX_SIZE = 1 cm/px so cm == px.
# Axle origin in ego pixels: (RCX, RCY).
# ---------------------------------------------------------------------------

# Body footprint: 180 mm forward of axle, 180 mm aft. Caster is extra.
BODY_LEN_CM = BODY_FRONT_CM + BODY_AFT_CM
BODY_WID_CM = 200.0 * _TG.HULL_HALF_W_M   # 30.8 along +Y (twin chassis; was 33.0)
BODY_CX_CM = 0.5 * (BODY_FRONT_CM - BODY_AFT_CM)
BODY_CY_CM = 0.0

# Drive wheels at the axle. Box is a bit fatter than the 85 mm radius
# tire so self-returns do not leak as obs.
WHEEL_LEN_CM = 20.0           # along +X
WHEEL_WID_CM = 10.0           # along +Y (tread)
WHEEL_L_CX_CM = 0.0
WHEEL_L_CY_CM = WHEEL_HALFTRACK_CM    # left (+Y)
WHEEL_R_CX_CM = 0.0
WHEEL_R_CY_CM = -WHEEL_HALFTRACK_CM   # right (−Y)

# Rear caster: body ends at 180 mm, caster rim at 200 mm.
CASTER_LEN_CM = CASTER_AFT_CM - BODY_AFT_CM
CASTER_WID_CM = 8.0
CASTER_CX_CM = -0.5 * (BODY_AFT_CM + CASTER_AFT_CM)
CASTER_CY_CM = 0.0

# Mast column (self-hits, not floor-clear). Size/offset from URDF (~3 cm post
# near back wall); slightly fattened so depth self-returns get stripped.
MAST_LEN_CM = 8.0
MAST_WID_CM = 8.0
MAST_CX_CM = 100.0 * _TG.MAST_X_M     # -15.75 twin chassis_mast (was -13.0)
MAST_CY_CM = 0.0

# Legacy aliases (consumers still import these names)
ROBOT_W = int(round(BODY_LEN_CM))   # front-back
ROBOT_H = int(round(BODY_WID_CM))   # side-to-side
ROBOT_CX_OFF = -78            # x offset of axle from crosshair center
WHEEL_HALF = int(round(WHEEL_LEN_CM / 2.0))  # compat import only

# Safety scan pads (NOT baked into self-mask boxes). Rear pad is 0 —
# the caster box already is the aft face. Lat pad brings FOOT out to
# ~25 cm (wander-clear) from the axle.
FOOT_PAD_FWD = 0
FOOT_PAD_BWD = 0
FOOT_PAD_LAT = 2

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
CASTER_BOX = _ego_box_from_axle(CASTER_CX_CM, CASTER_CY_CM, CASTER_LEN_CM, CASTER_WID_CM)
MAST_BOX = _ego_box_from_axle(MAST_CX_CM, MAST_CY_CM, MAST_LEN_CM, MAST_WID_CM)

# Named pixel aliases for body (compat)
BODY_X0, BODY_Y0, BODY_X1, BODY_Y1 = BODY_BOX

# Legacy FOOT_* AABB around hull + safety pads (safety.py forward/back scans)
_hull = [BODY_BOX, WHEEL_L_BOX, WHEEL_R_BOX, CASTER_BOX, MAST_BOX]
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
    CASTER_BOX,     # rear caster
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
