"""Kevin tire profile switch (single source of truth: tire_profiles.json next to this file;
mirrored byte-identical in kevin-urdf-twin simready/tire_profiles.json). See docs/tire_profiles.md.

    KEVIN_TIRE=stock      (default; BotWheel stock tire, every constant exactly as before)
    KEVIN_TIRE=soft_v3_2  (alias: soft; KevinFlex v3.2 over-tire: +14.35 mm ride height, loaded R 0.100 m)

Read once at import: set the env var before starting main.py / sims. Every consumer reads it from here.
SAFETY: RIDE_DZ_M raises every camera above the floor. TD_FLOOR_CLIP (RS1 top-down depth floor reject) and
FW_CAM_HEIGHT (RS2 phys_h) add it, so open space stays correct. A wrong profile = obstacles 14 mm off:
with soft tires fitted but KEVIN_TIRE=stock, anything lower than ~5.5 cm (instead of ~4.1 cm) reads as floor.
"""
import json
import os

_D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tire_profiles.json")))
PROFILES = _D["profiles"]


def resolve(name=None):
    n = (name if name is not None else os.environ.get(_D["env"], "")).strip() or _D["default"]
    n = _D["aliases"].get(n, n)
    if n not in PROFILES:
        raise SystemExit("unknown tire profile %r (KEVIN_TIRE); use one of %s" % (n, list(PROFILES)))
    return n


NAME = resolve()
P = PROFILES[NAME]
STOCK = NAME == "stock"
RIDE_DZ_M = float(P["ride_dz_m"])               # chassis + all sensors rise vs stock (0.0 for stock)
WHEEL_R_LOADED_M = float(P["wheel_r_loaded_m"])  # rolling radius (odometry, ODrive turn/s <-> m/s)
# robot_config override of WHEEL_DIAMETER_CM; None = keep robot_config's own (tape-measured) value
WHEEL_DIAMETER_CM = P["robot_wheel_diameter_cm"]
WHEEL_MASS_KG = float(P["wheel_mass_kg"])
SUFFIX = P["suffix"]
TWIN_URDF = P["twin_urdf"]

if not STOCK:
    import sys
    print("[tire_profile] KEVIN_TIRE=%s: wheel R %.4f m, ride +%.2f mm (camera heights + depth floor clip follow)"
          % (NAME, WHEEL_R_LOADED_M, RIDE_DZ_M * 1000), file=sys.stderr)
