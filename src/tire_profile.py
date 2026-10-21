"""Compatibility view of the selected wheels (2026-10-21: no own logic any more).

The wheels are selected in robots/<robot>/robot.json "wheels" (env KEVIN_WHEELS; KEVIN_TIRE = legacy alias) and resolved
by src/robot_params.py, the single resolver.  This module only re-exports resolved values under the old names.
SAFETY: RIDE_DZ_M raises every camera above the floor; floor clips / camera heights come resolved from robot_params.
"""
import robot_params as _RP

PROFILES = _RP.wheels_name("stock", "stock")[2]["profiles"]


def resolve(name=None):
    """profile name (aliases resolved); None = the selected wheels.  Unknown -> RobotConfigError (argparse type ok)."""
    return _RP.V["wheels"] if name is None else _RP.wheels_name(None, name)[0]


NAME = _RP.V["wheels"]
P = PROFILES[NAME]
STOCK = NAME == "stock"
RIDE_DZ_M = _RP.V["ride_dz_m"]
WHEEL_R_LOADED_M = _RP.V["wheel_r_loaded_m"]
WHEEL_DIAMETER_CM = _RP.V["wheel_diameter_cm"]
WHEEL_MASS_KG = _RP.V["wheel_mass_kg"]
SUFFIX = _RP.V["profile_suffix"]
TWIN_URDF = _RP.V["twin_urdf"]
