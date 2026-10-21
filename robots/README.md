# Robot Inventory

Per-robot identity and hardware configuration. Each subdirectory (`kevin001/`, `kevin002/`, ...) holds one robot's `inventory.json`
(hardware identity: camera serials, CAN) and `robot.json` (the robot's configuration: `wheels` profile, drive rim targets,
floor references, body outline).

## robot.json and the resolved config

`src/robot_params.py` resolves `robot.json` + the selected profiles (`src/tire_profiles.json`, `src/twin_geometry.json`,
`src/calibration.json`) into ONE flat table; every value carries its units and source (`input: ...` or `derived: ...`).
Every process reads only that table and logs its hash and dump path (`run/resolved_config_<hash>.json`); recordings get a
copy (`robot_config.json`). Invalid or missing config = the process refuses to start.

    python -m src.robot_params show                     # full table
    python -m src.robot_params show --diff stock        # what the selected wheels change vs stock
    python -m src.robot_params show --wheels stock --json
    curl localhost:8080/api/robot_config(.txt)          # live, from nav

Select the robot with `KEVIN_ROBOT` (default kevin001) or `KEVIN_ROBOT_CONFIG=<path>`; override the wheels with
`KEVIN_WHEELS=<profile>` (`KEVIN_TIRE` = legacy alias; both set and different = error).

## Inventory Fields

- **robot_id**: Unique identifier
- **platform**: `kevin` (Kevin mobile base)
- **can_iface**: CAN interface name (e.g., `can1`)
- **cameras**: List of RealSense cameras with serial, role (`RS1` top-down, `RS2` forward), product name, notes
- **notes**: Free-form description (power mode, JetPack version, known issues)

## Usage

Set `KEVIN_ROBOT=kevin001` to load that robot's inventory for camera serial→role mapping and CAN interface. Falls back to autodetection (product name + `KEVIN_RS*_SERIAL` env vars) if unset.

Example:
```bash
KEVIN_ROBOT=kevin001 python scripts/bringup_check.py
KEVIN_ROBOT=kevin001 scripts/bringup_run.sh 0.3
```

## Calibration Policy

**Live `src/calibration.json` stays gitignored.** Inventory is identity + serials only. Auto-cal (`nav.autocal`) writes the live calibration file after parked 10s windows (pitch/roll/height from gravity + floor RANSAC). Twin stays nominal; only real robots get calibration.json.

Per-robot calibration backups may live under `robots/<robot_id>/calibration_<timestamp>.json` for reference, but the live file is never checked in.

## Related Docs

- **`docs/orin_setup.md`** — Host setup (JetPack, PyTorch, RealSense, CAN)
- **`docs/bringup.md`** — Bring-up checklist with calibration ruler tests
- **`src/twin_geometry.py`** — Geometry + calibration hook
