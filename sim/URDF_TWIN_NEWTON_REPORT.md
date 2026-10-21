# Kevin URDF twin → Newton bring-up report

Date: 2026-09-23 PT (i777)  
Branch: `kevin-ppo-live` (sim code reused; live Jetson left DISARMED)  
Twin: `sim/kevin-urdf-twin/` (from `sim/kevin-urdf-twin-sim.tgz`)

## 1) Sim status + how to run

### Status
- **URDF loads in Newton 1.5.1** via `ModelBuilder.add_urdf(..., floating=True, collapse_fixed_joints=True, enable_self_collisions=False)`.
- After collapse: **7 bodies / 7 joints / 12 DoF / ~250 shapes**.
- Continuous joints retained: L/R drive + front caster yaw/roll + gate caster yaw/roll.
- **Realtime drive/wander smoke: OK** — `--drive --urdf --seconds 8` completed **8.0 s sim in 4.29 s wall (~1.9× realtime)** on RTX 3090, kinematic open-floor path, GL viewer on `DISPLAY=:1`. Travel ≈ 0.41 m. No crash.
- Drive wheels mapped to velocity actuators; **casters frozen** (stiff position holds) for first bring-up.
- House: visual STL site + wall-box collisions (unchanged; raw STL not used as MuJoCo collider).
- RS1/RS2 poses taken from URDF optical-frame FK (`sim/urdf_twin_fk.py`) into `SimRsCameras.apply_urdf_poses`.

### How to run
```bash
cd /home/james/gits/awokeknowing/anglerdroid
# Extract twin (already done): URDF at sim/kevin-urdf-twin/urdf/kevin.urdf
# Mesh resolve: sim/kevin-urdf-twin/urdf/meshes -> ../meshes (symlink)

DISPLAY=:1 .venv-newton/bin/python sim/newton_kevin.py \
  --drive --urdf --seconds 40 --viewer gl --device cuda:0

# Optional: leave casters free (may hurt solver)
DISPLAY=:1 .venv-newton/bin/python sim/newton_kevin.py \
  --drive --urdf --urdf-free-casters --seconds 20 --viewer gl --device cuda:0

# Headless / null viewer smoke
DISPLAY=:1 .venv-newton/bin/python sim/newton_kevin.py \
  --drive --urdf --seconds 8 --viewer null --device cuda:0
```

### Code touchpoints (minimal)
- `sim/urdf_twin_fk.py` — URDF FK for cameras / wheels / AABB / audit
- `sim/newton_kevin.py` — `build_kevin_urdf()`, `--urdf` / `--urdf-path` / `--urdf-free-casters`
- `sim/newton_drive.py` — `--urdf` branch in `run_perception_drive`, `SimRsCameras.apply_urdf_poses`

## 2) Concrete model issues found during load/sim

### Load / paths
1. **Relative mesh paths** `meshes/*.stl` resolve against the **URDF directory**, not the twin root. Newton warned until `urdf/meshes -> ../meshes` symlink existed. Prefer `package://kevin_description/...` or documenting the symlink for 3d builder exports.
2. No missing-mesh errors after the symlink (botwheel_v14, RS bodies, casters, anker, canable, pc mounts all found).

### Collisions (critical for physics twin)
3. **Only 13 collision geoms** on links: ['front_caster_yaw', 'front_caster_wheels', 'wheel_left', 'wheel_right', 'canable', 'gate_caster_fork', 'gate_caster_wheel', 'respeaker', 'whibal_g7', 'anker_usb_hub', 'realsense_d435', 'realsense_d435i', 'rgb_camera'].
4. **Beams, mast, brackets, battery, ODrive housings, PC mounts have NO collision** — chassis can ghost through furniture/door jambs in full physics; kinematic footprint still uses separate boxes.
5. Visual geom count ≈ 236 (heavy for MuJoCo if ever used as colliders). Do **not** enable `parse_visuals_as_colliders`.

### Inertials
6. **23 links missing `<inertial>`** (mostly optical frames — OK; **respeaker** also missing — should get a small mass).
7. Collapsed free-body mass ≈ **12.5 kg** (wheels 2.2 kg each + base stack). Plausible vs “~12.4 kg” note.

### Joints / casters
8. Six continuous joints load correctly. Unfrozen casters are likely to cause **solver pain / tip** (prior box sim used a fixed skid for this reason). Bring-up freezes them with POS targets; note for later.
9. Drive joints `wheel_left_joint` / `wheel_right_joint` axis = +Y at hubs y=±0.1515 — correct for REP-103 diff-drive.

### CoM / floor / spawn
10. URDF axle at z=0 with wheel radius **0.08565 m**. Sim spawns floating base at **z=wheel_r** so tires kiss z=0. Pure physics smoke earlier showed wheels spinning with little translation (traction/contact tuning still weak); **kinematic wander path is what ran realtime**.
11. Beam AABB rear ≈ -0.200 m (includes aft structure) vs documented outer rear **−0.165**; front ≈ **+0.115**.

### Optical frames vs live constants
| Frame | Twin FK | Live / old box sim | Notes |
|-------|---------|--------------------|-------|
| RS1 depth optical z | **0.865 m** | `RS1_ORIGIN_Z=0.86`; old mast cam 0.90 | Twin −0.0245 x, look-down; live constant 5 mm low vs twin |
| RS1 look | −Z (down) | hardcoded look-down | Match after optical→sim (−Z look) map |
| RS2 depth optical | xyz≈(-0.130, 0.000, 0.382), pitch_down **27.0°** | `FW_PITCH_DOWN_DEG=27`; old sim used **25.6°** + `FW_CAM_HEIGHT=0.475` | Pitch matches live 27°; **height 0.382 vs live/old 0.475** (~9 cm) |
| RGB optical | on `beam_mast_top` | not used in drive path yet | Optional |

### Body size / track vs old box
| | Twin | Old box sim |
|--|------|-------------|
| Outer length | ~0.28–0.31 m (front +0.115, rear −0.165 doc) | 0.33 m (rear −0.18, front +0.15) |
| Width / track | width 0.30 m; **hub track 0.303 m** (y=±0.1515) | BODY_W 0.33; track ≈ 0.40 (0.33/2 + gap + wheel_w/2)×2 |
| Wheel radius | **0.08565** | 0.09 |
| Wheel width (collision cyl) | ≈0.045 | 0.06 |

## 3) Requested twin updates (for 3d builder) — actionable

1. **Add collision boxes for structure**: at minimum a base cage AABB (≈0.28×0.30×0.15), mast column, and mast top bar — without turning every visual beam into a collider.
2. **Export meshes with package-relative URIs** (`package://kevin_description/meshes/...`) or co-locate `meshes/` next to the URDF so consumers do not need a symlink.
3. **Give `respeaker` an inertial** (even 50 g + small diagonal inertia).
4. **Confirm RS2 height** against the real robot: twin optical z≈0.382 vs live `FW_CAM_HEIGHT=0.475`. If the robot is 0.475, update the twin mount; if the twin is right, update live constants / sim.
5. **Keep RS1 at z≈0.865** and sync `RS1_ORIGIN_Z` (0.86 → 0.865) in live/sim once twin is authoritative.
6. **Keep RS2 pitch at 27°** (already matches `FW_PITCH_DOWN_DEG`); drop the old 25.6° sim assumption.
7. **Optional: simplify visual mesh count** for sim exports (convex hulls / low-poly) — 236 visuals are fine for GL but slow if ever mis-parsed as colliders.
8. **Document caster trail/yaw damping** intended for dynamics; consider a “sim” URDF variant with casters fixed or as a low-friction skid sphere for Newton bring-up.
9. **Outer envelope note in URDF comments**: front +0.115 / rear −0.165 / track 0.303 / wheel_r 0.08565 so box-sim constants can be retired deliberately.
10. **Collision on Anker / CANable / RS bodies** already present — keep; add battery collision if the pack ever snags thresholds.

## 4) Blockers / follow-ups
- **None for URDF load + short kinematic wander.**
- Full **physics** wheel traction / caster dynamics not yet validated (casters frozen; kinematic open-floor commander still owns XY).
- Do not feed chassis truth into SLAM (unchanged).
- Live Jetson drive remains **DISARMED** (untouched).
