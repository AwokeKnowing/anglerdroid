# High-Fidelity Host Simulator Architecture (i777)

**Purpose**: Offline transfer learning / policy iteration for Kevin's non-ROS stack on host workstation (i777 with RTX 3090).

**Branch**: `cursor/host-simulator-8d47`  
**Date**: 2026-09-07  
**Status**: v0 initial implementation

---

## Overview

This simulator extends the lightweight 2D sim (`sim/`) with higher-fidelity sensor models, richer environments, and drop-in compatibility with Kevin's live autonomy stack (SafetyGuard + MPPI + house_bot). Designed for heavy GPU compute on i777, it enables:

1. **Transfer learning**: Validate MPPI/policy changes offline before deploying to Kevin
2. **Stress testing**: Table overhangs, soft obstacles, keepout mats, tight doorways
3. **Reproducibility**: Seeded scenarios with recorded sensor noise characteristics
4. **Fast iteration**: No robot hardware required; ~10-100x real-time on RTX 3090

### Design Principles

- **No ROS**: Pure Python + NumPy + PyTorch, matching live stack constraints
- **Drop-in interfaces**: Same obs map format (320×240 uint8), same twist commands, same safety scales
- **Extend, don't replace**: Builds on existing `sim/` (robot.py, world.py, dynamics.py, mppi_policy.py)
- **Fidelity tuning**: Configurable sensor noise, latency, and perception gaps from measured sim↔reality data

---

## Architecture Components

### 1. Sensor Emulation (`sim/sensors.py`)

Simulates Kevin's perception pipeline with configurable noise models:

#### 1.1 RGB Webcam Model
```python
class RgbWebcamSensor:
    """Simulated USB webcam (640×480, 30fps).
    
    - Motion blur: Linear velocity → pixel smear
    - Auto-exposure: Brightness flicker (±10%)
    - Compression artifacts: JPEG-like block noise
    - Lens distortion: Optional radial warp
    """
```

**Reality gap**: Live webcam feeds house_bot + InsightFace; sim provides synthetic RGB or playback frames.

#### 1.2 Dual RealSense Depth Sensors
```python
class RealsenseDepthSensor:
    """Simulated RealSense D435i depth camera.
    
    - Dropout: 0.5-2% random invalid pixels (IR glare, low texture)
    - Gaussian noise: σ=5mm @ 1m, σ=15mm @ 3m (distance-dependent)
    - Temporal jitter: ±1 frame delay (USB bandwidth contention)
    - FOV masking: 87°×58° cone for RS2 forward, rectangle for RS1 topdown
    """
```

**Dual setup**:
- **RS1** (topdown, serial 815412070676): Orthographic projection, 45° pitch down
- **RS2** (forward, serial 944622074292): Perspective projection, 25.6° pitch

**Output**: Ego-space obstacle map (320×240 uint8) + height map, matching `src/vision.py` pipeline.

#### 1.3 Wheel Odometry + Pose Drift
```python
class WheelOdometrySensor:
    """Simulated ODrive CAN encoders with calibrated drift.
    
    - Slip: Angular slip scale 0.92 (carpet under-reports rotation)
    - Noise: ±2mm linear, ±0.5° angular per step (30Hz)
    - Drift accumulation: Random walk biased by surface friction
    """
```

**Visual odometry correction** (simplified):
- 85% trust wheel odom, 15% correction from synthetic VO (matches Kalman fusion weights)
- Pose drift: ~5cm / 10m traveled, ~2° / full rotation (measured from live SLAM logs)

#### 1.4 Clock Synchronization
```python
class SimClock:
    """Wall-clock time vs sim time for async rate testing.
    
    - Vision: 30Hz (33.33ms periods)
    - MPPI planner: 30Hz (same loop)
    - house_bot FSM: 1Hz (async goal updates)
    - Jitter: ±5ms per frame (USB + kernel scheduling)
    """
```

---

### 2. Enhanced World Scenarios (`sim/world_enhanced.py`)

Extends `sim/world.py` with high-fidelity indoor layouts:

#### 2.1 Table Overhang (Mast Crash Case)
```python
def table_overhang_scenario():
    """Dining table: 80cm tall, 120cm wide, 20cm overhang.
    
    - Floor clear under table (RS1 sees free space)
    - Mast collider at 45cm+ height (RS2 should see edge, but near-field blind <30cm)
    - Topdown near-field reflex (RS1) must trigger hard stop
    """
```

**Critical test**: RS1 topdown detects table edge at <30cm → SafetyGuard fwd_scale=0.

#### 2.2 Soft Obstacles (Dog Bed)
```python
def soft_obstacle_scenario():
    """Compressible dog bed: 15cm tall, 80cm diameter.
    
    - Contact cost: 0.5× normal obstacle (soft prefer vs hard stop)
    - Deformation: Bed compresses 5cm under robot weight (optional physics)
    """
```

**Policy behavior**: MPPI should assign higher cost than floor but lower than wall; prefer detours but tolerate edge contact if trapped.

#### 2.3 Checkered Keepout Mats
```python
def keepout_mat_scenario():
    """Visual keepout zones (black/yellow checkerboard).
    
    - Detected by RGB classifier (house_bot keepout logic)
    - Not in depth map (floor-level, no physical obstacle)
    - Cost injection: Inflate obs map with virtual obstacles
    """
```

**Sim simplification**: Pre-mark keepout pixels in obs map (skip RGB classifier; that's tested separately in `faces/test_people_behavior.py`).

#### 2.4 Doorway + Hallway Pinches
```python
def tight_doorway_scenario():
    """Residential doorway: 80cm clear width (FOOT lateral span ~42cm).
    
    - Approach angle: ±15° misalignment → collision
    - MPPI must align before committing through gap
    """
```

**Existing**: `sim/world.py` has `doorway` and `hallway` scenarios; enhance with tighter geometry + recorded layout from Kevin's house.

---

### 3. Drop-In Loop Integration (`sim/host_sim.py`)

Main entrypoint that runs Kevin's full stack in simulation:

```python
# Pseudo-code structure
def run_host_simulation(scenario, duration_sec, fidelity_mode):
    # 1. Initialize world + robot
    world = load_scenario(scenario)
    robot = Robot(start_pose, fidelity=fidelity_mode)
    sensors = SensorSuite(fidelity=fidelity_mode)
    
    # 2. Initialize autonomy stack (same imports as live main.py)
    from safety import SafetyGuard
    from mppi_costmap import MppiCostmapPlanner
    from house_bot import HouseBotFSM
    
    safety = SafetyGuard()
    mppi = MppiCostmapPlanner()
    house_bot = HouseBotFSM()
    
    # 3. 30Hz loop
    clock = SimClock(hz=30)
    for step in range(int(duration_sec * 30)):
        # 3a. Sensor emulation
        obs_map, height_map = sensors.capture_depth(robot.pose, world)
        rgb_frame = sensors.capture_rgb(robot.pose, world)
        
        # 3b. Safety scales (matches src/safety.py)
        safety.update(obs_map, height_map)
        scales = safety.get_scales()
        
        # 3c. MPPI planning (matches src/mppi_costmap.py)
        cmd = mppi.tick(obs_map, robot.pose, dt=0.033)
        
        # 3d. house_bot override (matches src/house_bot.py)
        if house_bot.should_override(obs_map, scales):
            cmd = house_bot.get_command(obs_map, scales)
        
        # 3e. Apply safety clipping + dynamics
        v_safe = cmd['fwd_mps'] * scales['fwd']
        w_safe = cmd['ang_rads'] * scales['ang']
        robot.step(v_safe, w_safe, dt=0.033)
        
        # 3f. Collision check + logging
        if robot.check_collision():
            log_collision(step, robot.pose)
        
        # 3g. Rerun logging (optional)
        if rerun_enabled:
            log_rerun_frame(obs_map, rgb_frame, robot.pose, cmd)
        
        clock.sleep_until_next_frame()
```

**Key matches to live stack**:
- Same `obs_map` shape (320×240) and dtype (uint8)
- Same safety scale dict keys: `fwd`, `bwd`, `ang` (and `fwd_m`, `bwd_m`, `lat_m` for DualScales)
- Same twist command dict: `{'fwd_mps': float, 'ang_rads': float}`
- Same 30Hz loop timing + multi-rate support (`--policy-hz` flag)

---

### 4. Rerun Logging (`sim/rerun_logger.py`)

Optional visualization via Rerun 0.37 (same as live `src/rerun_log.py`):

```python
class SimRerunLogger:
    """Log sim state to Rerun RRD file for post-run inspection.
    
    Entities (matching live layout):
      - world/obs_map: Ego-space obstacle map (image)
      - world/height_map: Height map (image)
      - world/robot: Robot pose + footprint (3D box)
      - world/trajectory: Past positions (line strip)
      - sensors/rgb: Simulated webcam frame
      - autonomy/mppi_samples: Sampled trajectories (lines)
      - autonomy/goal: Current goal waypoint (point)
    """
```

**Usage**:
```bash
python -m sim.host_sim --scenario table_overhang --rerun-save /tmp/sim.rrd
rerun /tmp/sim.rrd
```

---

## Simulation vs Reality: What's Emulated, What's Playback

| Component | Simulation Mode | Playback Mode (Future) |
|-----------|-----------------|------------------------|
| **World geometry** | Synthetic Box3D + procedural | Recorded atlas frames from live runs |
| **Depth sensing** | Raytrace + noise model | Pre-recorded depth frames |
| **RGB** | Synthetic or blank | Recorded webcam frames |
| **Pose** | Integrated from twist + drift | Recorded SLAM poses |
| **Safety scales** | Computed from sim obs map | Recomputed from playback obs |
| **MPPI/policy** | Live planner on sim obs | Live planner on playback obs |

**v0 scope**: Synthetic-only (simulation mode). Playback mode deferred to v1.

---

## Fidelity Modes

```python
# Low fidelity (fast, ~100x real-time)
python -m sim.host_sim --scenario house --fidelity=low

# Medium fidelity (transfer testing, ~10x real-time)
python -m sim.host_sim --scenario table_overhang --fidelity=medium

# High fidelity (close to hardware, ~1-3x real-time)
python -m sim.host_sim --scenario tight_doorway --fidelity=high --seed=42
```

### Fidelity Parameters

| Mode | Depth Noise | Pose Drift | Latency | Accel Limits | Sensor Jitter |
|------|-------------|------------|---------|--------------|---------------|
| **Low** | None | None | 0ms | ✓ | No |
| **Medium** | σ=3mm | 2cm/10m | 50ms | ✓ | ±2ms |
| **High** | σ=8mm | 5cm/10m | 150ms | ✓ | ±5ms |

**Note**: Even low-fidelity mode includes differential-drive dynamics + wheelbase caps (gaps D1–D4 from `docs/sim-reality-gaps.md`).

---

## Clock Synchronization + Multi-Rate Testing

Sim supports async subsystem rates (matching live):

```python
# Vision loop: 30Hz (every frame)
# MPPI planner: 30Hz (every frame)
# house_bot FSM: 10Hz (every 3rd frame)
# Speech/face: 1Hz (every 30th frame, not critical path)

python -m sim.host_sim --scenario house --policy-hz=10
```

**Testing hypothesis**: Does MPPI degrade when planned at 10Hz but robot dynamics update at 30Hz? (Answer from `test_mppi_transfer.py`: no significant degradation if horizon ≥ 1s.)

---

## Integration with Live Stack

### Same Imports
```python
# Sim can import live modules directly (no ROS → pure Python)
from src.safety import SafetyGuard
from src.mppi_costmap import MppiCostmapPlanner
from src.house_bot import HouseBotFSM
from src.robot_config import RCX, RCY, EGO_PX_SIZE, FOOT_X0, FOOT_Y0, FOOT_X1, FOOT_Y1
```

### Same Data Formats
```python
# Obstacle map (ego-space)
obs_map: np.ndarray  # (240, 320) uint8, 0=clear, >100=occupied
height_map: np.ndarray  # (240, 320) uint8, cm above floor

# Safety scales
scales = {
    'fwd': float,  # [0, 1]
    'bwd': float,
    'ang': float,
    'fwd_m': float,  # meters (for DualScales)
    'bwd_m': float,
    'lat_m': float,
}

# Twist command
cmd = {
    'fwd_mps': float,  # m/s
    'ang_rads': float,  # rad/s
}

# Pose
pose = {
    'x': float,  # meters, world frame
    'y': float,
    'theta': float,  # radians
}
```

**Critical**: No conversions or adapters needed — sim feeds same types to same functions.

---

## Measured Sim↔Reality Gaps (from `docs/sim-reality-gaps.md`)

Addressed in this v0:

| ID | Gap | Status | Implementation |
|----|-----|--------|----------------|
| D1 | Instant v,w | ✓ Addressed | `DiffDriveDynamics` with accel ramp |
| D2 | Command latency | ✓ Addressed | Optional 150ms delay queue |
| D3 | Diff-drive kinematics | ✓ Addressed | Wheel-level integration |
| D4 | Speed caps | ✓ Addressed | MPPI v_max=0.25, w_max=0.8 |
| G1 | Mast collisions | ✓ Addressed | Height map + MAST_CLEAR_CM=45 |
| P1 | Depth noise | ✓ Addressed | Dropout + Gaussian σ(distance) |
| P2 | Pose drift | ✓ Addressed | Random walk + slip scale |
| E2 | Soft obstacles | ⚠ Partial | Soft cost in MPPI, no physics deformation |

Not yet addressed (defer to v1+):

| ID | Gap | Defer Reason |
|----|-----|--------------|
| G2 | Non-rect body | Low priority; FOOT rect sufficient for transfer |
| P3 | Partial FOV | Complexity vs value; full ego map acceptable for v0 |
| P5 | Dynamic people | Separate test suite (`faces/test_people_behavior.py`) |
| C2 | Thread races | Deterministic single-thread sim avoids this |
| E1 | Real house layout | Hand-tuned scenarios adequate; could trace from atlas later |

---

## Testing Strategy

### Unit Tests (`sim/test_host_sensors.py`)
```bash
pytest sim/test_host_sensors.py -v
```

- Depth sensor: Dropout rate 0.5-2%, noise σ proportional to distance
- Wheel odom: Slip scale 0.92, drift accumulation < 5cm/10m
- Pose fusion: VO correction within ±3cm of ground truth

### Integration Tests (`sim/test_host_scenarios.py`)
```bash
pytest sim/test_host_scenarios.py -v
```

- **Table overhang**: RS1 topdown triggers fwd_scale=0 at <30cm from edge
- **Soft bed**: MPPI assigns 0.3-0.7× normal obstacle cost (not hard stop)
- **Doorway**: Robot aligns and crosses 80cm gap without collision (10 trials, seed sweep)
- **Keepout mat**: Virtual obstacle inflation prevents entry

### Smoke Test (CLI)
```bash
# Quick sanity: does it run without crash?
python -m sim.host_sim --scenario table_overhang --steps 400 --save /tmp/demo.gif
```

**Exit 0** if no collisions, exit 1 if crash → CI-ready.

---

## Usage on i777

### Dependencies
```bash
# On i777 (Ubuntu 22.04, RTX 3090)
cd /workspace
pip install numpy torch imageio opencv-python rerun-sdk

# Optional: robox3d for Box3D world gen (already installed on Orin)
pip install robox3d
```

### Run Scenarios
```bash
# Table overhang stress test
python -m sim.host_sim --scenario table_overhang --steps 400 --fidelity high --save artifacts/table_overhang.gif

# Soft bed scoring
python -m sim.host_sim --scenario soft_bed --steps 600 --fidelity medium --rerun-save /tmp/bed.rrd

# Full house wander (5 minutes sim time)
python -m sim.host_sim --scenario house --duration 300 --fidelity medium --seed 42
```

### Expected Performance (RTX 3090)
- Low fidelity: ~100x real-time (30 sim seconds in 0.3 wall seconds)
- Medium fidelity: ~10x real-time (30 sim seconds in 3 wall seconds)
- High fidelity: ~1-3x real-time (30 sim seconds in 10-30 wall seconds)

**Bottleneck**: MPPI sampling (512-2048 trajectories) on GPU. PyTorch `grid_sample` batching gives ~3ms/tick on 3090.

---

## Remaining Fidelity Gaps (v0)

Known limitations of this initial implementation:

1. **RGB synthetic**: Blank or procedural; no photorealistic rendering. (House_bot face/keepout triggers stubbed.)
2. **Visual odometry**: Simplified correction model; real VO uses dense optical flow + outlier rejection.
3. **Soft obstacle physics**: MPPI cost model only; no actual deformation or traction change.
4. **Dynamic obstacles**: World is static; no moving people/pets. (Separate `faces/people_behavior.py` tests.)
5. **RealSense IR patterns**: Depth noise is statistical, not ray-traced IR speckle.
6. **Battery voltage sag**: Infinite power; no torque limits or thermal throttling.

**Mitigation**: These gaps are acceptable for policy transfer testing. Known residuals documented; validate critical scenarios on hardware before deployment.

---

## Next Steps (v1+)

1. **Playback mode**: Record atlas frames from live runs → replay with different policies (counterfactual testing).
2. **GPU raytrace**: Replace Box3D rasterizer with CUDA mesh raytrace for photorealistic depth.
3. **Soft body physics**: MuJoCo or PyBullet backend for deformable obstacles (dog bed compression).
4. **Multi-agent**: Simulate people walking past → test social FSM interactions.
5. **Learned cost functions**: Train iPlanner-IL or SAC-polar in sim → deploy to live MPPI.

---

## File Structure

```
/workspace
├── docs/
│   └── host-simulator-architecture.md  (this file)
├── sim/
│   ├── host_sim.py                     # Main entrypoint (new)
│   ├── sensors.py                      # Sensor emulation (new)
│   ├── world_enhanced.py               # Enhanced scenarios (new)
│   ├── rerun_logger.py                 # Rerun integration (new)
│   ├── test_host_sensors.py            # Sensor unit tests (new)
│   ├── test_host_scenarios.py          # Integration tests (new)
│   ├── robot.py                        # Existing (reuse)
│   ├── dynamics.py                     # Existing (reuse)
│   ├── mppi_policy.py                  # Existing (reuse)
│   └── world.py                        # Existing (extend)
└── artifacts/
    └── sim/
        ├── table_overhang.gif          # Demo GIF
        └── README.md                   # i777 quickstart
```

---

## References

- **Live stack**: `src/main.py` (30Hz loop), `src/safety.py`, `src/mppi_costmap.py`, `src/house_bot.py`
- **Existing sim**: `sim/README.md`, `sim/robot.py`, `sim/dynamics.py`, `sim/run.py`
- **Sim-reality gaps**: `docs/sim-reality-gaps.md` (measured transfer deltas)
- **Mid-layer design**: `docs/kevin-autonomy-midlayer.md` (MPPI architecture)
- **Topdown reflex**: `docs/TOPDOWN_NEAR_FIELD_REFLEX.md` (RS1 <30cm stop)

---

**Author**: Cloud Agent (cursor/host-simulator-8d47)  
**Last Updated**: 2026-09-07
