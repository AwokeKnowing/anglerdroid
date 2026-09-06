# Kevin Autonomy Mid-Layer: Design & Integration

**Robot**: AwokeKnowing/anglerdroid (Jetson Orin NX)  
**Date**: 2026-09-06  
**Status**: Investigation + Scaffold  
**Constraint**: NO ROS/ros2/nav2

## Executive Summary

This document maps the existing 30Hz perception-to-control pipeline and proposes a **LocalExecutive** mid-layer that accepts high-level goals (x,y coordinates or "wander") and continuously generates velocity commands using the current occupancy map and pose. The design preserves existing teleop/UI control and keeps the 30Hz map generation intact.

---

## 1. Current Architecture: 30Hz Loop

### 1.1 Main Loop Flow (`main.py`)

Target: **30 fps** (33.33 ms/frame budget)

```
1. Capture frames (RealSense RS1 top-down + RS2 forward, RGB webcam)
2. Process depth → ego-space occupancy map (Vision thread)
3. Update pose (wheel + visual odometry fusion)
4. Update global map (GPU-accelerated evidence accumulation)
5. Compute safety scales (forward/backward/angular collision avoidance)
6. Handle gamepad input (if active, overrides all automation)
7. Execute pending tool calls from Gemini agent (twist_for, navigate, stop)
8. If no gamepad & no twist_for active:
   - Call navigator.compute_twist(atlas) → (fwd_mps, ang_rads) or None
   - Send commands to wheelbase
9. Throttle to 30 fps
```

**Key timing**: Typical loop processing ~10-15 ms, wait ~18-23 ms.

### 1.2 Vision Pipeline (`vision.py`)

**Capture Thread** (separate from main loop, runs continuously):

```
Input:  RealSense RS1 (815412070676) top-down, RS2 (944622074292) forward
        Depth decimated to ~320x240 pointclouds

Process:
  - RS1: Orthographic projection (Z-threshold for floor/obstacle)
    → obs1, known1 (320x240 uint8)
  - RS2: Pitch-rotated bird's-eye view (25.6° - 90° = -64.4° pitch)
    → obs2, known2 (320x240 uint8, then rotated CW 90°)
  - Combine with alignment offsets:
    TD_X_OFFSET = -75 px (adjustable)
    FW_X_OFFSET = TD_X_OFFSET + 132 = 57 px
  - Apply field-of-view masks (RS1 rectangle, RS2 80° cone)
  - Robot footprint forced clear

Output:
  - obs_combined (320x240 uint8): obstacle height in cm (0=free, 1-100=obstacle)
  - known_combined (320x240 uint8): 0=unobserved, 255=observed
  - persistent_obs: obs_combined + projected global map obstacles
  - Updated every ~33ms (30 Hz target)
```

**Ego-Space Coordinate System**:
- Robot center: `(RCX=81, RCY=119)` in 320×240 frame
- Robot faces **RIGHT** (positive X)
- Pixel size: `EGO_PX_SIZE = 0.01 m` (1 cm/px)
- Robot footprint: 30px wide (fwd/back) × 42px tall (lateral) = ~30cm × 42cm

**Atlas Layout** (640×480 sent to UI):
- Top-left: Webcam RGB (320×240)
- Top-right: RS1 color (320×240)
- Bottom-left: RS2 color (320×240)
- Bottom-right: Obstacle map visualization (320×240)

**Atlas Layout** (actually 960×960 internally before scaled for UI):
- Row 0-239: camera feeds
- Row 240-959: global map (960×720)

### 1.3 Pose Estimation (`pose.py`)

**Frames**:
- World frame: x=right, y=up, θ=0 facing +x, CCW positive
- Units: meters, radians
- Origin: robot start position

**Fusion**:
- Wheel odometry (primary): CAN bus from ODrive encoders (read at ~30Hz)
- Visual odometry (correction): GPU-accelerated dense optical flow from RS2 forward camera
- Kalman filter per-frame fusion with:
  - Physical plausibility gates
  - Confidence thresholding
  - Agreement checks (visual vs. wheel)
  - Mahalanobis gating
- Surface compensation: `ANGULAR_SLIP_SCALE = 0.92` (carpet under-reports rotation)

**State**:
- `pose.x, pose.y, pose.theta` (float, meters/radians)
- History: 300 recent poses for trajectory rendering

**Update Rate**: Every capture frame (~30 Hz)

### 1.4 Current Navigator (`navigator.py`)

**Algorithm**: Vector Field Histogram (VFH-lite)

**Input**:
- Goal heading (degrees): 0=forward, 90=left, -90=right, None=stop
- Obstacle map: bottom-right 320×240 quadrant of atlas

**Process** (per frame):
1. Extract obstacle pixels → polar coordinates from robot center
2. Build 36-bin angular histogram (10° per bin) of obstacle density
3. Weight by proximity: closer obstacles → higher cost
4. Find clearest direction near goal heading (score = 0.6×clearance - 0.4×heading_cost)
5. Set forward speed based on nearest obstacle ahead:
   - `STOP_RANGE = 18 px` (18 cm) → full stop
   - `DANGER_RANGE = 40 px` (40 cm) → ramp MIN_FWD to MAX_FWD
   - Beyond → MAX_FWD
6. Set turn rate: `angular = heading_error_deg × TURN_GAIN`

**Limits**:
- `MAX_FWD = 0.20 m/s`
- `MIN_FWD = 0.06 m/s` (below ~0.05 doesn't move on carpet)
- `MAX_ANG = 0.6 rad/s`
- `TURN_GAIN = 0.015 rad/s per degree`

**Output**: `(forward_mps, angular_rads)` or `None` (inactive/oscillation)

**Execution Time**: ~1 ms (pure numpy)

**Special Cases**:
- Heading >90° → pure spin in place (no forward motion)
- Oscillation detection: >12 sign changes in 20 frames (~0.67s) → auto-stop

**API**:
```python
navigator.set_goal(heading_deg: float | None)  # Set/clear goal
navigator.clear_goal()                         # Stop navigation
navigator.is_active() -> bool                  # Check if active
navigator.get_goal() -> float | None           # Current goal heading
navigator.compute_twist(atlas: np.ndarray) -> (float, float) | None
```

### 1.5 Safety Guard (`safety.py`)

**Purpose**: Directional collision avoidance via velocity scaling

**Scans** (per frame):
- Forward: from front edge rightward until first obstacle
- Backward: from rear edge leftward until first obstacle
- Lateral (up/down): from robot edges (diagonal extent ~26px radius)

**Scaling**:
- S-curve smoothstep: `clearance_scale(px)` → [0.0, 1.0]
  - `px ≤ 1` → 0.0 (full stop)
  - `px ≥ 30` → 1.0 (full speed)
  - 1 < px < 30 → smoothstep interpolation
- Applied in wheelbase: `effective_vel = commanded_vel × safety_scale`
- Moving AWAY from obstacle is always allowed (direction-aware)

**Output**:
- `fwd_scale, bwd_scale, ang_scale` (float [0, 1])
- Updated every frame, propagated to wheelbase via main loop

### 1.6 WheelBase (`wheelbase.py`)

**Hardware**:
- ODrive motor controllers via CAN (can1)
- Differential drive: wheelbase = 34 cm, wheel diameter = 17.13 cm
- Battery: 7S LiPo (~30-42V), monitored via VBUS

**Command API**:
```python
wheelbase.twist(forward_mps: float, angular_rads: float)
wheelbase.set_wheel_vels(left_tps: float, right_tps: float)
wheelbase.twist_for(fwd, ang, duration_secs, ramp_in_secs, ramp_out_secs)
wheelbase.cancel_twist_for()
wheelbase.stop()
```

**Safety Mechanisms**:
1. **Command staleness timeout**: 0.5s (app-level watchdog)
   - If non-zero command isn't refreshed → immediate zero, transition to IDLE
2. **ODrive hardware watchdog**: 2.0s (backstop, should never trip)
3. **Safety scaling**: Applied to all commands before sending to motors
4. **Idle detection**: Zero velocity for 5s + low torque → IDLE state

**Command Rate**: Main loop calls at 30 Hz; wheelbase tracks `_last_command_time`

### 1.7 Global Map (`globalmap.py`, `gpu_render.py`)

**Grid**:
- Size: 960×720 pixels
- Resolution: 2 cm/px (PX_SIZE = 0.02 m)
- Coverage: 19.2m × 14.4m
- Origin: (480, 360) = world (0, 0)

**Values**:
- `UNKNOWN_VAL = 128` (gray)
- `< OBS_THRESH (90)` = obstacle
- `> FREE_THRESH (190)` = free
- Evidence accumulation: ±50 per observation (clamped [0, 255])

**Updates**:
- Every frame: ego-space obs/known projected to global via pose
- GPU-accelerated: `gmap_update_gpu()` in `gpu_render.py`
- Free evidence only trusted within 2m (range mask) to avoid alignment drift

**SLAM** (`slam.py`):
- Pose-graph SLAM with loop closure
- Keyframes: 20 cm or 10° spacing
- Rotation-invariant radial descriptors for place recognition
- Scan-matching verification + Gauss-Newton optimization
- Memory budget: 200-500 MB (in-memory, no database)

---

## 2. Integration Points for LocalExecutive

### 2.1 Cleanest Hook

**Location**: `main.py` lines 157-163

Current code:
```python
elif not wb.is_twist_for_active():
    twist = navigator.compute_twist(atlas) if atlas is not None else None
    if twist is not None:
        tools.twist(twist[0], twist[1])
    else:
        tools.set_wheel_vels(0.0, 0.0)
```

**Proposed replacement** (behind `--auto-local` flag):
```python
elif not wb.is_twist_for_active():
    if args.auto_local and local_executive.is_active():
        # LocalExecutive mode: accept goal (x,y) or "wander", compute velocities
        cmd = local_executive.tick(
            obs_map=vis._persistent_obs,  # 320x240 ego-space + global projection
            pose=(vis._pose.x, vis._pose.y, vis._pose.theta),
            dt=LOOP_DT
        )
        if cmd is not None:
            tools.twist(cmd['fwd_mps'], cmd['ang_rads'])
        else:
            tools.set_wheel_vels(0.0, 0.0)
    else:
        # Existing navigator (VFH heading-based)
        twist = navigator.compute_twist(atlas) if atlas is not None else None
        if twist is not None:
            tools.twist(twist[0], twist[1])
        else:
            tools.set_wheel_vels(0.0, 0.0)
```

**Why this hook?**:
- Non-blocking: main loop already owns command timing
- Never interrupts capture thread
- Respects gamepad/twist_for priority
- Access to latest obs_map, pose, safety scales
- Same command rate as existing navigator (30 Hz)

### 2.2 Data Access

**Available at hook**:
```python
# From vision instance (vis)
obs_map = vis._persistent_obs          # (240, 320) uint8, ego+global obstacles
pose = (vis._pose.x, vis._pose.y, vis._pose.theta)  # (float, float, float) meters/radians
dt = LOOP_DT                           # 1/30 = 0.0333 s

# From wheelbase (wb)
wb._safety_fwd, wb._safety_bwd, wb._safety_ang  # Current safety scales [0, 1]

# Robot geometry (robot_config.py)
from robot_config import (
    RCX, RCY,           # Robot center in ego frame (81, 119)
    EGO_PX_SIZE,        # 0.01 m/px
    WHEELBASE_M,        # 0.34 m
    WHEEL_RADIUS_M      # ~0.08565 m
)
```

**Units & Frames**:
- **Ego-space**: 320×240 px, robot at (81, 119) facing RIGHT, 1px = 1cm
- **World-space**: meters, origin at start, x=right, y=up, θ=0 facing +x (CCW)
- **Velocity commands**: `(fwd_mps, ang_rads)` in m/s and rad/s
- **Pose**: `(x, y, theta)` in meters and radians (world frame)

### 2.3 Command Rate & Timing

**Target**: 30 Hz (matching main loop)

**Acceptable**: 10 Hz minimum (every 3rd frame)

**Budget**: ~2-3 ms per tick (to stay within 33ms frame budget after vision processing)

**Constraints**:
- Must not block capture thread
- Must not allocate large buffers in hot path
- Must refresh command before 0.5s staleness timeout

---

## 3. Proposed Architecture: LocalExecutive

### 3.1 API Contract

```python
class LocalExecutive:
    """
    Non-ROS mid-layer for autonomous navigation.
    Accepts high-level goals and computes velocity commands at ~30Hz.
    """
    
    def __init__(self):
        """Initialize with default parameters."""
        pass
    
    def set_goal(self, x: float, y: float) -> None:
        """
        Set a world-space goal position (meters).
        The executive will drive toward rolling ~0.8-1.0m subgoals.
        """
        pass
    
    def set_wander_mode(self, enabled: bool) -> None:
        """
        Enable/disable wander mode (explore, avoid obstacles).
        Disables explicit goal if enabled.
        """
        pass
    
    def cancel(self) -> None:
        """
        Cancel current goal/wander mode.
        Next tick() will return None (zero velocity).
        """
        pass
    
    def is_active(self) -> bool:
        """Return True if goal or wander mode is set."""
        pass
    
    def tick(self, obs_map: np.ndarray, pose: tuple, dt: float) -> dict | None:
        """
        Compute velocity command for this frame.
        
        Args:
            obs_map: (H, W) uint8 ego-space obstacles (0=free, 1-100=height cm)
            pose: (x, y, theta) in world frame (meters, radians)
            dt: time since last tick (seconds)
        
        Returns:
            {'fwd_mps': float, 'ang_rads': float} or None if inactive.
            Caller applies safety scaling (already integrated in wheelbase).
        """
        pass
    
    def get_debug_state(self) -> dict:
        """Return debug info for visualization (trajectories, subgoals, etc.)."""
        pass
```

### 3.2 Geometric v0: DWA or MPPI-lite

**Initial Implementation**: Dynamic Window Approach (DWA) simplified

**Why DWA?**:
- Well-proven, numerically stable
- Efficient (10-100 trajectories, 0.5-1s horizon)
- Easy to tune (velocity/acceleration limits already known)
- Straightforward upgrade path to learned local planner

**Algorithm** (per tick):

1. **Generate velocity candidates**:
   - Sample (v, ω) grid around current velocity
   - Apply kinematic limits (wheelbase differential drive)
   - Apply dynamic limits (acceleration, deceleration)
   - Typical: 7 linear × 9 angular = 63 candidates

2. **Simulate trajectories**:
   - Forward integrate for 0.5-1.0s (15-30 steps @ 30Hz)
   - Store (x, y, θ) sequence per candidate
   - Pre-allocate trajectory buffer (reuse every frame)

3. **Score each trajectory**:
   ```
   score = w_goal × goal_cost + w_obs × obstacle_cost + w_smooth × smoothness_cost
   
   goal_cost = distance to rolling subgoal (0.8-1.0m ahead on path to goal)
   obstacle_cost = min clearance in ego-space projection (heavy penalty < 20cm)
   smoothness_cost = angular/linear acceleration magnitude
   ```

4. **Select best** (highest score, or zero if all blocked)

5. **Output** first velocity in best trajectory

**Parameters** (tunable via LocalExecutive constructor):
```python
horizon_sec = 1.0           # Trajectory lookahead
v_samples = 7               # Linear velocity samples
w_samples = 9               # Angular velocity samples
v_max = 0.25                # m/s (slightly above current MAX_FWD for growth)
w_max = 0.8                 # rad/s (slightly above current MAX_ANG)
subgoal_dist = 0.8          # m (rolling subgoal distance)
min_clearance = 0.20        # m (obstacle cost kicks in)
accel_max = 0.5             # m/s² (comfortable)
alpha_max = 1.5             # rad/s² (comfortable)
```

**Performance Target**: 2-3 ms/tick (to fit in 33ms budget)

**Advantages**:
- No dependency on global path planner (pure reactive)
- Handles dynamic obstacles (re-plans every frame)
- Smooth velocity profiles (simulates acceleration)
- Easy to visualize (draw sampled trajectories)

### 3.3 Upgrade Path: Neural Local Planner

**After DWA v0 is stable**, replace scoring function with learned policy:

**Option A: Value network** (faster)
```python
# Replace trajectory scoring loop:
trajectories = generate_trajectories(v_samples, w_samples, horizon_sec)
obs_patches = extract_ego_patches(obs_map, trajectories)  # 64x64 around each traj
goal_vectors = compute_goal_vectors(pose, goal, trajectories)

# Forward pass (batched, ~1ms on Jetson Orin NX)
with torch.no_grad():
    values = model(obs_patches, goal_vectors)  # (N_traj,) scores

best_idx = torch.argmax(values)
return trajectories[best_idx, 0]  # First velocity in best trajectory
```

**Option B: Direct policy** (end-to-end, simpler)
```python
# Single forward pass:
obs_patch = obs_map[ego_crop]  # 128x128 centered on robot
goal_local = world_to_ego(goal, pose)  # (dx, dy) in ego frame

with torch.no_grad():
    action = model(obs_patch, goal_local)  # (v, ω) directly

return {'fwd_mps': action[0], 'ang_rads': action[1]}
```

**Training**:
- Imitation learning: log (obs, goal, velocity) tuples from DWA v0 runs
- Behavior cloning: supervised learning (L2 loss on velocity)
- Fine-tuning: DAgger (collect corrections, retrain)
- Deployment: ONNX or TorchScript for ~1ms inference

**Model Size**: ResNet-18 or EfficientNet-B0 backbone + MLP head (~15-25 MB)

**Advantages**:
- Learns implicit costmap understanding (edges, textures, height)
- Generalizes beyond hand-tuned DWA parameters
- Can encode style (aggressive vs. cautious)

**Risks**:
- Requires data collection + training pipeline
- Harder to debug than geometric planner
- Must validate safety (keep DWA as fallback)

### 3.4 Wander Mode

**Goal**: Autonomous exploration without explicit waypoint

**Strategy**: Frontier-based local wandering

1. **Detect frontiers** in ego-space:
   - Boundary pixels: `known=0` adjacent to `known=255`
   - Group into clusters (connected components)
   - Score by: distance, angular cost, estimated open area

2. **Select frontier cluster**:
   - Prefer clusters 1-2m away (reachable horizon)
   - Bias toward current heading (reduce zig-zag)

3. **Set virtual goal** at cluster centroid

4. **DWA/neural planner drives toward goal**

5. **Re-select frontier** every N frames (e.g., 30 frames = 1s)

**Fallback**: If no frontiers (fully explored local area):
- Rotate in place 90° (expose new camera FOV)
- OR back up 0.5m (escape local minimum)

---

## 4. Implementation Scaffold

See `src/local_executive.py` (minimal stub implementation).

### 4.1 File Structure

```
src/
├── local_executive.py       # New: LocalExecutive class (DWA v0)
├── main.py                  # Modified: add --auto-local flag, call local_executive.tick()
├── navigator.py             # Unchanged (existing VFH remains available)
├── vision.py                # Unchanged
├── wheelbase.py             # Unchanged
├── tools.py                 # Optional: add local_executive getters
└── robot_config.py          # Unchanged
```

### 4.2 Main Loop Modification

```python
# main.py, line ~40, add argument:
parser.add_argument("--auto-local", action="store_true",
                    help="Enable LocalExecutive mid-layer (default: VFH navigator)")

# main.py, line ~100, after tools.init():
from local_executive import LocalExecutive
local_executive = LocalExecutive() if args.auto_local else None

# main.py, lines 157-163, replace with:
elif not wb.is_twist_for_active():
    if local_executive and local_executive.is_active():
        cmd = local_executive.tick(
            obs_map=vis._persistent_obs,
            pose=(vis._pose.x, vis._pose.y, vis._pose.theta),
            dt=LOOP_DT
        )
        if cmd is not None:
            tools.twist(cmd['fwd_mps'], cmd['ang_rads'])
        else:
            tools.set_wheel_vels(0.0, 0.0)
    else:
        twist = navigator.compute_twist(atlas) if atlas is not None else None
        if twist is not None:
            tools.twist(twist[0], twist[1])
        else:
            tools.set_wheel_vels(0.0, 0.0)
```

### 4.3 Tool Call Extensions

**UI → Main Loop** (via Gemini/agent):

Add to `tools.py` or handle in main loop tool dispatch:
```python
# New tool calls:
elif name == "local_goto":
    x = cargs.get("x", 0.0)
    y = cargs.get("y", 0.0)
    if local_executive:
        local_executive.set_goal(x, y)
        
elif name == "local_wander":
    enable = cargs.get("enable", True)
    if local_executive:
        local_executive.set_wander_mode(enable)

elif name == "local_cancel":
    if local_executive:
        local_executive.cancel()
```

---

## 5. Open Questions & Risks

### 5.1 Coordinate Transforms

**Question**: When goal is set in world frame (x, y), how to efficiently compute rolling subgoal in ego frame for DWA scoring?

**Answer**: Pre-compute world→ego transform using current pose:
```python
# World goal (x_goal, y_goal) → ego goal (dx_ego, dy_ego)
dx_world = x_goal - pose_x
dy_world = y_goal - pose_y
cos_t = math.cos(pose_theta)
sin_t = math.sin(pose_theta)
dx_ego = dx_world * cos_t + dy_world * sin_t
dy_ego = -dx_world * sin_t + dy_world * cos_t
```

Rolling subgoal: clamp `|dx_ego|` to 0.8-1.0m range.

### 5.2 Trajectory Collision Checking

**Question**: DWA simulates 15-30 steps × 63 trajectories = ~1000-2000 pose checks. How to efficiently check collision in ego-space?

**Options**:
1. **Lazy per-step check**: For each (x, y, θ) in trajectory, transform robot footprint to ego, sample obs_map
2. **Pre-render trajectory mask**: Rasterize full trajectory, single obs_map lookup
3. **Conservative bounding circle**: Check only robot radius (simpler, slightly pessimistic)

**Recommendation**: Start with bounding circle (r=25cm, covers diagonal), upgrade to footprint if needed.

```python
def trajectory_collision_cost(traj, obs_map, pose, ego_px_size=0.01):
    """
    traj: (N_steps, 3) [x, y, theta] in world frame
    obs_map: (H, W) ego-space obstacles
    Returns: min_clearance_m (< 0.2 m → high cost)
    """
    robot_radius_px = int(0.25 / ego_px_size)  # 25 cm → 25 px
    min_clearance = float('inf')
    
    for (x, y, theta) in traj:
        # World to ego
        dx_ego, dy_ego = world_to_ego(x - pose[0], y - pose[1], pose[2])
        col = int(RCX + dx_ego / ego_px_size)
        row = int(RCY - dy_ego / ego_px_size)
        
        # Sample circle around robot
        if 0 <= row < obs_map.shape[0] and 0 <= col < obs_map.shape[1]:
            patch = obs_map[
                max(0, row - robot_radius_px):min(obs_map.shape[0], row + robot_radius_px),
                max(0, col - robot_radius_px):min(obs_map.shape[1], col + robot_radius_px)
            ]
            if patch.max() > 100:  # Obstacle detected
                clearance = ... # distance to nearest obstacle in patch
                min_clearance = min(min_clearance, clearance)
    
    return min_clearance * ego_px_size  # Convert back to meters
```

### 5.3 Performance Budget

**Question**: Can DWA fit in 2-3 ms on Jetson Orin NX?

**Back-of-envelope**:
- 63 trajectories × 20 steps = 1260 pose integrations (~1-2 µs each, numpy) → ~2 ms
- Collision checks: 1260 bounding-circle samples → ~1 ms
- Scoring: 63 dot products → <0.1 ms
- **Total: ~3-4 ms** (tight but feasible)

**Mitigation**:
- Reduce samples (5 linear × 7 angular = 35 trajectories) → ~2 ms
- Use every-other-frame planning (15 Hz) if needed
- Profile on real hardware; numpy/numba acceleration if bottleneck

### 5.4 Global Map Staleness

**Question**: persistent_obs includes projected global map. If robot returns to old area after SLAM loop closure, global map may have shifted. Does this break local planning?

**Risk**: Medium. Loop closure corrections are typically small (< 10 cm) for good odometry, but accumulated drift before closure can be 50-100 cm.

**Mitigation**:
1. LocalExecutive trusts **ego-space obstacles** (0-2m range) more than global projection
   - Weight obs_combined (fresh sensor data) higher than persistent_obs delta
2. Global map used only for obstacle memory beyond current sensor FOV (e.g., "corner I saw 2s ago")
3. If loop closure happens, worst case: local planner sees temporary phantom obstacle, re-plans around it
4. No catastrophic failure: safety guard still enforces hard stops for ego-space obstacles

**Validation**: Test loop closure during live navigation, verify no erratic behavior.

### 5.5 Integration with Existing UI/Agent

**Question**: How does high-level agent (Gemini/vLLM) set goals?

**Proposed Flow**:

1. **Agent** (via `ui.py` tool calls):
   - Reads current pose: `tools.get_vision()._pose.x/y/theta`
   - Decides goal (e.g., "move 2m forward" → compute world-space x_goal, y_goal)
   - Sends tool call: `{"name": "local_goto", "args": {"x": x_goal, "y": y_goal}}`

2. **Main loop** (in tool dispatch):
   - Calls `local_executive.set_goal(x_goal, y_goal)`

3. **LocalExecutive**:
   - Stores goal, activates planning
   - Each tick: compute subgoal, run DWA, return velocity
   - When close to goal (< 10 cm) → auto-cancel, return None

4. **Agent** (monitors):
   - Polls pose periodically (e.g., every 1s)
   - Detects goal reached or stuck (pose not changing)
   - Sends new goal or cancels

**Advantage**: Agent doesn't need to understand DWA/trajectories; just sets waypoints.

### 5.6 Failure Modes

**Stuck in local minimum**:
- DWA scores all trajectories as blocked → returns None → robot stops
- Agent timeout (e.g., 5s no progress) → cancels goal, tries different waypoint
- Wander mode fallback: rotate or back up

**Sensor occlusion**:
- Camera blocked (e.g., bright light, dust) → obs_map mostly unknown
- Safety guard still active (ego-space projection from global map)
- Worst case: safety scales → 0, robot stops (safe default)

**Pose drift**:
- Wheel odometry drifts (carpet slip)
- Visual odometry corrects, but not perfect
- SLAM loop closure reduces long-term drift
- Local planner only needs pose accurate to ~10 cm (subgoal radius)
- If drift > 50 cm: agent may need to re-localize or adjust goals

**Collision with low obstacles**:
- Depth cameras have minimum height threshold (~5 cm)
- Very flat obstacles (cables, low curbs) may not register
- Mitigation: safety guard still scans ego-space; wheelbase motor current limiting
- Risk: Medium (unavoidable with current sensors; not unique to LocalExecutive)

---

## 6. Testing Plan

### 6.1 Unit Tests

1. **LocalExecutive API**:
   - `set_goal()` / `cancel()` / `is_active()` state machine
   - `tick()` returns None when inactive
   - `tick()` returns valid velocity dict when active

2. **Coordinate transforms**:
   - World→ego conversion matches `pose.py` conventions
   - Rolling subgoal clamping (0.8-1.0m ahead)

3. **DWA trajectory generation**:
   - Samples cover reachable (v, ω) space
   - Kinematic constraints satisfied (wheelbase, acceleration)

4. **Collision checking**:
   - Static obstacle map → trajectory collision detected
   - Free space → trajectory valid

### 6.2 Integration Tests

1. **No-op mode** (`--auto-local` flag off):
   - Verify existing navigator still works
   - No performance regression

2. **Inactive mode** (`--auto-local` on, no goal set):
   - LocalExecutive returns None
   - Robot idle (zero velocity)

3. **Static goal** (open space):
   - Set goal 2m ahead
   - Robot drives straight
   - Stops within 10 cm of goal

4. **Obstacle avoidance**:
   - Set goal through/behind obstacle
   - Robot plans detour
   - Reaches goal without collision

5. **Wander mode**:
   - Enable wander, place in open area
   - Robot explores, avoids walls
   - No crashes over 60s

### 6.3 Hardware Tests (Jetson Orin NX)

1. **Performance profiling**:
   - Measure `tick()` execution time (target < 3 ms)
   - Verify 30 fps main loop maintained
   - Check CPU/GPU utilization

2. **Live navigation**:
   - Start in known room
   - Set goal 5m away (different room)
   - Verify: smooth motion, no oscillation, reaches goal

3. **Gamepad override**:
   - Start autonomous navigation
   - Grab gamepad → robot stops auto, manual control works
   - Release gamepad → auto resumes

4. **Safety validation**:
   - Approach wall at speed
   - Verify safety guard slows/stops before collision
   - LocalExecutive respects safety scales

### 6.4 Acceptance Criteria

- [ ] `--auto-local` flag compiles and runs without errors
- [ ] Existing teleop/VFH navigator unaffected when flag off
- [ ] `local_goto` tool call sets goal, robot drives toward it
- [ ] `tick()` execution time < 5 ms (target 2-3 ms)
- [ ] No crashes in 10 minutes of continuous wander mode
- [ ] Gamepad override works in all modes
- [ ] Safety guard integration (no collisions in stress test)

---

## 7. Next Steps

### Phase 1: Minimal Scaffold (1-2 days, IN PROGRESS)
- [x] Create `docs/kevin-autonomy-midlayer.md`
- [x] Add `src/local_executive.py` stub (API only, no DWA yet)
- [x] Add `--auto-local` flag to `main.py`
- [x] Integrate stub into main loop (inactive mode)
- [ ] Test compilation and no-op behavior

### Phase 2: DWA v0 Implementation (3-5 days)
- [ ] Implement trajectory generation (kinematic sampling)
- [ ] Implement collision checking (bounding circle)
- [ ] Implement scoring function (goal + obstacle + smoothness)
- [ ] Tune parameters on real robot (v_samples, w_samples, weights)
- [ ] Validate: straight-line goal, obstacle avoidance

### Phase 3: Wander Mode (2-3 days)
- [ ] Implement frontier detection (ego-space boundary)
- [ ] Implement cluster scoring and selection
- [ ] Integrate with DWA (virtual goal)
- [ ] Test: continuous exploration without getting stuck

### Phase 4: UI Integration (1-2 days)
- [ ] Add `local_goto`, `local_wander`, `local_cancel` tool calls
- [ ] Update Gemini prompt with new tools
- [ ] Test agent-driven navigation scenarios

### Phase 5: Neural Upgrade (Optional, 5-10 days)
- [ ] Data collection: log (obs, goal, velocity) from DWA runs
- [ ] Train value network or policy network
- [ ] Integrate inference (ONNX/TorchScript)
- [ ] A/B test: DWA vs. neural, measure success rate & smoothness

---

## 8. References

### Code Files Analyzed
- `src/main.py` – 30Hz loop, tool dispatch
- `src/vision.py` – Capture thread, depth processing, safety
- `src/navigator.py` – VFH reactive planner
- `src/wheelbase.py` – Motor control, safety watchdogs
- `src/pose.py` – Kalman pose fusion
- `src/safety.py` – Collision avoidance scaling
- `src/globalmap.py` – World-frame occupancy grid
- `src/slam.py` – Pose-graph SLAM
- `src/robot_config.py` – Geometry constants
- `src/tools.py` – High-level API wrappers

### Key Constants
```python
# Timing
TARGET_FPS = 30           # main.py
LOOP_DT = 1/30            # 0.0333 s

# Ego-space
FRAME_W, FRAME_H = 320, 240
RCX, RCY = 81, 119        # Robot center (facing RIGHT)
EGO_PX_SIZE = 0.01        # 1 cm/px

# Robot physical
WHEELBASE_M = 0.34        # 34 cm
WHEEL_RADIUS_M = 0.08565  # 17.13 cm diameter

# Navigator (VFH)
MAX_FWD = 0.20            # m/s
MAX_ANG = 0.6             # rad/s
STOP_RANGE = 18           # px (18 cm)
DANGER_RANGE = 40         # px (40 cm)

# Safety
COMMAND_STALE_TIMEOUT = 0.5  # seconds

# Global map
MAP_W, MAP_H = 960, 720
PX_SIZE = 0.02            # 2 cm/px
```

---

## 9. Revision History

- **2026-09-06**: Initial investigation and scaffold (this document)

---

## Appendix A: Example Usage

### Running with LocalExecutive

```bash
# Default: existing VFH navigator
python src/main.py --rs1 815412070676 --rs2 944622074292

# Enable LocalExecutive mid-layer
python src/main.py --rs1 815412070676 --rs2 944622074292 --auto-local

# In Python (agent tool call):
# Set goal 2 meters forward (world frame)
tools.call_tool("local_goto", {"x": pose.x + 2.0, "y": pose.y})

# Enable wander mode
tools.call_tool("local_wander", {"enable": True})

# Cancel navigation
tools.call_tool("local_cancel", {})
```

### Monitoring

```python
# In main loop (optional debug print every 90 frames):
if local_executive and local_executive.is_active() and frame_id % 90 == 0:
    dbg = local_executive.get_debug_state()
    print("LocalExec: goal=(%.2f,%.2f) subgoal=(%.2f,%.2f) cmd=(%.2f,%.2f)" % (
        dbg.get('goal_x', 0), dbg.get('goal_y', 0),
        dbg.get('subgoal_x', 0), dbg.get('subgoal_y', 0),
        dbg.get('cmd_fwd', 0), dbg.get('cmd_ang', 0)))
```

---

**End of Document**
