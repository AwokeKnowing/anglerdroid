# Kevin Autonomy Mid-Layer: Design & Integration

**Robot**: AwokeKnowing/anglerdroid (Jetson Orin NX)  
**Date**: 2026-09-06  
**Status**: Phase 1 (mppi-costmap-v0) – Scaffold + Literature Review  
**Constraint**: NO ROS/ros2/nav2

## Executive Summary

This document maps the existing 30Hz perception-to-control pipeline and proposes a **LocalExecutive** mid-layer based on **MPPI (Model Predictive Path Integral Control)** on the live 2D costmap. The executive accepts async map-frame goals (x,y) or "wander" mode and continuously generates velocity commands using non-blocking mailbox pattern. The design preserves existing teleop/UI control and keeps the 30Hz map generation intact.

**Experiment**: `mppi-costmap-v0`

### Implementation Roadmap (Literature Review):
- **Phase 1 (Current)**: MPPI on 2D costmap — CUDA or Torch, differential drive
- **Phase 2**: iPlanner-IL or SAC-polar (learned MPC / RL policies)
- **Phase 3**: ViNT/NoMaD + Doctor (visual navigation + safety filter over MPPI base)

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

### 2.1 Cleanest Hook (Implemented)

**Location**: `main.py` lines 157-177

**Non-blocking mailbox pattern**:
- `set_goal(x, y)` called from UI/tool dispatch (main thread)
- Main loop calls `local_executive.tick()` at 30Hz
- Vision/capture thread never waits or blocks on LocalExecutive

**Current code** (behind `--auto-local` flag, default OFF):
```python
elif not wb.is_twist_for_active():
    # LocalExecutive mode (if enabled and active)
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
        # Default: VFH navigator (existing, heading-based)
        twist = navigator.compute_twist(atlas) if atlas is not None else None
        if twist is not None:
            tools.twist(twist[0], twist[1])
        else:
            tools.set_wheel_vels(0.0, 0.0)
```

**Why this hook?**:
- Non-blocking: main loop already owns command timing, no waits
- Never interrupts capture thread
- Respects gamepad/twist_for priority (higher precedence)
- Access to latest obs_map, pose, safety scales
- Same command rate as existing navigator (30 Hz)
- Fallback: existing VFH navigator when `--auto-local` OFF

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

## 3. Architecture: MPPI-based LocalExecutive

### 3.1 API Contract (Implemented)

```python
class LocalExecutive:
    """
    MPPI-based local planner for autonomous navigation (non-ROS).
    Accepts async map-frame (x,y) goals, computes velocity commands from 
    ego-space 2D costmap. Non-blocking mailbox design.
    
    Experiment: mppi-costmap-v0
    """
    
    def __init__(self, horizon_sec=1.0, n_samples=512, v_max=0.25, w_max=0.8,
                 mppi_temperature=1.0, mppi_noise_sigma_v=0.1, mppi_noise_sigma_w=0.3):
        """
        Initialize with MPPI parameters.
        
        Args:
            horizon_sec: Lookahead time (1.0s typical)
            n_samples: MPPI trajectory samples (512-2048)
            v_max, w_max: Velocity limits (m/s, rad/s)
            mppi_temperature: λ parameter (lower = more aggressive)
            mppi_noise_sigma_v, mppi_noise_sigma_w: Process noise std
        """
    
    def set_goal(self, x: float, y: float) -> None:
        """
        Set a map-frame goal position (meters). Non-blocking mailbox write.
        Thread-safe: Vision/capture never waits.
        """
    
    def set_wander_mode(self, enabled: bool) -> None:
        """Enable/disable frontier-based wander mode."""
    
    def cancel(self) -> None:
        """Cancel goal/wander (next tick returns None)."""
    
    def is_active(self) -> bool:
        """True if goal or wander mode set."""
    
    def tick(self, obs_map: np.ndarray, pose: tuple, dt: float) -> dict | None:
        """
        Compute velocity command (non-blocking, ~30Hz).
        
        Main loop calls this each frame. Reads latest goal from mailbox,
        runs MPPI on costmap window, returns command.
        
        Args:
            obs_map: (H, W) uint8 ego-space (0=free, 1-100=obstacle height cm)
            pose: (x, y, theta) map frame (meters, radians)
            dt: time step (0.033s typical)
        
        Returns:
            {'fwd_mps': float, 'ang_rads': float} or None if inactive
        """
    
    def get_debug_state(self) -> dict:
        """Debug info: goal, subgoal, trajectories, costs."""
```

### 3.2 Phase 1: MPPI on 2D Costmap (mppi-costmap-v0)

**Chosen Approach**: Model Predictive Path Integral (MPPI) control

**Why MPPI?** (Literature review conclusion):
- **Smooth control**: Soft weights over trajectory samples (no hard selection like DWA)
- **Parallelizable**: Embarrassingly parallel rollouts → CUDA/Torch efficient
- **Information-theoretic**: Optimal under certain stochastic assumptions
- **Proven on hardware**: Used in autonomous driving (AUTORALLY, etc.)
- **Upgrade path**: Direct plugin for learned cost functions (Phase 2-3)

**Algorithm** (per tick, ~30Hz):

1. **Sample control sequences**: 
   ```
   U[i,k] ~ N(u_prev[k], Σ)  for i=1..n_samples, k=1..n_steps
   Σ = diag([σ_v², σ_w²])
   ```
   - Typical: 512-2048 samples, 30 steps (1s horizon @ 30Hz)
   - **CUDA hook**: `cuMPPI.sample_controls(n_samples, n_steps, noise_sigma)`
   - **Torch hook**: `torch.randn(n_samples, n_steps, 2).cuda() * sigma + u_prev`

2. **Rollout trajectories** (batched):
   ```
   x[k+1] = f(x[k], u[k], dt)  # Differential drive kinematics
   ```
   - Parallel integration for all samples
   - **CUDA hook**: `cuMPPI.rollout_batch(U, x0, costmap_texture)`
   - **Torch hook**: Batched forward pass with costmap interpolation

3. **Compute costs**:
   ```
   S[i] = ∑_{k=0}^{T-1} running_cost(x[k], u[k]) + terminal_cost(x[T])
   
   running_cost = w_obs × obstacle(x[k]) + w_smooth × ||u[k]||²
   terminal_cost = w_goal × ||x[T] - x_subgoal||²
   ```
   - **Obstacle cost**: Sample costmap at (x,y), high penalty if occupied
   - **Goal cost**: Distance to rolling subgoal (0.8-1.0m ahead)
   - **Smoothness**: Penalize large velocities/accelerations

4. **Weight trajectories**:
   ```
   w[i] = exp(-S[i] / λ) / Z,  where Z = ∑ exp(-S[i] / λ)
   ```
   - λ (temperature) controls exploration (lower = more greedy)

5. **Weighted mean control**:
   ```
   u* = ∑ w[i] × U[i,0]  (first control in each sequence)
   ```
   - Smooth blending of all samples (not hard max like DWA)

**Parameters** (tunable via LocalExecutive constructor):
```python
horizon_sec = 1.0           # Lookahead (30 steps @ 30Hz)
n_samples = 512             # Trajectory samples (increase for smoothness)
v_max = 0.25                # m/s
w_max = 0.8                 # rad/s
subgoal_dist = 0.8          # Rolling subgoal distance (m)
mppi_temperature = 1.0      # λ (lower = more aggressive)
noise_sigma_v = 0.1         # Linear velocity noise std
noise_sigma_w = 0.3         # Angular velocity noise std
```

**Performance Target**: 3-5 ms/tick on Jetson Orin NX
- CUDA MPPI: ~2-3 ms (512 samples, 30 steps)
- PyTorch MPPI: ~4-5 ms (with costmap batched interpolation)
- Geometric stub (Phase 1): <1 ms (pure pursuit fallback)

**Implementation Status**:
- ✅ API complete, non-blocking mailbox
- ✅ Geometric controller stub (pure pursuit for testing)
- ✅ MPPI stubs with integration hooks clearly marked:
  - `_mppi_sample_trajectories()` → CUDA/Torch sampling
  - `_mppi_rollout_batch()` → Batched rollout + cost
  - `_mppi_compute_weights()` → Softmax weighting
  - `_mppi_weighted_control()` → Output blending
- ⏳ Next: Replace numpy stubs with CUDA/Torch parallel implementation

### 3.3 Phase 2: Learned MPC (iPlanner-IL or SAC-polar)

**After MPPI v0 is stable**, upgrade cost function or policy:

**Option A: iPlanner-IL** (Imitation Learning over MPPI)
- **Paper**: "Learning to Plan via Imitation and Practice" (Bronstein et al.)
- **Architecture**: ResNet encoder + learned cost function
- **Training**: Imitate expert MPPI trajectories in diverse environments
- **Deployment**: Replace `_mppi_running_cost()` with neural network
  ```python
  # Learned cost network (batched over MPPI samples)
  obs_patches = extract_costmap_windows(X, obs_map)  # (n_samples, T, 64, 64)
  costs = cost_network(obs_patches, goals)  # (n_samples,)
  # Rest of MPPI weighting unchanged
  ```
- **Advantages**: 
  - Keeps MPPI structure (explainable, safe)
  - Learns implicit cost from demonstrations
  - Faster than hand-tuning cost weights

**Option B: SAC-polar** (Soft Actor-Critic, polar coordinates)
- **Architecture**: Policy network maps (ego-obs, goal) → (v, ω) directly
- **Training**: RL in simulation (Isaac Gym or similar)
- **Deployment**: Replace entire MPPI loop
  ```python
  obs_patch = obs_map[ego_crop]  # 128x128 ego-space
  goal_ego = world_to_ego(goal, pose)
  with torch.no_grad():
      action, _ = policy(obs_patch, goal_ego)
  return {'fwd_mps': action[0], 'ang_rads': action[1]}
  ```
- **Advantages**:
  - End-to-end policy (no explicit trajectory rollout)
  - ~1ms inference (faster than MPPI)
- **Risks**:
  - Less interpretable than MPPI
  - Requires extensive sim training
  - **Mitigation**: Keep MPPI as safety filter (Phase 3)

### 3.4 Phase 3: Visual Policies + Safety Filter (ViNT/NoMaD + Doctor)

**Foundation**: MPPI base layer from Phase 1

**Visual Navigation Transformer (ViNT) or NoMaD**:
- **Papers**: 
  - ViNT: "Visual Navigation Transformer" (Shah et al., 2022)
  - NoMaD: "Navigating Autonomous Driving with Map-less Dense Vision" (2023)
- **Architecture**: Vision transformer maps RGB image → subgoal waypoint
- **Training**: Large-scale outdoor navigation datasets
- **Deployment**: 
  ```python
  # High-level: ViNT proposes subgoal from RGB
  subgoal_ego = vint_model(rgb_image, goal_image)
  
  # Mid-level: MPPI tracks subgoal on costmap
  cmd = mppi_executive.tick_with_subgoal(obs_map, pose, subgoal_ego)
  ```

**Doctor (safety filter)**:
- **Paper**: "Doctor: Diffusion Model for Safe Robot Navigation" (Wu et al., 2023)
- **Purpose**: Safety wrapper around learned policies
- **Method**: Diffusion model learns "safe control distribution" from expert demos
- **Deployment**:
  ```python
  # Learned policy proposes action
  action_proposed = learned_policy(obs, goal)
  
  # Doctor filters through safety diffusion
  action_safe = doctor_model.filter(action_proposed, obs_map, pose)
  
  # MPPI provides backup if Doctor rejects
  if doctor_model.is_safe(action_safe):
      return action_safe
  else:
      return mppi_executive.tick(obs_map, pose, dt)  # Fallback
  ```

**Advantages**:
- Visual policies: Direct RGB → waypoint (no explicit costmap needed)
- Doctor: Provable safety guarantees over learned policies
- MPPI fallback: Always have geometric baseline if learned model fails

**Risks**:
- Model size: ViNT/NoMaD are large (~100-500MB)
- Inference time: ~10-20ms on Jetson Orin NX
- Generalization: Trained on outdoor data, may need fine-tuning for indoor

### 3.5 Wander Mode (All Phases)

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

## 4. Literature Review Summary

**Phase Ranking** (from literature + hardware constraints):

| Phase | Approach | Pros | Cons | Timeline |
|-------|----------|------|------|----------|
| **1** | **MPPI on costmap** (current) | Parallelizable, smooth, proven | Hand-tuned costs, ~3-5ms | 1-2 weeks |
| **2** | iPlanner-IL or SAC-polar | Learned costs/policy, faster | Requires training data/sim | 2-3 weeks |
| **3** | ViNT/NoMaD + Doctor | Visual, generalizable, safe | Large models, slower | 4-6 weeks |

**Key Decision Points**:
1. **MPPI vs. DWA**: MPPI chosen for smooth control + CUDA efficiency
2. **Costmap vs. Vision**: Costmap first (Phase 1-2), vision later (Phase 3)
3. **Safety**: Geometric baseline (MPPI) always available as fallback

**References**:
- MPPI: Williams et al. (2017) "Information-Theoretic Model Predictive Control"
- AUTORALLY: Williams et al. (2018) "Aggressive Driving with MPPI"
- iPlanner: Bronstein et al. (2022) "Learning to Plan via Imitation and Practice"
- SAC: Haarnoja et al. (2018) "Soft Actor-Critic"
- ViNT: Shah et al. (2022) "Visual Navigation Transformer"
- NoMaD: Sridhar et al. (2023) "Navigating Autonomous Driving with Map-less Dense Vision"
- Doctor: Wu et al. (2023) "Diffusion Model for Safe Robot Navigation"

## 5. Implementation Scaffold (Phase 1)

See `src/local_executive.py` (MPPI stubs + geometric controller).

### 5.1 File Structure

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

### 5.2 Main Loop Modification (Implemented)

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

### 5.3 Tool Call Extensions (Implemented)

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

## 6. Open Questions & Risks

### 6.1 Coordinate Transforms (Implemented)

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

### 6.2 MPPI Costmap Sampling (Phase 1 Key Risk)

**Question**: MPPI samples 512-2048 trajectories × 30 steps = ~15k-60k costmap lookups. How to efficiently sample costmap in ego-space?

**Options**:
1. **CUDA texture sampling**: Upload costmap to GPU texture, use hardware interpolation
   - Fast: ~0.1ms for 50k lookups on Jetson Orin NX
   - Requires CUDA MPPI implementation
2. **PyTorch grid_sample**: Batched bilinear interpolation on GPU
   - Fast: ~0.5-1ms for 50k lookups
   - Works with Torch MPPI
3. **Numpy indexing (current stub)**: Serial per-point lookup
   - Slow: ~10-20ms for 50k lookups
   - OK for Phase 1 testing, replace for full MPPI

**Recommendation**: PyTorch grid_sample for initial MPPI (easier to prototype than CUDA), upgrade to CUDA texture if performance critical.

```python
# PyTorch costmap sampling (batched)
def sample_costmap_batch(traj_batch, obs_map_tensor, pose):
    """
    traj_batch: (n_samples, n_steps, 3) [x, y, theta] in map frame
    obs_map_tensor: (1, 1, H, W) torch tensor on GPU
    Returns: (n_samples, n_steps) obstacle costs
    """
    # Transform map positions to ego-space grid coordinates
    dx = traj_batch[..., 0] - pose[0]
    dy = traj_batch[..., 1] - pose[1]
    dx_ego = dx * np.cos(pose[2]) + dy * np.sin(pose[2])
    dy_ego = -dx * np.sin(pose[2]) + dy * np.cos(pose[2])
    
    # Normalize to [-1, 1] for grid_sample
    grid_x = (RCX + dx_ego / EGO_PX_SIZE) / obs_map_tensor.shape[3] * 2 - 1
    grid_y = (RCY - dy_ego / EGO_PX_SIZE) / obs_map_tensor.shape[2] * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (n_samples, n_steps, 2)
    
    # Batched bilinear sample
    obs_vals = F.grid_sample(obs_map_tensor, grid, align_corners=False)
    return obs_vals.squeeze()  # (n_samples, n_steps)
```

### 6.3 Performance Budget (MPPI)

**Question**: Can MPPI fit in 3-5 ms on Jetson Orin NX?

**Back-of-envelope** (512 samples, 30 steps):
- **CUDA MPPI** (cuMPPI library):
  - Sampling: ~0.2 ms (GPU kernel)
  - Rollout: ~1.5 ms (parallel integration + texture sampling)
  - Weighting: ~0.3 ms (reduction kernel)
  - **Total: ~2-3 ms** ✓ Fits comfortably
- **PyTorch MPPI**:
  - Sampling: ~0.5 ms (torch.randn)
  - Rollout: ~2.0 ms (batched ops + grid_sample)
  - Weighting: ~0.5 ms (softmax + sum)
  - **Total: ~3-4 ms** ✓ Fits with margin
- **Numpy stub (current)**:
  - Serial rollout: ~15-20 ms
  - **Too slow for real-time**, OK for Phase 1 testing

**Recommendation**: Start with PyTorch MPPI (easier to prototype), profile on hardware, upgrade to CUDA if needed.

**Mitigation if too slow**:
- Reduce samples: 256 instead of 512 → ~2ms PyTorch
- Reduce horizon: 20 steps instead of 30 → ~2.5ms PyTorch
- Every-other-frame planning: 15 Hz → double budget to 6-10ms

### 6.4 Global Map Staleness (Same as before)

**Question**: persistent_obs includes projected global map. If robot returns to old area after SLAM loop closure, global map may have shifted. Does this break local planning?

**Risk**: Medium. Loop closure corrections are typically small (< 10 cm) for good odometry, but accumulated drift before closure can be 50-100 cm.

**Mitigation**:
1. LocalExecutive trusts **ego-space obstacles** (0-2m range) more than global projection
   - Weight obs_combined (fresh sensor data) higher than persistent_obs delta
2. Global map used only for obstacle memory beyond current sensor FOV (e.g., "corner I saw 2s ago")
3. If loop closure happens, worst case: local planner sees temporary phantom obstacle, re-plans around it
4. No catastrophic failure: safety guard still enforces hard stops for ego-space obstacles

**Validation**: Test loop closure during live navigation, verify no erratic behavior.

### 6.5 Integration with Existing UI/Agent (Implemented)

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

### 6.6 Failure Modes (MPPI-specific)

**Stuck in local minimum**:
- All MPPI samples have high cost → weighted mean → very slow creep or stop
- **Advantage over DWA**: Soft blending still produces gentle motion (less binary)
- Agent timeout (e.g., 5s no progress) → cancels goal, tries different waypoint
- Wander mode fallback: rotate or back up

**Temperature tuning**:
- λ too low (e.g., 0.1): Over-aggressive, picks single best sample (noise-sensitive)
- λ too high (e.g., 10.0): Over-cautious, blends all samples equally (slow)
- **Sweet spot**: λ=1.0-2.0 for smooth reactive navigation

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

## 7. Testing Plan

### 7.1 Unit Tests (Phase 1, implemented)

1. **LocalExecutive API** ✅:
   - `set_goal()` / `cancel()` / `is_active()` state machine
   - `tick()` returns None when inactive
   - `tick()` returns valid velocity dict when active (geometric stub)

2. **Coordinate transforms** ✅:
   - World→ego conversion matches `pose.py` conventions
   - Rolling subgoal clamping (0.8-1.0m ahead)

3. **MPPI stubs** ✅:
   - `_mppi_sample_trajectories()`: samples shape (n_samples, n_steps, 2)
   - `_mppi_rollout_batch()`: costs + trajectories correct shapes
   - `_mppi_compute_weights()`: weights sum to 1
   - `_mppi_weighted_control()`: returns (v, ω) tuple

4. **Geometric controller stub** ✅:
   - Pure pursuit drives toward goal
   - Simple obstacle stop (replace with MPPI later)

### 7.2 Integration Tests (Phase 1 next)

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

### 7.3 Hardware Tests (Jetson Orin NX)

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

### 7.4 Acceptance Criteria

- [x] `--auto-local` flag compiles and runs without errors
- [x] Existing teleop/VFH navigator unaffected when flag off
- [x] Unit tests pass (API + MPPI stubs)
- [ ] `local_goto` tool call sets goal, geometric stub drives toward it
- [ ] `tick()` execution time < 5 ms (geometric stub ~1ms, MPPI target 3-5ms)
- [ ] PyTorch/CUDA MPPI implementation (replaces stubs)
- [ ] No crashes in 10 minutes of continuous wander mode
- [ ] Gamepad override works in all modes
- [ ] Safety guard integration (no collisions in stress test)

---

## 8. Next Steps

### Phase 1: MPPI Scaffold (Complete ✅)
- [x] Update `docs/kevin-autonomy-midlayer.md` with MPPI architecture
- [x] Add `src/local_executive.py` with MPPI API + stubs
- [x] Add `--auto-local` flag to `main.py`
- [x] Integrate into main loop (non-blocking mailbox)
- [x] Geometric controller stub (pure pursuit for testing)
- [x] Unit tests pass
- [x] Tool calls: `local_goto`, `local_wander`, `local_cancel`

### Phase 2: PyTorch MPPI Implementation (1-2 weeks)
- [ ] Implement `_mppi_sample_trajectories()` with PyTorch
- [ ] Implement `_mppi_rollout_batch()` with batched kinematics
- [ ] Implement costmap sampling with `torch.nn.functional.grid_sample`
- [ ] Tune MPPI parameters on real robot (n_samples, λ, noise_sigma)
- [ ] Profile performance (target 3-5 ms/tick)
- [ ] Validate: straight-line goal, obstacle avoidance, smooth control

### Phase 3: CUDA MPPI Upgrade (Optional, 1-2 weeks)
- [ ] Port to cuMPPI library or custom CUDA kernels
- [ ] Costmap texture sampling (hardware interpolation)
- [ ] Profile: target 2-3 ms/tick (512-2048 samples)
- [ ] A/B test: PyTorch vs. CUDA performance

### Phase 4: Wander Mode (1 week)
- [ ] Implement frontier detection (ego-space boundary)
- [ ] Integrate with MPPI (virtual goal)
- [ ] Test: continuous exploration without getting stuck

### Phase 5: Learned MPC (Phase 2 roadmap, 2-4 weeks)
- [ ] Data collection: log (obs, goal, velocity) from MPPI runs
- [ ] Train iPlanner-IL cost network or SAC-polar policy
- [ ] Integrate inference (PyTorch on Jetson)
- [ ] A/B test: MPPI vs. learned, measure success rate & smoothness

### Phase 6: Visual Navigation (Phase 3 roadmap, 4-6 weeks)
- [ ] Integrate ViNT or NoMaD (RGB → waypoint)
- [ ] Doctor safety filter (diffusion model)
- [ ] MPPI fallback when learned model fails
- [ ] Test: generalization to novel environments

---

## 9. References

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

### Academic Papers (Literature Review)

**MPPI**:
- Williams, G. et al. (2017). "Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving." IEEE T-RO.
- Williams, G. et al. (2018). "Aggressive Driving with Model Predictive Path Integral Control." ICRA.
- Wagener, N. et al. (2019). "Information Theoretic MPC using Neural Network Dynamics." NeurIPS Workshop.

**Learned Local Planning**:
- Bronstein, E. et al. (2022). "Learning to Plan via Imitation and Practice." CoRL.
- Haarnoja, T. et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL." ICML.

**Visual Navigation**:
- Shah, D. et al. (2022). "Visual Navigation Transformer." arXiv:2206.03398.
- Sridhar, A. et al. (2023). "Navigating Autonomous Driving with Map-less Dense Vision (NoMaD)." ICRA.
- Wu, P. et al. (2023). "Doctor: Diffusion Model for Safe Robot Navigation." CoRL.

**General Mobile Robot Navigation**:
- Fox, D. et al. (1997). "The Dynamic Window Approach to Collision Avoidance." IEEE RA-M.
- Faust, A. et al. (2018). "PRM-RL: Long-range Robotic Navigation Tasks by Combining RL and Sampling-based Planning." ICRA.

## 10. Revision History

- **2026-09-06 (v1)**: Initial investigation and DWA scaffold
- **2026-09-06 (v2)**: Literature review complete, pivot to MPPI, Phase 1-3 roadmap

---

## Appendix A: Example Usage (Phase 1)

### Running with LocalExecutive

```bash
# Default: existing VFH navigator
python src/main.py --rs1 815412070676 --rs2 944622074292

# Enable LocalExecutive mid-layer
python src/main.py --rs1 815412070676 --rs2 944622074292 --auto-local

# In Python (agent tool call):
# Set goal 2 meters forward (map frame)
tools.call_tool("local_goto", {"x": pose.x + 2.0, "y": pose.y})
# Geometric stub will drive toward goal (replace with MPPI in Phase 2)

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
