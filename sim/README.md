# Lightweight 2D Simulator for Kevin (Anglerdroid)

**Status**: LIVE ROBOT DRIVING IS STOPPED. Use this simulator for offline policy development.

## Purpose

Practice local planning + look-before-leap / recover policy WITHOUT ROS, WITHOUT RealSense, WITHOUT ODrive.

Created after Kevin crashed into a couch when house_bot recover/commit overrode SafetyGuard hard stops.

## Architecture

Pure Python + NumPy. Optional matplotlib/imageio for visualization.

### Components

- **`world.py`**: Simple house floorplans with binary occupancy + height maps
- **`robot.py`**: Differential-drive kinematics with SafetyGuard-equivalent
- **`policy.py`**: Pluggable policy interface + HouseBotLite reference
- **`run.py`**: CLI for running scenarios headless

### Geometry

Matches real ego map from `src/robot_config.py`:
- Frame: 320×240 pixels
- Robot center: (81, 119) in ego frame
- Ego pixel size: 0.01 m/px (1 cm/pixel)
- Robot footprint: 30×42 pixels with safety pads
- Robot faces RIGHT (+X) in ego frame

### SafetyGuard Behavior

**CRITICAL**: Hard stop when scale=0:
- `fwd_scale=0` → forward motion BLOCKED
- `bwd_scale=0` → backward motion BLOCKED  
- `ang_scale=0` → angular motion BLOCKED

No policy can override these hard stops. This prevents the couch crash scenario where recover/commit forced motion through obstacles.

Scales computed from directional clearance scans (matching `src/safety.py`):
- Forward: scan from front edge rightward
- Backward: scan from rear edge leftward
- Lateral: scan above/below for spin clearance

## Usage

### Smoke Test (Empty Scenario)

```bash
python -m sim.run --steps 200 --scenario empty
```

Expected: ✅ PASSED: No collisions

### Couch Pinch Scenario

```bash
python -m sim.run --steps 400 --scenario couch_pinch --policy housebot
```

Tests recover behavior when robot faces couch with limited clearance.

### Full House

```bash
python -m sim.run --steps 1000 --scenario house --policy housebot --save house.gif
```

### Available Options

```bash
python -m sim.run --help
```

- `--steps N`: Number of simulation steps (default: 200)
- `--scenario {empty,couch_pinch,house}`: Scenario to run
- `--policy {random,housebot,stop}`: Policy to use
- `--hz FREQ`: Simulation frequency in Hz (default: 30)
- `--save PATH`: Save visualization as GIF or PNG strip
- `--render`: Enable rendering (requires --save or display)

## Writing Custom Policies

Implement the `Policy` interface:

```python
from sim.policy import Policy

class MyPolicy(Policy):
    def reset(self):
        """Called at episode start."""
        pass
    
    def act(self, obs, height, safety_scales, pose):
        """Compute action from observation.
        
        Args:
            obs: (H,W) uint8 ego obstacle map (0=clear, 255=occupied)
            height: (H,W) uint8 ego height map (cm above floor)
            safety_scales: dict with 'fwd', 'bwd', 'ang' (0-1 scales)
            pose: dict with 'x', 'y', 'theta' (world coords)
        
        Returns:
            (v, w) - linear velocity (m/s), angular velocity (rad/s)
        """
        v = 0.1  # forward at 10 cm/s
        w = 0.0  # no rotation
        return v, w
```

Then add to `policy.py:create_policy()` factory.

## HouseBotLite Reference Policy

Implements look-before-leap with recover fallback:

1. **Cruise**: Drive forward when `fwd_scale > 0.5`
2. **Stuck**: Stop when `fwd_scale ≤ 0.5` for >5 steps
3. **Backup**: Reverse if `bwd_scale > 0` for 20 steps
4. **Spin**: Rotate if `ang_scale > 0` until `fwd_scale > 0.8` or timeout
5. **Commit**: Return to cruise ONLY when forward path is clear

**Key safety property**: NEVER commands motion when corresponding scale=0.

## Scenarios

### Empty

Clear space, robot starts at origin. Zero collisions expected.

### Couch Pinch

Couch directly ahead, walls on sides. Robot must:
- Detect forward blockage (`fwd_scale → 0`)
- Back up (if `bwd_scale > 0`)
- Spin to find exit
- Commit forward only when clear

### House

Full house layout:
- Perimeter walls
- Couch (40 cm tall)
- Table (80 cm tall, mast-clear required)
- Chairs (45 cm tall)

## Performance

Runs on CPU in seconds:
- Empty 200 steps: ~0.1s
- Couch 400 steps: ~0.2s
- House 1000 steps: ~0.5s

No GPU required.

## Dependencies

Minimal:
```bash
pip install numpy
```

Optional (for visualization):
```bash
pip install imageio opencv-python
```

## Hypothesis: Couch Crash Root Cause

The real crash likely occurred when recover/commit logic overrode `fwd_scale≈0`:

```python
# UNSAFE (pre-crash behavior)
v = 0.15  # force forward
if safety.fwd_scale == 0:
    v = 0.15  # commit anyway! ❌ CRASHES

# SAFE (this simulator enforces)
v = 0.15  # request forward
if safety.fwd_scale == 0:
    v = 0.0  # hard stop ✅
```

This simulator treats any collision with `fwd_scale=0` as a test failure, proving the policy respects safety boundaries.

## Next Steps

1. ✅ Smoke test passes
2. Develop policy variations in sim
3. Tune recover thresholds
4. Validate zero collisions across scenarios
5. Port proven policy to real robot
6. Test on hardware with guardian supervision

Live robot stays stopped until policy is proven collision-free in simulation.
