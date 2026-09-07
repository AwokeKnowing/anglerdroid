# Loop Hardening: Rock-Solid 30 Hz Main Loop

## Goal

Ensure the main control loop maintains a **rock-solid 30 Hz** (33.3ms per frame) by protecting the critical path from expensive droppable work like Rerun logging, frame copies, and async subsystem overhead.

## Problem

Heavy vision processing (3D reconstruction, face recognition, InsightFace), Rerun logging with frame copies, and future neural RL could steal cycles from the critical safety and motion control path, causing frame drops and sluggish response.

## Solution: Per-Stage Timing + Budget Shedding

### Architecture

Three components work together:

1. **`loop_timing.py`**: Lightweight timing instrumentation and budget tracking
2. **Instrumented `main.py`**: Critical and droppable stages explicitly marked
3. **Runtime shedding**: Droppable stages skipped when >85% of frame budget consumed

### Stage Priority Levels

#### Critical Path (Always Run)
- **`atlas`**: Get latest 2D obstacle map and depth gate
- **`safety`**: Propagate safety scales (fwd/bwd/angular)
- **`planner`**: Gamepad input + LocalExecutive/VFH + Navigator twist computation

These stages **never** get skipped — they're the robot's core perception → planning → action pipeline.

#### Droppable Stages (Skip if Over Budget)
- **`rerun`**: RGB/depth/obs logging to Rerun viewer (already throttled to 5 Hz by `--rerun-hz`, but can be shed entirely)
- **`tool_calls`**: Agent tool execution (twist_for, goto_xy, etc.)

When accumulated time exceeds **85% of 33.3ms (28.3ms)**, droppable stages are skipped that frame.

### Budget Threshold

**85%** of frame budget (28.3ms) chosen to:
- Leave ~5ms safety margin for loop overhead, sleep scheduling jitter
- Shed early enough that critical path still has breathing room
- Avoid cutting too aggressively (would skip work unnecessarily)

## How to Read the FPS Line

Every 90 frames (~3 seconds at 30 Hz), the loop prints timing stats:

```
  fps=29.8  process=18.2 ms  wait=15.1 ms  (budget 33.3 ms)  nav=45°  local=vfh
    stages:[atlas=90,safety=90,planner=90,rerun=88,tool_calls=12]  shed:[rerun=2]
```

### First Line
- **fps**: Actual frame rate (should be ~30.0)
- **process**: Average loop processing time per frame
- **wait**: Average sleep time per frame (if negative or 0, you're overrunning)
- **budget**: Target frame budget (33.3ms)
- **nav**: Active navigation heading (degrees)
- **local**: LocalExecutive mode (vfh, mppi, wander)

### Second Line (Stage Breakdown)
- **stages**: Count of how many times each stage ran in the last 90 frames
  - `atlas=90, safety=90, planner=90` → critical path ran every frame ✅
  - `rerun=88` → Rerun was shed 2 frames (or already throttled by `--rerun-hz`)
  - `tool_calls=12` → Tool calls only present in 12 of 90 frames
- **shed**: How many times each droppable stage was skipped due to budget overrun
  - `shed:[rerun=2]` → Rerun was shed 2 times in the last 90 frames
  - Empty `shed:[]` means nothing was shed (good!)

### Healthy Output Example
```
  fps=30.1  process=22.5 ms  wait=10.8 ms  (budget 33.3 ms)
    stages:[atlas=90,safety=90,planner=90,rerun=18,tool_calls=5]
```
- FPS at target 30 Hz ✅
- Process time well under budget (22.5ms < 33.3ms) ✅
- All critical stages ran every frame ✅
- No sheds needed ✅

### Overrun Example (Shedding Active)
```
  fps=29.6  process=31.8 ms  wait=1.5 ms  (budget 33.3 ms)  
    stages:[atlas=90,safety=90,planner=90,rerun=72,tool_calls=8]  shed:[rerun=18]
```
- FPS slightly below target (29.6 < 30) ⚠️
- Process time near budget limit ⚠️
- Rerun was shed 18 times to keep critical path alive ✅
- **Good**: Shedding is working — critical path protected

### Bad Example (Sustained Overrun)
```
  fps=24.3  process=38.2 ms  wait=0.0 ms  (budget 33.3 ms)
    stages:[atlas=90,safety=90,planner=90,rerun=0,tool_calls=0]  shed:[rerun=90,tool_calls=15]
```
- FPS well below target (24.3 < 30) ❌
- Process time exceeds budget even with everything shed ❌
- **Problem**: Critical path itself is too slow — need to optimize atlas/safety/planner

## What Gets Shed (Priority Order)

1. **Rerun logging** — already throttled by `--rerun-hz`, but can be shed entirely if budget tight
2. **Tool calls** — agent commands executed only when budget allows

Shedding is frame-by-frame: if one frame is heavy (e.g., atlas took 28ms), droppable stages skip that frame, but next frame they get a fresh chance.

## Implementation Details

### loop_timing.py

```python
from loop_timing import FrameBudget

budget = FrameBudget(budget_ms=33.3, shed_threshold=0.85)

# At start of each frame
budget.reset_frame()

# Critical stage (always runs)
with budget.stage("atlas"):
    atlas, ts = tools.get_atlas()

# Droppable stage (skipped if over budget)
if budget.should_run("rerun"):
    with budget.stage("rerun"):
        # expensive frame copies and logging
        ...
```

### Stage Timer Context Manager

The `with budget.stage("name"):` context manager:
1. Records start time on entry
2. Records duration and updates accumulated time on exit
3. Zero overhead when not in context

### Budget Check

`budget.should_run("stage_name")`:
- Returns `True` if stage is critical OR budget not exceeded
- Returns `False` if stage is droppable AND budget exceeded
- Increments shed counter when skipping

## Testing

### Unit Tests

```bash
cd src
python3 test_loop_timing.py
```

Tests cover:
- Basic timing accuracy
- Critical stages always run
- Droppable stages shed when over budget
- Accumulated time tracking
- Frame reset behavior
- Lifetime statistics

### Integration Test (Smoke Test)

```bash
cd src
python3 main.py --no-wheelbase --no-rerun
```

Should start and run without errors. Check console for:
- "AnglerDroid v2 main loop (30 fps)" startup message
- FPS reports every 90 frames
- Stage breakdown in reports
- No Python exceptions

Press `Ctrl+C` to stop cleanly.

## Design Trade-offs

### Why 85% threshold?
- **Too low** (e.g., 50%): Shed work unnecessarily, waste available cycles
- **Too high** (e.g., 95%): Not enough margin, still get frame drops
- **85%** balances protection and utilization

### Why not make tool_calls critical?
- Most tool calls are from conversational agent (not time-critical)
- Gamepad input bypasses tool calls entirely (direct control path)
- Safety-critical "stop" commands are handled via gamepad or direct twist(0,0)
- Can be promoted to critical in `loop_timing.py` if needed

### Why not shed parts of critical path?
- Safety and motion control are non-negotiable
- Shedding atlas or planner would make robot blind or unresponsive
- Better to shed entire droppable stages than degrade critical path

### What about people_live and house_bot?
- Run in separate daemon threads, don't block main loop directly
- Their `get_frames()` calls (expensive) happen in main loop's rerun stage
- Rerun stage is droppable, so heavy frame copies can be shed
- Speech and face detection continue in background threads regardless

## Future Work

### Additional Droppable Stages
- `slam_keyframe`: SLAM keyframe insertion (currently in vision thread)
- `3d_recon`: 3D reconstruction updates
- `neural_rl`: Neural network inference for RL policies

### Adaptive Threshold
- Monitor sustained overruns and adjust threshold dynamically
- E.g., if shedding every frame for 10 seconds, raise threshold to 90%

### Stage-Level Profiling
- Add per-stage statistics (min/max/p95 timing)
- Detect which stage is the bottleneck
- Log slow frames for post-analysis

### Preemptive Shedding
- If atlas took 25ms, shed rerun before even checking budget
- "Look-ahead" prediction: if planner is VFH (fast) vs MPPI (slow)

## References

- **Main loop**: `src/main.py` — 30 Hz control loop with instrumentation
- **Timing module**: `src/loop_timing.py` — Budget tracking and shedding logic
- **Tests**: `src/test_loop_timing.py` — Unit tests for timing system
- **Rerun throttle**: `--rerun-hz` CLI flag (default 5 Hz) — already reduces Rerun overhead
- **Vision threads**: `src/vision.py` — RealSense + depth processing runs async

## Summary

The hardened loop ensures **30 Hz is sacred** for the critical safety and motion control path. Expensive logging and async work get shed when the budget is tight, but the robot always knows where it is, where obstacles are, and how to move safely.

**Key insight**: It's better to skip a Rerun frame or an agent tool call than to let the safety scales or planner lag behind real-time. The 2D obstacle map remains ground truth at 30 Hz.
