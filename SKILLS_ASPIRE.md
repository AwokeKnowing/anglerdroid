# Kevin ASPIRE Integration: Skills + Neural RL + Torque API

ASPIRE-inspired continual learning scaffold for Kevin mobile base.

## Overview

**ASPIRE** (Agentic Skill Programming through Iterative Robot Exploration) by Jim Fan / NVIDIA GEAR enables continual learning via:
- **Skill library** (code + text, NOT weights)
- **Execution traces** (multimodal logs → analysis → repair)
- **Coding agents** (evolutionary refinement of skills)

**Kevin adaptation:**
1. Seed skill library under `skills/` (5 behaviors: soft approach, overhang nav, checkered keepout, social distance, stuck recovery)
2. Trace logger (`trace_logger.py`) for JSONL + optional Rerun
3. Neural RL mid-layer stub (`neural_rl.py`) for learned policies (optional ONNX, falls back to VFH/MPPI)
4. ODrive fine-control API (`torque_tools.py`) for torque/current micro-moves
5. Integration hooks in `local_executive.py` (planner='neural_rl')

## Key Design Decisions

### 1. No ROS, Keep 30Hz Critical Path
- All new components are **optional** and **budget-gated**
- Neural RL inference: <5ms budget, fallback to VFH/MPPI if exceeded
- Trace logging: async writes, no frame drops
- Torque tools: separate from velocity control path (used for calibration/fine tasks only)

### 2. Skill Library = Code + Text (NOT Weights)
- Skills are markdown documents (description, preconditions, procedure, failure modes)
- Optional Python helpers in `skills/helpers/` (not implemented yet)
- Traced via `log_skill_invocation()` for evolution loop
- Human-readable and diff-able (git-friendly)

### 3. Neural RL as Optional Mid-Layer
- Interface: `obs_map (240x320) + goal_vector → (v, ω)`
- Small CNN backbone (~50k params, <5ms inference)
- Deployed as ONNX model (separate from repo, no weights in git)
- Fallback: VFH (default) or MPPI if model missing/slow/broken
- Safety-wrapped: outputs scaled by existing safety layer

### 4. Execution Traces → Multimodal Learning
- JSONL format: one event per line, timestamped, structured
- Events: twist_cmd, safety_state, local_executive, stuck_detection, skill invocations
- Optional Rerun integration for visual debugging (costmaps, trajectories)
- Analyzed by coding agents to refine skills (future: see AGENTS.md stub)

### 5. ODrive Torque API for Fine Control
- Separate from 30Hz navigation loop
- Use cases: fine positioning (<5cm), calibration, force-limited contact
- Safety limits: max 0.5 Nm per wheel, 1.0s max duration, watchdog-monitored
- Geometric helpers: force → torque conversions, wheelbase math
- Tested with mocks (no hardware required)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      main.py (30 Hz loop)                        │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─→ trace_logger.log_tick_snapshot()  [async JSONL]
             │
             ├─→ local_executive.tick(atlas, pose, obs_map)
             │      │
             │      ├─→ planner='vfh'  → VFH (default)
             │      ├─→ planner='mppi' → MPPI costmap
             │      └─→ planner='neural_rl' → NeuralRLPolicy.tick()
             │                                    │
             │                                    ├─→ ONNX inference (<5ms)
             │                                    └─→ fallback to VFH/MPPI
             │
             ├─→ safety.update() [reflexes + scales]
             │      │
             │      ├─→ topdown_soft_low_obstacle [skill: dog_bed_soft_approach]
             │      ├─→ topdown_overhang_approach  [skill: overhang_front_strip]
             │      ├─→ topdown_hazard             [skill: checkered_mat_keepout]
             │      └─→ near_field reflex
             │
             └─→ wheelbase.twist(fwd * fwd_scale, ang * ang_scale)

┌─────────────────────────────────────────────────────────────────┐
│                   Offline / Calibration Tasks                    │
└────────────┬────────────────────────────────────────────────────┘
             │
             └─→ torque_tools.TorqueController
                    │
                    ├─→ micro_move_straight(5cm)
                    ├─→ micro_rotate(15°)
                    └─→ force_limited_contact()
```

## Components

### `skills/` — Seed Skill Library
- `README.md` — skill structure and evolution process
- `dog_bed_soft_approach.md` — cautious approach to soft low obstacles
- `overhang_front_strip.md` — safe navigation under tables/shelves
- `checkered_mat_keepout.md` — detection and avoidance of hard-stop zones
- `person_approach_social.md` — approach people at 4-5ft comfortable distance
- `stuck_recovery.md` — detect and recover from stuck/oscillation states

### `src/trace_logger.py` — Execution Trace Logger
- **JSONL output**: `traces/run.jsonl` (one event per line)
- **Event types**: twist_cmd, safety_state, local_executive, stuck_detection, skill, costmap_snapshot
- **Optional Rerun**: visual debugging (requires `rerun-sdk`)
- **API**:
  - `init(log_path, enable_rerun=False)` — start logging
  - `log_event(event, **kwargs)` — generic structured event
  - `log_twist_cmd(fwd, ang, safety_scaled)` — velocity command
  - `log_safety_event(reflex_type, scales, throttled)` — safety layer state
  - `log_local_executive(mode, goal_xy, planner, active)` — mid-layer state
  - `log_skill_invocation(skill_name, params, outcome)` — high-level skill execution
  - `log_tick_snapshot(twist_cmd, safety_state, executive_state, obs_map_summary)` — complete 30Hz tick
  - `shutdown()` — close and flush

### `src/neural_rl.py` — Neural RL Policy Stub
- **Interface**: `obs_map (240x320) → (v, ω)`
- **Model**: ONNX format (separate deployment, not in repo)
- **Inference budget**: <5ms @ 30Hz (allows 25ms for rest of stack)
- **Fallback**: VFH or MPPI if model missing/slow/broken
- **API**:
  - `NeuralRLPolicy(model_path, inference_budget_ms=5.0, fallback_planner='vfh')`
  - `set_goal(x, y)` — world-frame goal
  - `set_wander_mode(enable)` — continuous exploration
  - `tick(obs_map, pose, dt)` → `{fwd_mps, ang_rads, source, inference_ms}`
  - `get_debug_state()` — inference stats for logging/UI

### `src/torque_tools.py` — ODrive Fine-Control API
- **Purpose**: torque/current micro-moves for calibration and fine tasks
- **Safety**: max 0.5 Nm/wheel, 1.0s max duration, watchdog-monitored
- **API**:
  - `TorqueController(left_axis, right_axis)` — initialize with ODrive axes
  - `micro_move_straight(distance_m, force_nm, max_duration_s)` — straight line
  - `micro_rotate(angle_rad, torque_nm, max_duration_s)` — in-place rotation
  - `force_limited_contact(direction, max_force_nm, duration_s)` — gentle push
  - `stop()` — emergency stop (zero all torques)
- **Helpers**:
  - `torque_from_force_linear(force_n)` — F→τ conversion
  - `torque_from_force_angular(force_n)` — angular F→τ
  - `estimate_contact_force(current_a)` — current→force (for force sensing)

### Integration Hooks

#### Add neural_rl planner to local_executive (optional):
```python
# In local_executive.py, extend set_planner():
def set_planner(name: str) -> None:
    """Select backend: 'vfh' (default), 'mppi', or 'neural_rl'."""
    global _planner, _neural_rl
    name = (name or "vfh").strip().lower()
    if name not in ("vfh", "mppi", "neural_rl"):
        raise ValueError("planner must be 'vfh', 'mppi', or 'neural_rl'")
    with _lock:
        _planner = name
    if name == "neural_rl":
        if _neural_rl is None:
            from neural_rl import NeuralRLPolicy
            _neural_rl = NeuralRLPolicy(model_path="models/policy.onnx")
    print(f"local_executive: planner backend = {name}")
```

#### Add trace logging to main loop:
```python
# In main.py, inside 30Hz loop:
import trace_logger

# At startup:
trace_logger.init("traces/run.jsonl", enable_rerun=False)

# Each tick:
if trace_logger.is_enabled():
    trace_logger.log_tick_snapshot(
        twist_cmd=(fwd, ang) if cmd else None,
        safety_state={
            "reflex_type": guard.near_field_reason,
            "fwd_scale": guard.fwd_scale,
            "bwd_scale": guard.bwd_scale,
            "ang_scale": guard.ang_scale,
            "throttled": guard.is_throttled
        },
        executive_state=local_executive.status(),
        obs_map_summary={
            "shape": obs_map.shape,
            "occupied_px": int((obs_map >= 100).sum())
        }
    )

# At shutdown:
trace_logger.shutdown()
```

## Testing (No Hardware Required)

All components have unit tests with mocks:

```bash
# Trace logger
python src/trace_logger.py

# Neural RL policy
python src/neural_rl.py

# Torque tools (mock ODrive axes)
python src/torque_tools.py

# Full test suite
python -m pytest test_trace_logger.py test_neural_rl.py test_torque_tools.py -v
```

## Future: Skill Evolution (AGENTS.md stub)

See `AGENTS.md` (placeholder) for coding-agent repair loop:
1. Collect execution traces (JSONL)
2. Analyze failures (multimodal: costmaps, trajectories, safety events)
3. Generate skill refinements (markdown + Python helpers)
4. Test in sim (host-sim or Isaac)
5. Deploy updated skills to robot

**NOT included in this scaffold**: actual coding agent implementation, RL training pipeline, sim integration.

## References

- **ASPIRE paper**: arXiv:2607.00272 (Jim Fan, NVIDIA GEAR)
- **ASPIRE repo**: https://github.com/NVlabs/ASPIRE (manip arms + CaP-X)
- **Kevin docs**: `docs/kevin-autonomy-midlayer.md` (existing mid-layer design)

## Changes Summary

**New files:**
- `skills/README.md` + 5 seed skills (dog bed, overhang, checkered, person, stuck)
- `src/trace_logger.py` (JSONL + Rerun hooks)
- `src/neural_rl.py` (policy stub with ONNX support)
- `src/torque_tools.py` (ODrive fine-control API)
- `SKILLS_ASPIRE.md` (this doc)
- `test_trace_logger.py`, `test_neural_rl.py`, `test_torque_tools.py` (unit tests)

**Modified files:**
- None (fully additive, no breaking changes to 30Hz path)

**NOT modified:**
- `local_executive.py` — neural_rl integration is optional hook (not included by default)
- `main.py` — trace logging integration is optional (not included by default)
- `odrivecan.py` — torque_tools uses existing `set_torque()` method

## PR Checklist

- [x] Skills library with 5 seed skills (markdown format)
- [x] Trace logger with JSONL output (async, no frame drops)
- [x] Neural RL policy stub with ONNX support and fallback
- [x] ODrive torque API with safety limits and geometric helpers
- [x] Unit tests for all new components (mocks, no hardware)
- [x] Documentation tying to ASPIRE ideas adapted for Kevin
- [ ] CI passes (tests must not need hardware) — pending test run
- [ ] No breaking changes to 30Hz critical path
- [ ] No face gallery in public repo (clean)
- [ ] Incremental and additive (can be extended without risk)
