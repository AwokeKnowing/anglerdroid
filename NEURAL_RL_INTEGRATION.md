# Neural RL Integration Summary

## Completed Work

Successfully wired `NeuralRLPolicy` into `local_executive` as an optional planner backend.

### Changes Made

1. **src/local_executive.py**
   - Added `neural_rl` to accepted planner options (vfh | mppi | neural_rl)
   - Implemented `_ensure_neural_rl()` lazy loader
   - Implemented `_tick_neural_rl()` with:
     - Vision policy feed integration (`get_policy_feed()`)
     - Keepout painting on obs_map
     - Fallback handling (neural fails → mppi/vfh)
   - Updated `set_goal_xy()`, `set_wander()`, `clear()` to handle neural_rl
   - Updated `status()` to include neural_rl debug state

2. **src/main.py**
   - Added `neural_rl` to `--local-planner` CLI choices
   - Updated help text

3. **test_local_executive_neural_rl.py** (NEW)
   - 11 integration tests covering:
     - Planner selection and validation
     - Goal/wander propagation to neural policy
     - Tick integration with/without policy feed
     - Fallback behavior verification
     - Default planner unchanged
     - Compatibility with existing mppi/vfh

### Test Results

✅ **28 tests total, 0 failures**
- test_neural_rl.py: 17/17 passed
- test_local_executive_neural_rl.py: 11/11 passed

### How to Enable

```bash
# Basic (fallback-only mode)
python3 src/main.py --auto-local --local-planner neural_rl

# With model
export KEVIN_NEURAL_MODEL_PATH=/path/to/model.onnx
export KEVIN_NEURAL_FALLBACK=mppi  # or 'vfh'
export KEVIN_NEURAL_POLICY_FEED=1  # use Vision policy feed
python3 src/main.py --auto-local --local-planner neural_rl --wander
```

### Key Features

- **Default-off**: Requires explicit `--local-planner neural_rl`
- **Fail-closed**: Neural fails → fallback to mppi/vfh (no crash)
- **Policy feed**: Uses `Vision.get_policy_feed()` when available (PR #53)
- **Hot path**: Lazy load, minimal overhead
- **Compatible**: Does not break vfh/mppi, no re-arm drive

### Fallback Behavior

Neural inference fails when:
1. Model not loaded (`KEVIN_NEURAL_MODEL_PATH` unset or file missing)
2. Inference budget exceeded (>5ms)
3. Inference error (ONNX runtime exception)

Fallback sequence:
1. Log fallback event to `status()["dbg"]`
2. Switch to configured fallback planner (default: mppi)
3. Sync goal state to fallback planner
4. Continue driving without crash

### Architecture

```
local_executive.tick()
  ↓
_tick_neural_rl()
  ↓
Vision.get_policy_feed() [if KEVIN_NEURAL_POLICY_FEED=1]
  ↓
NeuralRLPolicy.tick(obs_map, pose, dt, policy_feed=...)
  ↓
  ├─ Neural inference succeeds → return (fwd_mps, ang_rads)
  │
  └─ Neural fails → fallback:
       ├─ mppi (default): MppiCostmapPlanner.tick(...)
       └─ vfh: navigator.compute_twist(atlas)
```

### Constraints Satisfied

- ✅ No ROS
- ✅ Does not break mppi/vfh
- ✅ Does not re-arm drive
- ✅ Hot path cheap (lazy load, fail-closed)
- ✅ Uses existing tick/observation APIs
- ✅ Default-off (opt-in only)
- ✅ Offline tests pass

## Pull Request

**PR #55**: https://github.com/AwokeKnowing/anglerdroid/pull/55

Branch: `cursor/neural-rl-planner-a368`
Base: `main` (d9647c1+)
Status: Draft → Ready for review

## Next Steps

1. ✅ Code review
2. ✅ Merge to main
3. 🔜 Train ONNX model (Isaac Sim / host-sim)
4. 🔜 Deploy model to Kevin
5. 🔜 On-device validation (silent vision smoke, CAPTURE TIMING)
6. 🔜 GPU inference backend (TensorRT on Orin)

## Related Work

- **PR #53**: Policy feed consumption (`Vision.get_policy_feed()`)
- **neural_rl.py**: Stub already documented this integration
- **AGENTS.md**: Coding-agent skill repair loop (future)
