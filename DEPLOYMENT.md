# LocalExecutive Deployment Guide (Phase 1)

**Branch**: `cursor/kevin-autonomy-midlayer-6803`  
**Status**: Ready for robot deployment (Phase 1 scaffold complete)

## What's Included

### 1. Documentation
- **`docs/kevin-autonomy-midlayer.md`**: Complete architecture analysis
  - Maps the existing 30Hz loop (capture → occupancy → pose → commands)
  - Documents data shapes, units, frames, and timing constraints
  - Proposes DWA-based geometric planner (Phase 2) + neural upgrade path (Phase 4)
  - Identifies integration points and open risks

### 2. Code (Scaffold)
- **`src/local_executive.py`**: LocalExecutive class (API complete, stub implementation)
  - `set_goal(x, y)` - Set world-space goal (meters)
  - `set_wander_mode(enabled)` - Enable/disable exploration
  - `cancel()` - Stop navigation
  - `tick(obs_map, pose, dt)` - Compute velocity command (returns None in Phase 1)
  - Unit tests pass

- **`src/main.py`**: Integration with 30Hz loop
  - `--auto-local` flag (default OFF)
  - Calls `local_executive.tick()` when active (lines 157-177)
  - Respects gamepad/twist_for priority
  - New tool calls: `local_goto`, `local_wander`, `local_cancel`

- **`src/tools.py`**: Convenience wrappers for LocalExecutive

## Deployment to Robot

### Option A: Merge to main (Recommended for direct deploy)
```bash
# On robot or local dev machine with push access:
git fetch origin
git checkout main
git merge origin/cursor/kevin-autonomy-midlayer-6803
git push origin main
```

### Option B: Test on branch first
```bash
# On robot:
git fetch origin
git checkout cursor/kevin-autonomy-midlayer-6803
cd src
python main.py --rs1 815412070676 --rs2 944622074292 --auto-local
# (LocalExecutive will print "initialized (STUB)" but won't move robot yet)
```

## Behavior (Phase 1)

**With `--auto-local` OFF** (default):
- Existing VFH navigator works normally
- No changes to teleop or agent control

**With `--auto-local` ON**:
- LocalExecutive initializes (prints "STUB - Phase 1")
- `tick()` returns None → robot idle until goal set
- Tool calls `local_goto`, `local_wander` accepted but no motion yet
- Safe to test: no unexpected behavior (stub is inactive)

## Next Steps (Phase 2)

**Implement DWA in `local_executive.py`**:
1. Fill in `tick()` method:
   - Compute rolling subgoal (0.8-1.0m ahead toward goal)
   - Generate velocity samples (7 linear × 9 angular = 63)
   - Simulate trajectories (20 steps × 0.033s = 0.66s horizon)
   - Check collisions (bounding circle in ego-space)
   - Score: goal progress + obstacle clearance + smoothness
   - Return best (fwd_mps, ang_rads)

2. Test on robot:
   - Set goal 2m ahead: `local_goto(pose.x + 2, pose.y)`
   - Verify smooth approach and stop
   - Test obstacle avoidance (chair/table in path)

3. Tune parameters:
   - `v_max`, `w_max` (speed limits)
   - `subgoal_dist` (look-ahead distance)
   - Scoring weights (goal vs. obstacle vs. smoothness)

## Files Changed
```
docs/kevin-autonomy-midlayer.md  (new, 1300+ lines)
src/local_executive.py           (new, 450+ lines)
src/main.py                      (modified, +30 lines)
src/tools.py                     (modified, +25 lines)
```

## Testing Checklist

- [x] Unit tests pass (`python src/local_executive.py`)
- [x] Syntax valid (`python -m py_compile src/main.py`)
- [x] Git committed and pushed
- [ ] Deploys to robot without errors
- [ ] `--auto-local` flag accepted
- [ ] Existing navigator still works (flag OFF)
- [ ] LocalExecutive stub initializes (flag ON)

## Support

**Documentation**: Read `docs/kevin-autonomy-midlayer.md` for full details.

**Questions**:
- Units/frames: Section 1.2 (ego-space), Section 1.3 (world frame)
- Integration: Section 2 (where to hook, data access, timing)
- DWA algorithm: Section 3.2 (trajectory generation, scoring)
- Risks: Section 5 (coordinate transforms, collision checking, performance)

**Debugging**:
- LocalExecutive prints debug info every 90 frames (3 seconds)
- Check main loop prints: `local=(x, y)` shows current goal
- `get_debug_state()` returns trajectory data for visualization

---

**Ready for deployment. Merge to main when ready to test on robot.**
