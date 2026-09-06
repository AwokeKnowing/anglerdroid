# LocalExecutive Deployment Guide (mppi-costmap-v0)

**Branch**: `cursor/kevin-autonomy-midlayer-6803`  
**PR**: https://github.com/AwokeKnowing/anglerdroid/pull/3  
**Status**: Phase 1 scaffold complete, ready for MPPI implementation  
**Experiment**: `mppi-costmap-v0`

## What's Included

### 1. Architecture Documentation
- **`docs/kevin-autonomy-midlayer.md`**: Complete MPPI-based design (1400+ lines)
  - Maps the existing 30Hz loop (capture → occupancy → pose → commands)
  - Documents MPPI algorithm (sampling, rollout, cost, weighting)
  - Phase roadmap: Phase1=MPPI, Phase2=iPlanner-IL/SAC-polar, Phase3=ViNT/NoMaD+Doctor
  - Performance targets: CUDA 2-3ms, PyTorch 3-4ms on Jetson Orin NX
  - Literature review: Williams (MPPI), Bronstein (iPlanner), Shah (ViNT), Wu (Doctor)

### 2. Code (MPPI Scaffold)
- **`src/local_executive.py`**: LocalExecutive class (600+ lines)
  - Complete API: `set_goal(x, y)`, `set_wander_mode()`, `cancel()`, `tick()`
  - Non-blocking mailbox pattern (thread-safe, map-frame goals)
  - Geometric controller stub (pure pursuit for Phase 1 testing)
  - MPPI stubs with clear CUDA/Torch integration hooks:
    - `_mppi_sample_trajectories()` → GPU sampling placeholder
    - `_mppi_rollout_batch()` → Batched integration + costmap
    - `_mppi_running_cost()`, `_mppi_terminal_cost()` → Cost functions
    - `_mppi_compute_weights()` → Softmax weighting
    - `_mppi_weighted_control()` → Output blending
  - Unit tests pass ✅

- **`src/main.py`**: Integration with 30Hz loop
  - `--auto-local` flag (default OFF)
  - Non-blocking `tick()` in main loop (lines 157-177)
  - Respects gamepad/twist_for priority
  - Tool calls: `local_goto(x, y)`, `local_wander(enable)`, `local_cancel()`

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

## Behavior (Phase 1 Scaffold)

**With `--auto-local` OFF** (default):
- Existing VFH navigator works normally
- No changes to teleop or agent control

**With `--auto-local` ON**:
- LocalExecutive initializes (prints "mppi-costmap-v0, Phase 1 STUB")
- Geometric controller stub active (pure pursuit toward goal)
- Tool calls `local_goto(x, y)` → robot drives toward map-frame goal
- Safe to test: stub uses simple proportional control, stops if obstacle ahead

**MPPI Status**: Stubs in place with clear integration hooks (Phase 2)

## Next Steps (Phase 2: MPPI Implementation)

**Replace numpy stubs with PyTorch/CUDA MPPI** (1-2 weeks):

1. **PyTorch MPPI rollout**:
   - Implement batched control sampling: `torch.randn(n_samples, n_steps, 2)`
   - Batched trajectory integration (differential drive kinematics)
   - Costmap sampling with `torch.nn.functional.grid_sample`
   - Cost computation (obstacle + goal + smoothness)
   - Softmax weighting + weighted mean output

2. **Tune on robot**:
   - `n_samples`: 256-512 (balance smoothness vs. speed)
   - `temperature` (λ): 1.0-2.0 (balance exploration vs. exploitation)
   - `noise_sigma_v`, `noise_sigma_w`: 0.1, 0.3 (process noise)

3. **Profile performance**:
   - Target: 3-5ms/tick on Jetson Orin NX
   - If too slow: reduce samples or horizon
   - If fast enough: increase samples for smoother control

4. **Validate**:
   - Straight-line goals (smooth approach and stop)
   - Obstacle avoidance (detour around furniture)
   - No oscillation or jitter
   - Respects safety scales

**Optional: CUDA upgrade** (Phase 3, if PyTorch too slow):
- Port to cuMPPI library or custom CUDA kernels
- Texture sampling for costmap (hardware interpolation)
- Target: 2-3ms/tick (512-2048 samples)

## Files Changed (Phase 1)
```
docs/kevin-autonomy-midlayer.md  (new, 1400+ lines) - MPPI architecture + lit review
src/local_executive.py           (new, 600+ lines) - MPPI scaffold + stubs
src/main.py                      (modified, +30 lines) - Integration
src/tools.py                     (modified, +25 lines) - Tool wrappers
DEPLOYMENT.md                    (this file) - Deployment guide
```

## Testing Checklist (Phase 1)

- [x] Unit tests pass (`python src/local_executive.py`)
- [x] Syntax valid (`python -m py_compile src/main.py`)
- [x] Git committed and pushed
- [x] PR created (https://github.com/AwokeKnowing/anglerdroid/pull/3)
- [ ] Deploys to robot without errors
- [ ] `--auto-local` flag accepted
- [ ] Existing VFH navigator still works (flag OFF)
- [ ] Geometric stub drives toward goal (flag ON)
- [ ] Gamepad override works

## Testing Checklist (Phase 2 - MPPI)

- [ ] PyTorch MPPI implementation complete
- [ ] Costmap sampling with grid_sample working
- [ ] `tick()` execution time 3-5ms on Jetson Orin NX
- [ ] Smooth control (no jitter)
- [ ] Obstacle avoidance (detours around furniture)
- [ ] No crashes in 10 minutes continuous navigation
- [ ] Safety guard integration validated

## Support

**Documentation**: Read `docs/kevin-autonomy-midlayer.md` for full details:
- Section 1: 30Hz loop architecture
- Section 3.2: MPPI algorithm (sampling, rollout, cost, weighting)
- Section 4: Literature review (MPPI vs. DWA vs. learned methods)
- Section 6.2: Costmap sampling strategies (CUDA texture vs. PyTorch grid_sample)
- Section 6.3: Performance analysis (timing breakdown)
- Section 8: Phase 2-3 roadmap (iPlanner, ViNT, NoMaD, Doctor)

**MPPI Integration Hooks** (see `src/local_executive.py`):
- `_mppi_sample_trajectories()` → Replace with `torch.randn()` or cuMPPI
- `_mppi_rollout_batch()` → Batched integration + `F.grid_sample()` costmap
- `_mppi_running_cost()` → Obstacle penalty from costmap lookup
- `_mppi_terminal_cost()` → Goal distance at trajectory end
- `_mppi_compute_weights()` → Softmax (already correct)
- `_mppi_weighted_control()` → Weighted mean (already correct)

**Debugging**:
- LocalExecutive prints params on init (n_samples, λ, noise_sigma)
- Geometric stub prints commands every tick (when active)
- Main loop prints goal every 3 seconds: `local=(x, y)`
- `get_debug_state()` returns trajectories, costs, weights (for viz)

**References**:
- Williams et al. (2017) "Information-Theoretic MPC" - Core MPPI paper
- Williams et al. (2018) "Aggressive Driving with MPPI" - AUTORALLY hardware validation
- PyTorch grid_sample docs: https://pytorch.org/docs/stable/nn.functional.html#grid-sample

---

**PR ready for review: https://github.com/AwokeKnowing/anglerdroid/pull/3**  
**Merge when ready to test geometric stub, then implement PyTorch MPPI (Phase 2)**
