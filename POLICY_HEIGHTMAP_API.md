# Policy Heightmap Tensor API

**Status**: ✅ Implemented  
**Author**: James AwokeKnowing  
**Date**: 2026-09-07  
**Target**: Tesla FSD-style labeled ego heightmap for neural/MPPI mid-layer at 30 Hz

---

## Overview

This API exports Kevin's GPU-resident heightmap as a clean labeled tensor observation suitable for neural or MPPI policy consumption. The goal is to feed a policy with a rich, structured 2.5D representation of the robot's surroundings at 30 Hz, similar to Tesla's FSD occupancy network.

## Design Principles

1. **GPU-first**: Heightmap is processed on GPU and kept resident where practical
2. **Labeled layers**: Clear semantic separation (ego reactive, global predictive, known/unknown)
3. **Zero-copy aspiration**: Current CPU readback (~0.5ms), future CuPy/PyTorch GPU tensors
4. **Safety gates**: `topdown_ok` and `slam_locked` flags protect against bad observations
5. **Minimal API surface**: Single method returns all needed layers + metadata

## API Summary

### `Vision.get_policy_observation()` → dict

Returns labeled heightmap observation with 5 layers + metadata:

```python
obs = vision.get_policy_observation()

# Reactive ego-space layers (current frame, 30 Hz)
obs['ego_obs']        # (240, 320) uint8: obstacle heights 0-100 cm
obs['ego_known']      # (240, 320) uint8: observation mask (0=unknown, 255=known)

# Persistent fused layers (ego + SLAM history)
obs['ego_persistent'] # (240, 320) uint8: binary obstacle mask (includes history)
obs['ego_height']     # (240, 320) uint8: obstacle heights with SLAM fusion

# Predictive global-map layer (SLAM history projected to ego frame)
obs['global_projected_conf']  # (240, 320) uint8: confidence (0-89=obs, 90-190=unk, 191-255=free)

# Safety & context metadata
obs['metadata'] = {
    'ego_px_size': 0.01,           # 1 cm/px
    'robot_cx': 81, 'robot_cy': 119,  # robot center in ego frame
    'topdown_ok': bool,            # RS1 depth working (safety critical)
    'slam_locked': bool,           # SLAM pose reliable (trust global map)
    'pose_x', 'pose_y', 'pose_theta',  # world-frame pose
    'timestamp_capture', 'timestamp_mono',
}
```

### Layer Semantics

| Layer | Encoding | Use Case |
|-------|----------|----------|
| `ego_obs` | 0=floor, 1-100=height (cm) | Immediate collision avoidance |
| `ego_known` | 0=unobserved, 255=known | Trust mask for ego_obs |
| `ego_persistent` | 0=free, 255=obstacle | Short-horizon planning (~1-2m) |
| `ego_height` | 0-100 cm | Persistent obstacle heights |
| `global_projected_conf` | 0-89=obs, 90-190=unk, 191-255=free | Long-horizon planning (trust when `slam_locked=True`) |

### Safety Flags

- **`topdown_ok=False`** → **Immobilize**: No depth data, all motion unsafe
- **`slam_locked=False`** → **Local nav only**: SLAM pose unreliable, only use ego layers

## Integration Pattern

```python
# At 30 Hz in your policy loop
obs = vision.get_policy_observation()
if obs is None or not obs['metadata']['topdown_ok']:
    # SAFETY: No valid depth, stop all motion
    return zero_twist()

# Reactive collision avoidance (ego layers)
ego_cost = compute_collision_cost(obs['ego_obs'], obs['ego_known'])

# Predictive planning (global layer, only if SLAM locked)
if obs['metadata']['slam_locked']:
    global_cost = compute_path_cost(obs['global_projected_conf'])
else:
    # Graceful degradation: local nav only
    global_cost = 0.0

# Fuse reactive + predictive
total_cost = ego_cost + global_cost
vel_cmd = mppi.optimize(total_cost)
```

## Performance

| Operation | Current | Future (GPU zero-copy) |
|-----------|---------|------------------------|
| Ego layers | ~0.2 ms | 0 ms (in-place view) |
| Global projection | ~0.3 ms | ~0.1 ms (GPU-only) |
| **Total** | **~0.5 ms** | **~0.1 ms** |

Current implementation uses CPU numpy arrays with minimal readback overhead. Future work: CuPy/PyTorch DLPack interface for true zero-copy GPU tensors when the policy runs on GPU.

## Code Changes

### Added to `gpu_render.py`

- `get_policy_heightmap()`: Metadata + global map accessor
- `get_global_heightmap_gpu()`: Read conf + height layers from GPU

### Added to `vision.py`

- `get_policy_observation()`: Main policy-facing API (fuses ego + global layers)

### Tests

- `test_policy_heightmap_api.py`: Unit tests for structure, semantics, safety flags

## Neural Policy Design (North Star)

### Input Stack

```
[ego_obs, ego_known, ego_persistent, global_projected_conf]
→ (4, 240, 320) uint8 tensor
```

### Architecture Ideas

1. **Dual-stream CNN**:
   - Ego stream (reactive): shallow, fast inference
   - Global stream (predictive): deeper, contextual reasoning
   - Fuse at bottleneck before velocity decoder

2. **Transformer-based**:
   - Patch embedding over heightmap tiles
   - Cross-attention between ego and global features
   - Output velocity tokens or MPPI cost weights

3. **Hybrid MPPI**:
   - Neural network predicts cost weights per layer
   - MPPI optimizer uses weights to score rollouts
   - Combines learning with explicit physics constraints

### Training Strategy

- **Sim-to-real**: Train in Isaac Sim with domain randomization (sensor noise, lighting, object diversity)
- **Self-supervised**: Use SLAM history as pseudo-labels for predictive layer
- **Behavior cloning**: Bootstrap from teleoperation or scripted skills
- **RL fine-tuning**: Reward = progress + safety (no collisions, respect keepouts)

## Migration Path

1. ✅ **Phase 1** (this PR): Expose labeled heightmap API at 30 Hz
2. **Phase 2**: CuPy/PyTorch zero-copy interface for GPU policies
3. **Phase 3**: Neural policy prototype (offline training in sim)
4. **Phase 4**: Deploy + online fine-tuning on Kevin

## Related Docs

- `AGENTS.md`: Orin optimization principles, 30 Hz budget
- `JAMES_ARCHITECTURE.md`: Vision thread split (30 Hz critical path, ~3 Hz extras)
- `gpu_render.py`: GPU-resident heightmap implementation
- `vision.py`: Capture loop integration

---

**Summary**: This API provides a clean, Tesla FSD-style labeled heightmap observation for neural/MPPI policies at 30 Hz. Current implementation uses CPU numpy (~0.5ms), with clear path to GPU zero-copy tensors when the policy stack is ready.

## Buffer lifetime
`get_policy_observation()` returns **preallocated buffers**. Contents are valid until the next call; copy if you need to retain a frame.
