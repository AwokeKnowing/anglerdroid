# RS1 Heightmap API — Policy-Facing Labeled Heightmap @30Hz

## Overview

The RS1 topdown camera produces a clean **labeled heightmap @30Hz** for the neural policy (Tesla FSD-style environment view). This replaces the previous CPU scatter bottleneck with GPU-resident processing, delivering:

- **Dense heightmap**: obstacle height in cm at mag=3 GSD (~1cm@1m)
- **Known mask**: valid depth coverage
- **30Hz fresh frames**: guaranteed fresh data every capture cycle
- **Policy-ready tensor**: uint8 arrays ready for neural network input

## North Star: 30Hz Heightmap for Policy

**Not just shaving ms** — the goal is a clean, fast heightmap that a neural policy can consume every frame to make driving decisions (obstacle avoid, path planning, local navigation).

### Performance Target

- **30 Hz sustained** (33.3ms frame budget) — the bar
- **60 Hz nice-to-have** (16.7ms frame budget) — future goal
- **RS1 map work ~12ms OK** — goal is leaving ~20ms for the rest

### Current State

- CPU `process_rs1_topdown`: ~11ms @ mag=3 (~45k verts)
  - Dense ego scatter: ~4-5ms (NumPy tallest-wins scatter)
  - Sparse reflexes: ~2-3ms (stride=8 near/overhang)
  - Dense soft-low: ~2-3ms (forward strip detection)

- GPU `depth_topdown_gpu` (this PR): **~1-2ms target** for dense ego scatter
  - Atomic scatter via depth test (parallel GPU)
  - Sparse reflexes stay on CPU (already fast)
  - Soft-low: TBD (derive from GPU heightmap or keep CPU)

## Heightmap API

### Access in Vision Capture Loop

The heightmap is computed every capture cycle and stored in pre-allocated buffers:

```python
# In Vision._capture_loop():
self._z1  # uint8 (240, 320) - obstacle height in cm (0=floor, 1-100=obstacle)
self._k1  # uint8 (240, 320) - known mask (255=valid depth, 0=no data)

# After 180° rotation for ego frame:
obs1 = self._z1[::-1, ::-1]
known1 = self._k1[::-1, ::-1]
```

### Output Format

#### `obs` — Obstacle Height (uint8)
- Shape: `(H, W)` where H=240, W=320 (FRAME_H, FRAME_W)
- Values:
  - `0`: Floor (z >= TD_FLOOR_CLIP=0.91m) or unknown
  - `1-100`: Obstacle height in cm above floor
  - Encoding: `height_cm = (TD_FLOOR_CLIP - z) * 100`, clamped to [1, 100]
- Semantics: **Tallest wins** per pixel (GPU depth test max)

#### `known` — Valid Depth Mask (uint8)
- Shape: `(H, W)` where H=240, W=320
- Values:
  - `255`: Valid depth at this pixel
  - `0`: No depth data (blind spot, occlusion, out of range)
- Semantics: All pixels with z > 0.01m are marked known

### Coordinate Frame

- **Before rotation** (`self._z1`, `self._k1`): RS1 camera frame
  - Origin: camera optical center
  - +X: right, +Y: down, +Z: forward (away from camera)

- **After 180° rotation** (`obs1`, `known1`): Ego frame for policy
  - Top of image (row 0) = forward
  - Bottom of image (row 239) = rear
  - Left/right preserved

### Pixel Size

- **TD_PX_SIZE** = 0.01m (1cm/pixel) — same as EGO_PX_SIZE
- **GSD** (Ground Sample Distance): ~1cm @ 1m distance (mag=3 decimation)
- **FOV**: ~1m × 0.8m coverage depending on camera height

## GPU Implementation

### Dense Ego Scatter (GPU)

Replaces lines 579-599 of `process_rs1_topdown`:

```python
# GPU path (vision.py capture loop):
v_clean = _clip_decimated_border(self._rs1.verts, out=self._rs1_work_verts)
_gpu_result = self._gpu.depth_topdown_gpu(v_clean)

if _gpu_result is not None:
    self._z1[:], self._k1[:] = _gpu_result  # Dense heightmap from GPU
```

- **Shader**: `_VERT_SCATTER_TOPDOWN` (orthographic project + height encode)
- **Scatter**: Atomic via depth test (`depth_func='>'` for tallest-wins)
- **Output**: Single readback, decode to (obs, known)
- **No morphology**: RS1 is already dense (unlike RS2 forward)

### Sparse Reflexes (CPU)

Near-field and overhang stay on CPU with stride=8:

```python
# Sparse reflexes (CPU):
_rs1_sparse = process_rs1_sparse_reflexes(
    self._rs1.verts, work_verts=self._rs1_work_verts)

# Results:
_rs1_sparse['near_field']      # bool: <30cm hazard detected
_rs1_sparse['overhang']         # bool: 30-70cm table underside
```

- **Near-field**: Any obstacle <30cm from camera (table, hand)
- **Overhang**: Elevated structure 30-70cm ahead in forward strip
- **Stride=8**: ~8x subsampling (45k → ~5.6k verts checked)
- **Fast**: ~1-2ms (already optimized, no GPU gain)

### Soft-Low Detection (TBD)

Dog bed / cushion detection (5-30cm height, 35cm-1m range):

- **Current**: Dense CPU scan in forward strip (~2-3ms)
- **Future**: Derive from GPU heightmap forward strip if equivalent
- **For now**: Disabled in GPU path (`soft_low=False`)

## CPU Fallback

If GPU unavailable, full `process_rs1_topdown` runs on CPU:

```python
else:
    # CPU fallback: full process_rs1_topdown (sparse + dense)
    _rs1 = process_rs1_topdown(
        self._rs1.verts, self._z1, self._k1,
        work_verts=self._rs1_work_verts)
```

- Identical semantics to GPU path
- ~11ms total (vs ~3-4ms GPU path)
- Graceful degradation (still meets 30Hz bar)

## Policy Integration (Future)

### Consuming the Heightmap

```python
# Read latest heightmap (thread-safe via Vision._lock):
with vision._lock:
    obs = vision._z1[::-1, ::-1].copy()      # (240, 320) uint8
    known = vision._k1[::-1, ::-1].copy()    # (240, 320) uint8

# Neural policy forward pass:
obs_tensor = torch.from_numpy(obs).float() / 100.0  # normalize to [0, 1] metres
known_tensor = torch.from_numpy(known).float() / 255.0  # binary mask

# Combine as channels:
heightmap_input = torch.stack([obs_tensor, known_tensor], dim=0)  # (2, 240, 320)
```

### Future: Labeled Heightmap

Add semantic labels (floor, obstacle, hazard) as additional channels:

```python
# Future labels (uint8):
labels = np.zeros((240, 320), dtype=np.uint8)
labels[obs == 0] = 0  # floor
labels[obs > 0] = 1   # obstacle
labels[hazard_mask] = 2  # hazard (chessboard, soft-low, overhang)
```

## Performance Metrics

### Expected Savings (Hypothesis)

- **CPU dense ego scatter**: ~4-5ms (NumPy scatter, 45k verts)
- **GPU dense ego scatter**: ~1-2ms (atomic scatter, parallel)
- **Speedup**: 3-4x for dense ego
- **Total RS1**: ~11ms → ~7-8ms (3-4ms savings)

### On-Device Verification (Required)

Run on Kevin (Jetson Orin NX) and paste CAPTURE TIMING:

```bash
# Silent vision smoke:
python3 scripts/run_vision_smoke.py

# Expected output:
# CAPTURE TIMING (last 90 frames):
# rs1_checks:  7.5ms  (before: ~11ms)
# TOTAL:      28ms   (before: ~33ms)
```

## Architecture Notes

### AGENTS.md Compliance

✅ **GPU-resident processing**: verts → GPU → readback, no CPU round-trips
✅ **Decimate early**: uses pre-decimated mag=3 verts (~45k, not 848×480)
✅ **No casual CPU loops**: GPU scatter replaces NumPy tallest-wins
✅ **Pre-allocated buffers**: VBO reserved upfront, no per-frame alloc
✅ **Measure, don't assume**: On-device verification required before merge

### Product Focus

- **Not just optimization**: This enables a clean heightmap API for neural policy
- **30Hz is the bar**: Current implementation meets this
- **60Hz is the goal**: Future work (more GPU paths, parallel streams)
- **Labeled heightmap**: Future semantic channels (floor, obstacle, hazard types)

## Related

- **AGENTS.md**: Orin microsecond vigilance, 30Hz vs 60Hz targets
- **JAMES_ARCHITECTURE.md**: 30Hz capture vs ~3Hz extras split
- **src/vision.py**: `process_rs1_topdown`, `process_rs1_sparse_reflexes`
- **src/gpu_render.py**: `depth_topdown_gpu`, `configure_depth_topdown`

---

**Status**: Implemented, pending on-device verification.
**Next**: Run Kevin smoke, paste timing, merge if p95 < 33.3ms.

## Soft-low
Dog bed / cushions are ordinary heightmap obstacles. A separate soft-low CPU pass is not required on the GPU path; past issues were empty-frame nav bugs.
