# Top-Down Near-Field Safety Reflex

## Overview

The near-field safety reflex is a high-priority safety mechanism that prevents the robot from driving under tables, into overhead obstacles, or being blocked by hands near the top-down camera.

## Incident Background

Kevin crashed his mast while driving under a table. The floor obstacle map looked clear (forward/mid scores ≈1, mast score ≈0), but the table underside was only ~15cm from the top-down RealSense camera (RS1). The robot drove forward because:

1. The floor beneath the table appeared free (no floor-level obstacles)
2. The table top was clipped or treated as free after height thresholding
3. Only the raw RS1 range-to-camera would catch "15cm from topdown"

The existing pipeline focused on floor obstacles and didn't have a reflex for overhead hazards.

## Solution: Near-Field Reflex

### Design Principles

1. **Reflex Priority**: Runs BEFORE floor-obstacle logic (high priority)
2. **Distance Threshold**: Objects closer than ~30cm to RS1 trigger the reflex
3. **Forward Stop**: Immediately zeros forward motion when triggered
4. **Escape Allowed**: Backward motion still allowed if rear is clear
5. **Lightweight**: Cheap min-depth / close-pixel count (suitable for 30Hz loop)

### Implementation

#### Detection (`vision.py`)

```python
def check_topdown_near_field(verts, threshold_m=0.30, min_pixels=50):
    """Check if top-down camera sees a close object (near-field hazard reflex).
    
    Detects table undersides, hands, or any object closer than threshold_m
    to the RS1 camera.
    
    Returns: (triggered, close_count, min_z)
    """
```

The function:
- Takes RS1 point cloud (Nx3 array: X, Y, Z in metres)
- Filters valid depth points (Z > 0.01m)
- Counts points with Z < 0.30m (30cm threshold)
- Triggers if ≥50 close points detected (filters noise)

#### Safety Response (`safety.py`)

```python
def update(self, obs_map, yaw_delta, fwd_delta, height_cm=None, topdown_near_field=False):
    """
    topdown_near_field: True when top-down camera sees object <30cm overhead.
                       Triggers immediate forward stop; reverse allowed if bwd_clear.
    """
```

When `topdown_near_field=True`:
- `_fwd_scale = 0.0` (immediate forward stop)
- `_bwd_scale` computed normally from rear obstacles
- `_ang_scale` computed normally from lateral obstacles
- `_near_field_reason = "topdown_near_field"` (for logging)

### Integration

The reflex is integrated into the vision capture loop:

```python
# RS1 processing in _capture_loop
if self._rs1 and self._rs1.ok and self._rs1.verts is not None:
    # Check near-field FIRST (reflex before floor logic)
    triggered, close_count, min_z = check_topdown_near_field(self._rs1.verts)
    self._topdown_near_field = triggered
    
    # Then process floor obstacles
    z1, k1 = depth_topdown(self._rs1.verts)
    ...

# Pass to SafetyGuard
self._safety.update(self._persistent_obs, fused_yaw, fused_fwd,
                   height_cm=self._persistent_height,
                   topdown_near_field=self._topdown_near_field)
```

## Behavior

### Normal Operation (No Near-Field Hazard)

- All motion allowed (fwd=1.0, bwd=1.0, ang=1.0)
- Standard floor-obstacle avoidance active
- Robot navigates normally

### Near-Field Reflex Triggered (e.g., Hand or Table Underside)

1. **Forward motion**: Immediately stopped (fwd_scale=0.0)
2. **Backward motion**: Allowed if rear is clear
3. **Angular motion**: Allowed (can rotate in place)
4. **Logging**: "NEAR-FIELD REFLEX triggered — close_px=N min_z=Xm"

### Escape Behavior

If the robot triggers the reflex:
- **Rear clear**: Can reverse away from the hazard
- **Rear blocked**: Cannot reverse, but can rotate
- **Planning**: Higher-level planner should command reverse when reflex fires

## Testing

Comprehensive unit tests in `test_topdown_near_field.py`:

1. ✅ Patch at 15cm → triggers reflex
2. ✅ Patch at >40cm → no trigger
3. ✅ Hand-sized patch → triggers reflex
4. ✅ Mixed near/far patches → detects near
5. ✅ Noise filtering (20 points) → no trigger
6. ✅ SafetyGuard stops forward when triggered
7. ✅ SafetyGuard allows motion when clear
8. ✅ Near-field + rear obstacle → both directions blocked
9. ✅ depth_topdown handles mixed clouds
10. ✅ Empty/invalid clouds handled gracefully

Run tests: `python3 test_topdown_near_field.py`

## Parameters

### Tunable Parameters

| Parameter | Value | Location | Purpose |
|-----------|-------|----------|---------|
| `threshold_m` | 0.30m | `check_topdown_near_field()` | Distance threshold for "too close" |
| `min_pixels` | 50 | `check_topdown_near_field()` | Minimum points to trigger (noise filter) |

### Recommended Values

- **threshold_m**: 0.30m (30cm)
  - Larger than incident distance (15cm) with safety margin
  - Small enough to not trigger on normal ceilings (~2.5m)
  
- **min_pixels**: 50
  - Hand-sized patch: ~60 points
  - Table underside: 100+ points
  - Filters sensor noise: <20 points

## Relationship to Existing Safety

### Top-Down Lost Immobilize

**Existing**: If RS1 loses depth entirely (known_px < 800), robot immobilizes completely (fwd=0, bwd=0, ang=0).

**Near-Field Reflex**: If RS1 sees valid depth but something is very close, stop forward only (fwd=0, bwd=normal, ang=normal).

Both mechanisms:
- Use RS1 top-down camera
- Have higher priority than floor obstacles
- Log clear diagnostic messages

### Mast/Tall Obstacle Inflation

**Existing**: `build_safety_occ()` inflates obstacles ≥45cm tall (MAST_CLEAR_CM) to prevent mast collisions with table tops.

**Near-Field Reflex**: Detects table *undersides* that aren't visible as floor obstacles.

Complementary:
- Mast inflation: "Don't drive toward tall floor obstacles"
- Near-field reflex: "Don't drive under low overhead obstacles"

## Logging and Diagnostics

### Vision Properties

```python
vision.topdown_near_field        # bool: reflex triggered
vision.near_field_close_count    # int: number of close points
vision.near_field_min_z          # float: closest distance (metres)
```

### SafetyGuard Properties

```python
safety.topdown_near_field        # bool: reflex active
safety.near_field_reason         # str: "topdown_near_field" or None
safety.fwd_scale                 # float: 0.0 when reflex active
```

### Log Messages

**Reflex triggered** (every 30 frames while active):
```
vision: NEAR-FIELD REFLEX triggered — close_px=100 min_z=0.150m (table/hand <30cm from topdown)
safety: NEAR-FIELD REFLEX — topdown sees close object (<30cm), fwd=0.0, computing bwd/ang normally
```

**Reflex cleared**:
```
vision: NEAR-FIELD REFLEX cleared — close_px=5 min_z=0.450m (after 15 frames)
```

## Future Enhancements

### Potential Improvements

1. **Graduated response**: Instead of hard 0.0, scale forward speed by distance (15cm=0.0, 30cm=0.3, 40cm=1.0)
2. **Spatial mapping**: Track WHERE the close patch is (front, side, above robot center)
3. **Temporal filtering**: Require N consecutive frames to reduce false positives
4. **Integration with planner**: HouseBot automatically commands reverse when reflex fires

### Live Deployment

The implementation is complete and tested offline. To deploy:

1. ✅ Code implemented in `vision.py` and `safety.py`
2. ✅ Unit tests passing
3. ⏳ Integration testing in sim (if available)
4. ⏳ Live testing with drive DISARMED
5. ⏳ Live testing with drive ARMED in controlled environment
6. ⏳ Full deployment

**Current Status**: Ready for integration and sim testing. Drive stays DISARMED for live until validated.

## References

- **Incident**: Kevin drove under table, RS1 at 15cm from underside
- **Files Modified**:
  - `src/vision.py`: Added `check_topdown_near_field()`, integrated into capture loop
  - `src/safety.py`: Added near-field parameter to `SafetyGuard.update()`
  - `test_topdown_near_field.py`: Comprehensive unit tests
- **Related Issues**: Table underside detection, overhead obstacle avoidance, mast collision prevention
