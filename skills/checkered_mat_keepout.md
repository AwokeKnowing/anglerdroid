# Skill: Checkered Mat Hard-Stop

## Description
Detect and avoid checkered mat zones that mark:
- Dangerous areas (stairs, ledges)
- Off-limits spaces (human work areas)
- Furniture protection zones

**Special case: Door mat** (`checkered_door` keepout)
- Black/white checkered mat by front door marks high RL stuck risk zone
- Detection runs at ~3fps in vision extras loop (NOT in 30Hz capture)
- Named keepout `checkered_door` persists in map for path planning avoidance
- RL policy should learn to avoid this zone (high stuck reward penalty)

Uses RS1 top-down RGB pattern recognition for immediate hard-stop reflex.

## Preconditions
- RS1 top-down RGB camera operational
- Checkered pattern detector enabled in vision extras loop (~3fps)
- `topdown_hazard` reflex active in safety layer
- SLAM locked for persistent keepout marking (optional, reflex works without)

## Procedure
1. **Detection phase**
   - RS1 RGB detects high-contrast checkered pattern
   - Pattern analysis confirms grid structure (not noise)
   - Vision layer sets `topdown_hazard=True`

2. **Reflex response**
   - Safety layer zeros `fwd_scale=0.0` immediately
   - Reverse and angular authority maintained (escape capable)
   - No forward motion until pattern clears field of view

3. **Recovery phase**
   - If engaged: reverse 20-30cm to clear pattern
   - If approaching: replan route around keepout zone
   - Log event for map-based keepout persistence (if SLAM locked)

## Success Criteria
- Forward stop within <1 frame of pattern detection
- No entry into checkered zone
- Reverse escape functional during reflex
- Keepout map updated (persistent avoidance)

## Failure Modes
- **Pattern noise**: false positive on similar textures → conservative (stop is safe)
- **Late detection**: pattern appears very close → rely on immediate reflex + reverse
- **Sensor failure**: RGB dropout → fall back to floor obstacle map only
- **Already on mat**: stuck in keepout → stuck recovery prioritizes reverse

## Integration Points
- **vision.py**: `_vision_extras_loop()` runs RGB detections at ~3fps (separate from 30Hz capture)
- **checkered_mat.py**: pattern detection algorithm (RS1 RGB, NO findChessboardCorners in 30Hz loop)
- **safety.py**: `topdown_hazard` flag, hard-stop reflex
- **keepouts.py**: map-frame keepout polygons (persistent avoidance, e.g. `checkered_door`)
- **local_executive**: path planning must avoid keepout zones
- **neural_rl.py**: RL policy learns stuck risk penalty for `checkered_door` zone

## Trace Markers
```json
{"event": "checkered_detected", "confidence": 0.95, "distance_cm": 45}
{"event": "checkered_hardstop", "fwd_scale": 0.0, "reason": "topdown_hazard"}
{"event": "keepout_updated", "zone_id": "checkered_01", "world_xy": [2.3, -1.5]}
```
