# Skill: Overhang Front-Strip Navigation

## Description
Safely navigate under tables, shelves, and elevated structures by:
- Early detection of overhangs (30-70cm ahead) via RS1 top-down camera
- Forward stop BEFORE committing to drive under
- Escape capability via reverse if already engaged

Prevents mast collision when floor appears clear but overhead structure blocks.

## Preconditions
- RS1 top-down camera operational
- Mast height clearance tracking active (MAST_CLEAR_CM = 45cm)
- `topdown_overhang_approach` detection enabled

## Procedure
1. **Early warning phase** (30-70cm ahead)
   - RS1 detects elevated structure in forward cone
   - Safety layer sets `topdown_overhang_approach=True`
   - Safety zeros `fwd_scale=0.0` immediately

2. **Assessment phase**
   - Stop forward motion (reverse and angular still available)
   - Height map analysis: can mast fit under?
   - If clearance < MAST_CLEAR_CM (45cm): refuse entry
   - If clearance >= MAST_CLEAR_CM: consider low-speed approach

3. **Decision**
   - **Navigate around**: replan to avoid overhang zone
   - **Low-speed entry**: reduce speed, monitor height continuously
   - **Abort**: reverse out if already partially under

## Success Criteria
- No mast collision with table undersides, shelves
- Forward stop triggered 30-70cm before overhang
- Escape path (reverse) functional during assessment
- Height clearance margin maintained (>= 45cm)

## Failure Modes
- **Late detection**: overhang appears <30cm → near-field reflex catches it
- **Clearance misjudgment**: enter too-low space → immediate back-out
- **Sensor noise**: false overhang alarm → timeout and retry
- **Stuck under table**: height sensor shows <45cm → stuck recovery with reverse priority

## Integration Points
- **safety.py**: `topdown_overhang_approach` flag, `near_field_reason` tracking
- **vision.py**: RS1 depth + height map for overhead structure detection
- **local_executive**: goal planning must avoid overhang keepout zones
- **keepouts.py**: map-frame overhang zones (if SLAM locked)

## Trace Markers
```json
{"event": "overhang_detected", "distance_cm": 55, "clearance_cm": 40}
{"event": "overhang_approach_stop", "fwd_scale": 0.0, "escape_available": true}
{"event": "overhang_decision", "action": "navigate_around"}
```
