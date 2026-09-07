# Skill: Dog Bed Soft Approach

## Description
Cautiously approach soft, low obstacles (dog beds, cushions, pillows) that:
- Are 5-30cm height (detectable by RS1 top-down depth)
- Are at medium distance (35cm-1m forward)
- Don't require hard-stop reflex (unlike bumps/checkered mats)

Strategy: attenuated forward velocity (0.3x normal) for slow, safe contact.

## Preconditions
- RS1 top-down depth sensor operational
- `topdown_soft_low_obstacle` detection active in safety layer
- Forward path clear of hard obstacles

## Procedure
1. **Detection phase**
   - RS1 depth detects low, soft obstacle in forward cone
   - Safety layer sets `topdown_soft_low_obstacle=True`
   - Safety attenuates `fwd_scale=0.3` (not zero)

2. **Approach phase**
   - Continue forward at reduced velocity (0.3x)
   - Maintain full angular and backward authority
   - Monitor odometry for forward progress

3. **Contact handling**
   - Soft contact: wheels may climb slightly or compress cushion
   - Hard contact: safety hard-stop reflexes take over (checkered mat, bump)
   - Progress stalls: stuck detection triggers recovery

## Success Criteria
- Forward velocity reduced to ~0.3x during approach
- No crash or aggressive contact
- Escape path (reverse) remains available
- Can navigate around or over soft obstacle safely

## Failure Modes
- **False positive**: hard obstacle misclassified as soft → rely on hard-stop reflexes as backup
- **Stuck on cushion**: wheels lose traction → stuck recovery skill
- **Sensor dropout**: RS1 depth fails → fall back to floor obstacle map

## Integration Points
- **safety.py**: `topdown_soft_low_obstacle` flag, `fwd_scale` attenuation
- **vision.py**: RS1 depth analysis for soft low obstacle classification
- **local_executive**: continue goal-directed motion with safety-scaled velocity

## Trace Markers
```json
{"event": "soft_low_obstacle_detected", "distance_cm": 65, "height_cm": 12}
{"event": "soft_approach_active", "fwd_scale": 0.3, "cmd_fwd": 0.12}
{"event": "soft_approach_complete", "outcome": "navigated"}
```
