# Skill: Person Approach (Social Distance)

## Description
Approach humans at comfortable social distance (4-5 feet / 1.2-1.5m) by:
- Face detection and tracking (RGB camera)
- Social FSM state management (greeting, following, conversation)
- Velocity reduction in final approach zone
- Stop at social boundary (no encroachment)

Maintains safe, non-threatening interaction distance.

## Preconditions
- Face detection operational (RGB camera or face gallery)
- Social FSM initialized (see `social_fsm.py`)
- Person location tracked in world frame (SLAM locked preferred)
- Local executive available for goal-directed navigation

## Procedure
1. **Detection phase**
   - Face detected via RGB camera + face recognition
   - Person location estimated (depth + pose)
   - Social FSM transitions to `approaching` state

2. **Navigation phase**
   - Set goal 1.5m from person (outer social boundary)
   - Local executive navigates toward goal
   - Monitor distance continuously via face tracking

3. **Approach zone** (1.5-2.5m)
   - Reduce velocity to 50% of normal
   - Increase face tracking frequency
   - Prepare to stop at 1.2m (inner boundary)

4. **Social boundary** (1.2-1.5m)
   - Stop forward motion (goal reached)
   - Maintain facing orientation toward person
   - Transition to `conversation` or `following` state
   - Angular adjustments allowed to track person movement

## Success Criteria
- Stop distance: 1.2-1.5m from person
- Smooth deceleration (no abrupt stop)
- Face centered in view (±15 degrees)
- Person not startled or encroached upon

## Failure Modes
- **Person moves**: replan goal to new location, maintain distance
- **Face lost**: timeout 3 seconds, enter `searching` state
- **Obstacle blocks**: navigate around, retry approach from new angle
- **Too close**: immediate back-out 20-30cm, re-approach

## Integration Points
- **social_fsm.py**: state machine (idle → approaching → conversation)
- **people_live.py**: face detection and tracking
- **local_executive**: goal-directed navigation with safety scaling
- **safety.py**: maintain full obstacle avoidance during approach

## Trace Markers
```json
{"event": "person_detected", "name": "Alice", "distance_m": 3.2}
{"event": "approach_started", "goal_xy": [4.5, 2.1], "target_dist": 1.5}
{"event": "approach_zone_enter", "distance_m": 2.3, "velocity_scale": 0.5}
{"event": "social_boundary_reached", "distance_m": 1.35, "state": "conversation"}
```
