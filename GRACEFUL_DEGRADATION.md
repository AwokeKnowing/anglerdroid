# Graceful Degradation - Navigation Without Full SLAM

**Philosophy:** Kevin must navigate in a small house WITHOUT requiring perfect SLAM or a locked global map.

## Priority Reframe (from James)

### ❌ **Old Thinking**
- Fix SLAM first
- Block everything until SLAM perfect
- No roaming without locked global map

### ✅ **New Thinking**
- **Eventually:** SLAM + reconstruction (nice-to-have)
- **Near-term:** Navigate with:
  1. Local obstacle avoidance (depth, near-field, RGB topdown)
  2. Landmarks in ego/sensor frame (NOT map-frame)
  3. Local planning (MPPI ~1m goals)
- **DO NOT block roam on locked global map**

---

## What Works When SLAM Bad

### ✅ **Always Works (Local Navigation)**

**Obstacle Avoidance:**
- Depth-based occupancy map (ego-frame)
- Near-field reflex (topdown close objects)
- RGB topdown hazards (bump, checkered floor)
- Height-based inflation
- Safety guard (fwd/bwd/ang scaling)

**Local Planning:**
- MPPI costmap (~1m rolling goals)
- VFH local planner
- Gamepad override
- Twist commands

**Ego-Frame Landmarks:**
- Face detection (InsightFace)
- Room cues (visual features)
- Named hazards (relative to robot)
- Topdown RGB patterns

### ❌ **Disabled When SLAM Not Locked**

**Map-Frame Operations:**
- Global navigation goals (xy in world frame)
- Map-frame keepout marking
- Persistent map updates
- Loop closure optimization

**What SLAM Lock Requires:**
- Encoders working (`enc=True`, age <1s)
- Visual tracking OR recent visual (<5s)
- Top-down depth working

---

## Immobilization Rules

### 🚨 **Hard Immobilize (Safety Critical)**

**Stuck Detection:**
```python
if wheels_spinning and visual_shows_zero_motion:
    # STOP IMMEDIATELY
    safety_scales = 0.0
```

**Top-Down Lost:**
```python
if not topdown_depth_ok:
    # NO BLIND DRIVING
    safety_scales = 0.0
```

### ⚠️ **Degraded Mode (NOT Immobilized)**

**SLAM Not Locked:**
```python
if not slam_locked:
    # Local navigation still works
    # Map-frame ops disabled
    log("Map-frame operations disabled")
    log("Local navigation still works")
```

**Encoder Fallback:**
```python
if enc=False:
    # Skip SLAM updates (pose unreliable)
    # But local obstacle avoidance continues
    skip_slam_updates = True
```

---

## Navigation Modes

### 🟢 **Full Capability (SLAM Locked)**

```
Sensors: depth ✓, encoders ✓, visual ✓, topdown ✓
Local: obstacle avoid ✓, MPPI ✓, landmarks ✓
Map: global nav ✓, keepouts ✓, reconstruction ✓
```

### 🟡 **Degraded Mode (SLAM Not Locked)**

```
Sensors: depth ✓, encoders ✗, visual ~, topdown ✓
Local: obstacle avoid ✓, MPPI ✓, landmarks ✓
Map: global nav ✗, keepouts ✗, reconstruction ✗
```

**Use case:** Small house roaming without perfect odometry

### 🔴 **Emergency Stop (Stuck or Blind)**

```
Condition: wheels spinning OR topdown lost
Action: immobilize (safety_scales = 0.0)
```

---

## Landmark System (Ego-Frame)

### Philosophy
Map-frame keepouts don't work without SLAM. Use **ego-frame landmarks** instead.

### Examples

**Face Landmarks:**
```python
# "James is 2m ahead at 15° left"
face_landmark = {
    'name': 'James',
    'distance_m': 2.0,
    'bearing_deg': 15.0,
    'confidence': 0.9,
    'timestamp': now,
}
```

**Topdown RGB Patterns:**
```python
# "Checkered floor at 0.5m ahead"
floor_landmark = {
    'type': 'checkered_floor',
    'distance_m': 0.5,
    'bearing_deg': 0.0,
    'action': 'stop_forward',
}
```

**Room Cues:**
```python
# "Kitchen detected (visual features)"
room_landmark = {
    'name': 'kitchen',
    'confidence': 0.7,
    'features': ['countertop', 'stove', 'fridge'],
}
```

### Integration

**Avoid ego-frame hazards:**
```python
for landmark in ego_landmarks:
    if landmark['type'] in ['bump', 'checkered_floor']:
        if landmark['distance_m'] < 0.3:
            stop_forward_motion()
```

**Navigate to ego-frame goals:**
```python
# "Go toward James"
face = find_face('James')
if face:
    goal_bearing = face['bearing_deg']
    mppi.set_heading_goal(goal_bearing)
```

---

## Log Signatures

### 🟢 **Full Capability**
```
🟢 SLAM: LOCKED
encoder: enc=True age=0.05s
Local navigation: ✓
Map operations: ✓
```

### 🟡 **Degraded Mode**
```
🔴 SLAM: NOT LOCKED (encoder_failed)
   → Map-frame operations disabled (keepouts, global nav)
   → Local navigation still works (obstacle avoid, MPPI)
encoder: enc=False
Local navigation: ✓
Map operations: ✗
```

### 🔴 **Emergency Stop**
```
🚨 STUCK DETECTED (count=1)
   Commanded: 0.52m, Actual: 0.04m
⚠️  vision: STUCK — autonomous motion disabled
OR
⚠️  vision: TOPDOWN LOST — immobilized
```

---

## API

### Check Capabilities

```python
# Vision
slam_locked = vis.slam_locked          # True if map-frame ops work
is_stuck = vis.is_stuck                # True if wheels spinning
can_navigate_local = True              # Always (unless stuck/blind)
can_navigate_global = vis.slam_locked  # Only when SLAM locked
```

### Landmark Management

```python
# Add ego-frame landmark
landmarks.add_ego({
    'type': 'face',
    'name': 'James',
    'distance_m': 2.0,
    'bearing_deg': 15.0,
})

# Query nearest hazard
hazard = landmarks.nearest_hazard(max_distance=1.0)
if hazard and hazard['distance_m'] < 0.3:
    stop_forward()
```

---

## Phased Implementation

### ✅ **Phase 1 (This PR)**
- Stuck detection (critical)
- SLAM lock status (informational, not blocking)
- Graceful degradation (local works without SLAM)
- Documentation

### 📋 **Phase 2 (Next)**
- Ego-frame landmark system
- Face landmark integration
- Topdown RGB hazard landmarks
- Room cue detection

### 📋 **Phase 3 (Later)**
- Full SLAM reconstruction
- Persistent map with relocalization
- Map-frame keepouts (when SLAM reliable)
- Global navigation with map

---

## Success Criteria

### Near-Term (Small House)
- ✅ Navigate ~5m reliably
- ✅ Avoid obstacles (depth + RGB topdown)
- ✅ Detect/avoid checkered floor, bumps
- ✅ Respond to faces, rooms
- ✅ ~1m MPPI goals work
- ❌ Don't need locked global map

### Long-Term (Eventually)
- ✅ Full SLAM with loop closure
- ✅ Persistent map across sessions
- ✅ Global navigation goals
- ✅ Map-frame keepouts
- ✅ Reconstruction quality

---

## Related

- **Stuck detection:** `STUCK_DETECTION_INCIDENT.md`
- **SLAM lock:** `CRITICAL_SLAM_FIXES.md`
- **RGB topdown:** PR bc-51a08189
- **This PR:** #11 cursor/slam-robustness-improvements-cfbd

---

**Philosophy:** Build a robot that works in the real world with imperfect sensors, not one that only works in simulation with perfect SLAM.
