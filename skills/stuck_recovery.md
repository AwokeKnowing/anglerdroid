# Skill: Stuck Detection and Recovery

## Description
Detect and recover from stuck states:
- Wheel slip (commanding motion, no odometry progress)
- Oscillation (repeated small forward-backward or left-right cycles)
- Pinned state (safety scales all motion to zero)

Uses multi-strategy recovery: back-out, lateral shuffle, replan route.

## Preconditions
- Odometry tracking operational (wheelbase + IMU/SLAM)
- Safety layer providing fwd/bwd/ang scales
- Stuck detector initialized (see stuck detection implementation)

## Procedure
1. **Detection phase**
   - Monitor velocity commands vs. odometry progress
   - Detect slip: `|cmd| > threshold` but `|odom_delta| < epsilon` for N frames
   - Detect oscillation: pose variance high, net progress low
   - Detect pinned: `fwd_scale < 0.15` and `bwd_scale < 0.15` for M frames

2. **Classification**
   - **Type A (slip)**: wheels spinning, no grip → back out and try new angle
   - **Type B (oscillation)**: safety scaling causing micro-moves → clear local_executive, replan
   - **Type C (pinned)**: fully immobilized by obstacles → rotate in place, find escape gap

3. **Recovery strategy**
   - **Back-out**: reverse 30-50cm at 0.2 m/s
   - **Lateral shuffle**: rotate 45-90 degrees, try new forward direction
   - **Replan route**: clear local_executive goal, set new waypoint avoiding stuck zone
   - **Emergency stop**: if recovery fails after 3 attempts, alert user

4. **Verification**
   - Monitor odometry during recovery moves
   - Success: forward progress resumes for >1m
   - Failure: repeat stuck after recovery → escalate to emergency stop

## Success Criteria
- Detect stuck within 2-3 seconds
- Recover within 10-15 seconds (3 attempts max)
- Resume normal navigation
- Log stuck event for skill learning

## Failure Modes
- **Persistent stuck**: obstacle configuration prevents all escapes → emergency stop, user alert
- **Sensor failure**: odometry dropout causes false stuck alarm → require sensor validity check
- **Recovery causes new stuck**: back into different obstacle → prioritize reverse during recovery
- **Oscillation amplification**: recovery worsens oscillation → reduce cmd velocities, add damping

## Integration Points
- **test_stuck_detection.py**: stuck detector implementation
- **odometry.py**: velocity integration and progress tracking
- **safety.py**: fwd/bwd/ang scale monitoring for pinned state
- **local_executive**: clear goals and replan after recovery
- **navigator.py**: reactive navigation during recovery moves

## Trace Markers
```json
{"event": "stuck_detected", "type": "slip", "duration_s": 2.3, "cmd_fwd": 0.4, "odom_delta": 0.02}
{"event": "recovery_started", "strategy": "back_out", "reverse_dist_cm": 40}
{"event": "recovery_move", "cmd": {"fwd": -0.2, "ang": 0.0}, "duration_s": 2.0}
{"event": "recovery_success", "attempts": 1, "resume_progress": true}
```
