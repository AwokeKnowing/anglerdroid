# Sim ↔ Kevin plant (2026-09-18)

Brief log so we do not re-guess this.

## Ego / policy rate

- **Was:** 80×60 built on extras (~3 Hz). Policy hold 160 ms (~6 Hz).
- **Now:** 80×60 every capture tick from this frame’s decimated verts
  (`label_frame_80`). PNG dump stays on extras. `RlNavLive` holds
  `DT_POLICY = 0.20` — infer at 5 Hz, same as sim.
- **Timed on Kevin 2026-09-18** (35k+45k verts, cached FOV/self/cone):
  `label_frame_80`+float+throttle **p50 9.7 ms, p95 10.3 ms**. That is
  not the 5 Hz hold. 30 Hz is easy. 60 Hz budget is 16.7 ms — the
  function fits, cameras are still 30.

## Delay

- Not the 1 s `twist_for` ramp (pilot bags). That is a UI profile.
- Policy path: infer → CAN → ODrive. Discrete match is **one 5 Hz tick**.
- `DELAY_STEPS = 1` (was random 0 or 1). Slew is still `A_MAX=1.61`
  from `vel_ramp_rate=3.0` turns/s².

## Slip (pose only; vec still reports slewed cmd)

From `sim/rl_nav/bags/wander_5min_pilot/cmds.jsonl`, 5 s batches,
safety free, no boppy collision:

| cmd | measured | math | ratio |
|-----|----------|------|-------|
| 0.10 m/s | 0.392 m, 0.379 m | 0.50 m | **0.784, 0.758** |
| 0.10 m/s toward boppy | 0.349 m | 0.50 m | 0.698 (dropped) |
| 0.25 rad/s, ang=1 | 70.4° | 71.6° | **0.983** |
| pinched ang 0.3–0.43 | 5–11° | 34–57° | safety, not slip |

**Sim:** `SLIP_V=0.77` `SLIP_W=0.98`. Yaw does **not** overshoot when
clear; earlier “overshoot” was hitting the boppy.

Wheels.npy cmd↔encoder xcorr is useless here (1 s ramp). Encoder `age_s`
p50 ≈ 13 ms.

## Still randomized

House topology, blobs, movers, per-episode `v/a/w_scale`. Not slip/delay.

## Not done

RS1 USB2 is 640×480@30 (848 fails). Webcam often missing. IMU pipeline
still fights the depth pipe. Static RS2 `Y` blit still 0.
