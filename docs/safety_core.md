# Safety core (PROPOSAL, sim only, branch safety-core-loop-2026-10-17, not merged, not on the Orin)

Design (James, 2026-10-17): a dedicated deterministic 30 Hz safety loop, separate from planner/nav, reacting on the
frame where an issue is seen; operability = fresh frames on cadence; last-resort braking only; all obstacles below
1 m collapsed to 2-D; the planner keeps ~2x the safety threshold clear; no creep after a stop (hard 5 cm bands).

## Processes
| | brain (nav.run) | safety core (nav/safety_core.py) |
|---|---|---|
| process | main | `multiprocessing` spawn child, exits (after zeroing the wheels) if the parent dies |
| cameras | RS1 + RS2 pipelines; dedicated pump threads publish every frame to `nav.bus` at once | reads RS1 verts + RS2 depth from `nav.bus` |
| wheels | none (sends desired v, w over `nav.bus`) | owns `WheelBase` (only writer to the ODrives), publishes wheel vel / currents back |
| per tick | SLAM, map, MPPI, **governor** (2x threshold) | operability, transform, bands, last-resort brake |

## Per tick (one RS1 frame)
1. operability: RS1 and RS2 age (time.monotonic() at decision - capture) <= 100 ms, frame interval <= 1.5 periods
   (50 ms: a dropped frame = missed cadence), RS1 view not occluded; otherwise stop (max brake ramp if moving).
2. RS1 top-down `Ego` labels (the brain's own code) + RS2 cone `Rs2Cloud(h_max=1.0)`, 2-D, motion-compensated to now
   by the wheel-odometry twist over each frame's age.
3. bands: 5 cm ahead of the masked footprint (x 0.145-0.195, |y| <= 0.21) and 5 cm at the rear edge of what RS1 sees
   behind the mask (learned from valid returns); obstacle in a band -> zero velocity toward it (held, no creep).
4. brake: sweep the command (or measured speed, if faster) along its arc for need(v) = v (T_tick + T_cmd) + d_stop(v,
   ramp 5); if it enters the 3.5 cm margin -> zero now with the vel ramp raised just as far as needed (3..5 turns/s^2).
   Otherwise the command passes unchanged (no throttling in the core).
5. governor (brain): largest of 8 fixed scales whose sweep keeps 2 x (need(v) + margin) clear.

## Real time: sim vs Orin
| | sim (i777, this branch) | Orin (to do, not deployed) |
|---|---|---|
| CPU pinning | `sched_setaffinity` to the last CPU: works, but CPU not isolated (all 8 cores ~75 % busy) | `isolcpus=`/`nohz_full=` kernel args for one A78AE core; IRQ affinity away from it |
| priority | `SCHED_FIFO` denied (RLIMIT_RTPRIO 0, no CAP_SYS_NICE); nice -10 denied | `/etc/security/limits.d` rtprio 60 for the robot user (or CAP_SYS_NICE on the binary); JetPack kernel is PREEMPT (PREEMPT_RT optional) |
| GC | `gc.freeze(); gc.disable()` in the core | same |
| BLAS | single-threaded in the core | same |
| RS1 ingest | sim shim in Python (decimation 11 ms after the sort fix, was 37 ms) in a brain-process thread | librealsense C; better: move RS1 ownership into the core process (RS1 is a separate device from RS2/IMU) |
