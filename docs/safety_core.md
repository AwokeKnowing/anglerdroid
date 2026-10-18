# Safety core (PROPOSAL, sim only, branch safety-core-loop-2026-10-17, not merged, not on the Orin)

Design (James, 2026-10-17): a dedicated deterministic 30 Hz safety loop, separate from planner/nav, reacting on the
frame where an issue is seen; operability = fresh frames on cadence; last-resort braking only; all obstacles below
1 m collapsed to 2-D; the planner keeps ~2x the safety threshold clear; no creep after a stop (hard 5 cm bands).

## Processes
| | brain (nav.run) | safety core (nav/safety_core.py) |
|---|---|---|
| process | main | `multiprocessing` spawn child, exits (after zeroing the wheels) if the parent dies |
| cameras | RS2 pipeline (SLAM needs color/IR/IMU); pump thread publishes RS2 depth to `nav.bus`; reads RS1 back from `nav.bus` (`_Rs1Bus`) | owns the RS1 pipeline (v2); reads RS2 depth from `nav.bus` |
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
| RS1 ingest | sim shim in Python, in the core process (v2): ingest 11 ms p50, Ego 13 ms p50 | librealsense C in the core process |

## Results (sim, i777, 2026-10-17; n=3 goal benches, same goals/world/dog keep-out; cur = PR #62 in-brain envelope)
Measured = from logs/GT in this sim; assumed = constants not measured here.

| quantity | value | measured / assumed |
|---|---|---|
| core loop period (goals) p50/p95/max | 33.2 / 40.3 / 560 ms (cur brain loop 40.8 / 59.4 / 644) | measured |
| frame age at decision (goals) p50/p95/max | 46 / 74 / 157 ms; >100 ms on 1.4 % of ticks (cur 69 / 93 / 698, 2.4 %) | measured |
| frame age at decision (VLN running) | 164 / 285 / 379 ms, >100 ms on 83 % of ticks (cur 97 / 169 / 648, 45 %) | measured: sim delivers frames late (RTF ~1.0) |
| core processing per tick (moving) | 27 ms p50, 53 ms p95 (ingest 11, Ego 13, RS2 1.4, decide 1) | measured |
| frames missing cadence while moving | RS1 6.6 %, RS2 10.3 % of ticks (0 % idle) | measured |
| T_TICK | 1/30 s | assumed (design) |
| T_CMD (CAN command latency) | 10 ms | assumed |
| braking, ramp 5 turns/s^2 | d = 0.0926 v + v^2/(2*4.26) | measured offline (brake_test) |
| margin | 3.5 cm | assumed (design) |
| bands | front x 0.145-0.195 m, rear x -0.345..-0.295 m | front fixed; rear learned from RS1 visibility |

Stop tests (box, GT gap at rest; 0 contacts in all): see the report; post-obstacle-stop creep <= 2 mm except when an
operability stop (frame age > 100 ms) paused the approach and the drive resumed once frames were fresh again.

Known sim limitation: under load (VLN + GPU) the sim's camera frames reach the core 100-280 ms after capture and RS1/RS2
miss the 30 Hz cadence while driving, so the strict operability rule stops Kevin repeatedly (stop-go; goals 2/12 vs
cur 5/12).  One diagnostic run with KEVIN_SC_CADENCE_MAX=0.083 (one dropped frame tolerated; age rule unchanged) got
3/4 goals.  This needs a real-time host (isolated core, SCHED_FIFO) or real-robot latency data, not a looser rule.

## Sim frame latency fix (2026-10-17, i777)

Update to the limitation above: it was the sim's frame path, not missing RT priority.  Rules are unchanged
(age > 100 ms or frame interval > 1.5/30 s -> inoperable).  Bench: goal_bench (kitchen/bedroom/couch + LightNav
'find the kitchen'), n = 3 each.  Before = r6 (old world, in-brain RS2 pump, no pinning).  After = lat4.

Root causes (traced with KEVIN_EMU_CAMLOG / KEVIN_SHIM_LOG / core log stage columns):
1. emu.world rendered RS1+RS2 depth and noise in a single thread, sharing the GIL with physics, the colour/IR/viewer
   renders and the CAN/ctl servers.  Under VLN the render queue reached 28 ms p95 and RS2 age at write 67 ms p95.
   Fix (slambench): emu.depthcam process, one thread per camera with its own renderer and its own CPU.
2. The core got RS2 depth from a brain thread that shared the GIL with SLAM/MPPI/VLN, so 15-20 % of RS2 frames
   missed the beat.  Fix: nav/camproc.py owns RS2 (video + IMU, the same single-owner layout RSUSB needs on the
   Orin) and writes depth to the core first.
3. The core sampled "newest RS2" once per RS1 tick, which aliased into skipped frames (rs2cam got 100 % of frames
   but the core saw 6 % gaps).  Fix: the core pairs each RS1 frame with the nearest RS2 capture, using a 2-deep ring
   and a <= 10 ms wait only when the paired frame is due.
4. The core needs about 24-31 ms of CPU per frame.  On a hyperthread shared with a render lane it overran 33 ms.
   Fix: simpin.sh gives the core physical core 3 (only the light rs2cam on sibling 7).  Depth lanes run on 6/2.
   Everything else runs on 0,1,4,5, with LightNav/ollama at nice +10 and the viewer at +5 (user-level, no root).

Measured (ms p50/p95/max; age = at decision, after the transform; missed = consumed-frame interval > 50 ms, % of
ticks; period after the first 4 s of each window, which is nav-restart startup with Kevin still):

| window | run | age RS1 | age RS2 | age > 100 ms | missed RS1 / RS2 | loop period | stops/inop events |
|---|---|---|---|---|---|---|---|
| goals | before r6 | 46/74/157 | 46/75/165 | 1.50 % | 0.23 / 1.47 % | 33.2/40.2 | cadence 69, age 16 |
| goals | after lat4 | 50/60/122 | 50/60/113 | 0.16 % | 0.37 / 0.37 % | 33.4/41.3/91 | cadence 17, age 2 |
| VLN | before r6 | 164/285/379 | 169/291/441 | 85.2 % | 2.41 / 12.20 % | 33.5/51.2 | age 150, cadence 38 |
| VLN | after lat4 | 50/66/130 | 50/66/126 | 0.26 % | 0.36 / 0.34 % | 33.1/43.9/87 | cadence 6, age 4 |

Ground truth (gtbench): before r6 goals 2/12, contacts 0, 3-D gap min 0.095 / p5 0.120 m.  After lat4 goals 7/12
(kitchen 3/3, bedroom 3/3, couch 0/3 [planner "unreachable", in every run before too], VLN 1/3 arrived), contacts 0,
3-D gap min 0.027 / p5 0.153 m.  The 2-D gap min of -0.005 m is the tool's clamp; it appears in every run.

Per-stage after (lat4, ms p50/p95/max): physics tick -> depthcam pick-up 1.3/7.8/81; render RS1 12.2/19.2 and RS2
8.7/16.4; noise 10.4/13.1; shm write 0.2-0.3; capture -> core RS1 wake 26/34-40; rs2cam wake 23/29-34, 0.00 % gaps;
core ingest 9.3/12.1, ego 12.0/14.8, RS2 pick+cloud 1.5/2.1, decide 1.5/2.9, total 24.4/29.8.  The missed beats
that remain (0.3-0.4 %) start upstream: depthcam publish intervals > 50 ms 0.35-0.43 % (physics/pick-up hiccups).
Real Orin RS2 for reference: capture -> Python 32.1/32.5/33.2 ms, 33.3 ms steady, 0 drops.  Sim capture -> Python is
26/34 ms: about the same median, more jitter.
