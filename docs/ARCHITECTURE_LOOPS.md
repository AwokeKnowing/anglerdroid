# Kevin loops: fixed-rate modules, explicit dataflow (James 21:47 / 22:21 / 05:38, Oct 2026)

Rule: every module is a process with ONE fixed-rate loop, reads only shm blocks, writes only its own shm blocks
(emu.shm.Block seqlock, CLOCK_MONOTONIC ns stamps, writer creates). No module calls into another; each is testable by
replaying its input blocks and checking its output blocks. Flat code, no frameworks.

## Loops

| # | loop | host | rate | in | out | owner of |
|---|---|---|---|---|---|---|
| 1 | safety core | robot/Orin (sim: cpu 3) | 30 Hz, hard | RS1/RS2 depth, wheels, IMU (robot frame) | motor gate | stopping. UNTOUCHED |
| 2 | nav core (local planner + odom) | robot | 30 Hz | live frames, odom, `kevin_goal_<src>` (selected), `kevin_maplayer` | `kevin_navstate`, drive cmd to core | the only motion authority above the core |
| 3 | goal producers | any | fixed, e.g. 250 ms | `kevin_navstate`, `kevin_task_<src>`, own sensors | `kevin_goal_<src>`, `kevin_stat_<src>` | proposing goals only |
| 4 | orchestrator | i777 | 1-4 Hz | user/agent tasks, producer stats | `kevin_task_<src>`, `kevin_goalsel` | choosing the producer |
| 5 | SLAM (world model) | i777 | async, continual | logged/streamed frames + odom | map + keyframe poses | consistency, not rate |
| 6 | localization | i777 | 1 Hz | `kevin_navstate`, SLAM map, live scan | `kevin_maplayer` (map<-odom + confidence + map tiles) | alignment check |

Measured now (sim, i777, 2026-10-18/19): core frame age p50 51 / p90 56 / max 90 ms, 0 % > 100 ms; nav rx age p50 52 / p99 88 ms,
perception p50 41 / p99 149 ms, control compute p50 10 / p99 77 ms, global plan 70-170 ms (in the nav tick, to move out);
LightNav prediction 220-480 ms; LightNav process loop lateness max 4 ms at 250 ms (step 1 below).

## Blocks (step 1 implemented in nav/goalio.py; all poses in the ODOMETRY frame)

```
kevin_navstate      nav -> all, 30 Hz : odom f8[3], t_od f8, mode u1, ack u8 (last goal gseq taken), map_T f8[3], map_c f4
kevin_task_<src>    orchestrator -> producer : tseq u8, active u1, text u1[256]
kevin_goal_<src>    producer -> nav : gseq u8, source u1, kind u1 (1 pose, 2 face, 3 stop), pose f8[3], vmax f4, ttl f4, tseq u8
kevin_stat_<src>    producer -> all : ticks, late_ms_max, late_ms_p99, period_ms, pred_ms, preds, errors, fps
kevin_goalsel       orchestrator -> nav : src u1, seq u8                                  (step 3)
kevin_maplayer      localization -> nav/core : T_map_odom f8[3], conf f4, t f8, map version u8  (step 4)
```
Goal semantics: nav acts on a gseq change whose tseq matches the current task; goals for a cancelled task are ignored.
vmax/ttl are carried but not yet enforced (step 2).

## Migration (small steps, each behind a flag, default flips only after bench parity)
1. **Goal interface + LightNav process** (`KEVIN_GOAL_SHM=1`, done 05:43 Oct 19): nav writes navstate + task, LightNav runs in
   `nav/lightnav_proc.py` (own process, spawned by nav, exits with it), 250 ms loop, model calls in a worker thread, writes the goal
   block. Verify: LightNav kitchen n>=3 + goal bench unchanged, then default on.
2. Enforce vmax + ttl in nav; scripted producer (`code+labels`: place goals) moves out of nav `_apply` into a producer process.
3. Orchestrator process owns task routing (`kevin_goalsel`); nav's chat/command parsing becomes a producer too.
4. Localization process publishes `kevin_maplayer`; nav's `map_check()` (KEVIN_NAV_DECOUPLE) moves there; nav uses the map only via
   T + conf. SLAM moves out of the nav process (submit becomes a block of frames+odom it consumes).
5. Global planning out of the nav tick (async planner process writing a path block), nav keeps only local driving.
6. Replay harness per module: record its input blocks, re-run, diff its output blocks.
