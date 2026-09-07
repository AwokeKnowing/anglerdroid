# AGENTS.md — Kevin / Anglerdroid (Orin-first)

## Non-negotiable: Orin microsecond vigilance

Making stuff work is easy. Making it work on Jetson Orin NX in the **most
optimized way** is the real challenge. Be hawkish. Measure. Never waste µs.

### Critical path (30 Hz = 33.3 ms hard budget)
1. **Fresh frames every tick** — RealSense + RGB configured for 30 Hz; exposure
   capped so AE cannot stretch past the frame period.
2. **GPU-resident processing** — get depth/color onto GPU and keep them there.
   No casual CPU round-trips (`get_data` → numpy → upload again) in the hot path.
3. **Decimation / projection phase order** — do the *cheapest* reduction as early
   as physically possible (SDK decimate / GPU downsample) *before* project,
   deproject, join, morph, scatter. Never process undecimated 848×480 views in
   Python/NumPy “because it was easier.”
4. **Depth reflexes only on capture** — near / overhang / soft-low. RGB image
   detections (brown border, faces, gestures) live in a **~3 Hz extras** loop or
   on **i777**, never in capture.
5. **i777 ship thread** — separate non-blocking thread: downsize / compress /
   queue frames or detections to i777. Capture must never stall on network or
   encode.
6. **Per-core accounting** — know what every ARM core and the GPU are doing each
   phase. Prefer persistent worker pools over creating pools/threads per frame.

### Past wizardry (do not casually undo)
- RealSense depth **decimation**: **measure verts on device**. On current JP/librealsense,
  `RS_DECIMATE_MAG=3` produced ~45k verts (≈3×3 blocks). Mag=8 → ~6k verts (old docstring).
  On this JP6 stack mag is **linear** (mag=3≈45k verts, mag=8≈6.5k). Numpy can beat SDK pc on CPU ms, but **detail metric above gates any mag change**. Default stays mag=3 until GSD/ego-fill probe says otherwise.

### Thread architecture (James 2026-09-07)
1. **Pre-allocate and mutate in place** — hot path must not alloc/copy per frame when
   a buffer can be reused. Prefer `np.copyto` / slice assign into preallocated arrays.
   Drop fresh `.copy()` / `asarray().copy()` on IR/color/atlas/reader paths where safe.
2. **Frame loop waits for synced camera frames** each tick (RealSense poll/wait). Do
   not busy-spin past frame sync. Webcam stays off the RS grab barrier (already landed).
3. **Faster odom thread** — a dedicated high-rate thread (~100-200 Hz) deals with
   wheel odometry + IMU yaw and **emits/publishes poses at ~30 Hz** (or lock-free
   latest snapshot) for the frame/capture loop to consume. Capture must not be the
   only place that integrates wheels/IMU.
4. **Keep IMU and depth separate** (already mostly true via `IMUPipeline`). Do not
   require IMU↔depth hardware sync. SLAM may later care about IMU stamps paired
   with clouds, but that is out of scope for this thread split.

### Hardware-first (GPU > CPU NumPy)
A “numpy algorithm” is a *shape*, not a placement. On Orin prefer **CuPy /
CUDA / ModernGL** (or other GPU-resident ops already in `gpu_render`) for
decimate / project / join / morph so pixels never bounce through CPU caches
for convenience. Architect for silicon: ARM cores, GPU, ISP/USB bandwidth —
measure each.



### Policy feed: labeled heightmap @ 30 Hz (FSD-style)
The product of the critical path is a **clean labeled ego heightmap** every
frame for the neural policy — not a pretty CPU atlas. Prefer GPU-resident
depth→heightmap (and later VO/viz on GPU). 30 Hz is enough; 60 Hz is headroom.
RS1 map work ~12 ms inside the 33 ms budget is fine; spend the remaining ~20 ms
on grab/sync, forward depth, safety, and policy I/O — ideally also on GPU.

### Depth detail metric (decimate is NOT “fewer verts = win”)
Obstacle safety projects depth into an ego grid at **`EGO_PX_SIZE = 1 cm/px`**.
Chasing a tiny pointcloud is wrong if we punch holes in that grid or starve
reflex pixel counts (`near_min_pixels`, `soft_min_pixels`).

**Stick-to targets (local sensing, top-down RS1):**
1. **GSD ≤ ego cell** out to the reflex horizon that matters — soft-low to
   ~1.0 m, overhang to ~0.7 m, near-field <0.3 m. Practical bar:
   ground sampling ≤ **1 cm at 1 m** (≈ ≤1.5 cm at 1.5 m). Rough JP6 map:
   mag3 ≈ 0.6 cm @1 m; mag4 ≈ 0.7; mag5 ≈ 0.9; mag6 ≈ 1.1; **mag8 ≈ 1.5 (fails bar)**.
2. **Ego fill** — in the valid FOV footprint, median hits/occupied-cell ≥ 1
   and hole fraction low enough that thin legs / dog-bed edges still fire
   soft-low / near reflexes (compare hit counts vs mag3 baseline on the same scene).
3. **GPU capacity is secondary** — ModernGL scatter into 320×240 is cheap once
   verts are on GPU; CPU deproject/topdown and reflex min_pixels are the real
   limits. Prefer early GPU-resident downsample that **preserves GSD**, not
   arbitrary vert starvation.

Only lower `RS_DECIMATE_MAG` after an on-device probe reports GSD + ego-fill +
reflex counts for the candidate vs the current baseline. Speed is a constraint
under that metric, not the objective.

### Decimate: measure, don’t assume
James has seen **numpy decimate beat the RealSense SDK filter** on Orin in
some configs. Treat SDK vs numpy as a bake-off: benchmark grab+pc+downstream
on device whenever changing it. Goal is **60 fps** when physics allows —
33.3 ms is the *floor* we refuse to miss, not the ambition.


- Pre-allocated buffers, queue size 1, poll-then-short-wait grabs.
- GPU path in `gpu_render.py` for forward depth / odom / gmap / atlas.

### Before declaring a vision change “done”
- [ ] Ran silent vision smoke on Kevin; pasted CAPTURE TIMING table
- [ ] TOTAL p95 < 33.3 ms (or explained the physical blocker)
- [ ] No new undecimated CPU loops on 848×480
- [ ] No new sync network/encode on capture thread
- [ ] Stated which cores/GPU own each new stage

### Related docs
- `JAMES_ARCHITECTURE.md` — 30 Hz vs ~3 Hz extras split
- `LOOP_HARDENING_SUMMARY.md` / `CAPTURE_FPS.md` — budget shedding notes
- `skills/checkered_mat_keepout.md` — brown border @ 3 Hz + named keepout

---

# AGENTS.md — Coding-Agent Skill Repair Loop (Stub)

**Status**: Placeholder for future implementation.

## Overview

ASPIRE coding-agent loop for continual skill improvement:
1. **Collect traces** → execution logs (JSONL) from robot runs
2. **Analyze failures** → multimodal analysis (costmaps, trajectories, safety events)
3. **Generate repairs** → LLM/coding-agent proposes skill edits
4. **Test in sim** → validate in host-sim or Isaac before deployment
5. **Deploy** → update skill library (markdown + Python helpers)

## Architecture (Planned)

```
traces/run.jsonl
    ↓
[Trace Analyzer]  ← multimodal: costmaps, trajectories, safety reflexes
    ↓
[Failure Detector] ← stuck events, safety throttling, goal timeouts
    ↓
[Coding Agent] ← LLM (Gemini, Claude, GPT) + code generation
    ↓
skills/<skill_name>.md (updated)
skills/helpers/<skill_name>.py (new or updated)
    ↓
[Sim Validator] ← host-sim or Isaac Sim
    ↓
[Deploy] → git commit, notify human for review
```

## Trace Analysis

**Input**: `traces/run.jsonl` with events:
- `twist_cmd` — velocity commands (fwd, ang)
- `safety_state` — reflex type, scales, throttled flag
- `local_executive` — mode, goal, planner, active
- `stuck_detection` — detected, type, duration
- `skill` — invocations with params and outcomes
- `costmap_snapshot` — obstacle map summaries

**Output**: Failure signatures:
- Stuck: `stuck_detection=True` for >3s
- Oscillation: high `ang_rads` variance, low net progress
- Safety pinned: `fwd_scale < 0.15` for >5s
- Goal timeout: `local_executive.active=True` but `distance_to_goal` not decreasing

## Failure Detector

Patterns to detect:
1. **Stuck slip**: commands issued, no odometry progress
2. **Oscillation**: repeated small forward-backward cycles
3. **Pinned**: safety scales all motion to near-zero
4. **Skill failure**: skill invocation followed by stuck/timeout
5. **Reflex loop**: same reflex (e.g., `topdown_near_field`) firing repeatedly

## Coding Agent Prompt Template

```
You are a robotics coding agent refining skills for Kevin mobile base.

TASK: Improve skill `{skill_name}` based on execution trace failure.

TRACE SUMMARY:
- Event count: {event_count}
- Duration: {duration_s}s
- Failure mode: {failure_mode}
- Safety reflexes: {reflex_types}
- Stuck events: {stuck_count}

CURRENT SKILL:
{current_skill_markdown}

PROBLEM:
{failure_description}

INSTRUCTIONS:
1. Analyze the trace to identify root cause
2. Propose skill refinements (markdown updates or new Python helper)
3. Explain trade-offs and edge cases
4. Keep changes minimal and testable
5. Preserve existing safety constraints

OUTPUT:
- Updated skill markdown
- Optional: new/updated Python helper in skills/helpers/
- Test plan for sim validation
```

## Sim Validator

Before deploying to robot:
1. Load updated skill into sim (host-sim preferred, Isaac Sim optional)
2. Replay failure scenario from trace
3. Verify: skill succeeds, no new failures introduced
4. Run regression: existing skills still pass

Sim scenarios:
- Dog bed approach (soft low obstacle)
- Table overhang navigation (height clearance)
- Checkered mat avoidance (RGB detection)
- Person approach (social distance)
- Stuck recovery (back-out + replan)

## Deployment

After sim validation:
1. Git commit skill changes with descriptive message
2. Tag with version: `skill-v{version}-{skill_name}`
3. Notify human for review (PR or Slack)
4. If approved: merge to main, sync to robot

## Future Work

**Not included in scaffold:**
- Trace analyzer implementation (multimodal analysis)
- Failure detector heuristics
- LLM integration (Gemini API / Claude / GPT)
- Sim replay infrastructure
- Automated PR creation

**Related repos:**
- ASPIRE: https://github.com/NVlabs/ASPIRE (manip arms + CaP-X)
- CaP-X: multimodal trace → code generation
- Isaac Sim: NVIDIA sim platform (GPU-accelerated)

**Kevin-specific:**
- Host-sim: `sim/` directory (2D physics, RealSense mocks)
- Trace logger: `src/trace_logger.py`
- Skill library: `skills/`

---

**To enable**: Implement trace analyzer + coding agent + sim replay pipeline.
**Priority**: Start with manual trace analysis (human-in-loop) to validate approach.
