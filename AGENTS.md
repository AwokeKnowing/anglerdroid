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
