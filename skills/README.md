# Kevin Skill Library

ASPIRE-inspired continual learning: skills are code + text artifacts, NOT weights.

## Structure

Each skill is a markdown document with:
- **Description**: what the skill does and when to use it
- **Preconditions**: required state/context
- **Procedure**: step-by-step primitives or tool calls
- **Success criteria**: how to verify completion
- **Failure modes**: known issues and recovery strategies

Optional Python helpers in `helpers/` (imported by main loop or tools).

## Seed Skills

1. **dog_bed_soft_approach** — cautious approach to soft low obstacles (dog beds, cushions)
2. **overhang_front_strip** — safe navigation under tables/shelves with overhead clearance
3. **checkered_mat_keepout** — detection and avoidance of checkered mat hard-stop zones
4. **person_approach_social** — approach people at 4-5ft comfortable distance
5. **stuck_recovery** — detect and recover from stuck/oscillation states

## Evolution

Skills are refined through:
- Execution traces (JSONL logs → multimodal analysis)
- Coding-agent repair (see `AGENTS.md` stub)
- Human annotation of edge cases

New skills inherit from primitives or compose existing skills.
