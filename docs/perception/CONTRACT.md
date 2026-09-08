# Kevin Perception Contract (2026-09-07)

Mission: from raw RS1 (top-down) + RS2 (forward) + RGB, every frame publish
what is **non-self obstacle** vs **definitely clear**, in ≤20 ms on Orin,
then accumulate a drivable map, then dynamic-world SLAM.

## Per-frame ego labels (1 cm/px, 320×240, axle at RCX,RCY, +X forward)

Each cell is exactly one of:

| Label | Meaning | May drive? |
|-------|---------|------------|
| `UNKNOWN` | No trusted depth evidence this frame | No (unless map prior says clear) |
| `SELF` | Robot volume (body/wheels/mast). Not obstacle, not clear | N/A |
| `CLEAR` | Floor evidence — definitely free | Yes |
| `OBSTACLE` | Non-self surface above floor band | No |

**Hard rules**
1. `SELF` wins over everything. Geometric axle boxes (body 30×33, wheels 18×6×2, mast) are applied in ego after scatter. Self pixels must never enter obstacle or clear.
2. `CLEAR` is only from sensed floor (RS1 z ≥ floor_clip, in trust FOV). Never invent clear by punching the footprint.
   Dual `(obs,known)` compat: SELF encodes as `obs=0, known=0` (not known-clear).
3. `OBSTACLE` is only non-self. Height = floor_clip − z (cm), tallest wins.
4. RGB is not on the 30 Hz clear/obstacle critical path (faces/hazards ~3 Hz).
5. Budget: perception stages (RS1 project+label + RS2 project+label + fuse + self) ≤ **20 ms**. Grab/odom/render are outside this budget but must not starve it.

## Why the current stack fails the red-box dump

Today `obs`/`known` collapses three ideas:
- known∩¬obs → “free” (but under-robot **force-sets** known=255, obs=0 → fake CLEAR)
- known∩obs → obstacle (includes **self** until boxes clear it; leftover self stays black)
- ¬known → unknown

So the map lies in two directions: pretends clear under the chassis, and lets mast/body/wheel returns that miss the boxes pollute obstacles. James’s red box shows mixed black/yellow *inside* the robot — that is self leakage + invented clear, not a visualization bug.

## Accumulation (world map)

- Evidence grid: clear_evidence, obstacle_evidence, last_seen (dynamic decay).
- `SELF` never adds obstacle evidence.
- Dynamic objects: obstacle evidence decays when not re-observed; clear can reclaim.
- Do not assume static furniture forever (dog, people, chairs).
- Landed: `src/perception/evidence_map.py` (`EvidenceMap`). Per-update obstacle
  decay on cells not refreshed; clear decays slower and actively pulls obstacle
  evidence down on CLEAR hits so movers can free a cell. Optional live wire:
  `KEVIN_EVIDENCE_MAP=1` (default off) → `ego_ev:` metrics in vision.
- Gated planner feed (default off): `KEVIN_EGO_PLAN=1` and/or `KEVIN_EGO_LABELS=1`
  → costmap/planner/safety see ego-derived `(obs,known)` (`ego_plan: source=ego|evidence|legacy`).
  With `KEVIN_EVIDENCE_MAP=1`, prefer pose-warped `EvidenceMap.to_obs_known` in ego.
  SELF stays honest (not known-clear, not obstacle). Does **not** complete SLAM.

## SLAM (after ego labels are honest)

- Modern VO/SLAM that treats moving people/dog as outliers (robust residuals / dynamic masking).
- Prefer RGB-D + wheel + IMU; no ROS.
- Localization must not require a frozen house mesh, but a walls prior (STL) is a later assist.
- Wedge landed: `src/perception/dynamic_mask.py` — ego-label outlier mask for SLAM/VO
  (`SELF` always masked; ephemeral `OBSTACLE` vs pose-warped EvidenceMap prior
  when `KEVIN_EVIDENCE_MAP=1` has updates). Live gate `KEVIN_SLAM_DYNAMIC_MASK=1`
  (default off) zeros masked cells on self-SLAM keyframe obs; live
  ``slam_mask:`` metrics (SELF / ephemeral / ms) compute whenever ego
  labels fire — including ``--no-wheelbase`` when keyframe writes skip.
  Same gate also optionally zeros VO gray pixels whose RS2 depth samples
  project into masked ego cells (`apply_ignore_to_gray` /
  `build_forward_ignore_from_verts`; ``vo_ignore:`` metrics). Color UV is a
  depth-grid remap (not `rs.align`) — gap noted; still no invented CLEAR.
  Full RGB-D+wheel+IMU dynamic-tolerant SLAM still TODO.

## Delivery order

1. Honest ego labeler + self exclusion — landed (RS1 `label_rs1_ego` + SELF boxes)
2. Fuse RS2 without false clear; keep ≤20 ms — `fuse_rs2_into_ego` (cone+free-range CLEAR; SELF wins; no under-chassis CLEAR)
3. Accumulated drivable map with decay — landed (`EvidenceMap` / `evidence_map.py`; dynamic obstacle decay + clear reclaim)
4. Dynamic-tolerant SLAM — wedge: dynamic mask from ego (`dynamic_mask.py`); full RGB-D+wheel+IMU still TODO

