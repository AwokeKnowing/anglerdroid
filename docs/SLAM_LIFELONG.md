# Lifelong kslam: consistency over many days of continual mapping (design, 2026-10-19)

Status: design only, nothing here is implemented yet. It follows James t600: kslam is good and fast, so keep its core
(wheel+gyro odometry, 2 cm log-odds grid, correlative scan matching (CSM) on the likelihood field, Manhattan heading) and
make it **trustworthy over days**. There are four parts: (1) reject bad data, (2) back out of bad data once it has got in,
(3) close loops on revisits, (4) lock stable structure so new data can't drag it.

## 0. Why today's kslam can't do this (nav/slam.py)
- **One global grid, written in place.** Every accepted scan is added to the same log-odds cells. A bad pose
  smears the map permanently: there is no record of which scan wrote which cell, so nothing can be undone.
- **Acceptance uses fixed constants, not measured error:** score > 0.5, score > median + 0.04, gain over the guess
  > 0.02, |d| < 6 cm / 1.5°, partial step k = 0.3, clamp 2 cm / 0.5°. Those thresholds are tuning, not statistics.
  They bound damage per scan but not over time: 100 slightly wrong scans still drag the map.
- **Correction is local only.** The search window is ±12 cm / ±4°. Drift beyond that on a long loop is never pulled back.
  Revisits re-paint the old place at the drifted pose (the wall "doubling" failure).
- **Every cell is equally plastic.** A wall seen for 3 days weighs the same as a chair seen once.
- **Session start:** `load` + `relocalize` exists (grid sweep + CSM refine, returns margin) but the gate is a constant.
  A wrong relocalization writes the new session straight into the old map.

## 1. Representation
- **Odometry stream** (unchanged, 30 Hz): pose_odom(t) plus a **covariance that grows with motion**.
  Measured model: σ_xy = a·dist + b, σ_θ = c·|rot| + d·dist + gyro-bias random walk. Fit a..d from logs vs GT;
  on the real robot fit them from loop-closure residuals, because there is no GT.
- **Submaps:** small rigid local grids (2 cm, same log-odds code). A new submap starts after ~3 m of travel or 90° of rotation
  (assumed, to be measured: long enough to be distinctive, short enough that odom drift inside it stays below 1 cell).
  Each submap is anchored to one **pose-graph node**. Scans are integrated into the active submap only.
- **Scan journal:** append-only, per accepted scan: (node id, t, pose in submap, compressed points, flags). Grids are
  a cache that can always be rebuilt from the journal. This is what makes rollback exact.
- **Pose graph:** nodes = submap anchors (plus scan keyframes inside the active submap). Edges = odometry (measured
  covariance), scan-to-submap matches, loop closures. Each closure edge has a **switch variable** so it can be turned off.
  2-D with a few hundred nodes per day: Gauss–Newton/LM solves in milliseconds.
- **Structure layer (anchors):** line segments (walls, big furniture faces) fitted from submap occupied cells, each with
  an endpoint covariance, an observation count, the sessions it was seen in, and a free-space-contradiction count.
- **Volatile layer:** everything else (chairs, the dog, people, doors). Used for planning and short-term matching.
  It **decays** (log-odds relax toward unknown with a half-life) and never anchors the graph.
- **Global map for nav** = render of submaps at their optimized poses + anchors. Published as map->odom (T) plus a
  confidence. Nav keeps using `pose_map = T ∘ pose_odom`, so the interface stays the same.

## 2. Bad-data detection (all gates are statistical; the one free choice is a false-reject probability)
Per frame, before matching (sensor sanity, cheap):
- frame age / stamp monotonic / no repeated frame id;
- depth valid fraction and floor-plane fit: RS2 floor normal vs the twin pitch must agree within its measured σ
  (catches a bumped camera or a tilted robot);
- wheel-vs-gyro yaw rate: the residual must fit the measured slip σ (catches wheel slip and a stuck wheel);
- IMU magnitude: catches pick-up and kidnap.

Per match:
- **Innovation test.** ν = match − odom prediction, S = P_odom(since last fix) + R_match. Accept if
  νᵀS⁻¹ν < χ²₃(1−α). α = 1 % is the single chosen number. R_match comes from the curvature of the CSM score
  surface (already computed over the window), calibrated so the normalized errors are ~χ² on GT logs.
- **Degeneracy:** reject the unobservable direction when the score surface is flat along it (corridor, single wall).
  Use the eigenvalues of R_match instead of the "flat" constant.
- **Fusion** replaces k = 0.3 and the clamps: a Kalman/graph update weighted by S, not a fixed fraction.

Over time (catches slow poison that passes the per-match tests):
- **NIS window:** mean normalized innovation over the last N matches. It must lie inside its χ² band; outside means the model is wrong.
- **Self-consistency of the active submap:** re-score the last K scans against the submap built WITHOUT them.
  A drop beyond the measured spread means the recent data disagree with older data.

## 3. States, checkpoints and backtracking
```
          ok matches                     innovation/NIS fail, or no fix for D m
  TRACKING ─────────────▶ TRACKING  ───────────────────────────────▶ DEAD_RECKONING
     ▲   │ every confirmed window: CHECKPOINT                           │ odom σ < map ambiguity: keep trying local CSM
     │   │                                                              │ σ too big / kidnap evidence (IMU, big ν with odom static)
     │   ▼ poison detected (sec. 2, over time)                          ▼
     │  ROLLBACK: drop journal entries after the checkpoint,         RELOCALIZING: frozen map, multi-hypothesis
     │  rebuild the active submap, re-run those scans in             global search (existing relocalize),
     │  DEAD_RECKONING mode (match only, no writes)                  accept only if margin > measured ambiguity
     └──────────────────────────────────────────────────────────── and 3 consecutive agreeing fixes
```
- **Writes are allowed only in TRACKING** with pose σ below half a cell (or a measured bound). Outside TRACKING, scans are kept
  in the journal as *pending*, but nothing goes into the grids.
- **Checkpoint** = (journal offset, graph state, T) taken when the last W scans all passed AND a later scan has re-observed
  their cells consistently. So a checkpoint is data that has been **confirmed**, not merely accepted. It is cheap: an offset plus a small graph.
- **Rollback** is exact: rebuild the active submap from the journal up to the checkpoint. That is one submap (~3 m of
  scans), so it takes milliseconds to ~100 ms. Earlier submaps are never touched; they are frozen once closed.
- **Kidnap:** IMU pick-up, or strong consistent matches at a pose outside the odom uncertainty while odometry says
  nearly stationary. Go to RELOCALIZING immediately. Do not let the matcher walk there.
- Nav gets the state plus σ. It drives on the map only in TRACKING, and on odom with a growing margin otherwise. That is the existing
  `decouple`/`map_ok` hook.

## 4. Loop closure on revisits
- **Candidates:** older submaps whose footprint is inside the current pose's uncertainty ellipse (from the graph), plus
  a coarse place descriptor (rotation-invariant range histogram of the submap) for revisits after large drift.
- **Match** the current submap (not a single scan: more structure, less aliasing) against the candidate with CSM over a
  window sized by the relative uncertainty (3σ). Report score, margin over the 2nd-best peak, and R from the curvature.
- **Verify:** (a) margin > measured self-similarity of that place (house symmetry), (b) innovation test vs the graph's
  relative covariance, (c) **temporal consistency**: ≥2 independent submaps give the same correction within σ.
- **Add** with a switchable constraint. Optimize. If the switch turns off (the closure disagrees with the rest), the edge is
  dropped and logged. A wrong closure can't permanently bend the map.
- Submaps move rigidly with their nodes, so the global render updates without re-integrating scans.

## 5. Locking in structure (walls can't be dragged)
- **Promotion:** a segment becomes an anchor once seen from ≥N distinct poses in ≥M sessions, with an endpoint σ below
  a bound and its direction consistent with the Manhattan frame (that estimator already exists). N, M and the σ bound are
  assumed until measured on multi-session logs.
- **Use:** anchors enter matching as a separate high-information term (point-to-line), and as graph landmarks. Strong
  but finite information: one session's error can't move them, but a consistent multi-session correction can.
- **Demotion (moved furniture, removed walls):** only by accumulated **free-space evidence** through the anchor (rays
  passing through it) over ≥K views in ≥2 sessions. The segment goes back to the volatile layer with hysteresis. Occupancy
  that merely appears (a box in front of a wall) never moves an anchor; it lands in the volatile layer.
- **Volatile decay:** the half-life is per class of evidence (seen-once vs seen-daily), measured from multi-session change logs.

## 6. Multi-session
- On boot: load anchors + submaps + graph, enter RELOCALIZING. The new session builds a **provisional** submap chain
  (own graph component) until a verified closure (sec. 4) links it to the old map. If none ever does, it stays a separate
  fragment and is merged later when one appears. Nothing from a new session writes into old submaps.
- **Compaction (nightly):** merge overlapping old submaps, drop journal points already summarized in anchors, cap disk.

## 7. Measured vs assumed
| item | value | status |
|---|---|---|
| wheel+gyro odom, L8 70 s | ATE 0.217 m, heading max 7.4° | measured (sim; real-robot odometry will be worse) |
| odom per keyframe step (0.25 m / 20°), sim | p50 0.9 mm / 0.10° | measured, sim only |
| kslam L8 | ATE 0.088 m / max 0.174 / 3.4° | measured |
| planar point-to-point ICP rotation under-estimate | 2–3 % of the rotation step, also on clean renders | measured (sec. 9) |
| odom covariance model a..d | – | to measure: sim GT + real loop residuals |
| CSM R from score curvature | – | to calibrate (NIS on GT logs) |
| α = 1 % false reject | – | the one chosen parameter |
| submap size 3 m / 90°; anchor N, M, K; decay half-life | – | assumed, to be measured |

## 8. Test plan
- **Data:** L8 (70 s) plus L9/L10/L11 (120 s, routes A/B/C with repeated kitchen/bedroom/couch revisits, recorded
  2026-10-19 19:03+) form the baseline test set. Multi-session = run L9, save, then L10 with load + relocalize, then L11.
  Add real-robot logs as soon as recording on the Orin is allowed (read-only capture).
- **Offline replay harness:** feed the logged nav ticks plus depth into the SLAM process at log rate. Faults are injected in replay, so
  every run is repeatable and compared against its fault-free twin.
- **Fault injection:**
  - corrupt frames: all-zero depth, random depth, depth scaled ×1.1, 5° extrinsic yaw/pitch, repeated or stale frame, stamp jump;
  - odometry: wheel slip (one wheel ×0.5 for 2 s), stuck wheel, gyro bias step 0.5°/s;
  - **kidnapped robot:** splice log B into log A at a different pose (and a live sim teleport);
  - **moved furniture:** sim world variant with a moved couch, table or chairs between sessions (needs an emu option);
    a door opened/closed; the dog walking through;
  - **multi-session restarts:** a cold start at a different pose; a start in an unmapped room (must stay provisional);
    power-cut mid-session (journal must recover).
- **Metrics:** ATE / max / heading vs GT; map wall p50/p90 (metrics.mapq); bad-data **false-accept** and false-reject
  rate; detection latency (s); rollback correctness (map IoU vs the fault-free run); loop-closure precision (must be
  1.0) and recall; anchor drift across sessions (cm); time to relocalize; CPU ms per scan, RSS.
- **Pass criteria (proposed):**
  - with any single fault, ATE ≤ fault-free + 2σ_odom over the fault window;
  - zero accepted wrong closures;
  - anchors move < 1 cell over 3 sessions;
  - runs real time on one i777 core.

## 9. Root cause of the depth-ICP −0.20°/keyframe yaw bias (checked 2026-10-19)
The hypothesis was a twin/sim mismatch. **There is none:**
- **Mounts:** MuJoCo camera poses vs twin_geometry (from the live model /dev/shm/kevin_emu_model.mjb) agree.
  RS1 differs by 0.045 mm and 0.000°; RS2 by 0.03 mm and 0.000°. Pitch is 27.0°; on L8 the RS2 true-mount randomization was off
  (rs2_true = nominal).
- **Intrinsics:** fovy 58.0° → fx = fy = 432.97 for both cameras, matching the logged streams. Principal point (w−1)/2,
  (h−1)/2 with integer pixel indices.
- **Render vs ray-cast:** rendered z minus mj_ray z at sampled pixel centres has a median of −0.03 mm on RS1 and −3.4 mm
  (−0.24 %) on RS2. There is no left/right asymmetry, so there is no half-pixel or fx error.
- **Timing:** logged depth vs a render at the GT pose matches best at Δt = 0 (tested ±17/33/66 ms on 4 frames), so stamps are aligned.
- **The ICP itself:** started at the GT increment, it under-estimates rotation by 2.8 % of the yaw step on logged depth and
  **2.0 % on noise-free renders**. The bias is in the point-to-point scan-to-keyframe ICP (partial-overlap pull toward
  zero motion), not in the sim. On L8 (spin-heavy, mostly one direction) that gives the mean −0.2°/keyframe.

Nothing was changed in the sim, so kslam results are unchanged. A fix belongs in the matcher: point-to-line or
likelihood-field (kslam-style) scoring, and scan-to-submap instead of scan-to-keyframe.
- **Side finding:** logged vs rendered depth agree < 2 cm on ~90 % of pixels when the robot is still, but only 27–63 % at
  0.45–0.5 m/s. The likely cause is body pitch during acceleration (the render used zero pitch); not yet verified.

## 10. Phase 1 status (2026-10-19 19:45 PT): offline replay, work in progress
Code: `nav/lslam/` (model.py: SE(2), odom model, chi2 gate, Kalman fuse; journal.py; submap.py; core.py: state machine),
`emu/tools/slamlog/lslam_replay.py` (calib / run + faults corrupt, rescale, slip, gyro), `tests/test_lslam.py`.
The tests pass: chi2 / SE(2), **exact submap rebuild from the journal after truncation**, and a wrong-place scan rejected → DEAD_RECKONING →
back to TRACKING within 5 frames. Not wired into nav. Loop closure / pose-graph optimization is phase 2; submap anchors are a chain.

Measured on L8, used as the training log (calib):
- **Odometry, per ~0.1 s increment:** sd_xy = 1.42 % of distance + 0.5 mm; sd_th = 1.57 % of rotation + 1.23°/m.
- **RS2 floor height:** median −0.7 mm, sd (MAD) 4.2 mm.
- **CSM error with a GT-built reference** (online submap policy): heavy-tailed. A χ²-trimmed core has sd 3.6 / 4.8 mm and 0.26°, but 39 % of
  matches fall outside it, and their median error is only 2.8 cm / 0.46°. That is not a few gross failures but a wide, real error.
  The suspected cause is body pitch while driving (it also showed up as the depth-vs-render mismatch in §9). The fraction outside the core
  is higher with bigger references (59 % at ≥30 scans vs 33 % below).

First L8 run with the trimmed R: ATE 0.172 m (kslam 0.088, odom 0.217). Only 23 matches were accepted, 607 rejected, and the run was 92 % in
DEAD_RECKONING. Mean NIS of accepted matches was 5.3 (should be 3), and 7 % of frames failed the sensor gate (should be 1 %). Both point
to the same thing: R and the floor sd are underestimated. **The ATE requirement (not worse than kslam) is not met yet.**
Next:
- R = the untrimmed sample covariance (the code is changed; the re-calib + rerun is in progress);
- consider a pitch-aware scan (floor-plane-referenced heights) so the error is smaller rather than just modelled;
- then the L9–L11 runs and the fault matrix.

### 10.1 Update 2026-10-19 20:00 PT
- **Scans are levelled per frame:** a robust floor-plane fit on RS2 (and RS1) gives tilt and offset, and heights are taken from that plane.
  The sensor gate is χ²(3) on the plane parameters vs their measured spread (L8: tilt sd 0.61° / 0.52°, offset 4.3 mm). A fit tilted > 11° counts as no floor.
  - Effect on the match error (untrimmed, online submap policy): sd 15.9 / 17.6 mm, 0.79° → 8.6 / 14.7 mm, 0.35°.
- **Submaps split by distance only:** splitting every 90° of rotation left new submaps with no overlap with the scan, so there was no reference.
- **No score peak = unmapped view, not a rejection.** It is written only if the pose is within the local match window.
- **The matcher step was too coarse:** lin/6 reached 10 cm against a 3 cm likelihood field, so most "flat" results were the matcher missing the peak.
  Now steps are 3 cm / 1° as in kslam, the window is capped at 0.3 m / 8°, and beyond that the state is RELOCALIZING (5 cm / 2°).
- **Odometry model is now random-walk:** variance grows with distance and rotation, fitted on 3 s windows; the intercept is GT stamp jitter and is dropped.
  The per-0.1 s fit was inflated by GT jitter.

## 11. Topological roadmap (design, 2026-10-19 21:00 PT; James 20:58 / 21:00) — prototype offline, not live

Idea: the metric map only has to be good LOCALLY.  Globally the robot keeps a graph that is only accurate enough to
plan routes; drift is reset at every node by relocalizing against that node's own stored local submap.  Live view
(RS1/RS2 + safety core, unchanged) does local planning and obstacle avoidance between nodes.

**Nodes** = junctions, doorways and room centres.  **Edges** = driven paths between nodes, each storing approximate
length, heading (snapped to the Manhattan frame when that is consistent with its uncertainty) and the pose covariance
accumulated along it (lslam odometry model, §7).  Node payload:
- local submap (lslam submap / occupancy patch, ±2.5 m) + its levelled scan — for scan relocalization on arrival;
- a few key images chosen for distinctiveness and coverage (greedy: max min-descriptor-distance to already chosen keyframes,
  across headings), with a global place descriptor each (DINOv2-SALAD, `~/.cache/torch/hub/checkpoints/dino_salad.ckpt`,
  dinov2_vitb14 already on disk);
- a bag of detected objects with counts (the detector LightNav/VLN already use; open-vocab if available), run on i777,
  never in the safety loop.

**Construction (offline prototype on L8–L12):**
1. Occupancy grid from the lslam/kslam poses + levelled scans (hits + ray-carved free).
2. Manhattan frame = dominant wall direction (orientation histogram of wall-cell gradients, mod 90°).
3. Free space eroded by the robot radius → skeleton → spurs shorter than the local clearance pruned (standard skeleton
   pruning; not a tuned length).  Junction = skeleton branch point (degree ≥ 3); doorway = clearance local minimum along an
   edge that pinches below the clearance of both adjacent regions; room centre = clearance maximum per region.
4. Edges = skeleton segments between nodes, kept only if driven (the trajectory passes along them).

**Use:**
- On arrival at a node (and for loop closure): candidate gate = place descriptor similarity (top-k nodes within the
  pose-covariance ellipse), then scan-to-submap relocalization (lslam matcher, χ² gate at ALPHA, §2); the accepted correction
  resets the pose to the node frame.
- Natural requests ("go to the couch"): objects/places → node by bag-of-objects + text match → route = Dijkstra on the
  graph (edge cost = length) → live local planner drives edge by edge.

**Metrics (sim, against the casita ground truth: wall STL section + `envs.casita` places):**
- graph correctness: nodes found vs the same pipeline on the GT wall grid (match within 0.5 m), wrong edges = edges
  crossing GT walls;
- relocalization at nodes: position/heading error before vs after, n / p50 / p95;
- routes between random node pairs: valid (collision-free on the GT grid, inflated by the robot radius) fraction;
- place recognition precision/recall at nodes on held-out drives; ~10 natural object/place queries, top-1 / top-3 correct node.

Status: design only; results go in §11.1.
