# Safety core: the designed-deceleration intervention rule (design note, 2026-10-20, PROPOSAL, nothing changed)

**Criterion (James, t610 to t612).** Safety prevents collisions; it does not enforce safe speeds. The planner owns slow speeds near
obstacles, with its margin about 2x the safety margin. The core must not pass a command that would force it to decelerate **harder than its
designed deceleration** to stop **before the safety margin**, after allowing for the reaction lead time. The designed deceleration is a friction ellipse
of **0.2 g longitudinal and 0.1 g lateral**. The core intervenes only when passing the command would violate this by the next tick, and then it brakes at the designed
deceleration. Every other planner command passes untouched. The test is not "contact", and it is not "eventually collides".
It is "can a designed-decel stop still finish outside the margin if I pass this tick?". Spins and straight or arc motion get the same test.
The escape creep cap was rejected.

`safety_core.py` and `governor.py` are not edited. Everything below is for approval.

## 1. Rule, stated precisely

If this tick passes (v, w), use the measured twist where it is faster in the same direction:
1. **Reaction lead `T_R`.** The command continues until the next decision can change it (one tick, `T_TICK` = 33.3 ms), plus decision-to-CAN
   (`T_CMD` = 10 ms, assumed; logged as `t_sent - t_dec`), plus the actuator lag at the brake ramp. The measured fit gives 98.7 ms at ramp 3, and this
   value must be re-measured at the design ramp. Total ≈ **142 ms** of "still moving at the command".
2. **Designed brake.** The core decelerates to (0, 0) within the ellipse `(a_long / 0.2g)² + (a_lat / 0.1g)² ≤ 1`. On an arc,
   `a_lat = v²κ` shrinks while braking. A pure spin is limited at the wheels: α = 0.2g / (track/2) = **10.8 rad/s²**. The designed ramp at
   the ODrive is 0.2g / (2π r_wheel) = **3.65 turns/s²** (nominal; r = 85.65 mm). Today's ramps are 3 (normal) and 5 (max).
3. **Sweep** the footprint along hold-for-`T_R`, then the designed stop, and nothing beyond it.
4. **Violation** means the sweep comes within `MARGIN` (3.5 cm, `SAFETY_MARGIN`) of any point.
5. **Decision.** If there is no violation, pass the command untouched at any speed. If there is a violation, the command would need more than the designed decel next tick, so
   intervene now: zero the command with the ramp at the designed decel, and the stop lands outside the margin.

Because the next tick re-runs the test, the core always acts at least one full tick before the designed stop becomes impossible.

## 2. Current core mapped onto the rule

| Item | Current core (`decide`) | vs. the rule |
|---|---|---|
| Structure | It sweeps hold-for-(T_TICK + T_CMD), then a measured stop along the commanded arc, and **passes unchanged or zeroes**. There is no throttling. | **Matches.** |
| Lead | `T_TICK + T_CMD` = 43 ms, plus the actuator lag inside the stop fit (92.6 ms at ramp 5), ≈ 136 ms. | Matches; the rule's ≈ 142 ms at the design ramp is about the same. |
| Margin | It stops the sweep at `MARGIN` = 3.5 cm. | **Matches.** |
| Pass-test decel | `need_dist` uses the **ramp-5 fit: 4.26 m/s² ≈ 0.43 g** (the "MAX" ramp). | **Differs.** It passes commands that need about 2x the designed decel to stop outside the margin. |
| Brake executed | Ramp raised from 3 up to 5 "as far as needed". | **Differs.** The rule brakes at the designed decel. With the pass test fixed, the core never needs more, and ramp 5 stays a fault reserve only. |
| Lateral limit | None. The constant-curvature brake is assumed, and the ODrive ramps each wheel separately. | **Missing.** Add the ellipse, and integrate the brake per wheel (on an arc the slower wheel stops first). |
| Spin | `need_angle` with α = 4.26 / 0.181 = 23.5 rad/s² (ramp-5 basis). | Same structure; it should use **10.8 rad/s²**. |
| Hard bands | Any point in the 5 cm strip ahead of or behind the footprint gives v toward it = 0, **at any speed**. | **Differs.** This is a distance keep-out. Under the rule a band point is an ordinary point, and a command passes if a designed stop finishes outside the margin. |
| Escape (sd0 < MARGIN) | It passes any command whose **global** sweep-min `mn ≥ sd0`, at **any** speed. | **Differs** (section 3). |
| Operability (age > 100 ms, cadence, no depth) | STOP. | Orthogonal; keep. |
| Governor (brain) | It scales (v, w) so the sweep keeps 2x(need + MARGIN) clear, and uses the same `mn ≥ sd0` escape. | The scaling is the planner owning slow speeds ("planner margin 2x"). Keep it, on the new `need`. The escape clause changes as in the core. |

Stop distance plus margin, current (4.26 m/s² basis) vs. rule (0.2 g, `T_R` with 98.7 ms lag), straight line. These are computed from the code's sim-measured fits.
The lag at the design ramp is **assumed** equal to the ramp-3 lag.

| v (m/s) | current (m) | rule (m) | difference (m) |
|---|---|---|---|
| 0.05 | 0.042 | 0.043 | +0.001 |
| 0.10 | 0.050 | 0.052 | +0.002 |
| 0.20 | 0.067 | 0.074 | +0.007 |
| 0.30 | 0.086 | 0.101 | +0.014 |
| 0.45 | 0.120 | 0.151 | +0.031 |
| 0.60 | 0.159 | 0.212 | +0.053 |

Spin stop angle at design α: 4.7° at 0.5 rad/s, 10.8° at 1.0 and 19.3° at 1.57. At planner speeds the rule is nearly identical to
today. At speed it intervenes earlier, as designed. Wall-following and squeezing conflicts at low speed therefore come from the **bands** and the **escape**
rule, not from the brake test.

### Lead time vs. the 30 Hz / 100 ms budget
* The 30 Hz loop sets the decision lead at 33 ms + 10 ms. The actuator lag, about 0.1 s, is the largest term, and it is physical.
* Frame age: points are up to `AGE_MAX` = 100 ms old at decision. Measured on the route: p50 about 52 ms, p99 75 to 87 ms, max 95 to 116 ms.
  They are motion-compensated to *now* by wheel odometry, so age does not shorten the horizon. It does (a) delay a *newly appearing* obstacle by ≤ age + 1
  tick ≈ 133 ms, and (b) add compensation error. Proposal: no extra lead in the rule. Point uncertainty is covered by MARGIN, and the
  compensation residual at p99 age is measured (assumed below 1 cm at 0.3 m/s × 0.087 s × a few % odometry error) and must stay well inside MARGIN.

## 3. Inside the margin (L15 wedge, in-contact spins)

Measured (L15 core log): `code = escape` passed w up to 1.57 rad/s and v 0.15 m/s with sd0 between -1.7 and -4.3 cm, meaning a point was inside the
footprint. There is **no stopping room**: no designed stop can end outside a margin the robot is already inside. The rule therefore becomes,
over the same hold+designed-brake horizon:
* **Fail:** any motion that **reduces clearance** of any point already inside the margin, or brings a point from outside into it. That is,
  for each point, `sd_along(p) ≥ min(sd_now(p), MARGIN) - 1e-3`. A fast spin fails if it sweeps other points inward, at any speed.
* **Pass, untouched, at planner speed:** a motion that does not reduce any near point's clearance. Examples are a straight reverse out of the wedge,
  or nav/escape.py's planned rot -60° / back 12 cm chain.

Why today's rule passes the L15 spin (inferred from the code; per-point replay is test O1). `mn` is the *global* minimum, so the deepest point
(-4.3 cm) stays put while other points 1 to 3 cm outside are swept to about -3 cm, and -3 ≥ -4.3 passes. A point inside the footprint that cannot be real
(self-mask leak) would block everything. It is logged as `inside_fp` so the mask gets fixed instead of the rule hiding it.

Lateral 0.1 g on commands: if the planner commands an arc with v·w > 0.98 m/s², even the start of a designed brake leaves the
ellipse. Safety does not throttle that, because it is not its job. It brakes along the ellipse boundary (the longitudinal limit shrinks to fit), so the sweep is longer and
intervention comes earlier. **Question for James:** should such commands also be flagged to the planner (code `envelope`)?

## 4. Proposed diff summary (for approval; not applied)

`nav/safety_core.py`
1. Add `A_LONG = 0.2*9.81` and `A_LAT = 0.1*9.81` (envs `KEVIN_SC_A_LONG/LAT`), and `RAMP_DESIGN = A_LONG/(2π·WHEEL_R_M)`
   (3.65 turns/s²). Add a `BRAKE[RAMP_DESIGN]` fit from `emu.tools.brake_test`; until it is measured, assume the lag is the ramp-3 lag and the decel is A_LONG.
2. Add `brake_traj(v, w, v_m, w_m)`: hold for T_TICK + T_CMD, then actuator lag, then a per-wheel ramp stop, with a_long limited per step by
   `A_LONG·sqrt(1 - (a_lat/A_LAT)²)`. It returns the poses. It replaces the `need_dist`/`need_angle` arc sampling inside `decide` (the old
   functions stay).
3. Add `violates(P, poses)`: per-point (S, N) sd, true if `sd < min(sd_now, MARGIN) - 1e-3` anywhere. It uses the existing reach pruning.
4. `decide()`: one path for straight, arc and spin. `violates` gives `brake` with zero at `RAMP_DESIGN` (`collide` when sd0 < MARGIN).
   Otherwise `ok` (`escape` when sd0 < MARGIN, for log compatibility). Remove both `mn ≥ sd0` clauses and the unconditional band veto
   (band points join P; the rear band's visibility edge stays as an occupied blind-zone line, if James agrees).
   The ramp for coasting/idle braking also becomes `RAMP_DESIGN`. `RAMP_MAX` stays only for inop (stale frames, dead brain).
5. Append CODES `collide`, `inside_fp` and `envelope`, so the log indices are unchanged.

`nav/governor.py`
1. The same `brake_traj` + `violates` on brain points, with factor 2 on the horizon (the planner margin). The scale list stays (planner-owned speed).
2. Re-run the governor/core property test; 0/3000 governor-pass/core-reject is required.

Tools: `emu/tools/stuckdiag/sc_rule_offline.py` (new) replays logged core inputs through the old and new `decide`.

## 5. Sim test plan (n ≥ 3 each; old vs. new; fresh world and nav per run)

Offline (sim stopped, offline_gate):
* O1. L15 wedge ticks. Expected: escape-passed spins that sweep points inward become `collide`, and the escape.py plan stays `ok`/`escape`.
* O2. L8 to L17 normal driving. List every tick where new ≠ old, with v, w and geometry. Expected: new intervenes earlier only at v ≥ 0.2 and
  passes band-only cases at creep speed.
* O3. Brake model: `brake_test` at RAMP_DESIGN (straight 0.1 to 0.6 m/s; arcs κ = 1, 3 1/m; spins 0.5 to 1.57 rad/s). This fits the lag and decel, and checks the
  per-wheel curvature drift and peak |a_long|, |a_lat| against 0.2 g / 0.1 g.

Live:
1. **Wedge** (L15 start 3.877, 1.566, yaw 1.238; "go to kitchen"; KEVIN_ESCAPE=1). Metrics: contacts, max penetration, time to clear,
   clearance-reducing commands reaching the wheels (expected 0), and goal reached.
2. **Wall-follow at planner speed:** 0.15 to 0.25 m/s parallel to the hallway wall at 6 cm, then 4.5 cm, clearance. Metrics: interventions per metre,
   commanded vs. achieved speed, min clearance, and contacts (must be 0). Expected: the old rule bands or brakes, the new rule passes untouched.
3. **Squeeze through a gap** of footprint + 2 × MARGIN + 1 cm at 0.1 m/s. Metrics: success, interventions, and contacts (must be 0).
4. **Control: raw head-on** approaches (planner slow-down bypassed) at 0.1, 0.3 and 0.45 m/s, plus a 1.5 rad/s spin beside a chair leg. Metrics:
   rest gap ≥ MARGIN, and peak measured decel ≤ 0.2 g (no harder braking was ever needed).

Acceptance: 0 contacts in tests 2 to 4, rest gap ≥ MARGIN in test 4, peak brake ≤ 0.2 g / 0.1 g, the wedge cleared with fewer contacts than old,
and no core frame-age regression. All results are reported as measured vs. assumed. The design-ramp lag, the odometry residual and the lateral behaviour remain assumptions until O3.
