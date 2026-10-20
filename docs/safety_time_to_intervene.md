# Safety core vs. the time-to-intervene model (design note, 2026-10-20, PROPOSAL, nothing changed)

**Principle (James, t610/t611):** the safety core prevents collisions. It does not enforce safe speeds. The planner owns
slow speeds near obstacles; its margin is about 2x the safety margin. The core's question is bounded. Continue the current
command for the lead time, then brake gracefully to v=0, w=0 within the traction (friction-ellipse) limits. Sweep the
footprint along **that horizon only**. Intervene only if that braking trajectory contacts an obstacle; otherwise pass the
command untouched. Spins and straight or arc motion get the same test. "Eventually collides" is not a criterion, because every
forward command eventually collides. The escape creep cap (|v| <= 0.05, |w| <= 0.3 inside the margin) was rejected under this principle.

`safety_core.py` and `governor.py` are not edited. This note is a proposal for approval.

## 1. What the core does today (nav/safety_core.decide, nav/governor.governor)

| Rule | Today | vs. bounded braking-trajectory test |
|---|---|---|
| Brake check (arc and spin) | Sweep the commanded arc at constant curvature (the measured motion, if faster) for `need = v(T_TICK+T_CMD) + d_stop(v, ramp 5)`, where `d_stop = 0.0926 v + v²/(2·4.26)`. Pass if the sweep stays `MARGIN` = 3.5 cm clear; otherwise zero with the ramp raised as needed. A spin uses `need_angle` with `alpha_max = 4.26/(0.181) = 23.5 rad/s²`. | **Same structure.** It is already "continue, then brake, sweep only that horizon, pass untouched or zero" (no throttling). Three differences: (a) contact is defined as 3.5 cm, not about 0; (b) the brake is modelled at constant curvature, while the ODrive ramps each wheel separately (see section 2); (c) there is no explicit friction-ellipse check. |
| Hard bands | Any point in the 5 cm strip ahead of or behind the footprint makes v toward it 0, at any speed. | **Not bounded by the trajectory.** It is a speed-independent keep-out; at 0.05 m/s the braking trajectory reaches only about 0.6 cm. |
| Escape | When `sd0 < MARGIN`, pass any command with global sweep-min `mn >= sd0`, at any speed. | **Wrong test.** It uses the global minimum instead of per-point penetration change (section 3). |
| Operability | Frame age > 100 ms, missed cadence, or no depth: STOP. | Orthogonal; keep. |
| Governor (brain) | Scales (v, w) 1 down to 0.1 so the sweep keeps 2x(need + MARGIN) clear, then rotate-only, then stop. Uses the same `mn >= sd0` escape. | The scaling is the planner owning slow speed (factor 2 ≈ "planner margin 2x"). Keep it. The escape clause has the same flaw as the core's. |

### Lead time / threshold the current core implies, and the 30 Hz / 100 ms budget

Horizon modelled today, for continuing the command before the wheels decelerate:
* `T_TICK` = 33.3 ms. If this tick passes the command, the next decision can stop it at the earliest one frame later.
* `T_CMD` = 10 ms (assumed decision-to-CAN; logged as `t_sent - t_dec`).
* The actuator lag in the measured stop fit is 92.6 ms at ramp 5 (98.7 ms at ramp 3). This is physical: ramp shape and motor loop. It is not a choice.
* So the modelled "keep going" time is **≈ 43 ms decision lead + 93 ms actuator lag ≈ 136 ms**, then 4.26 m/s² decel.
* The 3.5 cm MARGIN adds the equivalent of `MARGIN/v` of extra lead: 700 ms at 0.05 m/s, 350 ms at 0.1, 117 ms at 0.3,
  78 ms at 0.45. The total equivalent lead is therefore 0.73 s, 0.38 s, 0.15 s and 0.11 s (computed from the code's brake fits, sim-measured).
  At planner speeds the margin, not the dynamics, decides. That is why the core acts like a distance keep-out when wall-following or squeezing.

Relation to the 100 ms budget: obstacle points reach the decision up to `AGE_MAX` = 100 ms old. Measured live:
frame-age p50 ≈ 52 ms, p99 75 to 87 ms, max 95 to 116 ms on the route. The points are motion-compensated to *now* with wheel odometry,
so frame age does not shorten the horizon. It does bound how late a *newly appearing* obstacle can enter the test
(≤ age + 1 tick ≈ 133 ms), and it adds compensation error ≈ odometry error × age. With a 100 ms reaction threshold,
the decision lead stays the 43 ms that the 30 Hz loop actually needs, and the 100 ms covers the rest
(frame-age slack plus a check one tick early). That means: intervene when continuing for `T_LEAD` = 100 ms, then braking, would
contact. 100 ms ≥ 43 ms always holds, so the core still acts at least one full tick before it is too late.

## 2. Proposed test (bounded braking trajectory)

For the current command (v_c, w_c), taking the measured twist where it is faster in the same direction:
1. **Trajectory.** Hold (v_c, w_c) for `T_LEAD` (default 0.10 s, env `KEVIN_SC_T_LEAD`; must be ≥ T_TICK + T_CMD). Then
   play the **measured brake response**: actuator lag, then each wheel ramps to 0 at the chosen ramp. The ODrive ramps
   each axis on its own, so on an arc the slower wheel stops first and the curvature changes during the stop. The brake
   is integrated per wheel (vl, vr), not along a constant-curvature arc. For a pure spin and for straight motion both reduce to today's
   formulas.
2. **Traction (friction ellipse).** Along the brake, require `sqrt(a_t² + (v·w)²) ≤ μg` with μ taken from the measured skid
   onset: ramp 8 skidded and ramp 5 did not, on carpet in sim. If ramp 5 would violate it (fast arc), use the largest
   ramp that does not; the horizon simply gets longer. The core never assumes it can brake harder than traction allows.
3. **Sweep** the footprint over that trajectory only, 1 cm / 0.02 rad steps as today.
4. **Contact** means any point comes within `EPS` of the footprint. EPS covers perception and pose error only, not comfort. Proposed 1.5 cm,
   an **assumption** to be measured: motion-compensation residual at p99 frame age plus sim depth noise.
5. **Decision.** If there is no contact on the horizon, **pass untouched**, whatever the speed. If there is contact, intervene now: zero, with the
   ramp raised as far as traction allows (current brake path). Nothing in between.
6. **Bands:** points in the band strips go into the same test, with no separate veto. Open question for James: keep the bands
   only as a *blind-zone* guard (unseen rear space treated as occupied at the visibility edge), or drop them.

Equivalent pass distance (straight line, ramp 5): `v·(T_LEAD + 0.0926) + v²/(2·4.26) + EPS` gives 2.5 cm at 0.05 m/s,
3.5 cm at 0.1, 5.8 cm at 0.2, 8.3 cm at 0.3 and 12.7 cm at 0.45. Today's values are 4.2 / 5.0 / 6.7 / 8.6 / 12.0 cm. The two are nearly identical
at speed and much smaller at creep speed.

## 3. In-contact case (L15 wedge)

Measured (L15 core log): `code = escape` passed w up to 1.57 rad/s and v 0.15 m/s with sd0 between -1.7 and -4.3 cm. A point was inside the
footprint, so stopping room was zero.

Under the bounded test, a robot already inside EPS has no horizon to spare. Any trajectory pose that *increases
penetration* is contact, and the command is rejected at any speed. Per-point rule over the same bounded horizon:
for every point, `sd_along(p) ≥ min(sd_now(p), EPS) - 1e-3`. No point outside EPS may enter it, and no point inside may get deeper.
A motion that *reduces* penetration for the near points passes untouched at planner speed. Examples are a straight reverse out of
the wedge, or nav/escape.py's planned rot -60° / back 12 cm chain.

Why today's rule passes the L15 full-speed spin: this is inferred from the code and has not yet been replayed per point. `mn` is the *global* minimum. The deepest
point (-4.3 cm) stays at about -4.3 cm during the spin while other points 1 to 3 cm outside are swept into the hull, for example to -3 cm.
Since -3 ≥ -4.3, the clause holds. The per-point rule rejects that spin and still allows the clearance-increasing exit. A point that
cannot be real (self-mask leak) would block every motion. It gets its own log code (`inside_fp`) so the mask gets fixed,
instead of a rule hiding it.

## 4. Proposed diff summary (for approval; not applied)

`nav/safety_core.py`
1. Add constants `T_LEAD = 0.10` (`KEVIN_SC_T_LEAD`, asserted ≥ T_TICK + T_CMD), `EPS = 0.015` (`KEVIN_SC_EPS`) and `MU_G`
   (traction limit from the skid measurement).
2. Add `brake_traj(v, w, v_m, w_m, ramp)`, which returns the poses for hold-for-T_LEAD followed by a per-wheel lag+ramp stop, with the
   ramp lowered until the friction ellipse holds. It replaces `need_dist` / `need_angle` arc sampling in `decide`, and the old functions stay for callers.
3. Add `contact(P, poses)` (per-point (S, N) sd matrix, the vectorised `_sweep` without the min). It returns True if any point violates
   `sd ≥ min(sd_now, EPS) - 1e-3` (pruned to points within reach + R_FOOT, as today).
4. `decide()`: one path for straight, arc and spin. Idle/coast stays as is. Then `contact(P, brake_traj(...))`: contact gives
   `brake` (or `collide` when sd0 < EPS) with zero output; otherwise `ok` (`escape` if sd0 < EPS, kept for log compatibility). The two
   `mn >= sd0` clauses and the unconditional band veto are removed (bands: per James's decision).
5. CODES: append `collide` and `inside_fp`, so existing indices are unchanged. Log fields: `free`/`need` become horizon length and contact arclength.
6. The legacy unused `safety_core.governor()` is deleted or left as is, as James prefers.

`nav/governor.py`
1. The same `contact()` on brain points, over `brake_traj` with the planner horizon (factor 2 on T_LEAD + EPS + 3.5 cm planner margin).
2. Scaling stays (planner-owned speed). The property test is re-run: 0/3000 governor-pass / core-reject disagreements required.

Offline tools: `emu/tools/stuckdiag/sc_tti_offline.py` (new) replays logged core inputs through the old and new `decide`.

## 5. Sim test plan (n ≥ 3 each, old vs. new rule, fresh world and nav per run)

Offline first (sim stopped, via offline_gate):
* O1. L15 wedge ticks. Expected: escape-passed spins that sweep points inward become `collide`, and escape.py plan primitives stay `ok`/`escape`.
* O2. L8 to L17 normal driving. Count new ≠ old. Expected: new passes more at v ≤ 0.15, and zero new passes have a world contact
  within the horizon (checked against the world contact log).
* O3. Brake-model check: predicted per-wheel stop poses vs. `emu.tools.brake_test` on arcs (0.3 m/s at κ = 0, 1, 3 1/m), to confirm the
  curvature drift during the stop.

Live:
1. **Wedge** (L15 start 3.877, 1.566, yaw 1.238; "go to kitchen"; KEVIN_ESCAPE=1). Metrics: contacts and max penetration,
   time to clear, number of penetration-increasing commands reaching the wheels (expected 0 under new), and goal reached.
2. **Wall-follow at planner speed:** 0.15 to 0.25 m/s parallel to the hallway wall at 6 cm, then 4 cm. Metrics: interventions per metre,
   commanded vs. achieved speed, min clearance, and world contacts (must be 0). Expected: old intervenes or bands, new passes untouched.
3. **Squeeze through a gap:** gap = footprint + 4 cm, then + 2 cm, at 0.1 m/s. Metrics: success, interventions, and contacts (must be 0).
4. **Control: raw head-on approach** (planner slow-down bypassed) at 0.1, 0.3 and 0.45 m/s, plus a 1.5 rad/s spin next to a chair leg.
   Metric: rest gap > 0 (≈ EPS). This shows the core alone prevents the collision.

Acceptance: 0 world contacts in tests 2 to 4, the wedge cleared with fewer contacts than old, and no frame-age regression in the core.
Every result is reported as measured vs. assumed. T_LEAD, EPS and μ remain assumptions until O3 and the noise and residual measurements are done.
