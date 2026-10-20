# Safety core vs. the time-to-intervene model (design note, 2026-10-20, PROPOSAL, nothing changed)

**Principle (James, 2026-10-20):** the safety core prevents collisions. It does not enforce safe speeds. The planner owns
slow speeds near obstacles; its margin is about 2x the safety margin. Each tick the core asks one question: if braking
started after a lead time, could the robot still stop before contact at its current speed? While the answer is yes,
the command passes untouched. When the remaining lead time drops to a threshold (about 100 ms), the core steps in and
forces the speed down fast. There is no blanket throttling near obstacles. Throttling of that kind conflicted with
wall-following and squeezing in past behaviour models. The escape creep cap (|v| <= 0.05, |w| <= 0.3 inside the margin)
was proposed and **rejected** under this principle.

`safety_core.py` and `governor.py` are not edited by this note. Everything below is a proposal for approval.

## 1. What the core does today (nav/safety_core.decide, nav/governor.governor)

| Rule | Today | Matches TTI? |
|---|---|---|
| Brake check | Sweep the commanded arc (or the measured one, if faster) for `need = v(T_TICK+T_CMD) + d_stop(v, ramp 5)` and pass if that sweep stays `MARGIN` (3.5 cm) clear of every point. Otherwise output zero and raise the ramp only as far as needed. | **Mostly.** It asks "can I still stop if I pass this tick?" and passes unchanged or zeroes (no gradual throttling). The difference is that the buffer is a **distance** (3.5 cm), not a **time**, and the built-in lead is only one tick plus latency (43 ms). |
| Rotation | Same check, using `need_angle(w)` on a spin in place. | Same as the brake check. |
| Hard bands | Any point in the 5 cm strip ahead of (or behind) the footprint means v toward it is 0, at any speed. | **No.** This is a speed-independent keep-out. At wall-follow or docking speed it vetoes motion that could stop in under 1 cm. |
| Escape | When `sd0 < MARGIN`, pass any command whose sweep min-sd `mn >= sd0`, at **any** speed. | **No, and unsafe.** It tests only the *global* minimum, so it is neither a collision test nor a speed rule (see section 3). |
| Operability | Frame age > 100 ms, missed cadence, or no depth: STOP. | Orthogonal. Keep it. |
| Governor (brain) | Scales (v, w) by 1 down to 0.1 so the sweep keeps 2x(need + MARGIN) clear, then rotate-only, then stop. The same `mn >= sd0` escape applies when inside the margin. | **Yes for scaling.** This is the planner owning slow speeds, with factor 2 = "planner margin about 2x safety". Its escape clause has the same flaw as the core's. |

Effective lead time of today's brake check. This is the time left before the core must act, i.e. `33 ms + MARGIN/v`:

| v (m/s) | need (m) | need + MARGIN (m) | equivalent lead (ms) |
|---|---|---|---|
| 0.05 | 0.007 | 0.042 | 733 |
| 0.10 | 0.015 | 0.050 | 383 |
| 0.20 | 0.032 | 0.067 | 208 |
| 0.30 | 0.051 | 0.086 | 150 |
| 0.45 | 0.085 | 0.120 | 111 |

These numbers are computed from the code's own brake fits (measured in sim, ramp 5). At planner speeds of 0.05 to 0.1 m/s, the
fixed 3.5 cm margin makes the core act 0.4 to 0.7 s early. That works as a distance keep-out and is what fights squeezing.
At 0.45 m/s it is already about 110 ms, which is close to the proposed rule.

## 2. Proposed rule (time-to-intervene)

For the commanded motion (or the measured one, if faster):

* `d_contact` is the arclength (or angle) along the sweep to the first pose where any point comes within `EPS` of the footprint.
  `EPS` covers perception and pose error only, not comfort. Proposed value: 1.5 cm, which is an **assumption**. It would be
  measured from the motion-compensation residual (frame age x speed) and the sim depth noise before it is adopted.
* `t_lead = (d_contact - v*T_CMD - d_stop(v, RAMP_MAX)) / |v|` is how long braking could still wait.
  For rotation, use the same formula with angle, `need_angle`, and `alpha_max`.
* If `t_lead > T_INTERVENE` (100 ms, plus one tick of decision period, i.e. checked as `>= T_INTERVENE + T_TICK`), **pass
  untouched**. Otherwise zero the command with the ramp raised as needed. This is the current brake path, unchanged.
* Equivalent pass distance: `v*(T_INTERVENE + T_TICK + T_CMD) + d_stop + EPS`. That is 2.5 cm at 0.05 m/s, 3.6 cm at
  0.1, 6.0 cm at 0.2, 8.6 cm at 0.3 and 13 cm at 0.45. It is about the same as today at speed, and much less at creep speed.
* **Bands** become ordinary points in the same test, so no speed-independent veto remains. One part is kept: the rear band's
  *visibility* learning. It still decides where the rear camera is blind, and blind rear space counts as occupied at
  the edge of visibility. *Decision for James:* keep the front/rear bands as a blind-spot guard only (points we cannot
  re-observe), or drop them entirely.
* **Escape → per-point collision test.** This replaces `mn >= sd0`. A command is *colliding* if, anywhere on its
  sweep up to `d_stop + lead`, **any point gets closer than `min(its own current sd, EPS)`**. In words: no point that is now
  outside EPS may come inside it, and no point that is already inside may get deeper. A colliding command is rejected at
  **any speed**. A non-colliding command is then handled by the t_lead rule, which passes it untouched at planner speed.
  There is no speed cap.
* The governor gets the same per-point clause in place of its `mn >= sd0` clause. Its 2x scaling stays as the planner's
  speed policy.

## 3. The in-contact full-speed spin (L15 wedge) under this model

Observed (L15, measured from the core log): `code = escape` passed w up to 1.57 rad/s and v 0.15 m/s with sd0 between -1.7 and -4.3 cm,
meaning a point was *inside* the footprint.

Why today's rule passes it (inferred from the code, not yet replayed per point): `mn` is the **global** minimum over all
points and all sweep poses. With the deepest point at -4.3 cm, a spin keeps that point at about -4.3 cm (it sits near the
rotation axis or on a convex corner), while *other* points 1 to 3 cm outside get swept into the hull. Their new depth,
for example -3 cm, is still >= -4.3, so `mn >= sd0` holds and the colliding spin passes at full speed.

Under the proposed per-point rule:
* A spin whose swept footprint takes any currently-outside point inside EPS, or pushes an inside point deeper, is
  colliding and is **rejected regardless of speed**. That holds whether it is a 0.3 rad/s creep or a 1.57 rad/s spin.
* A move that increases clearance for every close point (e.g. nav/escape.py's planned "rot -60°, back 12 cm" primitive
  chain, or a straight reverse out of the wedge) is **allowed at planner speed**, untouched.
* A point inside the footprint that cannot be real (self-mask leak, chassis returns) would block every motion. Today that
  case is hidden by the global-min rule. Under the new rule it must be logged as a code (`inside_fp`) so we fix the mask
  rather than the rule.

## 4. Proposed diff summary (for approval; not applied)

`nav/safety_core.py`
1. Add constants `T_INTERVENE = 0.10` (env `KEVIN_SC_T_INTERVENE`) and `EPS = 0.015` (env `KEVIN_SC_EPS`).
2. Change `_first_hit(m, steps)` so it can take a threshold. The default stays MARGIN for backward-compatible callers, and the
   core uses EPS.
3. Add `per_point_sweep(P, poses)`, which returns the (S, N) sd matrix (the vectorised `_sweep` without the `min`). Add
   `colliding(P, poses, eps) = any((sd_s < minimum(sd_0, eps) - 1e-3) & (sd_s < eps))`. Only points near the hull are
   evaluated, using the existing reach pruning.
4. `decide()`:
   * Band branch: replace the unconditional veto with "add band-strip points as occupied only where the band is blind"
     (pending James's band decision).
   * Translation and rotation: first `colliding(...)` over `need_tti + 0.02`, which gives code `collide` (new) and a zero
     output. Then pass if `free >= need_tti`, where `need_tti = need(v) + v*T_INTERVENE + EPS`. Otherwise `brake`.
   * Delete both `sd0 < MARGIN and mn >= sd0` escape clauses. The "escape" code is kept in CODES for log
     compatibility, now meaning "inside EPS, non-colliding, passed".
5. Add `"collide"` and `"inside_fp"` to CODES, appended so existing log indices are unchanged.
6. Delete the legacy unused `safety_core.governor()` (nav/governor.py is the live one), or leave it alone, as James prefers.

`nav/governor.py`
1. Same `colliding()` test on the brain point set in place of `mn >= sd0 - 1e-3`.
2. Keep 2x scaling but on `need_tti + MARGIN_PLANNER`, where MARGIN_PLANNER = 2 x EPS + 3.5 cm is today's planner comfort.
   The planner keeps owning slow speeds.
3. The property test (0/3000 disagreements with the core) is re-run on the new rule. Required: 0 disagreements.

`emu/tools/stuckdiag/escape_offline.py` (and a new `sc_tti_offline.py`)
* Replay L15, L14 and L17 core inputs through old and new `decide()` and report per-code counts, plus every tick where old
  passed and new rejects (and vice versa) with the point geometry.

## 5. Sim test plan (n >= 3 each; old rule vs. new rule, same seeds/starts; fresh world and nav per run)

Offline first (sim stopped, offline_gate):
* O1. Replay the L15 wedge ticks. Expected: every `escape` tick with |w| > 0.3 and a point swept inward becomes `collide`,
  and the escape.py plan's primitives stay `ok`/`escape`.
* O2. Replay L17, L14 and L8 to L13 (normal driving). Count ticks where new != old. Expected: new passes strictly more at
  v <= 0.15, and zero new passes that have a contact within 1 s in the world contact log.

Live (sim):
1. **Wedge** (L15 start 3.877, 1.566, yaw 1.238, "go to kitchen", KEVIN_ESCAPE=1). Metrics: world contact count and max
   penetration, time to clear, whether a full-speed spin with sd0 < 0 ever reaches the wheels (expected 0 under new),
   and success to goal.
2. **Wall-follow at planner speed.** Scripted drive 0.15 to 0.25 m/s parallel to the hallway wall at 6 cm, then 4 cm, clearance.
   Metrics: core interventions per metre, min clearance, world contacts (must be 0), and speed held vs commanded. Expected:
   old intervenes or bands at 4 cm, new passes untouched.
3. **Squeeze through a gap.** Two chairs (or env obstacles) at gap = footprint width + 4 cm, then + 2 cm, approached at
   0.1 m/s. Metrics: pass-through success, interventions, and world contacts (must be 0).
4. **Control: head-on approach** to a wall at 0.1, 0.3 and 0.45 m/s with no planner slow-down (raw command). Metric: stop gap
   at rest. It must be > 0 (no contact) and should be about EPS. This shows the core alone still prevents the collision.

Acceptance: 0 world contacts in tests 2 to 4, the wedge cleared with fewer contacts than old, and no frame-age regressions
(core p99 and max unchanged). Every result is reported as measured vs assumed (EPS and T_INTERVENE are assumptions until
the measurements above).
