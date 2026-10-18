# PROPOSAL (unmerged): safety stopping model from measured values

Branch `safety-measured-model-on-pr62-2026-10-17` (= PR #62 head bdf631d + one commit). Sim only (emu.world casita,
KEVIN_EMU_MODEL=full, KEVIN_EMU_NOISE=realistic, world pid started 16:23 PT with slambench f9e2036). Not run on the robot.

## Change (nav/safety.py, nav/run.py)
| term | current code (ASSUMED) | proposal (MEASURED) | source of the measurement |
|---|---|---|---|
| braking decel | 1.5 m/s² constant | 1.614 m/s² = 3 turns/s² × 2π × 0.08565 m | ODrive vel_ramp (src/wheelbase.py); stop test travel matches 1.61 |
| reaction | 0.10 s constant | per tick: (now − frame capture) + frame interval + 0.04 s | frame age at decision p50 63 ms / p95 80 ms (this world); frame interval 66 ms median; 0.04 s = cmd→sim 0–7 ms + wheel lag ≈31 ms |
| staleness | `now=t_frame` (never fires) | `now=time.monotonic()` | – |
| margin | 0.035 m (code tolerance) | unchanged 0.035 m | – |

## Stop gaps (box head-on and arc v=0.6, w=−0.4, box offset DL), sim GT, 2026-10-17 18:00–18:10 PT
predicted gap at rest = margin + modelled stop distance − measured travel (decision-frame capture → rest)
| run | v | DL | tr model | a model | modelled stop dist | measured travel | predicted gap | measured gap at rest | min gap over rest of 7 s drive cmd | contact |
|---|---|---|---|---|---|---|---|---|---|---|
| current 0.30 head-on | 0.30 | – | 0.100 assumed | 1.5 assumed | 0.060 | 0.080 | 0.015 | 0.021 | 0.021 | 0 |
| current 0.45 head-on | 0.45 | – | 0.100 assumed | 1.5 assumed | 0.113 | 0.129 | 0.019 | 0.008 | 0.008 | 0 |
| current 0.60 head-on | 0.60 | – | 0.100 assumed | 1.5 assumed | 0.180 | 0.199 | 0.016 | 0.053 | 0.016 | 0 |
| current arc | 0.60 | 0 | 0.100 assumed | 1.5 assumed | 0.180 | 0.201 | 0.014 | 0.011 | 0.011 | 0 |
| current arc | 0.60 | +0.2 | 0.100 assumed | 1.5 assumed | 0.180 | 0.223 | −0.008 | 0.051 | 0.017 | 0 |
| current arc | 0.60 | −0.2 | 0.100 assumed | 1.5 assumed | 0.180 | 0.161 | 0.054 | 0.025 | 0.025 | 0 |
| proposal 0.30 head-on | 0.30 | – | 0.191 measured | 1.614 measured | 0.085 | 0.073 | 0.047 | 0.058 | 0.058 | 0 |
| proposal 0.45 head-on | 0.45 | – | 0.195 measured | 1.614 measured | 0.150 | 0.127 | 0.058 | 0.099 | 0.040 | 0 |
| proposal 0.60 head-on | 0.60 | – | 0.232 measured | 1.614 measured | 0.251 | 0.210 | 0.076 | 0.104 | 0.038 | 0 |
| proposal arc | 0.60 | 0 | 0.203 measured | 1.614 measured | 0.233 | 0.170 | 0.098 | 0.067 | 0.034 | 0 |
| proposal arc | 0.60 | +0.2 | 0.210 measured | 1.614 measured | 0.238 | 0.213 | 0.060 | 0.080 | 0.026 | 0 |
| proposal arc | 0.60 | −0.2 | 0.229 measured | 1.614 measured | 0.249 | 0.174 | 0.110 | 0.078 | 0.033 | 0 |

The perceived clearance at the decision tick was always SMALLER than GT (−1.7 to −6.3 cm, i.e. conservative) for this
box. The last column is the existing "inside margin and not getting closer" behaviour under a continued drive
command (creep after the stop); both versions creep, the proposal ends at 2.6–3.8 cm instead of 0.8–2.5 cm.
The arc DL +0.2 case that touched earlier did not touch today (1 cm wall skin; world restarted 16:23).

## Bench (normal planned driving) — confounded, see Task A notes
benchB1–2 (proposal): 5.29 events/min (stop 15, limit 3, no_depth 4), goals 1/8, collisions 0.
pre1–2 (current): 10.59 events/min, goals 7/8, collisions 0. The goal failures in both are at the kitchen island,
whose top (z 0.684–0.744 m) is perceived but is not in the sim's 4-height collision slabs; not attributable to the
safety change on n=2. The proposal does NOT yet meet "never triggers in normal driving" (it triggers 5.3/min).

## r3 bench (sliced walls, n=3, PR #62 docs/bench_r3_2026-10-17.md)
goals 8/12 (current 6/12), 14.41 events/min (current 10.05): overhang (island top, 2-D only) 60, no_depth 22,
sweep 28, perceived-at-margin 11, at-margin 1, real 0.  3-D gap min 0.045, p5 0.125, 0 samples < 3.5 cm
(current min 0.015, 32 samples).  The 22 no_depth stops are frames 0.25-0.71 s old that current code drives on
(up to 0.54 m/s, 0.175 m blind).  Not yet "never triggers in normal driving".
