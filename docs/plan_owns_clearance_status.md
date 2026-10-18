# Task A status (NOT merged into PR #62): MPPI owns clearance

Bench (sim, casita, GT gap = SIL12 hull vs mesh sliced every 1 cm over 0.03-0.95 m, gtfield v2):
| variant | goals | events/min | non-ok ticks | GT min / p5 gap | collisions (sim) |
|---|---|---|---|---|---|
| baseline pre1-3 (PR #62) | 7/12 | 10.24 | 3.3 % | -0.005 / 0.005 | 0 |
| 065ceb9 postA1-3 | 2/12 | 13.49 | 20.3 % | -0.005 / -0.005 | 1 |
| dd9e932 postC1-2 | 0/8 | 6.13 | 74.2 % | -0.005 / -0.005 | 1 |
| 6c6c18e postD1-2 | 1/8 (safety log lost) | - | - | - | 0 |
Regression not fixed. Every variant (baseline too) wedges at the kitchen island: its top (mesh z 0.684-0.744 m) is
seen by RS1 only at ~0.16 m from the axle, ends inside the 2D footprint (RS1 points at body (0.11, -0.13..-0.15),
1.6 cm inside), and is NOT in emu.envs collision/GT (4 slices at 0.08/0.35/0.63/0.90 m), so the sim lets the robot
under it. The planner stop envelope cannot help with an obstacle first perceived inside it; the baseline gets out
more often (7/12 vs 0-2/12) for reasons not isolated.

## r3 (sliced walls, n=3 valid runs, see PR #62 docs/bench_r3_2026-10-17.md)
goals 7/12 (current 6/12), 9.96 events/min (current 10.05), non-ok ticks 9.5 % (2.9 %), 3-D gap min -0.005 /
p5 0.049 m with 129/5017 samples < 3.5 cm (current 32/3885), 1 sim contact (dustpan, v 0.02 m/s, in-place turn).
Retreat recovery: 2 of 4 goals that needed a retreat arrived (current 1/3).  Still not mergeable: it drives closer.
