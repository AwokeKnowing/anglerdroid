"""Offline stuck diagnosis from a slamlog (KEVIN_SLAM_REC slamjobs.pkl + KEVIN_RECORD nav ticks), 2026-10-20 (L15 stall).
All replay the live Slam2 map up to time t (s since the first nav tick), at the recorded nav pose:
  map_view.py <log> <t> <out.png>        map crop + GT walls + live scan + footprint + trail
  governor_probe.py <log> <t>            nav.governor verdict for probe commands (live points / map memory / both)
  footprint_sweep.py <log> <t>           true-footprint clearance vs in-place rotation and small moves
  mppi_offline.py <log> <t> <gx,gy> [n]  the live MPPI (nav.plan.Planner) on the replayed map, closed loop n steps
                                         (DIAG=1 sample feasibility, DIAG3=1 global path bearings)
  se2_escape.py <log> <t>                SE(2) lattice search (10 deg / 4 cm, 2 cm footprint clearance) to a spin-free pose
run with PYTHONPATH=.:src:nav/kplan from the repo root."""
