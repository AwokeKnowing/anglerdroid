# casitahouse31-walls-laser-zfix.stl: the house model in use since 2026-10-21 (worlds/casita/world.json)
Source: sim/casitahouse31-walls-laser.stl (laser plan, unchanged x/y). Only these height levels moved (sim frame, floor = wall bottom = 0):
- wall tops / ceiling junction 2.280 -> 2.190 (298 verts)   [fitted c = 2.1897 +- 0.0034, per frame 2.193 / 2.191 / 2.187]
- pass-through header 2.160 -> 2.073 (19 verts)             [fitted 2.0728 +- 0.0094, per frame 2.076 / 2.077 / 2.069]
- level 2.140 -> 2.053 (28 verts)                           [ASSUMED: moved with the header, not measured]
- door heads AND window tops 1.880 -> 1.923 (250 verts)     [measured on the front door only: 1.9234 +- 0.0039, per frame 1.926/1.924/1.923;
                                                              window tops/closet doors ASSUMED to share the door-head level]
Unchanged (not measured): sills 0.682/0.744/1.278, counter 0.802 (image says 0.63 at the model plan position, see report), shelves etc.
Fit: 3 mast frames (12:13, 12:05, 09:52, same pose), camera z fixed 0.99115 (robot_params mast_rgb_height_m), plan scale 1, shared f/k1, per-frame pose5.

Heights FITTED (measured) vs ASSUMED: fitted = wall tops 2.1897, header 2.0728, front-door head 1.9234; assumed = the 2.053 level
(moved with the header), window tops / closet doors (= door-head level), and every height not listed above (sills, counter, shelves).
Approved by James 2026-10-21 20:17 PDT as the model for everything (sim, viewer, nav map, maploc, lineloc, casita_fit), to be refined
toward a fully measured model over time. Rollback/comparison: sim/casitahouse-walls-fixed.stl (= casitahouse26.blend, tops 2.28) stays in
the repo; set worlds/casita/world.json walls_stl back, or KEVIN_WALLS_STL=sim/casitahouse-walls-fixed.stl for one process.
Plan note: zfix keeps the v31 laser plan, which differs from the old file (v26) in the kitchen/laundry counters and one bedroom-1 north
detail, so the sim's 2D wall raster is NOT byte-identical to the old one (see the commit message for the cell counts).
