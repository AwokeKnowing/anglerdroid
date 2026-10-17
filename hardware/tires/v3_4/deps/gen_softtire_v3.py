#!/usr/bin/env python3
"""KevinFlex soft over-tire v3: FDM TPU 95A, one part, no pins, no separate ring (2026-10-12).
Wheel frame = BotWheel v17 tire frame (axis Z, mount face z=0, stock tire z -7.2..-51.8, outboard -Z), mm.
PRINT: flip so the INBOARD face (z=-5.2) is on the bed; print direction = -Z (wheel). No supports.
  - 8 inboard tabs (2.0 thick, z -7.2..-5.2, on the bed) at hole positions k=0,5,..,35, each with a 90° countersunk
    Ø5.8 hole for an ST5.5x16 DIN 7982 / ISO 7050 countersunk self-tapper into the Ø5 blind sidewall hole.
  - thin cylindrical band (1.6 mm) on the stock crown (0.3 mm interference at the crown peak).
  - N_SP vertical extruded spokes + 1.6 mm smooth tread, all bed-up extrusions.
  - outboard 45° chamfer lip at the top of the band (r 85.35 -> 81.35 over 4 mm) stops the tire walking inboard.
Usage: gen_softtire_v3.py OUTDIR [parts...]"""
import math, os, sys
from pathlib import Path
import numpy as np, cadquery as cq
P = dict(R_bore=85.35, band_t=1.6, R_tread_in=102.2, tread_t=1.6, n_sp=20, sp_t=2.0, sweep_deg=7.0, bulge=3.0,
         zbed=-5.2, z0=-7.2, ztop=-51.8, lip=4.0, hole_r=75.2, n_holes=40, n_tabs=8, tab_w=14.0, tab_r_in=69.5, tab_t=2.0,
         screw_clear=5.8, csk_d=9.8)
def ha(k): return 2 * math.pi * (k + 0.5) / P['n_holes']
def wp(z): return cq.Workplane("XY").workplane(offset=z)
def spoke_poly(th_end):
    ra, rb = P['R_bore'] + P['band_t'] - 0.5, P['R_tread_in'] + 0.5; sw = math.radians(P['sweep_deg']); n = 16; pts = []
    for i in range(n + 1):
        s = i / n; r = ra + (rb - ra) * s
        th = th_end - sw + sw * s + P['bulge'] * math.sin(math.pi * s) / ((ra + rb) / 2); pts.append((r * math.cos(th), r * math.sin(th)))
    pts = np.array(pts); t = np.gradient(pts, axis=0); t /= np.linalg.norm(t, axis=1)[:, None]
    nrm = np.c_[-t[:, 1], t[:, 0]] * P['sp_t'] / 2
    return [tuple(p) for p in np.r_[pts + nrm, (pts - nrm)[::-1]]]
def tire(sector=None):
    H = P['ztop'] - P['zbed']                       # negative: extrude toward -Z (print up)
    ro, rbo = P['R_tread_in'] + P['tread_t'], P['R_bore'] + P['band_t']
    body = wp(P['zbed']).circle(ro).circle(P['R_tread_in']).extrude(H)
    body = body.union(wp(P['zbed']).circle(rbo).circle(P['R_bore']).extrude(H))
    for j in range(P['n_sp']):
        body = body.union(wp(P['zbed']).polyline(spoke_poly(2 * math.pi * j / P['n_sp'])).close().extrude(H))
    # outboard chamfer lip (45°, grows inward toward the top of the print)
    zl = P['ztop'] + P['lip']
    lip = cq.Workplane("XZ").polyline([(P['R_bore'] + 0.1, zl), (P['R_bore'] - P['lip'], P['ztop']), (rbo, P['ztop']), (rbo, zl)]).close() \
            .revolve(360, (0, 0, 0), (0, 1, 0))
    body = body.union(lip)
    for k in range(0, P['n_holes'], P['n_holes'] // P['n_tabs']):
        a = ha(k); c, s = math.cos(a), math.sin(a); L = P['R_bore'] + 0.8 - P['tab_r_in']; rm = P['tab_r_in'] + L / 2
        tab = wp(P['zbed']).center(rm * c, rm * s).rect(L, P['tab_w']).extrude(-P['tab_t']).rotate((rm * c, rm * s, 0), (rm * c, rm * s, 1), math.degrees(a))
        tab = tab.edges("|Z").fillet(2.0)
        x, y = P['hole_r'] * c, P['hole_r'] * s
        hole = wp(P['zbed'] + 1).center(x, y).circle(P['screw_clear'] / 2).extrude(-P['tab_t'] - 2)
        csk = wp(P['zbed']).center(x, y).circle(P['csk_d'] / 2).workplane(offset=-(P['csk_d'] - P['screw_clear']) / 2).circle(P['screw_clear'] / 2).loft()
        body = body.union(tab).cut(hole).cut(csk)
    if sector:
        a0, a1 = math.radians(-sector / 2), math.radians(sector / 2); R = 130
        body = body.intersect(wp(P['zbed'] + 1).polyline([(0, 0), (R * math.cos(a0), R * math.sin(a0)), (R, 0), (R * math.cos(a1), R * math.sin(a1))]).close().extrude(H - 2))
    return body
if __name__ == "__main__":
    dst = Path(sys.argv[1]); dst.mkdir(parents=True, exist_ok=True)
    parts = {'softtire_v3_tire_95a': lambda: tire(), 'softtire_v3_test_seg60_95a': lambda: tire(60)}
    for name in (sys.argv[2:] or parts):
        s = parts[name](); tol = float(os.environ.get("TOL", 0.05))
        cq.exporters.export(s, str(dst / f"{name}.step")); cq.exporters.export(s, str(dst / f"{name}.stl"), tolerance=tol, angularTolerance=tol * 3)
        print("wrote", name)
