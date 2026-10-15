#!/usr/bin/env python3
"""v3.1 = v3 + (a) 1.75 mm root fillets in the XY profile where spokes meet band/tread (closing), sweep 7->8° to offset
fillet stiffening; (b) bore follows the stock crowned tread (gen_botwheel_tire_v17 CROWN) with uniform 0.4 mm
interference, band = crowned inner skin 1.6 + cylindrical outer skin (r 85.35-86.95) with two sealed lens cavities
(walls >=1.6, cavity roofs <=45°), replacing the v3 chamfer lip (the crowned bore itself locks axially)."""
import math, sys
import numpy as np, cadquery as cq
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import gen_softtire_v3 as g
P = g.P; P['sweep_deg'] = 8.0; RF = 1.75; INT = 0.4; SK = 1.6
CROWN = [(79.5,-7.2),(80.6,-9.2),(81.7,-11.2),(82.4,-13.2),(83.2,-15.2),(83.8,-17.2),(84.8,-21.2),(85.4,-25.2),(85.65,-29.5),
         (85.4,-33.8),(84.8,-37.8),(83.8,-41.8),(83.2,-43.8),(82.4,-45.8),(81.7,-47.8),(80.6,-49.8),(79.5,-51.8)]
zc = np.array([z for r, z in CROWN]); rc = np.array([r for r, z in CROWN])
def crown(z): return float(np.interp(-z, -zc, rc))
ROB = P['R_bore'] + P['band_t']   # 86.95 outer band (spoke roots)
def spoke_ext(th_end):
    """v3.2: v3 spoke centreline + straight tangent extension inward to r 78.5 (rib = same wall, no step)."""
    ra, rb = P['R_bore'] + P['band_t'] - 0.5, P['R_tread_in'] + 0.5; sw = math.radians(P['sweep_deg']); n = 16; pts = []
    for i in range(n + 1):
        s_ = i / n; r = ra + (rb - ra) * s_
        th = th_end - sw + sw * s_ + P['bulge'] * math.sin(math.pi * s_) / ((ra + rb) / 2); pts.append((r * math.cos(th), r * math.sin(th)))
    pts = np.array(pts); d = pts[0] - pts[1]; d /= np.linalg.norm(d)
    k = 0; ext = []
    p = pts[0].copy()
    while np.hypot(*p) > 78.5:
        p = p + d * 1.0; ext.append(p.copy())
    pts = np.r_[np.array(ext[::-1]), pts]
    t = np.gradient(pts, axis=0); t /= np.linalg.norm(t, axis=1)[:, None]
    nrm = np.c_[-t[:, 1], t[:, 0]] * P['sp_t'] / 2
    return Polygon([tuple(q) for q in np.r_[pts + nrm, (pts - nrm)[::-1]]]).buffer(0)
def section():
    ro = P['R_tread_in'] + P['tread_t']
    ring = lambda a, b: Point(0, 0).buffer(b, 64).difference(Point(0, 0).buffer(a, 64))
    u = unary_union([ring(P['R_tread_in'], ro)] + [spoke_ext(2 * math.pi * j / P['n_sp']) for j in range(P['n_sp'])])
    return u.buffer(RF, 4).buffer(-RF, 4).buffer(0)
def rev(pts): return cq.Workplane("XZ").polyline(pts).close().revolve(360, (0, 0, 0), (0, 1, 0))
def bore_void():
    zs = np.linspace(P['z0'], P['ztop'], 23)
    return rev([(0, P['zbed'] + 1), (crown(P['z0']) - INT, P['zbed'] + 1)] + [(crown(z) - INT, z) for z in zs] + [(crown(P['ztop']) - INT, P['ztop'] - 1), (0, P['ztop'] - 1)])
def skin():
    zs = np.linspace(P['z0'], P['ztop'], 23)
    inner = [(crown(z) - INT, z) for z in zs]; outer = [(crown(z) - INT + SK, z) for z in zs[::-1]]
    return rev([(crown(P['z0']) - INT, P['zbed'])] + inner + outer + [(crown(P['z0']) - INT + SK, P['zbed'])])
def tire():
    s = section(); H = P['ztop'] - P['zbed']
    polys = list(s.geoms) if s.geom_type == 'MultiPolygon' else [s]
    body = None
    for pg in polys:
        b = g.wp(P['zbed']).polyline(list(pg.exterior.coords)[:-1]).close().extrude(H)
        for hole in pg.interiors:
            b = b.cut(g.wp(P['zbed'] + 1).polyline(list(hole.coords)[:-1]).close().extrude(H - 2))
        body = b if body is None else body.union(b)
    body = body.cut(bore_void()).union(skin())
    for k in range(0, P['n_holes'], P['n_holes'] // P['n_tabs']):
        a = g.ha(k); c, s_ = math.cos(a), math.sin(a); r_in_edge = crown(P['z0']) - INT
        L = r_in_edge + 0.8 - P['tab_r_in']; rm = P['tab_r_in'] + L / 2
        tab = g.wp(P['zbed']).center(rm * c, rm * s_).rect(L, P['tab_w']).extrude(-P['tab_t']).rotate((rm * c, rm * s_, 0), (rm * c, rm * s_, 1), math.degrees(a)).edges("|Z").fillet(2.0)
        x, y = P['hole_r'] * c, P['hole_r'] * s_
        hole = g.wp(P['zbed'] + 1).center(x, y).circle(P['screw_clear'] / 2).extrude(-P['tab_t'] - 2)
        csk = g.wp(P['zbed']).center(x, y).circle(P['csk_d'] / 2).workplane(offset=-(P['csk_d'] - P['screw_clear']) / 2).circle(P['screw_clear'] / 2).loft()
        body = body.union(tab).cut(hole).cut(csk)
    return body
if __name__ == "__main__":
    b = tire(); cq.exporters.export(b, 'softtire_v3_2_tire_95a.step'); cq.exporters.export(b, 'softtire_v3_2_tire_95a.stl', tolerance=0.05, angularTolerance=0.15); print('wrote')
