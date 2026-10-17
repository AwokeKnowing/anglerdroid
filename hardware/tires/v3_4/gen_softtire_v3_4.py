#!/usr/bin/env python3
"""v3.4 = v3.2 + outboard tab set (mirror of inboard lip+8 tabs, 16 screws/tire) + root fillets (RF 1.75) where spokes meet the crowned inner skin. In v3.2 the XY fillet was only
applied against the outer tread (the skin is a revolve unioned afterwards), leaving sharp acute spoke/skin corners
(narrow wedge slivers) along the whole height. Fill is lofted through 24 z-stations following crown(z); everything else
identical (0.4 mm interference crowned bore, tabs, csk, screw clearance)."""
import math, numpy as np, cadquery as cq
from shapely.geometry import Point, Polygon
import gen_softtire_v3_2 as v
P = v.P; DZ = 1.0
def _gaps(z):
    s = v.section(); r = v.crown(z) - v.INT
    ring = Point(0, 0).buffer(r + v.SK, 512).difference(Point(0, 0).buffer(r, 512))
    u = s.union(ring).difference(Point(0, 0).buffer(r, 512))
    gap = u.buffer(v.RF, 8).buffer(-v.RF, 8).difference(u).intersection(Point(0, 0).buffer(r + v.SK + 6, 256))
    gs = [gg.buffer(0.15).buffer(-0.05) for gg in getattr(gap, 'geoms', [gap]) if gg.area > 0.05]  # sink 0.1 mm into skin/spoke
    return sorted(gs, key=lambda q: math.atan2(q.centroid.y, q.centroid.x))
def _resample(pg, n=48):
    ring = np.array(pg.exterior.coords)[:-1]
    if Polygon(ring).exterior.is_ccw is False: ring = ring[::-1]
    k = np.argmin(np.hypot(ring[:, 0], ring[:, 1])); ring = np.r_[ring[k:], ring[:k]]   # start at innermost point
    from shapely.geometry import LineString
    L = LineString(np.r_[ring, ring[:1]]); return [tuple(L.interpolate(t, normalized=True).coords[0]) for t in np.linspace(0, 1, n, endpoint=False)]
def fillet_fill():
    zs = np.r_[P['zbed'], np.linspace(P['z0'], P['ztop'], 23)]
    secs = [_gaps(min(max(z, P['ztop']), P['z0'])) for z in zs]
    n = len(secs[0]); assert all(len(q) == n for q in secs), [len(q) for q in secs]
    sol = []
    for i in range(n):
        wires = [cq.Wire.makePolygon([cq.Vector(x, y, z) for x, y in _resample(sec[i])], close=True) for sec, z in zip(secs, zs)]
        sol.append(cq.Solid.makeLoft(wires, True))
    return sol
ZM = (P['z0'] + P['ztop']) / 2   # tread mid-plane
def tire():
    t0 = v.tire().val(); fl = fillet_fill(); print('fill pieces', len(fl), flush=True)
    b = cq.Workplane().add(t0)
    for k, f in enumerate(fl): b = b.union(cq.Workplane().add(f), clean=False); print('u', k, flush=True)
    b = b.cut(v.bore_void())
    # outboard tabs: mirror the inboard 2 mm lip + 8 tabs (z zbed..z0) about the mid-plane -> z ztop..ztop-2,
    # same angles (stock wheel blind holes are the same 40 x r75.2 pattern on both faces)
    lip = b.intersect(cq.Workplane("XY").workplane(offset=P['z0'] - 0.01).circle(150).extrude(P['zbed'] - P['z0'] + 0.02))
    lip = lip.mirror("XY", (0, 0, ZM))
    return b.union(lip).clean()
if __name__ == "__main__":
    b = tire(); cq.exporters.export(b, 'softtire_v3_4_tire_95a.step'); cq.exporters.export(b, 'softtire_v3_4_tire_95a.stl', tolerance=0.05, angularTolerance=0.15); print('wrote')
