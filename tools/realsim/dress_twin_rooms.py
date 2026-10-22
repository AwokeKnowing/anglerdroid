"""Dress the casita rooms the real frames have not shown yet (kitchen, hall, bedrooms, bathrooms, laundry) in
tools/realsim/dressing.json, so visual SLAM sees textured, furnished rooms -- all ASSUMED (not observed).

Rule: the RTX colour must agree with the sim depth (MuJoCo twin), so every solid piece here MIRRORS a twin geom
(emu/envs.py env_casita furniture + emu/realism.py casita_decor) at its exact pose/size; things the twin does not
have are surface-only (<= 1 cm proud: floor-zone textures, wall tile, wall art), so they cannot disagree with depth.
Idempotent: removes its own entries (name prefix "twin_") and re-adds them.
  usage: python tools/realsim/dress_twin_rooms.py [--dressing tools/realsim/dressing.json] [--assets ~/kevin_bench/m1/assets]
then rebuild the scene: tools/realsim/usd_scene.py --assets ... --out .../casita_slam.usda"""
import argparse, json, math, os, shutil, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO, os.path.join(REPO, "src")]
SRC = "ASSUMED (twin mirror): pose/size = MuJoCo twin geom (emu/envs.py env_casita / realism.casita_decor); look not observed"
SURF = "ASSUMED surface-only dressing (<= 1 cm proud, cannot disagree with sim depth); not observed"
EMU = os.path.expanduser("~/.cache/kevin_emu")
# CC0 ambientCG sets already fetched by emu/realism.py (pbr_src) + its procedural art (tex_real_v2)
MATS = {
    "twin_bedding": {"tex": "Fabric061_1K-JPG", "tint": [0.9, 0.9, 0.95], "tile": 0.8, "bump": 0.5, "roughness": 0.9},
    "twin_bedding2": {"tex": "Fabric026_1K-JPG", "tint": [0.8, 0.85, 1.0], "tile": 0.8, "bump": 0.5, "roughness": 0.9},
    "twin_floor_tile": {"tex": "Tiles107_1K-JPG", "tile": 1.2, "bump": 0.4, "roughness": 0.35},
    "twin_bath_tile": {"tex": "Marble012_1K-JPG", "tile": 0.6, "bump": 0.2, "roughness": 0.2},
    "twin_art0": {"tex": "TwinArt0", "normal": False, "tile": 0.8, "roughness": 0.5},
    "twin_art1": {"tex": "TwinArt1", "normal": False, "tile": 0.8, "roughness": 0.5},
    "twin_art3": {"tex": "TwinArt3", "normal": False, "tile": 0.6, "roughness": 0.5},
    "twin_poster": {"tex": "TwinPoster1", "normal": False, "tile": 0.5, "roughness": 0.6},
    "twin_cabinet": {"tex": "TwinCabinet", "normal": False, "tile": 0.8, "roughness": 0.5},
    "twin_book": {"tex": "TwinBook", "normal": False, "tile": 0.3, "roughness": 0.7},
    "twin_rug": {"tex": "TwinRug1", "normal": False, "tile": 1.0, "roughness": 0.95},
    "twin_switch": {"color": [0.85, 0.85, 0.82], "roughness": 0.4},
}
PNG = {"TwinArt0": "art0", "TwinArt1": "art1", "TwinArt3": "art3", "TwinPoster1": "poster1", "TwinCabinet": "cabinet", "TwinBook": "book", "TwinRug1": "rug1"}
FLOOR = {"kitchen_floor": "twin_floor_tile", "laundry_floor": "twin_floor_tile", "bath_floor": "twin_bath_tile"}
MAP = {"wood": "wood", "fabric2": None, "dark": "screen"}   # twin geom mat -> look (None: per-bed bedding)


def undressed(x, y):   # living room (real, dressed from frames) = south of the kitchen/hall line, west of the diagonal
    return y > 4.75 or (x > 3.4 and y > 3.0 and x + y > 8.3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dressing", default=os.path.join(REPO, "tools/realsim/dressing.json"))
    ap.add_argument("--assets", default=os.path.expanduser("~/kevin_bench/m1/assets")); a = ap.parse_args()
    from emu import envs
    import world_params as wp
    e = envs.build("casita", 0); ncnt = len(wp.counter_parts())
    D = json.load(open(a.dressing)); td = os.path.join(a.assets, "textures")
    for k in ("boxes", "furniture"): D[k] = [b for b in D.get(k, []) if not b["name"].startswith("twin_")]
    for k in list(D["materials"]):
        if k.startswith("twin_"): del D["materials"][k]
    # textures into the assets dir (outside git; sources: ambientCG CC0 via realism.fetch_pbr, realism procedural art)
    from PIL import Image
    for m in MATS.values():
        t = m.get("tex")
        if t in PNG:
            dst = os.path.join(td, t + "_Color.jpg")
            if not os.path.exists(dst): Image.open(os.path.join(EMU, "tex_real_v2", PNG[t] + ".png")).convert("RGB").save(dst, quality=92)
        elif t:
            sid = t.split("_")[0]
            for suf in ("Color", "NormalGL", "Roughness"):
                dst = os.path.join(td, "%s_%s.jpg" % (t, suf))
                if not os.path.exists(dst): shutil.copy(os.path.join(EMU, "pbr_src", sid, "%s_%s.jpg" % (t, suf)), dst)
    for k, m in MATS.items(): D["materials"][k] = dict(m, _src="ASSUMED look for undressed rooms (CC0 ambientCG / emu procedural art)")
    for z, look in FLOOR.items(): D["materials"][z] = dict(MATS[look], _src="ASSUMED tile (not yet seen in frames); was flat colour")
    boxes, nbed = [], 0
    for i, g in enumerate(e.geoms):
        if g.kind != "box" or 1 <= i <= ncnt: continue          # 0 = walls mesh; 1..ncnt = counters (usd_scene adds them)
        x, y, z = g.pos
        if not undressed(x, y): continue
        mat = MAP.get(g.mat, "wood")
        if g.mat == "fabric2": mat = "twin_bedding" if nbed == 0 else "twin_bedding2"; nbed += 1
        boxes.append(dict(name="twin_%s" % g.mat, c=[round(v, 4) for v in g.pos], size=[round(2 * v, 4) for v in g.size],
                          yaw_deg=round(math.degrees(g.yaw), 3), mat=mat, _src=SRC))
    dec = {"art0": "twin_art0", "art1": "twin_art1", "art3": "twin_art3", "poster1": "twin_poster", "cabinet": "twin_cabinet",
           "book": "twin_book", "outlet": "twin_switch", "switch": "twin_switch", "tile": "twin_bath_tile", "dark": "screen"}
    nplant = 0
    for kind, pos, size, yaw, mat, col in getattr(e, "decor", []):
        if not undressed(pos[0], pos[1]): continue
        if kind == "box" and mat in dec:
            boxes.append(dict(name="twin_decor_%s" % mat, c=[round(v, 4) for v in pos], size=[round(2 * v, 4) for v in size],
                              yaw_deg=round(math.degrees(yaw), 3), mat=dec[mat], _src=SRC))
        elif kind == "cylinder" and mat == "concrete":          # twin potted plant (pot + leaf spheres) -> CC0 potted plant
            D["furniture"].append(dict(name="twin_plant%d" % nplant, asset="ph/potted_plant_02/potted_plant_02_1k.usdc",
                                       pos=[pos[0], pos[1], 0], yaw_deg=0, _src=SRC)); nplant += 1
    # surface-only: bathroom wall tile wainscot (4 mm proud, 1.2 m), hall runner rug (3 mm, as the floor zones)
    for (x0, y0, x1, y1) in [z["poly"][0] + z["poly"][2] for z in wp.floor_zones()["zones"] if z["material"] == "bath_floor"]:
        for (cx, cy, sx, sy) in (((x0 + x1) / 2, y0 + 0.002, x1 - x0, 0.004), ((x0 + x1) / 2, y1 - 0.002, x1 - x0, 0.004),
                                 (x0 + 0.002, (y0 + y1) / 2, 0.004, y1 - y0), (x1 - 0.002, (y0 + y1) / 2, 0.004, y1 - y0)):
            boxes.append(dict(name="twin_bath_wainscot", c=[round(cx, 4), round(cy, 4), 0.6], size=[round(sx, 4), round(sy, 4), 1.2],
                              yaw_deg=0, mat="twin_bath_tile", _src=SURF + " (sits at the bath zone edge = wall line, ASSUMED)"))
    boxes.append(dict(name="twin_hall_runner", c=[2.0, 7.0, 0.0015], size=[0.8, 2.4, 0.003], yaw_deg=90, mat="twin_rug", _src=SURF))
    D["boxes"] += boxes
    json.dump(D, open(a.dressing, "w"), indent=1)
    print("twin_ boxes %d, plants %d, bedding %d -> %s" % (len(boxes), nplant, nbed, a.dressing))


if __name__ == "__main__":
    main()
