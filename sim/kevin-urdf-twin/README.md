# Kevin — MakerBeam XL differential-drive URDF twin (v2)

Approximate but recognizable digital twin of James's MakerBeam XL diff-drive robot.
Primitives only (box/cylinder). Units: **meters**. Frame: **x forward, y left, z up** (REP-103).

## Quick view (phone-friendly)

From the project root:

```bash
cd /workspace/kevin-urdf-twin
./serve.sh
# or: python3 -m http.server 8080
```

Then open:

- Local: http://localhost:8080/viewer/
- Phone (same LAN): http://&lt;host-ip&gt;:8080/viewer/

Drag to orbit · pinch/scroll to zoom · toolbar: reset / axes / grid.

## Layout

```
kevin-urdf-twin/
├── urdf/kevin.urdf      # robot model (stock beams + XL L-brackets)
├── viewer/index.html    # three.js + urdf-loader (CDN)
├── meshes/              # optional STLs (none required for v2)
├── refs/makerbeam-xl/   # XL bracket drawing refs (Drive may need auth)
├── previews/            # offline / headless previews
├── serve.sh
└── README.md
```

## MakerBeam XL stock rule

Cross-section: **15 × 15 mm (0.015 × 0.015 m)**.

**Allowed lengths only:** 300, 200, 150, 100, 50 mm.  
Every `beam_*` link in the URDF is exactly one of these (no cut-to-length custom sizes in v1).

### Beam BOM (v2)

| Stock length | Qty | Links |
|-------------:|----:|-------|
| **300 mm** | **9** | `beam_base_front`, `beam_base_rear`, `beam_base_mid`, `beam_axle_top`, `beam_top_front`, `beam_top_rear`, `beam_mast_lower`, `beam_mast_mid`, `beam_mast_upper` |
| **200 mm** | **0** | — (was mast upper; replaced by third 300) |
| **150 mm** | **5** | `beam_base_left_aft`, `beam_base_right_aft`, `beam_top_left_aft`, `beam_top_right_aft`, `beam_mast_top` |
| **100 mm** | **8** | `beam_base_left_fwd`, `beam_base_right_fwd`, `beam_top_left_fwd`, `beam_top_right_fwd`, `beam_riser_fl/fr/rl/rr` |
| **50 mm** | **0** | — (reserved for future) |

**Total MakerBeam XL pieces in model: 22** (+ `beam_axle_top`)

Resulting assemblies:

- **Base cage** (lower + upper tiers): width **300 mm**, rail span **250 mm**, outer length **280 mm** (see join style below)
- Four sets of **150+100** longitudinal rails: left/right × lower/upper
- Risers: four **100 mm** corner posts between tiers
- Mast height = **300 + 300 + 300 = 900 mm** + **150 mm** horizontal top bar (tip plumb with drive axle)

### Hardware brackets (not beam stock)

| Part | Qty | Notes |
|------|----:|-------|
| **XL flat L-bracket** (James 60×30×15×~2 mm) | **4** | `bracket_xl_{left,right}_{upper,lower}` — wheel mounts only |
| Placeholder silver corner cubes | **0** | **Removed** in v2 |

Stock MakerBeam catalog **101732** is an equal-leg ~45×45 right-angle; James’s install uses the **unequal 60/30 flat L** from his XL T+L drawing (15 mm section). Plate thickness **1.5 mm (James)**.

### Base join style (document — easy to flip)

**300 mm beams = outside end-caps** (front and rear). They define the **300 mm** base width (along Y).

The **150 + 100 mm** longitudinal rails are **sandwiched BETWEEN** those two 300s: they butt against the **inner faces** of the front and rear end-caps (they do **not** extend beyond the 300s).

| Measure | Value |
|---------|-------|
| Rail span (inner face → inner face) | **250 mm** = 100 + 150 |
| End-cap thickness (each) | **15 mm** |
| Outer length (outer face → outer face) | **280 mm** = 250 + 2×15 |
| Front outer / inner face | **+0.115 m** / **+0.100 m** |
| Rear outer / inner face | **−0.165 m** / **−0.150 m** |

If James's physical join is rails-outside instead, flip the end-cap origins by one beam thickness (±0.015 m).

### Orientation assumption (easy to flip later)

Along **+X (forward)**:

- **100 mm** segment **forward** of the axle join
- **150 mm** segment **aft** of the axle join (toward rear caster)
- Axle exactly at the **100\|150 join** (`x = 0`)

## Key dimensions

| Item | Value | Source |
|------|-------|--------|
| MakerBeam XL cross-section | **0.015 × 0.015 m** | Spec |
| Base width (Y, end-caps) | **0.300 m** | James (hard) |
| Rail span (X, sandwiched) | **0.250 m** | 100+150 James |
| Outer length (X, with end-caps) | **0.280 m** | 250 + 2×15 |
| Track (L/R wheel centers) | **0.330 m** | James (hard) |
| Drive axle x | **0.000 m** | 100\|150 join |
| Front of rails / rear of rails | **+0.100 m** / **−0.150 m** | Orientation assumption |
| Front/rear outer faces | **+0.115 m** / **−0.165 m** | End-cap outside |
| Wheel y (L / R) | **+0.165 / −0.165 m** | Track/2, REP-103 |
| Caster (approx) | **x ≈ −0.150 m** | Near aft rail end |
| Mast (stacked stock) | **0.90 m** | 300+300+300 James |
| Mast centerline x | **−0.150 m** | Center of aft 300 / aft rail end |
| Mast top-bar tip x | **0.000 m** | Plumb with drive-wheel centerline |
| Mast top bar | **0.150 m** along +X | Stock; tip at axle x |
| Axle height / ground | **0.0855 m** | BotWheel radius |
| Upper tier height | **0.100 m** | 100 mm risers |

### ODrive BotWheel

From [ODrive shop](https://shop.odriverobotics.com/products/botwheels) and official OnShape publication  
https://cad.onshape.com/publications/ffa16d9268e83ffcb5643cba/w/81f7a79746984b3bebf1d392/e/3101f331226b74a504e702ae  
(sign-in required for CAD export — **v2 uses multi-primitive hull**, not mesh):

| Spec | Value | Notes |
|------|-------|-------|
| Diameter | **171 mm (0.171 m)** | Official / OnShape |
| Radius | **0.0855 m** | Derived |
| Width | **~0.052 m** | **Guessed** (was 0.055) — not on public shop page |
| Mass | **2.2 kg** / wheel | Official |
| Mount | **4× M6 on 36 mm PCD** | Diamond; T/B on axle plane `x=0` |

Visual (v3 hub-motor): one tire cylinder (offset outboard) + inner hub flange + 5 spoke boxes + 4× M6 markers on hub face only — no nested full-radius cylinders.

### Wheel mount geometry (v4 — James photo bracket orientation)

| Item | Value | Notes |
|------|-------|-------|
| XL L-bracket | **60 × 30 × 15 × 1.5 mm** | Long / short / width / visual thickness |
| **Orientation (photo)** | **Long UP, short on rail** | Long = vertical upright; short = horizontal on beam outer face |
| Upright holes | **3×** (top Ø3 / **middle Ø6 M6** / bottom Ø3) | 15 mm pitch; M6 bolts to BotWheel flange |
| Short-leg fasteners | **2× T-slot bolts** | Point **away** from pair partner along beam (X) |
| Brackets per wheel | **2** | Mirrored pair on each side rail; verticals toward pair center |
| M6 hole spacing (James) | **50 mm along X** | Model: `x = ±0.025` m, `z = 0` (axle height) |
| Naming | `upper` = **+X (fwd)**, `lower` = **−X (aft)** | Both longs stick **up** (not Z-mirrored) |
| BotWheel opposite-hole spacing | **36 mm** | Official PCD; T/B on `x=0` at `z = ±0.018` |
| **50 vs 36** | **Conflict** | Bracket M6s at 50 mm (X); diamond T/B at 36 mm still drawn |
| Wheel / axle axis height | **`base_link` z = 0** | Midpoint / height of bracket M6s |
| Top diamond height | **z = +0.018 m** | |
| `beam_axle_top` | **300 mm** at **z = +0.018** | Spans left–right top diamond holes |
| Side-rail outer face | **y = ±0.150 m** | Beam centers ±0.1425 + half of 15 mm |
| L-plate center (Y) | **y = ±0.1515 m** | Outer face + 1.5 mm (½×3 mm plate); plate spans ±0.150..±0.153 |
| Tire inner face (mount) | **y = ±0.153 m** | Clears beam + bracket; hub flange sits here |
| **Stand-off math** | beam outer 0.150 → bracket 3 mm → tire inner 0.153 | Unchanged |

**Bracket orientation (James photo — truth):** Two flat XL L-brackets per wheel side on the outer face of the horizontal side rail. **Long legs vertical (up)** with three holes (top small, **middle enlarged M6**, bottom small) — those M6s bolt to the ODrive BotWheel mount flange. **Short legs horizontal** on the rail, bolted into the T-slot, pointing **away from each other** along the beam (X). Vertical flanges are parallel and toward the pair center; big-hole centers **50 mm** apart at axle height. Flat L lies in the **XZ** plane (thickness along Y) so M6 bolts aim along Y into the hub face.

Big-hole XYZ (`base_link`, meters):

| Link | X | Y | Z |
|------|--:|--:|--:|
| `bracket_xl_left_upper` | +0.025 | +0.1515 | 0 |
| `bracket_xl_left_lower` | −0.025 | +0.1515 | 0 |
| `bracket_xl_right_upper` | +0.025 | −0.1515 | 0 |
| `bracket_xl_right_lower` | −0.025 | −0.1515 | 0 |

### Other guessed sizes

| Part | Guessed size |
|------|--------------|
| Wheel width | 0.052 m |
| Rear caster wheel | Ø 0.050 × 0.018 m |
| Battery main / aux | 0.15×0.09×0.05 / 0.08×0.06×0.04 m |
| ODrive S1 + clear case | 0.07×0.055×0.03 m each |
| PC + heatsink | 0.14×0.12×0.04 m + fins |
| E-stop | Ø ~0.056 m mushroom |

## URDF naming

**Beams:** `beam_base_*`, `beam_top_*`, `beam_riser_*`, `beam_mast_*` (incl. `beam_mast_top`), `beam_axle_top`  
**Brackets:** `bracket_xl_{left,right}_{upper,lower}` (XL flat L, wheel mounts)  
**Mobility:** `wheel_left` / `wheel_right` (continuous), `caster_yaw` / `caster_fork` / `caster_wheel`  
**Electronics / sensors:** `battery`, `battery_aux`, `odrive_left`, `odrive_right`, `pc`, `heatsink`, `estop`, `respeaker`  
**Root:** `base_link`, `mast` (mast parent for stacked segments)


## Mast geometry (James hard constraints)

1. **Three 300 mm** MakerBeam XL segments stacked vertically = **900 mm** mast height (replaced prior 300+300+200 = 800 mm).
2. Mast sits on the **center of the aft 300 mm** base/upper rear crossbeam (**Y = 0**), centerline at **x = −0.150 m** (aft end of the 150 mm side rails / on the aft 300).
3. On top of the mast: a **150 mm** horizontal bar (`beam_mast_top`) extending **forward (+X)**.
4. The **forward tip** of that bar is exactly **plumb with the drive-wheel centerline** (virtual axle at **x = 0**). There is no physical axle (inhub BotWheel motors); the line is still the wheel axis / track midpoint.

| Mast measure | Value |
|--------------|-------|
| Mast origin (upper-tier plane) | **x = −0.150, y = 0, z = 0.100** m |
| Stack (relative z) | 0…0.300, 0.300…0.600, 0.600…0.900 |
| Top-bar center | mast-frame **(0.075, 0, 0.9075)** → world tip **x = 0** |
| Mast height (stock stack) | **0.900 m** above upper tier |

### Sensor placement (reattached)

| Sensor | Placement | Rationale |
|--------|-----------|-----------|
| **E-stop** (`estop`) | Mid-mast forward face (~z=0.450 in mast frame) | Reachable height; clear of top-bar sensors |

## Assumptions (v2)

1. Base beam plane at drive-axle height (`base_link` z=0).
2. Diff-drive axle at **x = 0** (100\|150 join); caster near aft rail end (~−0.150 m).
3. **300 mm end-caps outside**; 150+100 rails sandwiched between their inner faces.
4. Orientation: 100 mm forward, 150 mm aft of axle (flip later if wrong).
5. Upper tier mirrors lower rails + front/rear 300s on 100 mm risers.
6. All structural beams are uncut stock lengths only.
7. Mast = 3×300 mm on aft 300 centerline; 150 mm top bar tip at x=0 (plumb with wheels).
8. BotWheel mesh deferred (OnShape requires login); multi-primitive Ø171 mm hull.
9. XL L-bracket visual thickness 1.5 mm (stand-off visibility); catalog 101732 equal-leg ≠ James 60/30 drawing.
10. Bracket M6 spacing modelled at James’s **50 mm along X** (photo pair); official hub PCD opposite holes are **36 mm** (T/B diamond still drawn). Long-up / short-on-rail per photo.
11. Colors approximate; inertias are placeholders.

## Validation

- [x] All joints reference existing links
- [x] Continuous joints for L/R drive + caster yaw/roll
- [x] Every `beam_*` length ∈ {50,100,150,200,300} mm
- [x] Track = 0.330 m; width = 0.300 m; rails = 100+150 sandwiched
- [x] Mast = 3×300 = 900 mm at x=−0.150; top-bar tip at x=0
- [x] Corner silver cubes removed; 4× XL L-brackets + `beam_axle_top`
- [x] James photo long-up orientation; M6 pair at x=±0.025, z=0; viewer `?v=6`
- [x] Mobile viewer with orbit + axes

## Blockers / known limits

- **OnShape BotWheel mesh:** public URL redirects to sign-in; v1 uses primitives.
- **Headless Chrome WebGL:** box GPU/WebGL context unavailable for automated screenshots; use the live viewer or `previews/render_preview.py` (matplotlib orthographic).

## Confirm with James

1. **Orientation:** Is 100 mm forward / 150 mm aft correct, or flip?
2. **Join style:** Are the 300s outside end-caps (as modeled), or do rails sit outside the 300s?
3. **Upper tier:** Confirm dual-tier cage with four 150+100 sets (vs lower-only).
4. **Mid axle cross:** Model includes a lower `beam_base_mid` 300 mm at the axle join — present on the real robot?
5. **M6 spacing:** Is bracket M6-to-M6 really **50 mm**, or **36 mm** (BotWheel opposite holes)? Model shows both.
6. **L-bracket:** Photo-confirmed long-up / short-on-rail; confirm 60×30 stock vs upright showing 3 holes, and plate thickness (~2–3 mm).

## BOM delta (v1 → v2)

| Change | Delta |
|--------|------:|
| Placeholder corner cubes (`bracket_fl/fr/rl/rr`) | **−4** |
| XL flat L-brackets (wheel mounts) | **+4** |
| `beam_axle_top` 300 mm | **+1** |
| MakerBeam XL stock total | **21 → 22** |

## Next

- Resolve 50 mm vs 36 mm M6 spacing with a caliper on the real robot
- Export BotWheel STL from OnShape when logged in → `meshes/`
- Download real 101732 / James L STP into `refs/` (Drive folder needs auth)
- Flip orientation / join style if James confirms otherwise
- ROS 2 `robot_state_publisher` wrap

## BotWheel (OnShape-measured, 2026-09-19)

Source: [OnShape publication](https://cad.onshape.com/publications/ffa16d9268e83ffcb5643cba/w/81f7a79746984b3bebf1d392/e/49f4650575b85a5903fc4ddb)
Raw export: `meshes/botwheel_onshape_raw.stl` (8.8 MB). Viewer uses low-poly `meshes/botwheel.stl`.

| Spec | Value |
|------|------:|
| Overall OD | **171.3 mm** |
| Tire width | **44.6 mm** |
| Mount flange | **Ø50 × 5 mm** |
| Mount holes | **4 × Ø5 on PCD 36 mm** |
| Mount face → near sidewall | **7.2 mm** |
| Neck | **Ø20** |
| Sidewall | curved crown (not flat cylinder); low-poly lathe from OnShape profile |


## Inner longitudinals (added)
Two MakerBeam XL back-to-front rails parallel to the sides: each **100 mm + 150 mm**, centers **120 mm** apart (`y = ±0.060 m`), sandwiched between the front/rear 300 mm end-caps (same X layout as the outer side rails).
