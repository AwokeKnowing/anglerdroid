# WebGL Live Viewer Improvements (PR #62)

## Summary
Enhanced the WebGL live sim viewer under `emu/viewer/` to meet all four requirements:
1. ✅ Real yorkie from MuJoCo sim (replacing toy sphere)
2. ✅ Fixed camera frustum show/hide toggle
3. ✅ Per-camera view panels with toggles
4. ✅ No regressions

Commit: `eee2e2d`

## Implementation Details

### 1. Real Yorkie Geometry

**Problem**: Orange sphere dog marker did not match physics sim.

**Solution**: Built primitive yorkie matching `world.py` lines 197-206:
- **Capsule body**: radius 0.055m, from (-0.07,0,0.13) to (0.06,0,0.135), fur2 material (#7a5530)
- **Sphere head**: radius 0.048m at (0.11,0,0.175), fur material (#8c6540)
- **Capsule snout**: radius 0.018m, from (0.14,0,0.165) to (0.17,0,0.16)
- **Box ears**: 0.024×0.044×0.016m at (0.105,±0.028,0.225), left/right pair
- **4 Capsule legs**: radius 0.013m each at corners (±0.07,±0.03), vertical
- **Capsule tail**: radius 0.01m, from (-0.11,0,0.15) to (-0.15,0,0.2)

**Data flow**:
- `world.py` line 651-659: Extract dog position + convert quaternion to yaw
- Export as `dog: [x, y, th]` in `/dev/shm/kevin_sim.json`
- `live.html` consumes `/state.json` → updates `dogGroup` position + rotation
- Default: **shown when present** (changed from OFF by default)

### 2. Camera Frustums Fixed

**Root cause**: `frustums=[]` array was never populated, so checkbox did nothing.

**Fix**: Created 4 camera frustums at real twin mounts:

| Camera | Position (base frame, m) | FOV | Color | Notes |
|--------|--------------------------|-----|-------|-------|
| RS1 depth | (-0.007, 0.0, 0.8619) | 58° | Cyan | Top-down, points -Y (down) |
| RS1 color | (0.008, 0.0, 0.8619) | 42° | Green | Top-down, +15mm offset |
| RS2 depth | (-0.1301, 0.0, 0.3822) | 58° | Yellow | Forward, pitched 30° down |
| RS2 color | (-0.1301, 0.015, 0.3822) | 42° | Magenta | Forward, +15mm Y offset |

**Frustum design** (per James's profile prefs):
- Small wireframe-only pyramid ~10cm deep
- Cross-section plane near tip (for future live image slice)
- Faint FOV lines (line segments from origin to corners)
- Attached to `robotGroup` → move with kevin

**Toggle**: `#camvis` checkbox now works (frustums array populated).

### 3. Per-Camera View Panels

**UI**: 4 new checkboxes in controls panel:
- RS1 depth / RS1 color
- RS2 depth / RS2 color

**Panels**: HUD overlays (320×240px) at screen corners, toggle visibility.

**Endpoints**: `viewer.py` serves `/cam/<name>.jpg` from `/dev/shm/kevin_<cam>.jpg`:
- RS1 depth: `/cam/rs1_depth.jpg` → `/dev/shm/kevin_rs1_depth.jpg`
- RS1 color: `/cam/rs1_color.jpg` → `/dev/shm/kevin_rs1_color.jpg`
- RS2 depth: `/cam/rs2_depth.jpg` → `/dev/shm/kevin_rs2_depth.jpg`
- RS2 color: `/cam/rs2_color.jpg` → `/dev/shm/kevin_rs2_color.jpg`

**Polling**: JavaScript updates panels at 5 Hz when enabled.

**Gap**: Live camera JPEGs not yet available in this environment. Panels show working UI plumbing; images will display when sim writes JPEG thumbs to shm. Endpoint returns 404 until then.

### 4. No Regressions

Preserved:
- ✅ `kevin_lite.glb` robot (~738KB)
- ✅ `mjToThree` coordinate transform
- ✅ No toy cyan robot
- ✅ House unmirrored
- ✅ Text chat → `/chat`

Changes were simple and principled:
- Frustum fix: populate array that was always referenced but never filled
- Yorkie: match MuJoCo primitives 1:1
- Panels: additive UI, no existing code changed

## How Frustums Work

**Build** (lines 233-270 in `live.html`):
1. `makeFrustum(name, pos, fovy, color, pitch)` creates THREE.Group:
   - Wireframe LineSegments (8 edges from origin to 4 corners + 4 corner edges)
   - Cross-section PlaneGeometry at 90% depth (for future image texture)
2. Position set to twin mount (MuJoCo Z-up → Three.js Y-up)
3. Rotation: RS1 points down (rotate -90° X), RS2 pitched 30° down
4. Add to `robotGroup` (not scene) → inherit robot transform

**Toggle** (line 276):
```js
frustums=[rs1dFrustum,rs1cFrustum,rs2dFrustum,rs2cFrustum];
camvisEl.onchange=()=>{frustums.forEach(f=>f.visible=camvisEl.checked&&sceneLoaded)};
```
Previously `frustums=[]` was never mutated → checkbox did nothing.

**Why toggle works now**: Array populated before onchange binding.

## Yorkie Matches world.py

| Geom | world.py (lines 197-206) | live.html (lines 186-242) | Match? |
|------|--------------------------|---------------------------|--------|
| Body | `type="capsule" size="0.055" fromto="-0.07 0 0.13 0.06 0 0.135"` | `CapsuleGeometry(0.055, 0.13)` at midpoint | ✅ |
| Head | `type="sphere" size="0.048" pos="0.11 0 0.175"` | `SphereGeometry(0.048)` at (0.11,0.175,0) | ✅ |
| Snout | `type="capsule" size="0.018" fromto="0.14 0 0.165 0.17 0 0.16"` | `CapsuleGeometry(0.018, 0.03)` at midpoint | ✅ |
| Ears | `type="box" size="0.012 0.008 0.022" pos="0.105 ±0.028 0.225"` | `BoxGeometry(0.024,0.044,0.016)` × 2 | ✅ |
| Legs | `type="capsule" size="0.013" fromto="lx ly 0.11 lx ly 0.012"` × 4 | `CapsuleGeometry(0.013, 0.098)` × 4 | ✅ |
| Tail | `type="capsule" size="0.01" fromto="-0.11 0 0.15 -0.15 0 0.2"` | `CapsuleGeometry(0.01, 0.064)` rotated | ✅ |

Materials: `m_fur` → #8c6540 (brown), `m_fur2` → #7a5530 (darker brown).

## Camera Panel Endpoints

**Exist now**: `/cam/<name>.jpg` endpoints in `viewer.py` lines 186-196.

**Available**: None (sim does not yet write to `/dev/shm/kevin_<cam>.jpg`).

**Status**: Panels work; images will appear when sim adds JPEG export.

**Next steps** (out of scope):
- Add lightweight JPEG thumb export to `world.py` render loop
- Write to `/dev/shm/kevin_rs1_depth.jpg`, etc. from sim shm buffers
- Panels will automatically display once files exist

## Files Changed

1. **emu/world.py** (10 lines):
   - Extract dog yaw from mocap_quat (quaternion → yaw via atan2)
   - Export `dog: [x, y, th]` instead of `[x, y]`

2. **emu/viewer/live.html** (+171 lines):
   - Yorkie geometry (capsules, sphere, boxes) with brown materials
   - Camera frustums at RS1/RS2 mounts with correct FOVs
   - Per-camera panel UI + polling logic
   - Dog default shown when present
   - Frustums array populated (fix)

3. **emu/viewer/viewer.py** (+10 lines):
   - `/cam/<name>.jpg` endpoints (404 if shm images unavailable)

## Testing Notes

- **Frustum toggle**: Works (array populated)
- **Dog shown by default**: Yes (changed from OFF)
- **Yorkie matches world.py**: Geometries and materials match 1:1
- **Camera panels**: UI works; images 404 until sim writes shm JPEGs
- **No regressions**: kevin_lite.glb, mjToThree, house, chat all preserved

## Gap Documentation

**Camera images not served** in this environment:
- Sim does not yet write JPEG thumbs to `/dev/shm/kevin_<cam>.jpg`
- Panels show working UI plumbing (checkboxes, overlays, polling)
- Endpoints return 404 gracefully
- Images will display when sim adds JPEG export

**Why deferred**: Task specifies "if you cannot fully serve live camera JPEGs in this environment, still ship working frustum geometry + panel UI plumbing and document the gap."

**Recommendation**: Wire sim camera renders to shm in a follow-up (requires adding JPEG encode to `world.py` render loop).
