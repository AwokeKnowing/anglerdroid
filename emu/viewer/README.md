# Kevin Emu Viewer

Live WebGL sim viewer + chat interface for Kevin emulator.

## Usage

```bash
# Terminal 1: Start emulator
MUJOCO_GL=egl KEVIN_EMU_VIEW=1 python -m emu.world --env apartment

# Terminal 2: Start nav stack (sim mode)
KEVIN_HAL=sim python -m nav.run --port 8090

# Terminal 3: Start viewer
python emu/viewer/viewer.py 8095 8090
```

Open browser: http://localhost:8095

## Pages

- **`/`** (default) — Live Three.js sim with client-side orbit controls, polled state, chat
- **`/jpeg`** — Legacy server-side JPEG render with camera frustums (KEVIN_EMU_VIEW=1 required)
- **`/state.json`** — Polled state endpoint (robot, dog, goal, path) — tunnel-friendly
- **`/state`** — SSE endpoint for live state stream (may not work through cloudflared tunnel)
- **`/scene.json`** — Environment geometry export (boxes, cylinders, meshes with vertices/faces)

## Features

**Live WebGL viewer:**
- Three.js scene with orbit controls (drag to orbit, scroll to zoom)
- Real-time state updates via polling `/state.json` (~20 Hz, tunnel-friendly)
- Robot: Kevin GLB mesh (`assets/kevin_lite.glb`) loaded via GLTFLoader
  - Chassis + mast + wheels geometry from URDF dimensions
  - Fallback to box placeholder on load failure
  - Yellow heading arrow for direction indication
- Dog (brown sphere, when present)
- Goal marker (red cone, when set)
- Path visualization (orange line, when planned)
- Environment walls/furniture (trimesh exported as BufferGeometry)
- Translucent walls (60% opacity, see rooms from outside)
- Auto-fit camera to scene bounds (chase angle view)
- Frame house button (reset camera to show whole scene)
- Sky-blue background with bright outdoor lighting (no fog)
- Large floor and grid (scene extent + 8m margin)
- Camera frustum toggle (client-side show/hide, optional)
- Chat box (POST to nav `/api/command`)
- Console logging (F12 to debug scene export/load)

**Legacy JPEG viewer (`/jpeg`):**
- Server-side MuJoCo render with camera frustums
- Requires `KEVIN_EMU_VIEW=1`
- Slower but shows live camera feeds textured on frustum planes

## Architecture

**Static scene:** Environment geometry exported from `emu/envs.py` via `/scene.json`:
- **Boxes/cylinders:** kind, size, pos, yaw, color (from MATS palette)
- **Meshes (walls):** vertices/faces from trimesh STL, rendered as BufferGeometry
- **Extent:** `[xmin, ymin, xmax, ymax]` for auto-sizing floor/grid and camera positioning
- **Coordinate transform:** MuJoCo Z-up → Three.js Y-up via `[x,y,z]→[x,z,y]`

**Live state:** Polling `/state.json` reads `/dev/shm/kevin_sim.json` (robot x,y,yaw; dog; env) + nav `/api/state` (goal, path) every 50ms
- Tunnel-friendly (works through cloudflared quick tunnel)
- Fallback SSE `/state` available but may buffer through proxies

**Browser:** Three.js (CDN) with OrbitControls. Chat POSTs to existing `/chat` endpoint → nav `/api/command`

---

## Design Decisions

**i777 deploy root cause (2026-10-15):**
- Original viewer had `scene.fog = new THREE.Fog(..., 8, 15)` + fixed camera at `(-3, 2.5, 4)`
- Result: only nearby sofa visible; walls/Kevin/house fogged out beyond 8m
- Scene.json was fine (53 geoms + 1 wall mesh 3380 faces)
- Fix: remove fog, sky-blue background, auto-frame camera, translucent walls

**Current design:**
- No fog (entire scene visible)
- Sky-blue background (`0x87ceeb`) for outdoor daytime look
- Translucent walls (60% opacity, glass-house effect)
- Bright cyan robot + yellow arrow (high visibility)
- Auto-frame on scene load + manual "Frame house" button
- Large ground/grid (extent + 8m margin)

## Debugging

**If walls missing (dark void with furniture only):**

1. **Check server console** (where `python3 emu/viewer/viewer.py` runs):
   ```
   [viewer] exporting scene: env=casita, extent=(-0.2, -0.2, 9.9, 7.5), geoms=54
   [viewer] loading mesh: /workspace/sim/casitahouse-walls-fixed.stl
   [viewer] mesh loaded: 1523 verts, 3044 faces, scale=0.001
   [viewer] mesh exported: 1523 verts, 3044 faces
   [viewer] scene.json ready: 54 geoms
   ```
   - If missing: mesh export failed (trimesh not installed or STL not found)
   - If exception: see traceback for root cause

2. **Check browser console** (F12 in browser):
   ```
   Scene extent: 10.1m x 7.7m @ (4.8, 3.7)
   Loading mesh: 1523 verts, 3044 faces
   Scene loaded: 1 meshes, 53 boxes, 0 cylinders
   ```
   - If `0 meshes`: mesh export failed on server
   - If `0 verts, 0 faces`: mesh is empty
   - If exception: coordinate transform error

3. **Common fixes:**
   - Install trimesh: `pip3 install trimesh`
   - Check STL exists: `ls sim/casitahouse-walls-fixed.stl`
   - Verify coordinate system: MuJoCo Z-up → Three.js Y-up `[x,y,z]→[x,z,y]`

**Expected output for casita:**
- Server: 1523 verts, 3044 faces
- Client: 1 meshes, 53 boxes, 0 cylinders
- Walls visible (plaster color `0xdcd8cc`)
- Camera framing whole house from chase angle

**Reuse:** Patterns borrowed from `sim/kevin-urdf-twin/viewer/index.html` (Three.js URDF twin)

## Minimal Working

- Casita boxes + Kevin + dog render smoothly
- State updates move robot when sim runs
- Chat commands work (go to couch, forward 2, stop, etc.)
- Client-side orbit controls (no waiting on server frames)
- Prefer minimal working over perfect mesh fidelity

## Notes

- No WebSocket (SSE is simpler with stdlib http.server)
- No MuJoCo meshes in v1 (boxes/cylinders only)
- Camera frustum visualization is optional client-side toggle (no JPEG slices)
- Legacy `/jpeg` page remains as fallback but `/` defaults to WebGL
