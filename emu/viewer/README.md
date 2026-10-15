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

- **`/`** (default) — Live Three.js sim with client-side orbit controls, SSE state stream, chat
- **`/jpeg`** — Legacy server-side JPEG render with camera frustums (KEVIN_EMU_VIEW=1 required)
- **`/state`** — SSE endpoint for live state stream (robot, dog, goal, path)
- **`/scene.json`** — Environment geometry export (boxes, cylinders with colors)

## Features

**Live WebGL viewer:**
- Three.js scene with orbit controls (drag to orbit, scroll to zoom)
- Real-time state updates via Server-Sent Events (~20 Hz)
- Robot (green box with heading arrow)
- Dog (brown sphere, when present)
- Goal marker (red cone, when set)
- Path visualization (orange line, when planned)
- Camera frustum toggle (client-side show/hide, optional)
- Chat box (POST to nav `/api/command`)

**Legacy JPEG viewer (`/jpeg`):**
- Server-side MuJoCo render with camera frustums
- Requires `KEVIN_EMU_VIEW=1`
- Slower but shows live camera feeds textured on frustum planes

## Architecture

**Static scene:** Environment geometry exported from `emu/envs.py` via `/scene.json` (boxes/cylinders with sizes, poses, colors from MATS palette)

**Live state:** SSE stream reads `/dev/shm/kevin_sim.json` (robot x,y,yaw; dog; env) + nav `/api/state` (goal, path) every 50ms

**Browser:** Three.js (CDN) with OrbitControls. Chat POSTs to existing `/chat` endpoint → nav `/api/command`

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
