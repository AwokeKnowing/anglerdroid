# AnglerDroid v2

Minimal robot stack: 30 fps loop, tools (wheelbase + vision + ui), single-thread vision atlas.

## Layout (flat)

- `main.py` – entrypoint; 30 fps loop, reads frames + tool calls, optional rerun
- `tools.py` – wraps wheelbase, vision, ui; high-level API for AI loop
- `vision.py` – 2 RealSense + 1 RGB cam → 320×240 frames, 640×480 atlas (UL=rgb1, UR=rgbd1, LL=rgbd2, LR=top-down depth)
- `ui.py` – stub for audio/webview; server thread; `get_user_text()` / `get_pending_tool_calls()`
- `wheelbase.py` – final (do not change)

## Run locally (no robot)

```bash
cd anglerdroidv2
pip install -r requirements.txt
python main.py --no-wheelbase --rs1 "" --rs2 ""
```

With RealSense and cameras (set serials and device). **rgb1** = USB webcam (forward); use a device that is *not* a RealSense node. From `python3 debug_camera.py`: video0 = 600×800 → USB webcam; video4/6/10/12 = 480×640 → RealSense. So use `--rgb1 /dev/video0` (default) for the forward cam.

```bash
python main.py --no-wheelbase --rs1 815412070676 --rs2 944622074292 --rgb1 /dev/video0
```

With wheelbase (e.g. on robot):

```bash
python main.py --rs1 815412070676 --rs2 944622074292 --rgb1 /dev/video0
```

## Docker on Orin NX

Build and run on the Orin (ARM64):

```bash
cd anglerdroidv2
docker build -t anglerdroidv2 .
docker run --runtime nvidia --privileged -v /dev:/dev --network host -it anglerdroidv2
```

- `--privileged` and `-v /dev:/dev`: CAN, cameras, USB
- `--network host`: if you need LiveKit or other network services
- Serial numbers and `--rgb1` can be passed: `docker run ... anglerdroidv2 python3 main.py --rs1 XXX --rs2 YYY --rgb1 /dev/video0`

If `pyrealsense2` is not available in pip on Jetson, install the RealSense SDK from JetPack/apt and use the system Python or a venv that can see it; the Dockerfile tries `apt-get install librealsense2` and `pip install pyrealsense2` as fallback.

## Vision API

- `vision.frames`: list of 3 arrays (rgb1, rgbd1, rgbd2), each 320×240 RGB
- `vision.atlas`: 640×480 RGB (4 quads)
- `vision.timestamp`: last capture time
- Use `vision.read()` for a thread-safe copy: `(frames, atlas, timestamp)`

## Tool calls

The main loop consumes `tools.get_pending_tool_calls()`. Each item can be e.g. `{"name": "set_wheel_vels", "args": {"left_tps": 0.0, "right_tps": 0.0}}` or `{"name": "stop"}` / `{"name": "twist", "args": {"forward_mps": 0.0, "angular_rads": 0.0}}`. Execute these in the tight loop so low-level logic (e.g. obstacle stop) can override.

## UI / agent

`ui.py` is a stub. To integrate LiveKit + Gemini/Grok: run the agent in `ui`’s server thread; have it push user transcript with `set_user_text()` and tool calls with `push_tool_calls()`. The browser client connects to localhost; the app serves or connects to LiveKit from that thread.
