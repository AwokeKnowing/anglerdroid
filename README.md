# AnglerDroid v2

Minimal robot stack: 30 fps loop, tools (wheelbase + vision + ui), dual RealSense obstacle mapping, Gemini AI with voice.

## Project structure

All source lives in `src/`:

- `main.py` – entrypoint; 30 fps loop, reads frames + tool calls, tuning sliders
- `tools.py` – wraps wheelbase, vision, ui; high-level API for AI loop
- `vision.py` – depth processing (pure numpy); combines RS1 top-down + RS2 forward pointclouds into obstacle map
- `cameras.py` – RealSense and webcam hardware abstraction; pre-allocated numpy buffers
- `ui.py` – HTTPS + WSS server, Gemini AI (conversation + TTS), audio I/O
- `index.html` – browser UI: controls, chat/log tabs, mic capture, audio playback, animated eyes
- `prompt.txt` – Gemini system prompt
- `wheelbase.py` – differential drive controller (do not change)

## Fresh device setup

### 1. Dependencies

```bash
cd src
pip install -r requirements.txt
# On Jetson: pyrealsense2 may need the RealSense SDK from apt
```

### 2. Run

Without robot hardware (dev machine):
```bash
python main.py --no-wheelbase --rs1 "" --rs2 ""
```

With cameras and wheelbase (on robot):
```bash
python main.py --rs1 815412070676 --rs2 944622074292 --rgb1 /dev/video0
```

With Gemini AI:
```bash
export GEMINI_KEY="your-api-key"
python main.py --rs1 815412070676 --rs2 944622074292 --rgb1 /dev/video0
```

### 3. HTTPS + microphone setup

The server auto-generates a self-signed cert on first run (requires `openssl`).
This is needed because browsers require HTTPS for microphone access.

On the device browser, visit:
1. `https://localhost:8080` — accept the cert warning ("Proceed to localhost")
2. `https://localhost:8081` — accept the WSS cert warning (shows "Certificate accepted")
3. Go back to `https://localhost:8080` and reload

For remote access from another machine, replace `localhost` with the robot's IP.
Alternatively, use Chrome flag `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.

## Vision

Atlas layout (640x480): UL=webcam, UR=RS1 color, LL=RS2 color, LR=obstacle map.

Obstacle map (lower-right quadrant):
- **Black** = known clear (camera sees it, no obstacle)
- **Green** = RS1 (top-down) obstacle
- **Red** = RS2 (forward) obstacle
- **Yellow** = both cameras detect obstacle
- **Dim blue** = unknown (outside camera coverage)

Tuning sliders on the imshow window: `height_clip`, `td_xoff`, `fw_xoff`, `fw_pxmm`.

## Web UI

- **Controls**: twist_for sliders, STOP button, mic toggle, launch AI
- **Chat tab**: user/AI messages, state changes
- **Log tab**: mic events, debug output
- **Eyes**: click to go fullscreen, auto-blink and pupil movement
- **Mic**: captures audio via MediaRecorder, sends to Gemini for transcription
- **TTS**: AI responses spoken via Gemini TTS (high-quality voice)

## Docker on Orin NX

```bash
cd src
docker build -t anglerdroidv2 .
docker run --runtime nvidia --privileged -v /dev:/dev --network host -it anglerdroidv2
```

- `--privileged` and `-v /dev:/dev`: CAN bus, cameras, USB
- `--network host`: web UI, WebSocket
- Pass args: `docker run ... anglerdroidv2 python3 main.py --rs1 XXX --rs2 YYY`
