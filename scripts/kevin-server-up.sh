#!/bin/bash
# Kevin sim server on i777 (M1): MuJoCo world + nav.run (same brain as the robot, KEVIN_HAL=sim) + shared viewer (:8095)
# + local LLM (ollama) for the chat agent + LightNav-0 VLN (:8050, started separately, long-lived).
#   scripts/kevin-server-up.sh            start/restart world, nav, viewer (+ ollama if not running)
#   ENV=casita KEVIN_EMU_NOISE=realistic  overrides; KEVIN_LLM_MODEL=qwen2.5:7b-instruct; KEVIN_VLN_URL=ws://localhost:8050
# Remote: cloudflared tunnel --url http://localhost:8095 (WS /ws passes through). TODO(M2): auth before sharing the URL.
set -u
cd "$(dirname "$0")/.."
R=$PWD; LOGD=~/.kevin/emu; mkdir -p "$LOGD"
export KEVIN_LLM_URL=${KEVIN_LLM_URL:-http://localhost:11434/v1}
export KEVIN_LLM_MODEL=${KEVIN_LLM_MODEL:-qwen2.5:7b-instruct}
export KEVIN_VLN_URL=${KEVIN_VLN_URL:-ws://localhost:8050}
OLLAMA=${OLLAMA:-$HOME/ollama/bin/ollama}

# 1. local LLM (user-space ollama; models in ~/.ollama). Kept resident (keep_alive -1) so chat never waits on a load.
if ! (exec 3<>/dev/tcp/127.0.0.1/11434) 2>/dev/null; then
  OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 nohup "$OLLAMA" serve > "$LOGD/ollama.log" 2>&1 &
  sleep 3
fi
"$OLLAMA" list 2>/dev/null | grep -q "^${KEVIN_LLM_MODEL}" || "$OLLAMA" pull "$KEVIN_LLM_MODEL"
python3 - <<PY || echo "warn: LLM warm-up failed"
import json,urllib.request
urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/generate",json.dumps(dict(model="$KEVIN_LLM_MODEL",prompt="hi",keep_alive=-1,options=dict(num_predict=1))).encode()),timeout=120).read()
PY

# 2. VLN (LightNav-0) is long-lived and holds ~12 GB GPU: only check it.
python3 -c "import socket;socket.create_connection(('127.0.0.1',8050),2)" 2>/dev/null && echo "VLN up on :8050" || { echo "warn: LightNav-0 not on :8050, 'ask' will report no VLN"; unset KEVIN_VLN_URL; }

# 3. world + brain + viewer
pkill -f tour_clip.sh; pkill -f emu.eval; pkill -f "emu.world"; pkill -f "nav.run --port 8090"; pkill -f "viewer.py 8095"; sleep 2
MUJOCO_GL=egl KEVIN_EMU_MODEL=full KEVIN_EMU_VIEW=1 KEVIN_EMU_NOISE=${KEVIN_EMU_NOISE:-realistic} nohup ~/.venv-kevinsim/bin/python -m emu.world --env ${ENV:-casita} > "$LOGD/live_world.log" 2>&1 &
for i in $(seq 30); do [ -e /dev/shm/kevin_sim.json ] && [ /dev/shm/kevin_sim.json -nt "$LOGD/live_world.log" ] && break; sleep 1; done; sleep 3
KEVIN_HAL=sim nohup scripts/kevin-hal ~/.venv-kevinrobot/bin/python -m nav.run --port 8090 ${KEVIN_VLN_URL:+--vln $KEVIN_VLN_URL} > "$LOGD/live_nav.log" 2>&1 &
KEVIN_HAL=sim nohup scripts/kevin-hal ~/.venv-kevinrobot/bin/python emu/viewer/viewer.py 8095 8090 > "$LOGD/viewer.log" 2>&1 &
sleep 8
python3 -c "import urllib.request,json;s=json.load(urllib.request.urlopen('http://127.0.0.1:8095/state.json',timeout=5));print('viewer ok, mode',s.get('mode'),'pose',s.get('pose'))" || echo "warn: viewer not answering yet"
pgrep -f "cloudflared tunnel.*8095" >/dev/null || echo "note: no cloudflared tunnel for :8095 (start: ~/.local/bin/cloudflared tunnel --no-autoupdate --url http://localhost:8095)"
echo "local: http://localhost:8095/   logs: $LOGD/{live_world,live_nav,viewer,ollama}.log"
