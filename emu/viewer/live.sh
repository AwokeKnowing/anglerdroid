#!/bin/bash
# Live casita sim for the web viewer: full-detail twin + third-person view + nav (safety core + RS2 camera process)
# + viewer (:8095), CPU layout from emu/tools/simpin.sh (safety path on its own physical core).  Stops offline tours.
# LightNav (lightnav-serve :8050), ollama and cloudflared are separate services: started elsewhere, only re-pinned here.
cd "$(dirname "$0")/../.."
pkill -f tour_clip.sh; pkill -f emu.eval; pkill -f emu.world; pkill -f nav.run; pkill -f "viewer.py"; sleep 2
source emu/tools/simpin.sh
export KEVIN_VLN_URL=${KEVIN_VLN_URL:-ws://localhost:8050}   # LightNav server (task vln); nav reconnects if it is not up yet
export KEVIN_LLM_MODEL=${KEVIN_LLM_MODEL:-qwen2.5:7b-instruct} KEVIN_LLM_URL=${KEVIN_LLM_URL:-http://localhost:11434/v1} KEVIN_RGB_DEV=${KEVIN_RGB_DEV:-none}
[ -f ~/.kevin/emu/live_world.log ] && mv ~/.kevin/emu/live_world.log ~/.kevin/emu/live_world.prev.log
MUJOCO_GL=egl KEVIN_EMU_MODEL=full KEVIN_EMU_VIEW=1 KEVIN_EMU_NOISE=${KEVIN_EMU_NOISE:-realistic} setsid nohup $WORLD_PREFIX ~/.venv-kevinsim/bin/python -m emu.world --env ${ENV:-casita} > ~/.kevin/emu/live_world.log 2>&1 < /dev/null &
for i in $(seq 90); do grep -q "emu.depthcam: pid\|emu.world: casita" ~/.kevin/emu/live_world.log 2>/dev/null && break; sleep 1; done; sleep 3
KEVIN_HAL=sim setsid nohup $NAV_PREFIX scripts/kevin-hal ~/.venv-kevinrobot/bin/python -m nav.run --port 8090 --vln "$KEVIN_VLN_URL" > ~/.kevin/emu/live_nav.log 2>&1 < /dev/null &
KEVIN_HAL=sim setsid nohup $NAV_PREFIX nice -n 5 scripts/kevin-hal ~/.venv-kevinrobot/bin/python emu/viewer/viewer.py 8095 8090 > ~/.kevin/emu/viewer.log 2>&1 < /dev/null &
for i in $(seq 90); do grep -q "safety_core: pid" ~/.kevin/emu/live_nav.log 2>/dev/null && break; sleep 1; done
emu/tools/simpin.sh apply
echo started
