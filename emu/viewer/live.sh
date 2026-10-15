#!/bin/bash
# Live casita sim for the web viewer: full-detail twin + third-person view + nav + viewer (:8095).  Stops offline tours.
cd "$(dirname "$0")/../.."
pkill -f tour_clip.sh; pkill -f emu.eval; pkill -f emu.world; pkill -f nav.run; pkill -f "viewer.py"; sleep 2
MUJOCO_GL=egl KEVIN_EMU_MODEL=full KEVIN_EMU_VIEW=1 KEVIN_EMU_NOISE=${KEVIN_EMU_NOISE:-realistic} nohup ~/.venv-kevinsim/bin/python -m emu.world --env ${ENV:-casita} > ~/.kevin/emu/live_world.log 2>&1 &
sleep 10
KEVIN_HAL=sim nohup scripts/kevin-hal ~/.venv-kevinrobot/bin/python -m nav.run --port 8090 > ~/.kevin/emu/live_nav.log 2>&1 &
KEVIN_HAL=sim nohup scripts/kevin-hal ~/.venv-kevinrobot/bin/python emu/viewer/viewer.py 8095 8090 > ~/.kevin/emu/viewer.log 2>&1 &
echo started
