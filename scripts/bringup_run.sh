#!/bin/bash
# Real Kevin nav with bring-up caps + replay log.  Orin:  cd ~/anglerdroid && scripts/bringup_run.sh [vmax=0.3]
# Commands: web page http://<orin>:8080 (chat box), or  curl -s localhost:8080/api/command -d '{"text":"stop"}'
# Emulator rehearsal (i777): KEVIN_HAL=sim KEVIN_PY=~/.venv-kevinrobot/bin/python KEVIN_CALIB=/tmp/c.json scripts/bringup_run.sh 0.3
cd "$(dirname "$0")/.."
export KEVIN_HAL=${KEVIN_HAL:-real} KEVIN_VMAX=${1:-0.3} KEVIN_WMAX=${KEVIN_WMAX:-1.0} KEVIN_AUTOCAL=${KEVIN_AUTOCAL:-1}
[ "$KEVIN_HAL" = real ] && export CAN_IFACE=${CAN_IFACE:-can1}
export KEVIN_RECORD=${KEVIN_RECORD:-$HOME/kevin_logs/$(date +%Y%m%d_%H%M%S)}
PY=${KEVIN_PY:-$HOME/.venv-kevinnav/bin/python}; PORT=${KEVIN_PORT:-8080}

# Add venv bin to PATH (ninja for CUDA extension builds)
export PATH="$(dirname "$PY"):$PATH"

# On Jetson Orin (real hardware), verify CUDA is usable before starting nav
if [ "$KEVIN_HAL" = "real" ] && [ -f /etc/nv_tegra_release ]; then
    echo "Jetson Orin detected: verifying CUDA for planner..."
    if ! scripts/build_orin_cuda.sh; then
        echo "FATAL: CUDA not usable on Orin. Kevin planner REQUIRES GPU (CPU fallback forbidden)."
        echo "See docs/bringup.md for torch/JetPack setup instructions."
        exit 1
    fi
    echo
fi

mkdir -p "$KEVIN_RECORD"; echo "hal=$KEVIN_HAL vmax=$KEVIN_VMAX wmax=$KEVIN_WMAX autocal=$KEVIN_AUTOCAL log=$KEVIN_RECORD"
scripts/kevin-hal "$PY" -m nav.run --port $PORT 2>&1 | tee "$KEVIN_RECORD/nav.log"
