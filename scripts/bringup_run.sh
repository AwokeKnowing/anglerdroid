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

# Orin CPU layout (2026-10-20 stationary soaks, emu/tools/scbench/REALTEST_2026-10-21.md): safety core alone on cpu 7
# (KEVIN_SC_CPU, its own default) at nice -10 (needs /etc/security/limits.d/99-kevin-rt.conf; SCHED_FIFO measured worse),
# RS2 camera process on cpu 6, brain + its threads on 0-5.  Override: KEVIN_NAV_CPUS / KEVIN_RS2CAM_CPUS ('' = unpinned).
NAVPIN=()
if [ "$KEVIN_HAL" = real ] && [ -f /etc/nv_tegra_release ]; then
    export KEVIN_SC_CPU=${KEVIN_SC_CPU:-7} KEVIN_RS2CAM_CPUS=${KEVIN_RS2CAM_CPUS-6}; NAV_CPUS=${KEVIN_NAV_CPUS-0-5}
    [ -n "$NAV_CPUS" ] && NAVPIN=(taskset -c "$NAV_CPUS")
    [ "$(ulimit -e)" -lt 30 ] && echo "WARN: nice limit $(ulimit -e) -> safety core cannot get nice -10 (install deploy/orin/99-kevin-rt.conf, re-login)"
fi
mkdir -p "$KEVIN_RECORD"; echo "hal=$KEVIN_HAL vmax=$KEVIN_VMAX wmax=$KEVIN_WMAX autocal=$KEVIN_AUTOCAL log=$KEVIN_RECORD"
"${NAVPIN[@]}" scripts/kevin-hal "$PY" -m nav.run --port $PORT 2>&1 | tee "$KEVIN_RECORD/nav.log"
