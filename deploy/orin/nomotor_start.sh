#!/bin/bash
# stationary bench: full nav stack on real cameras, NO wheel base (hal/nomotor), sim-parity env. Extra env from caller.
# Run from ~/anglerdroid or set KEVIN_REPO. Copies live in ~/kevin_rollback/ on the Orin.
cd "${KEVIN_REPO:-$HOME/anglerdroid}"
export PATH=$HOME/.venv-kevinnav/bin:$PATH KEVIN_HAL=real KEVIN_ROBOT=kevin001 KEVIN_AUTOCAL=0 KEVIN_RGB_DEV=${KEVIN_RGB_DEV:-none}
export KEVIN_SC_DECEL=${KEVIN_SC_DECEL:-1} KEVIN_PLAN_LIMIT=${KEVIN_PLAN_LIMIT:-1} KEVIN_PLAN_CLEARANCE=${KEVIN_PLAN_CLEARANCE:-1} KEVIN_EGO_SHARED=${KEVIN_EGO_SHARED:-1}
mkdir -p ~/kevin_bench; LOG=~/kevin_bench/nav_$(date +%H%M%S).log; ln -sfn $LOG ~/kevin_bench/nav_latest.log
setsid nohup ${NAV_PREFIX} $HOME/.venv-kevinnav/bin/python hal/nomotor/run_nomotor.py --port 8080 > $LOG 2>&1 < /dev/null &
echo started $! log $LOG
