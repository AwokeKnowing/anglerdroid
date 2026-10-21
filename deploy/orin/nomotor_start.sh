#!/bin/bash
# stationary bench: full nav stack on real cameras, NO wheel base (hal/nomotor), sim-parity env. Extra env from caller.
# Run from ~/anglerdroid or set KEVIN_REPO. Copies live in ~/kevin_rollback/ on the Orin.
cd "${KEVIN_REPO:-$HOME/anglerdroid}"
export PATH=$HOME/.venv-kevinnav/bin:$PATH KEVIN_HAL=real KEVIN_ROBOT=kevin001 KEVIN_AUTOCAL=0 KEVIN_RGB_DEV=${KEVIN_RGB_DEV:-none}
export KEVIN_SC_DECEL=${KEVIN_SC_DECEL:-1} KEVIN_PLAN_LIMIT=${KEVIN_PLAN_LIMIT:-1} KEVIN_PLAN_CLEARANCE=${KEVIN_PLAN_CLEARANCE:-1} KEVIN_EGO_SHARED=${KEVIN_EGO_SHARED:-1}
mkdir -p ~/kevin_bench; LOG=~/kevin_bench/nav_$(date +%H%M%S).log; ln -sfn $LOG ~/kevin_bench/nav_latest.log
if [ "${KEVIN_VOICE:-1}" = "1" ] && ! pgrep -f "python -m nav.voice" >/dev/null; then   # nav/voice.py: TTS :8092 (nav /api/say forwards), ReSpeaker ears -> webhook; own process, low priority
  setsid nohup taskset -c ${KEVIN_VOICE_CPUS:-0-5} nice -n 10 $HOME/.venv-kevinnav/bin/python -m nav.voice >> ~/kevin_bench/voice.out 2>&1 < /dev/null &
  echo started voice $! log ~/kevin_bench/voice.log
fi
if [ "${KEVIN_SUPERVISE:-1}" = "1" ]; then   # nav/supervisor.py: camera reset + re-enumeration before every start, health watch, restarts, help webhook
  setsid nohup $HOME/.venv-kevinnav/bin/python -m nav.supervisor --port 8080 --log $LOG -- ${NAV_PREFIX} $HOME/.venv-kevinnav/bin/python hal/nomotor/run_nomotor.py --port 8080 >> ~/kevin_bench/supervisor.out 2>&1 < /dev/null &
  echo started supervisor $! log $LOG supervisor ~/kevin_bench/supervisor.log
else
  setsid nohup ${NAV_PREFIX} $HOME/.venv-kevinnav/bin/python hal/nomotor/run_nomotor.py --port 8080 > $LOG 2>&1 < /dev/null &
  echo started $! log $LOG
fi
