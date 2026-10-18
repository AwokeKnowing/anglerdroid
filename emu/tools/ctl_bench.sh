#!/bin/bash
# before/after: KEVIN_CTL=frame vs cmd; goal bench x3 + LightNav fridge x3 each
cd /home/james/gits/awokeknowing/anglerdroid
PY=~/.venv-kevinrobot/bin/python
for M in ${MODES:-pre cmd}; do
  if [ $M = pre ]; then export KEVIN_REPO=/home/james/gits/anglerdroid_pre; unset KEVIN_CTL; else unset KEVIN_REPO; export KEVIN_CTL=$M KEVIN_CMD_CYCLE=1; fi
  for i in 1 2 3; do
    echo "== $M bench $i $(date +%T)"
    (source emu/tools/simpin.sh; emu/tools/simpin.sh apply >/dev/null 2>&1; BENCH_DIRECT=1 $PY emu/tools/goal_bench.py ctl_${M}$i > ~/m1_backup/dbg/bench_ctl_${M}$i.log 2>&1); tail -3 ~/m1_backup/dbg/bench_ctl_${M}$i.log
  done
  for i in 1 2 3; do
    echo "== $M fridge $i $(date +%T)"
    $PY -c "from emu import gt; import json; S=json.load(open('/dev/shm/kevin_sim.json')); x,y,a=S['start']; gt.ctl('teleport',x=x,y=y,yaw=a)"; sleep 2
    (source emu/tools/simpin.sh; ~/m1_backup/dbg/restart_nav.sh >/dev/null 2>&1; emu/tools/simpin.sh apply >/dev/null 2>&1)
    $PY emu/tools/vln_drift.py ctl_${M}_fridge$i "go to the fridge" 150 2>&1 | tail -1
  done
done
echo "== done $(date +%T)"
