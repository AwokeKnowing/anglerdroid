#!/bin/bash
# interleaved n=3 bench: cur (PR #62), ta (Task A branch head), tb (Task B proposal); world = slambench 6bf0a69 (sliced walls)
# BENCH_DOG=keepout|off|on (default keepout: dog kept >= BENCH_DOG_R=1.5 m from the start pose), exported to goal_bench.py
cd ~/m1_backup/dbg; G=/home/james/gits/awokeknowing
declare -A REPO=([cur]=$G/anglerdroid [ta]=$G/anglerdroid-taskA [tb]=$G/anglerdroid-taskB)
for i in 1 2 3; do for v in cur ta tb; do
  t=r3_${v}$i; echo "$(date +%T) $t ${REPO[$v]}"
  KEVIN_REPO=${REPO[$v]} ~/.venv-kevinrobot/bin/python $G/anglerdroid/emu/tools/goal_bench.py $t > bench_$t.log 2>&1; cp ~/.kevin/emu/live_nav.log nav_$t.log
done; done
echo "$(date +%T) done"
