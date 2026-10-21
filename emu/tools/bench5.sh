#!/bin/bash
# interleaved n=3 goal bench: cur (PR #62 head) vs core (safety-core-loop proposal); world = slambench; dog keepout
cd ~/m1_backup/dbg; G=/home/james/gits/awokeknowing
[ -f safelog.jsonl ] && mv safelog.jsonl safelog_pre_r5_$(date +%H%M).jsonl
export KEVIN_SAFETY_CORE_LOG=$HOME/m1_backup/dbg/corelog_r5.jsonl
declare -A REPO=([cur]=$G/anglerdroid [core]=$G/anglerdroid-core)
for i in 1 2 3; do for v in cur core; do
  t=r5_${v}$i; echo "$(date +%T) $t ${REPO[$v]}"
  KEVIN_REPO=${REPO[$v]} ~/.venv-kevinrobot/bin/python $G/anglerdroid/emu/tools/goal_bench.py $t > bench_$t.log 2>&1; cp ~/.kevin/emu/live_nav.log nav_$t.log
done; done
cp safelog.jsonl safelog_r5.jsonl; echo "$(date +%T) done"
