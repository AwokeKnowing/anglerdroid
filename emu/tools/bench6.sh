#!/bin/bash
# completes the n=3 interleaved goal bench: cur3 (PR #62 head) + core v2 x3 (same goals/world as bench5)
cd ~/m1_backup/dbg; G=/home/james/gits/awokeknowing
export KEVIN_SAFETY_CORE_LOG=$HOME/m1_backup/dbg/corelog_r6.jsonl
declare -A REPO=([cur]=$G/anglerdroid [core]=$G/anglerdroid-core)
for t in r6_core1 r5_cur3 r6_core2 r6_core3; do v=${t#*_}; v=${v%[0-9]}
  echo "$(date +%T) $t ${REPO[$v]}"
  KEVIN_REPO=${REPO[$v]} ~/.venv-kevinrobot/bin/python $G/anglerdroid/emu/tools/goal_bench.py $t > bench_$t.log 2>&1; cp ~/.kevin/emu/live_nav.log nav_$t.log
done
echo "$(date +%T) done"
