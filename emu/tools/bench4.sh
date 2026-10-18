#!/bin/bash
# interleaved n=3 bench: cur (PR #62) vs st (safety-staleness-now-2026-10-17 proposal); world = slambench (floor fix,
# dog kept >= 1.5 m from the start pose via goal_bench BENCH_DOG=keepout); safety log -> safelog_r4.jsonl
cd ~/m1_backup/dbg; G=/home/james/gits/awokeknowing
[ -f safelog.jsonl ] && mv safelog.jsonl safelog_pre_r4_$(date +%H%M).jsonl
declare -A REPO=([cur]=$G/anglerdroid [st]=$G/anglerdroid-stale)
for i in 1 2 3; do for v in cur st; do
  t=r4_${v}$i; echo "$(date +%T) $t ${REPO[$v]}"
  KEVIN_REPO=${REPO[$v]} ~/.venv-kevinrobot/bin/python $G/anglerdroid/emu/tools/goal_bench.py $t > bench_$t.log 2>&1; cp ~/.kevin/emu/live_nav.log nav_$t.log
done; done
cp safelog.jsonl safelog_r4.jsonl; echo "$(date +%T) done"
