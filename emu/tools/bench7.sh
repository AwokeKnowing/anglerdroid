#!/bin/bash
# remaining core v2 goal benches (strict design: cadence 1.5 frames) + one diagnostic run tolerating one dropped frame
cd ~/m1_backup/dbg; G=/home/james/gits/awokeknowing
export KEVIN_SAFETY_CORE_LOG=$HOME/m1_backup/dbg/corelog_r6.jsonl
for t in r6_core2 r6_core3; do echo "$(date +%T) $t"
  KEVIN_REPO=$G/anglerdroid-core ~/.venv-kevinrobot/bin/python $G/anglerdroid/emu/tools/goal_bench.py $t > bench_$t.log 2>&1; cp ~/.kevin/emu/live_nav.log nav_$t.log; done
echo "$(date +%T) r6_coreC1 (KEVIN_SC_CADENCE_MAX=0.083, diagnostic)"
KEVIN_SC_CADENCE_MAX=0.083 KEVIN_REPO=$G/anglerdroid-core ~/.venv-kevinrobot/bin/python $G/anglerdroid/emu/tools/goal_bench.py r6_coreC1 > bench_r6_coreC1.log 2>&1; cp ~/.kevin/emu/live_nav.log nav_r6_coreC1.log
echo "$(date +%T) done"
