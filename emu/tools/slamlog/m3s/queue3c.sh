#!/bin/bash
# CAPPED mode with the sim UP (James 2026-10-19 14:09): MASt3R at nice 19 on cpus 0,1,4,5, torch memory fraction per
# CUDA process, optional per-frame GPU duty sleep; process group registered in /tmp/heavy.pgids so emu.tools.agewatch
# aborts it on any safety frame age >100 ms.  Early kill: >EK_RELOC reloc attempts in the first 20% of the log.
# Usage: SEQS="L3 L4" VARS="oc dc" FRAC=0.22 SLEEPMS=0 CPUS=0,1,4,5 NT=2 bash queue3c.sh   (SCHED_IDLE + nice 19)
export PATH=~/slamvenv/bin:$PATH TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=${NT:-2} MKL_NUM_THREADS=${NT:-2} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEQS=${SEQS:-"L3 L4"}; VARS=${VARS:-"oc dc"}; FRAC=${FRAC:-0.22}; SLEEPMS=${SLEEPMS:-0}; EK=${EK_RELOC:-40}; K=${K:-1}
mkdir -p ~/slamlogs/res/m3s
for L in $SEQS; do for v in $VARS; do
  D=~/slamlogs/${L}x_rs2; NF=$(ls $D/png | wc -l)
  case $v in
    r)  E="";; o) E="M3S_ODOM=$D/odom_prior.npy";;
    oc) E="M3S_ODOM=$D/odom_prior.npy M3S_ODOM_COAST=1 M3S_ODOM_KF=1";;
    dc) E="M3S_ODOM=$D/odom_prior.npy M3S_ODOM_COAST=1 M3S_ODOM_KF=1 M3S_DEPTH=$D/depth";;
  esac
  T=${L}_q3c${v}$K; echo "== m3s $T [$E] frac $FRAC sleep $SLEEPMS ms $(date +%T)"
  cd ~/slamcand/MASt3R-SLAM
  env $E M3S_MEM_FRAC=$FRAC M3S_FRAME_SLEEP_MS=$SLEEPMS setsid nice -n 19 chrt -i 0 taskset -c ${CPUS:-0,1,4,5} timeout -k 20 1200 python main.py --dataset $D/png --config config/base.yaml --calib config/kevin_rs2.yaml --no-viz --save-as $T > /tmp/m3s_$T.log 2>&1 &
  P=$!; sleep 1; PG=$(ps -o pgid= -p $P | tr -d ' '); echo $PG > /tmp/heavy.pgids; t0=$(date +%s); pk=0
  while kill -0 $P 2>/dev/null; do
    sleep 5; el=$(( $(date +%s) - t0 ))
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ' | sed "s/^/$(date +%T) /" >> /tmp/m3s_${T}_gpu.txt; echo >> /tmp/m3s_${T}_gpu.txt
    nr=$(grep -c RELOCALIZING /tmp/m3s_$T.log); fps=$(grep "^FPS" /tmp/m3s_$T.log | tail -1)
    # frames done ~ el*fps; first 20% of the log
    fr=$(python -c "import sys;f='$fps'.split(':')[-1].strip() or '0';print(int($el*float(f)))" 2>/dev/null || echo 0)
    if [ "$fr" -lt $((NF/5)) ] && [ "$nr" -gt $EK ]; then echo "EARLY KILL $T: $nr reloc attempts by frame ~$fr/$NF"; kill -- -$PG; pk=1; fi
  done; wait $P; rc=$?; : > /tmp/heavy.pgids
  echo "rc $rc wall $(( $(date +%s) - t0 )) s"; grep -E "frac|peak|killed|init scale|Out of memory|OutOfMemory|Error" /tmp/m3s_$T.log | head -6
  echo "reloc $(grep -c RELOCALIZING /tmp/m3s_$T.log) coast $(grep -c ^COAST /tmp/m3s_$T.log) odomkf $(grep -c ^ODOM_KF /tmp/m3s_$T.log) skipped $(grep -c '^Skipped' /tmp/m3s_$T.log)"
  echo "gpu max MiB per pid: $(python - <<P
import re,collections;m=collections.defaultdict(int)
for l in open('/tmp/m3s_${T}_gpu.txt'):
    for pid,mb in re.findall(r'(\d+), (\d+) MiB',l): m[pid]=max(m[pid],int(mb))
print(dict(m))
P
)"
  pkill -f "/home/james/slamvenv/bin/python -c from multiprocessing" && echo "WARN: orphans remained (killed)"
  cp /tmp/m3s_$T.log /tmp/m3s_${T}_gpu.txt ~/slamlogs/res/m3s/
  [ -f logs/$T/png.txt ] || { echo "no output"; continue; }
  cp logs/$T/png.txt ~/slamlogs/res/m3s/$T.txt
  (cd ~/gits/awokeknowing/anglerdroid && nice -n 19 python -m emu.tools.slamlog.ext_eval ~/slamlogs/$L rs2 $D ~/slamlogs/res/m3s/$T.txt tum@30 ~/slamlogs/res/m3s_$T.json > /dev/null) && python -c "import json;d=json.load(open('$HOME/slamlogs/res/m3s_$T.json'));print({k:d[k] for k in d if not isinstance(d[k],(list,dict))})"
done; done; echo "QDONE $(date +%T)"
