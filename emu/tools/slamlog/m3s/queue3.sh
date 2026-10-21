#!/bin/bash
# SIM-OFF ONLY (MASt3R ~17 GB GPU; LightNav must be stopped: 24.5 GB card).  Prepared 2026-10-19, NOT yet run.
# Variants (main.py env flags, see the comment block in ~/slamcand/MASt3R-SLAM/main.py; diff in m3s_main_kevin_prep2.patch):
#   r   : stock (no prior)                                    -> baseline, = queue2 r*
#   o   : M3S_ODOM rotation-only prior (scale-free)           -> tests "metric t on arbitrary-scale map" hypothesis for L4 o* regression
#   oc  : o + M3S_ODOM_COAST + M3S_ODOM_KF                    -> recovery through camera-log gaps during spins (L3 lost at 14 KF)
#   dc  : oc + M3S_DEPTH real-depth metric anchor (enables metric odom translation, logs KFSCALE)
# Usage: N=2 SEQS="L3 L4" VARS="oc dc o r" bash queue3.sh > queue3.out 2>&1      (every run hard-capped 1200 s)
export PATH=~/slamvenv/bin:$PATH TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
N=${N:-2}; SEQS=${SEQS:-"L3 L4"}; VARS=${VARS:-"oc dc o r"}
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
pgrep -f "lightnav-serve|emu.world" >/dev/null && { echo "REFUSING: sim/LightNav still running (GPU too small for both)"; exit 1; }
mkdir -p ~/slamlogs/res/m3s
for v in $VARS; do for L in $SEQS; do for k in $(seq $N); do
  D=~/slamlogs/${L}x_rs2; E=""
  case $v in
    r)  E="";;
    o)  E="M3S_ODOM=$D/odom_prior.npy";;
    oc) E="M3S_ODOM=$D/odom_prior.npy M3S_ODOM_COAST=1 M3S_ODOM_KF=1";;
    dc) E="M3S_ODOM=$D/odom_prior.npy M3S_ODOM_COAST=1 M3S_ODOM_KF=1 M3S_DEPTH=$D/depth";;
  esac
  T=${L}_q3${v}$k; echo "== m3s $T [$E] $(date +%T)"
  (cd ~/slamcand/MASt3R-SLAM && env $E timeout -k 20 1200 /usr/bin/time -f "WALL %e s" python main.py --dataset $D/png --config config/base.yaml --calib config/kevin_rs2.yaml --no-viz --save-as $T > /tmp/m3s_$T.log 2>&1
   grep -E "^FPS|WALL|killed|init scale" /tmp/m3s_$T.log | tail -3; echo "reloc $(grep -c RELOCALIZING /tmp/m3s_$T.log) coast $(grep -c ^COAST /tmp/m3s_$T.log) odomkf $(grep -c ^ODOM_KF /tmp/m3s_$T.log)")
  pkill -f "/home/james/slamvenv/bin/python -c from multiprocessing" && echo "WARN: orphans remained (killed)"
  cp /tmp/m3s_$T.log ~/slamlogs/res/m3s/$T.log
  [ -f ~/slamcand/MASt3R-SLAM/logs/$T/png.txt ] || { echo "no output"; continue; }
  cp ~/slamcand/MASt3R-SLAM/logs/$T/png.txt ~/slamlogs/res/m3s/$T.txt
  (cd ~/gits/awokeknowing/anglerdroid && python -m emu.tools.slamlog.ext_eval ~/slamlogs/$L rs2 $D ~/slamlogs/res/m3s/$T.txt tum@30 ~/slamlogs/res/m3s_$T.json > /dev/null) && python -c "import json;d=json.load(open('$HOME/slamlogs/res/m3s_$T.json'));print({k:d[k] for k in d if not isinstance(d[k],(list,dict))})"
done; done; done; echo "QDONE $(date +%T)"
