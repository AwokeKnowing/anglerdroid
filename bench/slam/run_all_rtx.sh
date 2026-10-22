#!/bin/bash
# SLAM-from-zero bench on one RTX log: every method, then trajectory (eval.py) + map (occmap.py) scores.
#   bench/slam/run_all_rtx.sh DATA/<seq> RESULTS/<seq>
set -x
D=$1; R=$2; H=$(dirname "$(realpath "$0")"); mkdir -p $R
RB=~/.venv-kevinrobot/bin/python; SB=~/.venv-kevinsim/bin/python; KV=~/slambench/z/kissvenv/bin/python
$RB $H/run_baselines.py $D $R
$RB $H/kslam.py $D $R
$SB $H/run_orb_rtx.py $D $R -j 2
$RB $H/run_rtabmap.py $D $R
$KV $H/run_kiss.py $D $R
$KV $H/run_viw_fg.py $D $R
$KV $H/run_viw_fg.py $D $R --no-visual
$RB $H/eval.py $D $R --md $R/traj.md
$RB $H/occmap.py $D $R
