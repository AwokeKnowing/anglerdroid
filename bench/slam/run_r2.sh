#!/bin/bash
# Round 2 pipeline for one recorded route: RTX render -> prep (registration + IMU sync) -> top methods -> scores.
#   bench/slam/run_r2.sh DATA/<route> RESULTS/<route>
#   methods: baselines (wheel, wheel+gyro, slam2), kslam, ORB-SLAM3 RGB-D, RTAB-Map (VO / wheel+IMU odom), viw_fg + wi_fg
#   (online gyro scale) and their round-1 ablations (--no-scale).
set -x
D=$1; R=$2; H=$(dirname "$(realpath "$0")"); mkdir -p $R
RB=~/.venv-kevinrobot/bin/python; SB=~/.venv-kevinsim/bin/python; KV=~/slambench/z/kissvenv/bin/python; OV=~/.venv-ovrtx/bin/python
[ -f $D/rtx.json ] || $OV $H/render_rtx.py $D
[ -f $D/imu_sync.csv ] || $RB $H/prep_rtx.py $D
$RB $H/run_baselines.py $D $R
$RB $H/kslam.py $D $R
$SB $H/run_orb_rtx.py $D $R --modes orb_rgbd -j 1
$RB $H/run_rtabmap.py $D $R
$KV $H/run_viw_fg.py $D $R
$KV $H/run_viw_fg.py $D $R --no-visual
$KV $H/run_viw_fg.py $D $R --no-scale
$KV $H/run_viw_fg.py $D $R --no-visual --no-scale
$RB $H/eval.py $D $R --md $R/traj.md
$RB $H/occmap.py $D $R
echo R2_DONE
