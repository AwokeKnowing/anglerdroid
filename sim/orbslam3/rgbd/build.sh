#!/bin/bash
# Build rgbd_kevin (ORB-SLAM3 rgbd_tum with viewer off, no real-time sleep) against an existing ORB_SLAM3 build.
# Reuses the compile/link flags CMake generated for rgbd_tum; adds the conda env lib dir (boost_serialization, crypto).
set -e; O=${ORB_SLAM3:-$HOME/gits/awokeknowing/ORB_SLAM3}; E=${ORB_ENV:-$HOME/.local/share/mamba/envs/orbslam3}; D=$O/build/CMakeFiles/rgbd_tum.dir; H=$(realpath "$(dirname "$0")")
FL=$(sed -n 's/^CXX_\(FLAGS\|DEFINES\|INCLUDES\) = //p' $D/flags.make | tr '\n' ' ')
LK=$(cat $D/link.txt | sed 's/^[^ ]* //; s/ "\?CMakeFiles\/rgbd_tum.dir\/[^ ]*\.o"\?//; s/ -o [^ ]*//')
cd $O/build && g++ $FL -c $H/rgbd_kevin.cc -o /tmp/rgbd_kevin.o && g++ /tmp/rgbd_kevin.o -o $H/rgbd_kevin $LK -L$E/lib -Wl,-rpath,$E/lib
