#!/bin/bash
# Build rtabmap_kevin against the robostack ros-humble-rtabmap library (micromamba env 'rtabmap', no ROS needed).
set -e; E=${RTABMAP_ENV:-$HOME/.local/share/mamba/envs/rtabmap}; H=$(realpath "$(dirname "$0")")
g++ -O2 -std=c++17 $H/rtabmap_kevin.cc -o $H/rtabmap_kevin -I$E/include/rtabmap-0.22 -I$E/include/opencv5 -I$E/include/eigen3 -I$(ls -d $E/include/pcl-* | head -1) -I$E/include \
  -L$E/lib -Wl,-rpath,$E/lib -lrtabmap_core -lrtabmap_utilite -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
