#!/bin/bash
# Build rtabmap_kevin against the robostack ros-humble-rtabmap library (micromamba env 'rtabmap', no ROS needed).
set -e; E=${RTABMAP_ENV:-$HOME/.local/share/mamba/envs/rtabmap}; H=$(realpath "$(dirname "$0")")
g++ -O2 -std=c++17 $H/rtabmap_kevin.cc -o $H/rtabmap_kevin -I$E/include/rtabmap-0.22 -I$E/include/opencv5 -I$E/include/eigen3 -I$(ls -d $E/include/pcl-* | head -1) -I$E/include \
  -L$E/lib -Wl,-rpath,$E/lib -lrtabmap_core -lrtabmap_utilite -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
# disk-lean runtime (i777 keeps ~30 GB free): vendor the shared-library closure, then the 6 GB env can go
#   R=~/slambench/z/rtabmap_rt/lib; mkdir -p $R; ldd $H/rtabmap_kevin | grep "$E" | awk '{print $3}' | xargs -I{} cp -L {} $R/   (+ any "not found", e.g. libcblas.so.3)
#   bench/slam/run_rtabmap.py sets LD_LIBRARY_PATH=$R

# rtabmap_stream: streaming driver for the online 24 h / lifelong bench (bench/slam/life24.py); built against the
# headers in ~/slambench/z/rtabmap_hdr and the runtime copy in ~/slambench/z/rtabmap_rt/lib
SE=${SE:-$HOME/slambench/z/rtabmap_hdr}; SR=${SR:-$HOME/slambench/z/rtabmap_rt/lib}
g++ -O2 -std=c++17 $H/rtabmap_stream.cc -o $H/rtabmap_stream -I$SE/include/rtabmap-0.22 -I$SE/include/opencv5 -I$SE/include/eigen3 \
    -I$SE/include/pcl-1.15 -I$SE/include $SR/librtabmap_core.so.0.22 $SR/librtabmap_utilite.so.0.22 $SR/libopencv_core.so.500 \
    $SR/libopencv_imgcodecs.so.500 -Wl,-rpath-link,$SR
