#!/usr/bin/env bash
# Build ORB-SLAM3 + Kevin TCP tracker into the user-prefix conda env (no sudo).
set -euo pipefail
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
PREFIX="${ORB_SLAM3_PREFIX:-$HOME/.local/share/mamba/envs/orbslam3}"
ORB="${ORB_SLAM3_ROOT:-$HOME/gits/awokeknowing/ORB_SLAM3}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="$HOME/.local/bin:$PREFIX/bin:$PATH"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$PREFIX/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$PREFIX/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

if [[ ! -x "$PREFIX/bin/pkg-config" ]]; then
  echo "missing conda env $PREFIX — create orbslam3 env first" >&2
  exit 1
fi
if [[ ! -f "$PREFIX/lib/cmake/Pangolin/PangolinConfig.cmake" && \
      ! -f "$PREFIX/lib64/cmake/Pangolin/PangolinConfig.cmake" ]]; then
  echo "Pangolin not installed in $PREFIX" >&2
  exit 1
fi

JOBS="${JOBS:-$(nproc)}"
cmake_cfg() {
  local src="$1" build="$2"
  mkdir -p "$build"
  cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DCMAKE_INCLUDE_PATH="$PREFIX/include" \
    -DCMAKE_LIBRARY_PATH="$PREFIX/lib" \
    -DCMAKE_CXX_FLAGS="-I$PREFIX/include" \
    -DOpenCV_DIR="$PREFIX/lib/cmake/opencv4" \
    -DEigen3_DIR="$PREFIX/share/eigen3/cmake" \
    -DPangolin_DIR="$PREFIX/lib/cmake/Pangolin"
  cmake --build "$build" -j"$JOBS"
}

echo "=== DBoW2 ==="
cmake_cfg "$ORB/Thirdparty/DBoW2" "$ORB/Thirdparty/DBoW2/build"
echo "=== g2o ==="
cmake_cfg "$ORB/Thirdparty/g2o" "$ORB/Thirdparty/g2o/build"
echo "=== Sophus ==="
cmake_cfg "$ORB/Thirdparty/Sophus" "$ORB/Thirdparty/Sophus/build"

if [[ ! -f "$ORB/Vocabulary/ORBvoc.txt" ]]; then
  tar -xf "$ORB/Vocabulary/ORBvoc.txt.tar.gz" -C "$ORB/Vocabulary"
fi

echo "=== ORB_SLAM3 ==="
cmake_cfg "$ORB" "$ORB/build"

echo "=== orbslam3_track ==="
cmake --build "$ORB/build" --target orbslam3_track -j"$JOBS"
ls -lh "$REPO/sim/orbslam3/orbslam3_track"
