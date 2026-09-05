# AnglerDroid – Jetson Orin NX (aarch64)
#
# Requires JetPack 6.x on the host for GPU/CUDA 12 support.
# Adjust BASE_IMAGE arg to match your installed JetPack:
#   JP 6.0 GA  →  nvcr.io/nvidia/l4t-jetpack:r36.3.0
#   JP 6.1     →  nvcr.io/nvidia/l4t-jetpack:r36.4.0
#   JP 6.2     →  nvcr.io/nvidia/l4t-jetpack:r36.4.0  (closest available tag)
#
# Build:  docker compose build          (~25 min first time, cached after)
# Run:    docker compose up
# Shell:  docker compose exec anglerdroid bash
#
# Hardware: reComputer Industrial J4012 (Orin NX 16GB)
# Flash guide: https://wiki.seeedstudio.com/reComputer_Industrial_Getting_Started

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── 1. System packages ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-dev \
        git git-lfs \
        cmake build-essential pkg-config \
        # EGL / OpenGL for moderngl headless rendering
        libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
        # librealsense2 build deps
        libusb-1.0-0-dev libglfw3-dev libssl-dev libudev-dev \
        # V4L2 webcam
        libv4l-dev \
        # CAN bus
        can-utils iproute2 \
        # TurboJPEG native lib
        libturbojpeg0-dev \
    && rm -rf /var/lib/apt/lists/*

# ── 2. librealsense2 + pyrealsense2 (source build for aarch64) ─
#    Pre-built pip wheels rarely exist for aarch64.
#    This layer is heavy (~20 min build) but cached after first run.
ARG LIBREALSENSE_VERSION=v2.56.2
RUN git clone --depth 1 -b ${LIBREALSENSE_VERSION} \
        https://github.com/IntelRealSense/librealsense.git /tmp/rs \
    && mkdir /tmp/rs/build && cd /tmp/rs/build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    && make -j$(nproc) \
    && make install \
    && PYDIR=$(python3 -c "import site; print(site.getsitepackages()[0])") \
    && cp -r /tmp/rs/build/wrappers/python/pyrealsense2 "$PYDIR/" \
    && ldconfig \
    && python3 -c "import pyrealsense2; print('pyrealsense2 OK:', pyrealsense2.__version__)" \
    && rm -rf /tmp/rs

# ── 3. Python packages ──────────────────────────────────────────
RUN pip3 install --no-cache-dir \
        numpy \
        opencv-python-headless \
        python-can \
        inputs \
        websockets \
        PyTurboJPEG \
        moderngl

# ── 4. cuVSLAM (NVIDIA GPU-accelerated visual SLAM) ─────────────
RUN git lfs install \
    && git clone --depth 1 https://github.com/NVlabs/pycuvslam.git /opt/pycuvslam \
    && pip3 install --no-cache-dir -e /opt/pycuvslam/bin/aarch64 \
    && python3 -c "import cuvslam; v,_,_ = cuvslam.get_version(); print('cuVSLAM OK:', v)"

WORKDIR /app
CMD ["python3", "src/main.py"]
