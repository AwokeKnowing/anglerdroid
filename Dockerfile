# AnglerDroid v2 - Orin NX (ARM64). Minimal deps: RealSense, Open3D, rerun, wheelbase (CAN).
# Build on Orin: docker build -t anglerdroidv2 .
# Run: docker run --runtime nvidia --privileged -v /dev:/dev --network host -it anglerdroidv2

ARG L4T_VERSION=r36.2.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libopencv-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libv4l-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheelbase + deps (flat layout)
COPY wheelbase.py odrivecan.py simplegamepad.py flat_endpoints.json ./
COPY vision.py ui.py tools.py main.py ./

# Python deps (Orin: use system or pip; pyrealsense2 may need dnf/apt on Jetson)
RUN pip3 install --no-cache-dir numpy opencv-python-headless open3d rerun-sdk python-can inputs

# RealSense: on Jetson often installed via apt or pre-installed in L4T
RUN apt-get update && apt-get install -y --no-install-recommends \
    librealsense2 librealsense2-utils \
    || true
RUN pip3 install pyrealsense2 || true

ENTRYPOINT ["python3", "main.py"]
