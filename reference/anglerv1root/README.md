# Kevin


This is a demo of real time speech to text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.

To install dependencies simply run
```
python3 -m venv venv
source venv/bin/activate
sudo apt install \
  git libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev libglfw3 libgl1-mesa-dev libglu1-mesa-dev \
  python3-pyaudio portaudio19-dev ffmpeg flac python3.8-venv python3-pip mosquitto mosquitto-clients
pip install -r requirements.txt
```
in an environment of your choosing.

also, enable mosquitto mqtt broker on startup even in desktop mode:
```
sudo ln -s /lib/systemd/system/mosquitto.service /etc/systemd/system/graphical.target.wants/mosquitto.service

```



for Jetson had to deal with versions
```
pip install numba==0.54


sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons libopenblas-dev 

export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

python3 -m pip install --upgrade pip; python3 -m pip install aiohttp numpy=='1.19.4' scipy=='1.5.3' export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"; python3 -m pip install --upgrade protobuf; python3 -m pip install --no-cache $TORCH_INSTALL


```
https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html


also cupy, cuda-python, pyopengl 
was hard but got help looking at images from dusty-nv images. TODO: make one anglerdroid docker:

`https://github.com/dusty-nv/jetson-containers/blob/master/packages/cuda/cuda-python/Dockerfile.builder`


```
pip install cupy-cuda11x
pip install pyopengl
pip install pyrr
pip install glfw
git clone --branch 11.8.x --depth=1 https://github.com/NVIDIA/cuda-python
cd cuda-python/
pip install numba
sudo apt-get update
pip3 install numpy
pip install --no-cache-dir --verbose -r requirements.txt
cd ..
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-tegra-repo-ubuntu2004-11-8-local_11.8.0-1_arm64.deb
sudo dpkg -i cuda-tegra-repo-ubuntu2004-11-8-local_11.8.0-1_arm64.deb
sudo cp /var/cuda-tegra-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
cd cuda-python/
python setup.py bdist_wheel --verbose
cp dist/cuda*.whl /opt
sudo cp dist/cuda*.whl /opt
cd ..
pip install --no-cache-dir --verbose /opt/cuda*.whl
pip show cuda-python && python3 -c 'import cuda; print(cuda.__version__)'
```






For more information on Whisper please see https://github.com/openai/whisper

The code in this repository is public domain.









```
        sense     (sensor input)
      perceive    (deep learning encode)
    emote         (calculate internal low level motivations like battery, connect, help )
  concern         (attention over world model with possible futures)
trust             (core policy. score futures. select goal)
  act             (evaluate actions to acheive goal)
    try           (select behavior)
      orchestrate (track progress and emit actions)
        react     (actuators output)
```


def invert()
spectator


tick
lambda: #Root
  lambda: #Chooser
    SensePainCondition() and HandlePainAction() or\
    IsMovingCondition()  and  or\
    FindMove()

