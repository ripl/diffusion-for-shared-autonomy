FROM nvidia/cudagl:11.4.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget curl software-properties-common

# Install python3.9
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt install -y python3.9 python3.9-dev python3.9-distutils
RUN ln -s `which python3.9` /usr/local/bin/python
RUN ln -s `which python3.9` /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py && rm get-pip.py

# "pip install mujoco" somehow tries to find pip by absolute path
RUN ln -s /usr/local/bin/pip /usr/bin/pip
RUN pip install mujoco

ENV PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    tree \
    zsh  \
    git \
    git-lfs \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev

RUN pip install --upgrade pip
RUN pip install pytest pytest-forked lz4 pyyaml
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Install typical dependencies
RUN pip install einops sklearn tqdm scipy h5py matplotlib

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    ffmpeg \
    libffi-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglfw3 \
    gcc \
    pciutils \
    xserver-xorg \
    xserver-xorg-video-fbdev \
    xauth

# install GLX-Gears (for debugging)
RUN apt-get update && apt-get install -y \
    mesa-utils \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

### Install mujoco ###
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

# ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# NOTE: Dockerfile is so counter-intuitive.
# `COPY ./vendor .` or `COPY ./vendor ./` copies the contents of ./vendor to .
# https://stackoverflow.com/a/72376058/7057866
WORKDIR /opt/mujoco_py
COPY ./vendor ./vendor

RUN mkdir -p /usr/local/etc/mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /usr/local/etc/mujoco \
    && rm mujoco.tar.gz
ENV MUJOCO_PY_MUJOCO_PATH="/usr/local/etc/mujoco/mujoco210"
ENV LD_LIBRARY_PATH="/usr/local/etc/mujoco/mujoco210/bin"

# deepmind control suite reads from this env var
ENV MJLIB_PATH="/usr/local/etc/mujoco/mujoco210/bin/libmujoco210.so"

# Compile-time library location
ENV LIBRARY_PATH=$LD_LIBRARY_PATH
RUN ln -s /usr/local/etc/mujoco $HOME/.mujoco \
    && mkdir -p /opt/mujoco_py/mujoco_py/binaries \
    && ln -s /usr/local/etc/mujoco /opt/mujoco_py/mujoco_py/binaries/linux

# From requirements-ext.txt
RUN pip install \
    "numpy>=1.20.0" \
    "numba>=0.52.0,<=0.53.1" \
    "scipy>=1.2.3" \
    "free-mujoco-py==2.1.6"

# Install wandb and its (optional) dependencies
RUN pip install \
    wandb \
    kornia \
    imageio \
    imageio-ffmpeg \
    moviepy

RUN apt-get update && apt-get install -y vim unzip xvfb patchelf cmake swig
RUN apt-get update && apt-get install -y libssl-dev libcurl4-openssl-dev  # Needed for pyCurl
RUN apt-get update && apt-get install -y \
    libglu1-mesa-dev \
    libopenmpi-dev  zlib1g-dev \
    qtdeclarative5-dev libglib2.0-0 \
    libosmesa6 \
    libglfw3-dev \
    net-tools \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN pip install gym[box2d]==0.22 pfrl
RUN pip install pygame opencv-python
ENV SDL_VIDEODRIVER="dummy"
RUN pip install params_proto termcolor pandas
RUN pip install ConfigArgParse trimesh scipy h5py matplotlib mplot3d-dragger pyglet

# block push environment dependencies (https://github.com/google-research/ibc)
RUN pip install \
    absl-py==0.12.0 \
    gin-config==0.4.0 \
    matplotlib==3.4.3 \
    mediapy==1.0.3 \
    opencv-python==4.5.3.56 \
    scipy==1.7.1 \
    tensorflow==2.6.0 \
    keras==2.6.0 \
    tf-agents==0.11.0rc0 \
    tqdm==4.62.2

# numpy >= 1.19 (?) is required before pip installing pybullet!!
RUN pip install -U numpy==1.21.3
RUN pip install pybullet==3.1.6
RUN pip install gym==0.22.0
RUN pip install scikit-image

RUN apt-get update && apt-get install -y \
    libcudnn8-dev \
    libcudnn8

# NOTE: Without this, singularity fails with
# OSError: [Errno 30] Read-only file system: b'/usr/local/lib/python3.9/dist-packages/mujoco_py/generated/mujocopy-buildlock'
RUN chmod a+rwx -R /usr/local/lib/python3.9/dist-packages/mujoco_py/generated

# Install customized d4rl env (This includes pointmaze)
RUN pip install git+https://github.com/takuma-yoneda/d4rl.git@484382d

# This compiles /usr/local/lib/python3.8/dist-packages/mujoco_py/cymj.pyx
RUN python -c 'import mujoco_py'
# RUN pip install -I gym==0.23.1

# Avoid TensorFlow error `TypeError: Descriptors cannot not be created directly.`
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV D4RL_SUPPRESS_IMPORT_ERROR=1

# Avoid `fatal: detected dubious ownership in repository at '/code'` by git
RUN git config --global --add safe.directory /code

ENTRYPOINT ["/opt/mujoco_py/vendor/Xdummy-entrypoint"]
