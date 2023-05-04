# Borrowed from https://github.com/openai/mujoco-py/blob/master/Dockerfile
# ------------------------------------------------------------------------
# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# Base images available at https://hub.docker.com/r/nvidia/cuda/tags
# Recommend using *cuddnnX-devel*
# If updating the base image, be sure to test on GPU.
#FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
# NOTE: cuda version should match version in nvidia-smi output
FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04

# Use bourne-again shell
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    zip \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3 \
    patchelf \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update

ENV LANG C.UTF-8

# Anaconda setup
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
RUN pip install --upgrade pip
RUN conda install python=3.9

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

# Lib references
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /mj_envs_vision
# Copy over just requirements.txt and bash scripts at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./vendor /mj_envs_vision/
COPY ./requirements.txt /mj_envs_vision/
COPY ./setup.bash /mj_envs_vision/
RUN echo "source /mj_envs_vision/setup.bash" >> /mj_envs_vision/tool_setup.bash
RUN echo "unset LD_PRELOAD" >> /mj_envs_vision/tool_setup.bash
RUN chmod +x /mj_envs_vision/tool_setup.bash
RUN /mj_envs_vision/tool_setup.bash
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -m pip install -U 'mujoco-py<2.2,>=2.1'
RUN python3 -m pip install git+https://github.com/aravindr93/mjrl.git 
# Install custom sb3 for compatibility with gym 0.24+
RUN python3 -m pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
RUN python3 -m pip install stable-baselines3[extra]

# compile mujoco
RUN python3 -c "import mujoco_py;print(mujoco_py.__version__)"


# Copy over project
COPY . /mj_envs_vision
RUN rm -rf dependencies/*
RUN git clone https://github.com/Kaixhin/PlaNet.git dependencies/PlaNet
RUN git clone https://github.com/vikashplus/Adroit.git dependencies/Adroit
RUN git clone https://github.com/bilkitty/pydreamer.git dependencies/DreamerV2
ENTRYPOINT ["python3", "/mj_envs_vision/vendor/Xdummy-entrypoint"]
