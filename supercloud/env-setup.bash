#!/bin/bash

# NOTE: run only once per login node

# install mujoco-py
module load anaconda/2023a
#mkdir -p /state/partition1/user/$USER
#python -m venv /state/partition1/user/$USER/mujoco_env
source /state/partition1/user/$USER/mujoco_env/bin/activate
pip install 'mujoco-py<2.2,>=2.1'
pip install "cython<3"
# install lib-glew (missing in interactive sessions)
#conda install -c conda-forge glew

# install other project deps
pip install git+https://github.com/aravindr93/mjrl.git
# install custom sb3 for compatibility with gym 0.24+
pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
pip install stable-baselines3[extra]

# install project
cd ~/mj_envs_vision
source setup.bash
unset LD_PRELOAD
pip install --no-cache-dir -r requirements.txt

# reference for copying to gpu nodes
cd /state/partition1/user/$USER/
zip -r mujoco_env.zip mujoco_env

# install glew
wget http://archive.ubuntu.com/ubuntu/pool/universe/g/glew/glew_2.2.0.orig.tar.xz
tar -xJf glew_2.2.0.orig.tar.xz
cd glew-2.2.0/build/cmake
cmake -DCMAKE_INSTALL_PREFIX=$HOME/glew-install/
make
make install
