#!/bin/bash

# install mujoco-py
module load anaconda/2023a
mkdir -p /state/partition1/user/$USER
python -m venv /state/partition1/user/$USER/mujoco_env
source /state/partition1/user/$USER/mujoco_env/bin/activate
pip install 'mujoco-py<2.2,>=2.1'
pip install "cython<3"

# install other project deps
pip install git+https://github.com/aravindr93/mjrl.git
# install custom sb3 for compatibility with gym 0.24+
pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
pip install stable-baselines3[extra]

# install project
cd ~/mj_envs_vision
source /mj_envs_vision/setup.bash
unset LD_PRELOAD
pip install --no-cache-dir -r requirements.txt

zip -r 
