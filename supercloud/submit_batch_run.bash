#!/bin/bash

echo "~~ working dir: '$(pwd)' ~~"

run_script="mj_envs_vision/mj_envs_vision/run.py mj_envs_vision/config/mini-ppo.json ppo"
#run_script="mj_envs_vision/mj_envs_vision/batch_run.py mj_envs_vision/config/offline.json"

# Set some useful environment variables
export MUJOCO_ENV_HOME=$HOME/software/mujoco/mujoco_env
export MUJOCO_ENV=/state/partition1/user/$USER/mujoco_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia:$HOME/.mujoco/mujoco210/bin

# Check if the environment exists on the local disk. If not copy it over from the home directory.
if [ ! -d "$MUJOCO_ENV" ]; then
    echo "~~ copying $MUJOCO_ENV_HOME to $MUJOCO_ENV ~~"
    mkdir /state/partition1/user/$USER
    cp -r $MUJOCO_ENV_HOME $MUJOCO_ENV
fi

# Load an anaconda module, then activate your mujoco environment
module load anaconda/2023a
source $MUJOCO_ENV/bin/activate

# Project setup
pip install --no-cache-dir -r $HOME/mj_envs_vision/requirements.txt
source $HOME/mj_envs_vision/setup.bash
unset LD_PRELOAD

echo "~~ check packages ~~"
pip list | grep torch
echo $PYTHONPATH

#SBATCH -c 4
#SBATCH --gres=gpu:volta:1 

echo "~~ python $run_script ~~"
python $run_script
squeue -u $USER 
