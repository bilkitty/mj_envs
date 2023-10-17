#!/bin/bash
#SBATCH -c 4
#SBATCH --gres=gpu:volta:1 

# Optional
##SBATCH --exclusive

echo "~~ working dir: '$(pwd)' ~~"

# TEST
run_script="mj_envs_vision/mj_envs_vision/run.py mj_envs_vision/config/mini-ppo.json ppo"

# Set some useful environment variables
export MUJOCO_ENV_HOME=/state/partition1/user/$USER
export MUJOCO_ENV=/state/partition1/user/$USER/mujoco_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia:$HOME/.mujoco/mujoco210/bin
export CFLAGS="-I/home/gridsan/bgithinji/glew-install/include"
#export CUDA_VISIBLE_DEVICES=0

# Check if the environment exists on the local disk. If not copy it over from the home directory.
if [ ! -d "$MUJOCO_ENV" ]; then
    echo "~~ extracting $HOME/mujoco_env.zip to $MUJOCO_ENV_HOME ~~"
    mkdir -p $MUJOCO_ENV_HOME
    unzip "$HOME/mujoco_env.zip" -d $MUJOCO_ENV_HOME
fi

echo "~~ python env: '$MUJOCO_ENV' ~~"

# Load an anaconda module, then activate your mujoco environment
module load anaconda/2023a
ls -l $MUJOCO_ENV/bin/activate
if [ $? -eq 0 ]; then
    source $MUJOCO_ENV/bin/activate
else
    echo "~~ failed to activate python env ~~"
    exit -1
fi

echo "~~ check packages ~~"
pip list | grep mjrl


# Project setup
pip install --no-cache-dir -r $HOME/mj_envs_vision/requirements.txt
source $HOME/mj_envs_vision/setup_supercloud.bash
unset LD_PRELOAD

echo "~~ CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES ~~"
python -c "import torch;print(f'~~ cuda.is_available: {torch.cuda.is_available()} ~~')"
python -c "import torch;print(f'~~ cuda.device_count: {torch.cuda.device_count()} ~~')"

echo "~~ python $run_script ~~"
python $run_script > "submit_$(date +%Y%m%d_%H%M).out"
