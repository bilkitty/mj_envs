#!/bin/bash

if [[ $1 == 0 ]];then 
    is_batch=False
else
    is_batch=True
fi

if [[ $2 == "" ]];then 
    run_script=mujoco-test.py
else
    run_script=$2
fi

# Set some useful environment variables
export MUJOCO_ENV_HOME=$HOME/software/mujoco/mujoco_env
export MUJOCO_ENV=/state/partition1/user/$USER/mujoco_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia:/home/gridsan/bgithinji/.mujoco/mujoco210/bin

# Check if the environment exists on the local disk. If not copy it over from the home directory.
if [ ! -d "$MUJOCO_ENV" ]; then
    echo "~~ copying $MUJOCO_ENV_HOME to $MUJOCO_ENV ~~"
    mkdir -p /state/partition1/user/$USER
    #cp -r $MUJOCO_ENV_HOME $MUJOCO_ENV
    unzip "mujoco_env.zip" -d $MUJOCO_ENV
fi

# Load an anaconda module, then activate your mujoco environment
module load anaconda/2023a
source $MUJOCO_ENV/bin/activate

if [[ $is_batch ]];then
    echo "~~ running as batch ~~"
    #SBATCH -c 4
    #SBATCH --gres=gpu:volta:1 
fi

python $run_script
squeue -u $USER 
