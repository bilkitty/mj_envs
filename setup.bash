export PYTHONPATH="$(pwd)/mj_envs_vision/:$PYTHONPATH"
export PYTHONPATH="$(pwd)/dependencies/PlaNet/:$PYTHONPATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
# NOTE: depending on the machine, use 'unset LD_PRELOAD' if 
#       GL missing error occurs.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
