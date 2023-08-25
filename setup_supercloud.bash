export PROJECT_ROOT="$HOME/mj_envs_vision"
export PYTHONPATH="$PROJECT_ROOT/:$PYTHONPATH"
export PYTHONPATH="$PROJECT_ROOT/dependencies/PlaNet/:$PYTHONPATH"
export PYTHONPATH="$PROJECT_ROOT/dependencies/Adroit/:$PYTHONPATH"
export PYTHONPATH="$PROJECT_ROOT/dependencies/DreamerV2/:$PYTHONPATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export CUDA_VISIBLE_DEVICES=0,1
# NOTE: if rendering offscreen, comment this line or use 'unset LD_PRELOAD'.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so
