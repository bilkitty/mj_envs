# Vision-based Adroit Environments
This repo is an adaptation of `mj_envs` dexterious manipulation task suite (see https://github.com/vikashplus/mj_envs.git).


## Setup Mods

package updates

* requirements.txt: gym, numpy, torch, torchvision, imageio, opencv_python

* setup.py: 'mujoco-py<2.2,>=2.1.2' (if using mujoco 2.1)

```
python3 -m pip install -r requirements.txt
```

## Quick Started

Download and setup Mujoco 2.1

Avaialble at https://github.com/deepmind/mujoco/releases/tag/2.1.2

```
mkdir -p /root/.mujoco/
tar -xzf /mujoco2.1.1.tar.gz --directory /root/.mujoco/
ln -s /root/.mujoco/mujoco-2.1.1 /root/.mujoco/mujoco2.1.1_linux || true
ln -s /root/.mujoco/mujoco2.1.1_linux /root/.mujoco/mujoco210

```

Download source

```
$ git clone --recursive https://github.com/bilkitty/mj_envs.git
```

Setup refs to (a) this package, (b) mujoco bins and (c) graphics driver libs for driver preload
```
export PYTHONPATH="<path/to/mj_envs_vision_vision>:$PYTHONPATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so
```

Visualisation test

```
$ python utils/visualize_env.py --env_name hammer-v0
```
