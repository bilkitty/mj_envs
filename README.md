# Mujoco Environments
`mj_envs` is a collection of environments/tasks simulated with the [Mujoco](http://www.mujoco.org/) physics engine and wrapped in the OpenAI `gym` API.

## Getting Started
`mj_envs` uses git submodules to resolve dependencies. Please follow steps exactly as below to install correctly.

1. Clone this repo with pre-populated submodule dependencies
```
$ git clone --recursive https://github.com/vikashplus/mj_envs.git
```
2. Install package using `pip`
```
$ pip install -e .
```
**OR**
Add repo to pythonpath by updating `~/.bashrc` or `~/.bash_profile`
```
export PYTHONPATH="<path/to/mj_envs_vision_vision>:$PYTHONPATH"
```
3. You can visualize the environments with random controls using the below command
```
$ python utils/visualize_env.py --env_name hammer-v0
```
**NOTE:** If the visualization results in a GLFW error, this is because `mujoco-py` does not see some graphics drivers correctly. This can usually be fixed by explicitly loading the correct drivers before running the python script. See [this page](https://github.com/aravindr93/mjrl/tree/master/setup#known-issues) for details.



requirements.txt updates
setup.py remove requirement 'mujoco-py<2.2,>=2.0' to avoid this error:
  numpy https://github.com/openai/mujoco-py/issues/607
    File "/home/bilkit/anaconda3/envs/mj_env/lib/python3.8/site-packages/mujoco_py/builder.py", line 125, in load_dynamic_ext
      return loader.load_module()
    File "mujoco_py/cymj.pyx", line 1, in init mujoco_py.cymj
  ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 96 from C header, got 80 from PyObject

