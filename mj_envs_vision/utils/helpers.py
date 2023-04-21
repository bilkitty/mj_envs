import os
import mjrl
import gym
import math
import torch
import time
import numpy as np
from PIL import Image, ImageSequence
from typing import List, Tuple
from matplotlib import pyplot as plt
from gym import make as gym_make
from gym.wrappers.time_limit import TimeLimit
from dependencies.PlaNet.env import GYM_ENVS
from mj_envs_vision.utils.wrappers import STATE_KEY
from mj_envs_vision.utils.wrappers import CustomPixelObservationWrapper, GuiObservationWrapper


DEFAULT_HZ = 30


class Metrics:
  def __init__(self, group_size=1):
    self._group_size = group_size
    self._aggregate_fn = lambda x: list(np.array(x).reshape(self._group_size, -1).mean(axis=0))

  def items(self) -> dict:
    return {k: self._aggregate_fn(v) for k, v in self.__dict__.items() if not k.startswith('_')}

  def update(self, metric: dict) -> None:
    """
    Imports values from input dictionary. If values are of type torch.Tensor,
    make sure to call .item() or .tolist() to move them to cpu. This is not
    performant, but will save memory.
    """
    raise NotImplementedError

class BasicTimer:
    CONV_FACTORS = dict(m=60*1e-9, s=1e-9, ms=1e-6, ns=1)
    def __init__(self, units='s'):
      if units not in self.CONV_FACTORS.keys():
        raise Exception(f"unkown units {units}")
      self.units = units
      self._timings = dict()
      self._names = list()

    def start(self, name):
      if name in self._names and self._timings[name][-1] >= 0:
        self._timings[name].append(-1 * time.time_ns())
      else:
        self._timings[name] = [-1 * time.time_ns()]
        self._names.append(name)

    def stop(self, name):
      if name in self._names and self._timings[name][-1] < 0:
        self._timings[name][-1] += time.time_ns()
        self._timings[name][-1] *= self.CONV_FACTORS[self.units]
      else:
        print(f"WARN: timer '{name}' was not started")

    def reset(self):
      self._timings = dict()
      self._names = list()

    def dump(self):
      return self._timings.copy()


# Helpers for gym envs
def is_from_adroit_suite(env_str: str):
  return isinstance(env_str, str) and env_str in ['door-v0', 'pen-v0', 'hammer-v0', 'relocate-v0']

def is_from_continuous_control_gym_suite(env_str: str):
  return isinstance(env_str, str) and env_str in GYM_ENVS

def is_valid_env(env_str: str):
  return is_from_continuous_control_gym_suite(env_str) or is_from_adroit_suite(env_str)

def action_size(env):
  return env.action_space.shape[0]

def observation_size(env):
  # aka, input channel size
  return env.observation_space.shape[-1]

def reset(env):
  return env.reset()

def step(env, action: torch.FloatTensor):
  """
  A helper that ensures the correct type is passed to env step.
  Note that unexpected behaviour will result if env takes
  action of type other than numpy.ndarray. (e.g., try dapg policies)
  """
  res = env.step(action.squeeze(dim=0).numpy())
  obs, reward, done = res[:3]
  # since it's undefined, default to success = false for standard gym envs
  success = res[-1]["goal_achieved"] if isinstance(env.unwrapped, mjrl.envs.mujoco_env.MujocoEnv) else False
  return obs, reward, done, success

def make_env(config):
    assert is_valid_env(config.env_name)
    render_kwargs = dict(render_mode='rgb_array', width=64, height=64)
    if is_from_adroit_suite(config.env_name):
      e = gym_make(config.env_name,
                   is_headless=config.nogui,
                   variation_type=config.variation_type,
                   **render_kwargs)
    else:
      e = gym_make(config.env_name,
                   **render_kwargs)
    #e.seed(config.seed)
    if isinstance(e, TimeLimit):
      e = e.env

    if not config.nogui:
      return GuiObservationWrapper(e)
    elif config.state_type == "observation":
      return CustomPixelObservationWrapper(e)
    elif config.state_type == "vector":
      return CustomPixelObservationWrapper(e, obs_key=STATE_KEY)
    else:
      raise Exception(f"Unsupported state type '{config.state_type}'")

# Helpers for processing samples drawn from experience
def flatten_sample(x):
  sh = x.shape # TODO: same as size?
  return x.view(sh[0] * sh[1], *sh[2:])

def expand(x_flat, sh):
  return x_flat.view(sh[0], sh[1], *x_flat.shape[1:])

def to_image_frame(obs: np.ndarray):
  """ converts image (used as model input) with pels in [-0.5, 0.5] to image with pels in [0, 255] """
  return (255 * (obs.transpose((1, 2, 0)) + 0.5)).astype('uint8')

def to_input_obs(frame: np.ndarray):
  """ converts image with pels in [0, 255] to image observation with pels in [-0.5, 0.5] for model input """
  return (frame.transpose((2, 0, 1)) / 255 - 0.5).astype('float')

def save_as_gif(frames: List[np.ndarray], gif_path: str, is_obs: bool=False, hz: float=DEFAULT_HZ):
  pils = list()
  for frame in frames:
    frame = to_image_frame(frame) if is_obs else frame.astype('uint8')
    pils.append(Image.fromarray(frame))

  durations = 1/hz * 1000
  pils[0].save(gif_path, append_images=pils, save_all=True, optimize=False, loop=1, duration=durations)

def grid_gif(gif_path: str, t_sample: int = 1, is_square: bool=True):
  n = max(2, math.ceil(math.sqrt(t_sample))) if is_square else t_sample
  m = max(2, math.floor(math.sqrt(t_sample))) if is_square else 1
  fig, ax = plt.subplots(m, n, figsize=(8, 5))
  if not is_square: ax = ax.reshape(1, -1)
  for x in range(0, t_sample):
    path = gif_path.replace("_0.gif", f"_{x}.gif")
    assert os.path.exists(path), f"{path}"
    im = Image.open(path).convert("RGB")
    ax[x // n, x % n].imshow(im)
    ax[x // n, x % n].axis('off')
    ax[x // n, x % n].set_xlabel(f"sample {x}")
    ax[x // n, x % n].set_box_aspect(1)

  fig.tight_layout(pad=0.2)
  fig.savefig(gif_path.replace("_0.gif", "_grid.png"))

def plot_rewards(rewards: List[Tuple], yaxis_label="total reward"):
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  ep = np.array([x[0] for x in rewards])
  rwd = np.array([x[1] for x in rewards])
  if len(rwd.shape) == 1:
    ax.plot(ep, rwd)
    rwd = rwd.reshape(-1, 1)
  else:
    mu, std, med = np.mean(rwd, axis=-1), np.std(rwd, axis=-1), np.median(rwd, axis=-1)
    ax.plot(ep, mu, linestyle='dashed', linewidth=0.3, label="mean")
    ax.plot(ep, med, linestyle='solid', linewidth=0.5, label="median")
    ax.fill_between(ep, mu - std, mu + std, alpha=0.05)

  ax.set_xlabel('epochs')
  ax.set_ylabel(f'{yaxis_label} n=({rwd.shape[-1]})')
  ax.legend(loc='upper right')
  ax.set_box_aspect(1)
  fig.tight_layout(pad=0.2)
  return fig

def plot_time(timings, max_epoch, y_axis_label="time"):
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(111)
  for k,v in timings.items():
    v = np.array(v)
    if len(v.shape) > 1:
      v = np.median(v, axis=0)
    ax.plot(np.linspace(0, max_epoch, v.shape[0], endpoint=False), v, linestyle='solid', linewidth=1, label=k)
    ax.scatter(np.linspace(0, max_epoch, v.shape[0], endpoint=False), v, marker='D', s=5)

  plt.xlabel('epochs')
  plt.ylabel(y_axis_label)
  plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
  ax.set_box_aspect(1)
  fig.tight_layout(pad=0.2)
  return fig

#
# primarily for debugging
#

def visualise_batch_from_experience(id, config, experience, out_dir):
  batch = experience.sample(min(config.batch_size, experience.idx - 1), min(config.chunk_size, experience.idx - 1))
  save_as_gif(batch[0].reshape(-1, *experience.observations.shape[1:]).cpu().numpy(),
              os.path.join(out_dir, f'experience_{id}.gif'),
              is_obs=True,
              hz=1)

def visualise_trajectory(id: str, trajectory: List, out_dir: str, prefix="trajectory"):
  save_as_gif([x[0].cpu().numpy() for x in trajectory], os.path.join(out_dir, f'{prefix}_{id}.gif'))

def visualise_state_trajectory(id: int, trajectory: List, out_dir: str, env: gym.Env):
  pass
