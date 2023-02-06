import os
import mjrl
import numpy as np
from PIL import Image
from typing import List, Tuple
from matplotlib import pyplot as plt
from gym import make as gym_make
from gym.wrappers.time_limit import TimeLimit
from dependencies.PlaNet.env import GYM_ENVS
from mj_envs_vision.utils.wrappers import CustomObservationWrapper
from mj_envs_vision.utils.wrappers import CustomPixelObservationWrapper


GIF_DURATION = 15


class Metrics:
  def __init__(self):
    self.total_return = list()


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

def step(env, action):
  if isinstance(env.unwrapped, mjrl.envs.mujoco_env.MujocoEnv):
    obs, reward, done, success = env.step(action)[:4]
  else:
    obs, reward, done = env.step(action)[:3]
    success = False # false = unknown
  return obs, reward, done, success

def make_env(config):
    assert is_valid_env(config.env_name)
    e = gym_make(config.env_name,
                 render_mode='rgb_array',
                 width=64,
                 height=64,
                 is_headless=config.nogui,
                 variation_type=config.variation_type)
    #e.seed(config.seed)
    if isinstance(e, TimeLimit):
      e = e.env
    if config.state_type == "vector":
      return CustomObservationWrapper(e)
    elif config.state_type == "observation":
      return CustomPixelObservationWrapper(e)
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

def save_as_gif(frames: List[np.ndarray], gif_path: str, is_obs: bool=False):
  pils = list()
  for frame in frames:
    frame = to_image_frame(frame) if is_obs else frame.astype('uint8')
    pils.append(Image.fromarray(frame))
  pils[0].save(gif_path, append_images=pils, save_all=True, optimize=False, loop=True, duration=GIF_DURATION)

def plot_rewards(rewards: List[Tuple], yaxis_label="total reward"):
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  ep = np.array([x[0] for x in rewards])
  rwd = np.array([x[1] for x in rewards])
  x = len(rwd.shape)
  if len(rwd.shape) == 1:
    ax.plot(ep, rwd)
    rwd = rwd.reshape(1, -1)
  else:
    mu, std, med = np.mean(rwd, axis=-1), np.std(rwd, axis=-1), np.median(rwd, axis=-1)
    ax.plot(ep, mu, linestyle='dashed', linewidth=0.3, label="mean")
    ax.plot(ep, med, linestyle='solid', linewidth=0.5, label="median")
    ax.fill_between(ep, mu - std, mu + std, alpha=0.05)

  ax.set_xlabel('epochs')
  ax.set_ylabel(f'{yaxis_label} n=({rwd.shape[-1]})')
  ax.legend(loc='upper right')
  return fig

#
# primarily for debugging
#

def visualise_batch_from_experience(id, config, experience, out_dir):
  batch = experience.sample(min(config.batch_size, experience.idx - 1), min(config.chunk_size, experience.idx - 1))
  save_as_gif(batch[0].reshape(-1, *experience.observations.shape[1:]).cpu().numpy(), os.path.join(out_dir, f'experience_{id}.gif'))

def visualise_trajectory(id, trajectory: List, out_dir):
  save_as_gif([x[0].cpu().numpy() for x in trajectory], os.path.join(out_dir, f'trajectory_{id}.gif'))
