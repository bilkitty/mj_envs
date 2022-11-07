import os
import numpy as np
from PIL import Image
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.pixel_observation import PixelObservationWrapper
from dependencies.PlaNet import memory
from dependencies.PlaNet import env
from mjrl.utils import gym_env
from mj_envs_vision.algos.baselines import Planet
from mj_envs_vision.algos.baselines import PlanetConfig


def train(config, env, experience, policy, optimiser):
  # sample from experience

  # update models

  # generate action

  # forward sim

  pass


def evaluate(config,  policy):
  # instantiate test env


  # close env
  pass


import torch
class EnvWrapper:
  def __init__(self, env, is_adroit):
    self.env = env
    self.is_adroit = is_adroit

  @property
  def action_size(self):
    return self.env.action_dim if self.is_adroit else self.env.action_size

  @property
  def observation_size(self):
    return self.env.observation_dim if self.is_adroit else self.env.observation_size

  def reset(self):
    return self.env.reset()

  def step(self, action):
    obs, reward, done = self.env.step(action)[:3]
    return obs, rwd, done

  def sample_action(self):
    if self.is_adroit:
      return torch.FloatTensor(self.env.action_space.sample())
    else:
      return torch.FloatTensor(self.env._env.action_space.sample())


def make_env(config):
  if config.env_name in ['door-v0', 'pen-v0', 'haammer-v0', 'relocate-v0']:
    e = gym_env.GymEnv(config.env_name)
    e.set_seed(config.seed)
    if isinstance(e.env, TimeLimit):
      e = e.env.env
    return EnvWrapper(e, is_adroit=True)
  else:
    assert config.env_name in env.GYM_ENVS
    e = env.GymEnv(config.env_name,
                      config.state_type == "vector",
                      config.seed,
                      config.max_episode_length,
                      config.action_repeat,
                      config.bit_depth)
    if isinstance(e._env, TimeLimit):
      e._env = e._env.env
    return EnvWrapper(e, is_adroit=False)

def to_image_frame(obs: np.ndarray):
  """ converts image observation with pels in [-0.5, 0.5] to image with pels in [0, 255] """
  return (255 * (obs.transpose((1, 2, 0)) + 0.5)).astype('uint8')

def to_input_obs(frame: np.ndarray):
  """ converts image with pels in [0, 255] to image observation with pels in [-0.5, 0.5] """
  return (frame.transpose((2, 0, 1)) / 255 - 0.5).astype('float')


if __name__ == "__main__":
  # TODO: sanity check
  #       compare metrics of this training loop with those of PlaNet/main.py

  # load user defined parameters
  config = PlanetConfig()
  config.load("mj_envs_vision/utils/test_config.json")
  print(config.str())

  # setup
  out_dir = os.path.join("results", f"train_{config.run_id}")
  os.makedirs(out_dir, exist_ok=True)
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  if torch.cuda.is_available() and not config.disable_cuda:
    config.device = torch.device('cuda')
    torch.cuda.manual_seed(config.seed)
  else:
    config.device = torch.device('cpu')


  # instantiate env
  E = make_env(config)

  # TODO: create worker setup and parallelise
  # initialise policy
  policy = Planet(config, E.action_size, E.observation_size)
  if config.models_path != "":
    policy.load_models(config.models_path)

  # initialise experience buffer (TODO: consider rewriting)
  experience = memory.ExperienceReplay(config.experience_size,
                                       config.state_type == "vector",
                                       E.observation_size,
                                       E.action_size,
                                       config.bit_depth,
                                       config.device)

  for n in range(config.seed_episodes):
    t, rwd, done = 0, 0.0, False
    obs = E.reset()
    while not done and t < config.max_episode_length:
      action = E.sample_action()
      experience.append(obs, action, rwd, done)
      obs, rwd, done = E.step(action)
      t = t + 1

  a, b = E.observation_size, E.action_size

  # visualise experience for debugging
  pils = list()
  batch = experience.sample(config.batch_size, config.chunk_size)
  for obs in batch[0].reshape(-1, *E.observation_size).cpu():
    pils.append(Image.fromarray(to_image_frame(obs.numpy())))

  pils[0].save(os.path.join(out_dir, f'experience_0.gif'),
               append_images=pils, save_all=True, optimize=False, loop=True, duration=len(pils) / 30)


  # for N episodes
  #   train on env
  #   (opt) evalute on test env
  #   collect experience

  # save performance metrics
  # visualise performance

  print("done :)")