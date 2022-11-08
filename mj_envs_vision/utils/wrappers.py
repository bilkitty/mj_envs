import torch
from mjrl.utils import gym_env
from dependencies.PlaNet import env
from gym.wrappers.time_limit import TimeLimit


class EnvWrapper:
  def __init__(self, env, is_adroit):
    self.env = env
    self.is_adroit = is_adroit

  @property
  def action_space(self):
    return self.env.action_space if self.is_adroit else self.env._env.action_space

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
    return obs, reward, done

  def sample_action(self):
    if self.is_adroit:
      return torch.FloatTensor(self.env.action_space.sample())
    else:
      return torch.FloatTensor(self.env._env.action_space.sample())


def make_env(config):
  if config.env_name in ['door-v0', 'pen-v0', 'hammer-v0', 'relocate-v0']:
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