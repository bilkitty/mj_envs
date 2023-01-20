import numpy
import torch
from gym import spaces
from gym import ObservationWrapper
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.step_api_compatibility import StepAPICompatibility


class StateActionSpec:
  def __init__(self, action_space, observation_space):
    self.action_dim = action_space.shape[0]
    self.observation_dim = observation_space.shape[0]

class CustomObservationWrapper(ObservationWrapper):
  def __init__(self, env, action_repeat=1):
    env = StepAPICompatibility(env, output_truncation_bool=True) # convert any envs from old ot new api
    super().__init__(env)
    self.env_spec = StateActionSpec(env.action_space, env.observation_space)
    self.action_repeat = action_repeat
    self.max_episode_length = 200 # TODO: dont hard code
    self.timer = 0

  def reset(self):
    self.timer = 0
    return super().reset()

  def step(self, action):
    # execute multiple repeats of action
    items = super().step(action)
    obs = items[0]
    self.timer += 1
    for i in range(self.action_repeat - 1):
      i_items = super().step(action)
      if items[2] == True or i_items[2] == True or self.timer > self.max_episode_length: # check done flag
        break
      self.timer += 1
      obs = i_items[0] # update obs
      items[1] += i_items[1] # accumulate rewards

    return obs, *items[1:]

class CustomPixelObservationWrapper(PixelObservationWrapper):
  def __init__(self, env, pixels_only=True, render_kwargs=None, action_repeat=1):
    env = StepAPICompatibility(env, output_truncation_bool=True) # convert any envs from old ot new api
    super().__init__(env, pixels_only, render_kwargs)
    self.env_spec = StateActionSpec(env.action_space, env.observation_space)
    self.action_repeat = action_repeat
    self.max_episode_length = 200 # TODO: dont hard code
    self.timer = 0

    # use pil image format
    test_obs = super().reset()[0]['pixels']
    space = super().observation_space['pixels']
    lo = int(space.low_repr) if space.dtype == numpy.integer else 0
    hi = int(space.high_repr) if space.dtype == numpy.integer else 1
    self.observation_space = spaces.Box(
      shape=test_obs.shape, low=lo, high=hi, dtype=space.dtype
    )

  def reset(self):
    # call base class to get pixels
    self.timer = 0
    items = super().reset()
    return torch.FloatTensor(items[0]['pixels'].copy()), *items[1:]

  def step(self, action):
    # call base class to get pixels & execute multiple repeats of action
    items = super().step(action)
    obs = items[0]
    self.timer += 1
    for i in range(self.action_repeat - 1):
      i_items = super().step(action)
      if items[2] == True or i_items[2] == True or self.timer > self.max_episode_length: # check done flag
        break
      self.timer += 1
      obs = i_items[0] # update obs
      items[1] += i_items[1] # accumulate rewards

    return torch.FloatTensor(obs['pixels'].copy()), *items[1:]
