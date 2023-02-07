import numpy
import torch
from gym import spaces
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.step_api_compatibility import StepAPICompatibility


PIXELS_KEY = "pixels"
STATE_KEY = "state"


class StateActionSpec:
  def __init__(self, action_space, observation_space):
    self.action_dim = action_space.shape[0]
    self.observation_dim = observation_space.shape[0]


class CustomPixelObservationWrapper(PixelObservationWrapper):
  def __init__(self, env, obs_key=PIXELS_KEY, render_kwargs=None, action_repeat=1):
    env = StepAPICompatibility(env, output_truncation_bool=True) # convert any envs from old ot new api
    pks = () if render_kwargs is None else ("pixels",)
    super().__init__(env, pixels_only=False, render_kwargs=render_kwargs, pixel_keys=pks) # TODO: too wasteful to keep both state/pels?
    self.env_spec = StateActionSpec(env.action_space, env.observation_space)
    self.action_repeat = action_repeat
    self.max_episode_length = 200 # TODO: dont hard code
    self.timer = 0
    self.obs_key = obs_key

    # if pixels, uses pil image format
    self.curr_obs = super().reset()[0]
    space = super().observation_space[obs_key]
    lo = int(space.low_repr) if space.dtype == numpy.integer else 0
    hi = int(space.high_repr) if space.dtype == numpy.integer else 1
    self.observation_space = spaces.Box(
      shape=self.curr_obs[obs_key].shape, low=lo, high=hi, dtype=space.dtype
    )

  def reset(self):
    self.timer = 0
    items = super().reset()
    self.curr_obs = items[0]
    return torch.FloatTensor(self.curr_obs[self.obs_key].copy()), *items[1:]

  def step(self, action):
    # execute multiple repeats of action
    items = super().step(action)
    self.curr_obs = items[0]
    self.timer += 1
    for i in range(self.action_repeat - 1):
      i_items = super().step(action)
      if items[2] == True or i_items[2] == True or self.timer > self.max_episode_length: # check done flag
        break
      self.timer += 1
      self.curr_obs = i_items[0] # update obs
      items[1] += i_items[1] # accumulate rewards

    return torch.FloatTensor(self.curr_obs[self.obs_key].copy()), *items[1:]

  def get_pixels(self) -> torch.FloatTensor:
    return torch.FloatTensor(self.curr_obs[PIXELS_KEY].copy())

  def get_state(self) -> torch.FloatTensor:
    return torch.FloatTensor(self.curr_obs[STATE_KEY].copy())
