import torch
import numpy as np
from dependencies.PlaNet import env
from dependencies.PlaNet import models

# TODO: replace these with own generic implementation
from dependencies.PlaNet import memory
from dependencies.PlaNet import planner
from mj_envs_vision.utils.config import Config

# TODO: base

class PlanetConfig(Config):
  def __init__(self):
    Config.__init__(self)
    self.belief_size = 200
    self.state_size = 30
    self.embedding_size = 200
    self.hidden_size = 200
    self.overshooting_distance = 50
    self.overshooting_kl_beta = 0
    self.overshooting_reward_scale = 0
    self.free_nats = 3


class Metrics: # todo: move out
  def __init__(self):
    self.total_return = list()

class PlanetMetrics(Metrics):
  def __init__(self):
    Metrics.__init__(self)
    self.observation_loss = list()
    self.reward_loss = list()
    self.kl_loss = list()
    self.time = list()

  def current_loss(self):
    return self.observation_loss[-1] + self.reward_loss[-1] + self.kl_loss[-1]

  def total_loss(self):
    return dict(observation_loss=[v.item() for v in self.observation_loss],
                reward_loss=[v.item() for v in self.reward_loss],
                kl_loss=[v.item() for v in self.kl_loss])

class Planet:
  def __init__(self, config, action_size, observation_size):
    self.planner = None
    self.params_list = list()
    self.models = dict(transition=models.TransitionModel(config.belief_size, config.state_size, action_size, config.hidden_size, config.embedding_size, config.activation_fn),
        observation=models.ObservationModel(config.state_type == "vector", observation_size, config.belief_size, config.state_size, config.embedding_size, config.activation_fn),
        encoder=models.Encoder(config.state_type == "vector", observation_size, config.embedding_size, config.activation_fn),
        reward=models.RewardModel(config.belief_size, config.state_size, config.hidden_size, config.activation_fn))

    # move models to device and collect parameters
    for m in self.models.values():
      m.to(device=config.device)
      self.params_list.extend(list(m.parameters()))


  def load_models(self, models_path: str):
    pass

  def update_models(self, state: np.ndarray):
    # state is a dictionary of belief, prior/post state, observation
    pass

  def get_action(self, state: np.ndarray):
    pass