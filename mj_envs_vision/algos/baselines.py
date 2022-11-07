import torch
import numpy as np
from torch import optim
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
    #self.planning_horizon = 12 # consider moving to planner?

class Planet:
  def __init__(self, config, action_size, observation_size):
    self.planner = None
    self.params_list = list()
    self.models = dict(transition_model=models.TransitionModel(config.belief_size, config.state_size, action_size, config.hidden_size, config.embedding_size, config.activation_fn),
        observation_model=models.ObservationModel(config.state_type == "vector", observation_size, config.belief_size, config.state_size, config.embedding_size, config.activation_fn),
        encoder_model=models.Encoder(config.state_type == "vector", observation_size, config.embedding_size, config.activation_fn),
        reward_model=models.RewardModel(config.belief_size, config.state_size, config.hidden_size, config.activation_fn))

    # move models to device and collect parameters
    for m in self.models.values():
      m.to(device=config.device)
      self.params_list.extend(list(m.parameters()))

    # initialise optimiser
    self.optimiser = optim.Adam(self.params_list, lr=config.learning_rate, eps=config.adam_epsilon)


  def load_models(self, models_path: str):
    pass

  def update_models(self, state: np.ndarray):
    # state is a dictionary of belief, prior/post state, observation
    pass

  def get_action(self, state: np.ndarray):
    pass