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
    # state_size = 30
    # embedding_size = 200
    # hidden_size = 200
    # overshooting_distance = 50
    # overshooting_kl_beta = 0
    # overshooting_reward_scale = 0
    # free_nats = 3
    # planning_horizon = 12
    pass

class Planet:
  def __init__(self, config):
    self.planner = None
    self.models = dict()
      #dict(transition_model=models.TransitionModel(config.belif_size, config.state_size, ...),
      #observation_model=models.ObservationModel(config.state_type == "vector", ...),
      #encoder_model=models.Encoder(config.state_type == "vector", ...),
      #reward_model=models.RewardModel(config.belief_size, config.state_size, config.hidden_size, config.activation_fn))

  def load_models(self, models_path: str):
    pass

  def update_models(self, state: np.ndarray):
    # state is a dictionary of belief, prior/post state, observation
    pass

  def get_action(self, state: np.ndarray):
    pass