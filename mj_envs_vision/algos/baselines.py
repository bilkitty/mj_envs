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
    # chunk_size = 50
    # state_size = 30
    # embedding_size = 200
    # hidden_size = 200
    # belief_size = 50
    # overshooting_distance = 50
    # overshooting_kl_beta = 0
    # overshooting_reward_scale = 0
    # free_nats = 3
    # planning_horizon = 12
    pass

class Planet:
  def __init__(self):
    self.planner = None
    self.transition_model = None
    self.observation_model = None
    self.encoder_model = None
    self.reward_model = None
    pass

  def update_models(self, state):
    # state is a dictionary of belief, prior/post state, observation
    pass

  def get_action(self, state):
    pass