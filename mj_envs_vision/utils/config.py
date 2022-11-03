# This file contains parameter specs and types for
# various algorithms, envs, training and test
# procedures

class Config:
  def __init__(self):
    # General parameters
    self.seed = 0
    self.disable_cuda = False
    # Algorithm parameters
    self.seed_episodes = 5
    self.max_episodes = 1000
    self.max_episode_length = 500
    self.experience_size = 1000000
    self.data_collection_interval = 100
    self.training_epochs = 1000
    self.test_interval = 100
    self.activation_fn = 'relu'
    self.action_noise = 0.3
    self.learning_rate = 3e-3
    self.learning_rate_factor = 0 # linearly ramp up to learning rate with this slope
    self.adam_epsilon = 1e-4
    self.grad_clip_norm = 1000
    # Environment parameters
    self.env_name = None
    self.action_repeat = 2
    self.state_type = 'observation'
    # bit_depth

    # TODO: select params after planning horizon

  def load(self, filepath: str):
    # TODO: use attributes
    pass

  def save(self, filepath: str):
    pass

  # TODO: get attr?