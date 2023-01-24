# This file contains parameter specs and types for
# various algorithms, envs, training and test
# procedures
import json


class Config(json.JSONEncoder):
  def __init__(self):
    super().__init__()

    # General parameters
    self.run_id = 0
    self.seed = 0
    self.device_type = 'cpu'
    self.disable_cuda = False
    self.models_path = ""
    self.nogui = True
    # Algorithm parameters
    self.seed_episodes = 2
    self.max_episodes = 1000
    self.max_episode_length = 500
    self.experience_size = 1000000
    self.sample_iters = 1000
    self.test_interval = 100
    self.activation_fn = 'relu'
    self.action_noise = 0.3
    self.learning_rate = 1e-3
    self.learning_rate_factor = 0 # linearly ramp up to learning rate with this slope
    self.adam_epsilon = 1e-4
    self.grad_clip_norm = 1000
    self.candidates = 1000
    self.top_candidates = 100
    self.checkpoint_interval = 100
    # Environment parameters
    self.env_name = None
    self.action_repeat = 2
    self.state_type = 'observation'
    self.bit_depth = 5
    # Memory parameters
    self.batch_size = 50
    self.chunk_size = 50

    # TODO: select params after planning horizon
  def default(self, o):
    # TODO: filter out base class attributes
    return o.__dict__

  def load(self, filepath: str):
    # TODO: use attributes
    fp = open(filepath, 'rb')
    if "json" in filepath:
        cfg = json.load(fp)
    else:
      raise Exception("only json configs are supported atm")

    # TODO: is it okay to use __dict__?
    for att, v in cfg.items():
      if att in list(self.__dict__.keys()):
        self.__dict__[att] = v
      else:
        print(f"No such config field, '{att}'.")

    fp.close()

  def save(self, filepath: str):
    fp = open(filepath, 'w')
    if "json" in filepath:
        json.dump(Config().encode(self), fp, indent=2)
    else:
      raise Exception("only json configs are supported atm")

  def str(self):
    s = "Parameters:\n"
    for att, v in self.__dict__.items():
      s += f"\t\t{att:<25} = {v}\n"
    return s

  # TODO: get attr?

class PlanetConfig(Config):
  def __init__(self):
    Config.__init__(self)
    self.belief_size = 200
    self.state_size = 30
    self.embedding_size = 1024
    self.hidden_size = 200
    self.overshooting_distance = 50
    self.overshooting_kl_beta = 0
    self.overshooting_reward_scale = 0
    self.free_nats = 3
    self.planning_horizon = 12
    self.optimisation_iters = 10

class PPOConfig(Config):
  def __init__(self):
    Config.__init__(self)
    self.model_type = "mlp"
