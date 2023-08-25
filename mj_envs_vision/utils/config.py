# This file contains parameter specs and types for
# various algorithms, envs, training and test
# procedures
import json
import os.path

import yaml


class Config:
  def __init__(self):
    super().__init__()

    # General parameters
    self.run_id = 0
    self.seed = 0
    self.device_type = 'cpu'
    self.device_is_multicore = False
    self.disable_cuda = False
    self.models_path = ""
    self.models_path_bootstrap = ""
    self.bootstrap_interval = 1
    self.bootstrap_mechanism = None
    self.bootstrap_factor_inv = 1
    self.log_path = None
    self.nogui = True
    # Algorithm parameters
    self.seed_episodes = 2
    self.max_episodes = 1000
    self.max_episode_length = 500
    self.experience_size = 1000000
    self.train_epochs = 1000
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
    self.variation_type = None
    self.image_width = 64
    self.image_height = 64
    self.enable_resize = True
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
    if isinstance(cfg, str):
      cfg = json.loads(cfg)

    for att, v in cfg.items():
      if att in list(self.__dict__.keys()):
        if "_path" in att:
          self.__dict__[att] = os.path.expandvars(v)
        else:
          self.__dict__[att] = v
      else:
        print(f"No such config field, '{att}'.")

    fp.close()

  def save(self, filepath: str):
    fp = open(filepath, 'w')
    if "json" in filepath:
        json.dump(self, fp, default=lambda x: x.__dict__, indent=2)
    else:
      raise Exception("only json configs are supported atm")

  def str(self):
    s = "Parameters:\n"
    for att, v in self.__dict__.items():
      s += f"\t\t{att:<25} = {v}\n"
    return s

  # TODO: get attr?

class DefaultPlanetConfig(Config):
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
    # collect_interval = 100 is analogous to training epochs

class DefaultPPOConfig(Config):
  def __init__(self):
    Config.__init__(self)
    self.model_type = "mlp"
    self.entropy = 0.0


class DefaultDreamerConfig(Config):

  # TODO: pare down and save as yaml in this proj
  default_config_path = os.path.expandvars("$PROJECT_ROOT/dependencies/DreamerV2/config/defaults.yaml")
  def __init__(self):
    Config.__init__(self)
    with open(DefaultDreamerConfig.default_config_path, 'r') as fp:
      multi_cfg = yaml.safe_load(fp)

    for att, v in multi_cfg['defaults'].items():
      self.__dict__[att] = v


def load_config(config_path, policy_type):
  if policy_type == "ppo":
    config = DefaultPPOConfig()
  elif policy_type == "planet":
    config = DefaultPlanetConfig()
  elif policy_type == "dreamer":
    config = DefaultDreamerConfig()
  else:
    config = Config()

  # Override with config file
  config.load(config_path)
  print(config.str())

  return config