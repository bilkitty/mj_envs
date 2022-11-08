import torch
import numpy as np
from dependencies.PlaNet import env
from dependencies.PlaNet import models
from dependencies.PlaNet.planner import MPCPlanner

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
  def __init__(self, config, action_size, observation_size, action_space):
    self.batch_size = config.batch_size
    self.belief_size = config.belief_size
    self.state_size = config.state_size
    self.action_noise = config.action_noise
    self.action_size = action_size
    self.action_space = action_space
    self.device = config.device
    self.initialise()
    self.models = dict(transition=models.TransitionModel(config.belief_size, config.state_size, action_size, config.hidden_size, config.embedding_size, config.activation_fn),
        observation=models.ObservationModel(config.state_type == "vector", observation_size, config.belief_size, config.state_size, config.embedding_size, config.activation_fn),
        encoder=models.Encoder(config.state_type == "vector", observation_size, config.embedding_size, config.activation_fn),
        reward=models.RewardModel(config.belief_size, config.state_size, config.hidden_size, config.activation_fn))

    # move models to device and collect parameters
    self.params_list = list()
    for m in self.models.values():
      m.to(device=config.device)
      self.params_list.extend(list(m.parameters()))

    # initialise mpc planner
    self.planner = MPCPlanner(action_size,
                              config.planning_horizon,
                              config.optimisation_iters,
                              config.candidates,
                              config.top_candidates,
                              self.models["transition"],
                              self.models["reward"],
                              float(action_space.low[0]),
                              float(action_space.high[0]))


  def load_models(self, models_path: str):
    pass

  def update_models(self, obs: np.ndarray):
    # state is a dictionary of belief, prior/post state, observation
    pass

  def initialise(self, *args, **kwargs):
    count = kwargs["count"] if "count" in list(kwargs.keys()) else self.batch_size
    self.b = torch.zeros(count, self.belief_size, device=self.device) # TODO: does this init matter?
    self.x = torch.zeros(count, self.state_size, device=self.device)
    self.a = torch.zeros(count, self.action_size, device=self.device)

  def act(self, obs: torch.Tensor):
    # state estimation and fwd prediction
    z = self.models["encoder"](obs.to(self.device)).unsqueeze(dim=0)
    action = self.a.unsqueeze(dim=0)
    fwd_pred = self.models["transition"](self.x, action, self.b, z)
    self.b, self.x = fwd_pred[0].squeeze(dim=0), fwd_pred[4].squeeze(dim=0)
    # action selection
    self.a = self.planner(self.b, self.x)
    return self.a

  def sample_action(self, obs: torch.Tensor):
    # apply uniform exploration noise
    self.a = self.act(obs) + self.action_noise * torch.rand_like(self.a)
    self.a.clamp_(float(self.action_space.low[0]), float(self.action_space.high[0]))
    return self.a
