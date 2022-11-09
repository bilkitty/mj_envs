import torch
import numpy as np
from dependencies.PlaNet import env
from dependencies.PlaNet import models
from dependencies.PlaNet.planner import MPCPlanner

# TODO: replace these with own generic implementation
from dependencies.PlaNet import memory
from dependencies.PlaNet import planner
from mj_envs_vision.utils.config import Config

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
    self.metrics = PlanetMetrics()
    # TODO: track prediction errors
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

    # global prior parameters for latent overshooting
    self.zero_mean = torch.zeros(config.batch_size, config.state_size, device=config.device)
    self.unit_var = torch.ones(config.batch_size, config.state_size, device=config.device)

  def load_models(self, models_path: str):
    pass

  def update(self, sample_batch: List, optimiser):
    assert len(sample_batch) == 4 # TODO: pack/unpack data
    obs, actions, rewards, not_done = sample_batch

    z_gt = expand(policy.models["encoder"](flatten_sample(obs[1:])), obs[1:].shape) # embeddings of gt next state
    b0 = torch.zeros(config.batch_size, config.belief_size, device=config.device) # TODO: does this init matter?
    x0 = torch.zeros(config.batch_size, config.state_size, device=config.device)
    fwd_pred = policy.models["transition"](x0, actions[:-1], b0, z_gt, not_done[:-1])
    beliefs, prior_x, prior_mean, prior_std = fwd_pred[:4]
    post_x, post_mean, post_std = fwd_pred[4:]
    # predict current reward and next obs
    r = expand(policy.models["reward"](flatten_sample(beliefs), flatten_sample(post_x)), beliefs.shape)
    o = expand(policy.models["observation"](flatten_sample(beliefs), flatten_sample(post_x)), beliefs.shape)
    # compute losses
    P = Normal(post_mean, post_std)
    Q = Normal(prior_mean, prior_std)
    nats = torch.full((1,), config.free_nats, dtype=torch.float32, device=config.device)
    mse_rewards = F.mse_loss(r, rewards[:-1], reduction='none')
    mse_pixels = F.mse_loss(o, obs[1:], reduction='none').sum(dim=(2, 3, 4))
    self.metrics.observation_loss.append(mse_pixels.mean(dim=(0, 1)))
    self.metrics.reward_loss.append(mse_rewards.mean(dim=(0, 1)))
    # TODO: include overshooting
    #state_prior = torch.distributions.Normal(zero_mean, unit_var)
    self.metrics.kl_loss.append(torch.max(kl_divergence(P, Q).sum(dim=2), nats).mean(dim=(0, 1)))
    L = self.metrics.current_loss()
    # update models
    optimiser.zero_grad()
    L.backward()
    nn.utils.clip_grad_norm_(policy.params_list, config.grad_clip_norm, norm_type=2)
    optimiser.step()


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
