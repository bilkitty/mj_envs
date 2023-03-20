from gym import Env
from typing import List
import os
import glob
import torch
import pickle
from torch import nn
from torch import optim
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence
from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from dependencies.PlaNet import models, memory
from dependencies.PlaNet.planner import MPCPlanner
from dependencies.PlaNet.env import _images_to_observation
from dependencies.DreamerV2.pydreamer.models.dreamer import WorldModel, ActorCritic

# TODO: replace these with own generic implementation
from mjrl.policies.gaussian_mlp import MLP
from mj_envs_vision.utils.helpers import Metrics
from mj_envs_vision.utils.helpers import expand, flatten_sample
from mj_envs_vision.utils.helpers import action_size, observation_size
from mj_envs_vision.utils.config import Config
from mj_envs_vision.utils.wrappers import StateActionSpec


SUPPORTED_POLICIES = ["default", "dapg", "planet", "ppo", "dreamer"]


def make_baseline_policy(config: Config, policy_type: str, env: Env, device: torch.device):
  assert policy_type in SUPPORTED_POLICIES, f"Unsupported policy type '{policy_type}'"
  if policy_type == "default":
    return MLPBaseline(config, env, is_random=True)
  elif policy_type == "dapg":
    return MLPBaseline(config, env)
  elif policy_type == "ppo": # TODO: explicitly cast config
    return PPOBaseline(config, env)
  elif policy_type == "planet":
    return Planet(config, action_size(env), observation_size(env), env.action_space, device)
  elif policy_type == "dreamer":
    return Dreamer(config, action_size(env), observation_size(env), device)

def make_policy_optimisers(config: Config, policy_type: str, policy):
  optims = list()
  if policy_type == "ppo" or policy_type == "planet":
    optims.append(optim.Adam(policy.params_list, lr=config.learning_rate, eps=config.adam_epsilon))
  elif policy_type == "dreamer":
    optims.append(torch.optim.AdamW(list(policy.agent.actor.parameters()), lr=config.adam_lr_actor, eps=config.adam_eps))
    optims.append(torch.optim.AdamW(list(policy.agent.critic.parameters()), lr=config.adam_lr_critic, eps=config.adam_eps))
    optims.append(torch.optim.AdamW(list(policy.worldmodel.parameters()), lr=config.adam_lr, eps=config.adam_eps))
  else:
    return list()
  return optims

class MLPBaseline:
  def __init__(self, config, custom_env: Env, is_random: bool = False):
    assert not is_random or (is_random and config.models_path is None), "cannot be random if policy was specified"

    # TODO: support for images
    self.name = "mlp"
    self.mlp = None
    self.is_random = is_random
    self.seed = config.seed
    self.env_name = config.env_name
    self.env_spec = StateActionSpec(custom_env.action_space, custom_env.observation_space)
    self.observation_shape = custom_env.observation_space.shape
    self.models_path = config.models_path
    self.metrics = Metrics()
    self.experience = None

  def reset(self, *args, **kwargs):
    pass

  def set_models_to_eval(self):
    pass

  def set_models_to_train(self):
    pass

  def load(self) -> str:
    if self.is_random:
      self.mlp = MLP(self.env_spec, hidden_sizes=(32, 32), seed=self.seed, init_log_std=-1.0)
    else:
      self.mlp = pickle.load(open(self.models_path, 'rb'))
      assert self.observation_shape[-1] == self.mlp.param_shapes[0][-1], "observation is incompatible with model input"
    return self.models_path

  def save(self, models_path: str):
    self.models_path = models_path
    pickle.dump(self.mlp, open(models_path, 'wb'))

  def update(self, sample_batch: List) -> None: # TODO: update to compute_loss
    pass

  def act(self, obs: torch.Tensor) -> torch.FloatTensor:
    return torch.FloatTensor(self.mlp.get_action(obs.numpy())[1]['evaluation'])

  def sample_action(self, obs: torch.Tensor) -> torch.FloatTensor:
    return torch.FloatTensor(self.mlp.get_action(obs.numpy())[0])


class PPOMetrics(Metrics):
  def __init__(self):
    Metrics.__init__(self)
    self.policy_gradient_loss = list()
    self.value_loss = list()
    self.entropy_loss = list()
    self.approx_kl = list()
    self.n_updates = list()
    self.clip_fraction = list()
    self.time = list()

  def total_loss(self):
    return dict(policy_gradient_loss=self.policy_gradient_loss,
                value_loss=self.value_loss,
                entropy_loss=self.entropy_loss,
                approx_kl=self.approx_kl)

class PPOBaseline:
  def __init__(self, config, custom_env: Env):
    self.name = "ppo"
    self.ppo = None
    self.env_name = config.env_name
    self.log_interval = config.checkpoint_interval
    self.learning_rate = config.learning_rate
    self.n_steps = config.sample_iters
    self.epochs = config.max_episodes
    self.batch_size = config.batch_size
    self.target_kl = None
    self.models_path = config.models_path
    self.metrics = PPOMetrics()

    if config.model_type == "mlp":
      policy_type = ActorCriticPolicy#(observation_space, action_space, lr_schedule=lambda x: 0.0)
    elif config.model_type == "cnn":
      policy_type = ActorCriticCnnPolicy#(observation_space, action_space, lr_schedule=lambda x: 0.0)
    else:
      raise Exception(f"unsupported model type {config.model_type}")

    self.ppo = PPO(policy_type,
              custom_env,
              self.learning_rate,
              n_steps=self.n_steps,
              n_epochs=self.epochs,
              batch_size=self.batch_size,
              target_kl=self.target_kl,
              tensorboard_log=config.log_path)

    self.params_list = [Parameter(torch.zeros(3))] # sb3 internally updates params

  def reset(self, *args, **kwargs):
    """ re-initialise generic policy """
    pass

  def set_models_to_eval(self):
    pass

  def set_models_to_train(self):
    pass

  def load(self) -> str:
    models_path = self.models_path
    if os.path.isdir(models_path):
      paths = glob.glob(os.path.join(models_path, "*.zip"))
      if len(paths) == 0:
        raise Exception(f"Failed to load models in {models_path}")
      else:
        models_path = sorted(paths)[-1]

    print(f"Loading pre-trained model '{models_path}'")
    self.ppo.load(models_path)

    return models_path

  def save(self, models_path: str):
    self.models_path = models_path
    self.ppo.save(models_path)

  def update(self, sample_batch: List) -> None:
    self.ppo.learn(self.n_steps, log_interval=self.log_interval, progress_bar=True)
    if self.ppo.logger.name_to_value and len(self.ppo.logger.name_to_value) > 0:
      self.metrics.value_loss.append(self.ppo.logger.name_to_value['train/value_loss'])
      self.metrics.entropy_loss.append(self.ppo.logger.name_to_value['train/entropy_loss'])
      self.metrics.policy_gradient_loss.append(self.ppo.logger.name_to_value['train/policy_gradient_loss'])
      self.metrics.approx_kl.append(self.ppo.logger.name_to_value['train/approx_kl'])
      self.metrics.n_updates.append(self.ppo.logger.name_to_value['train/n_updates'])
      self.metrics.clip_fraction.append(self.ppo.logger.name_to_value['train/clip_fraction'])

  def act(self, obs: torch.Tensor) -> torch.FloatTensor:
    action, state = self.ppo.predict(obs.numpy())
    return torch.FloatTensor(action)

  def sample_action(self, obs: torch.Tensor) -> torch.FloatTensor:
    a = self.act(obs)
    return a + 0.3 * torch.rand_like(a)


class PlanetMetrics(Metrics):
  def __init__(self):
    Metrics.__init__(self)
    self.observation_loss = list()
    self.reward_loss = list()
    self.kl_loss = list()
    self.time = list()

  def total_loss(self):
    return dict(observation_loss=self.observation_loss, reward_loss=self.reward_loss, kl_loss=self.kl_loss)


class Planet:
  def __init__(self, config, action_size, observation_size, action_space, device):
    self.name = "planet"
    self.free_nats = config.free_nats
    self.batch_size = config.batch_size
    self.chunk_size = config.chunk_size
    self.belief_size = config.belief_size
    self.state_size = config.state_size
    self.action_noise = config.action_noise
    self.action_size = action_size
    self.action_space = action_space
    self.device = device
    self.models_path = config.models_path
    self.metrics = PlanetMetrics()
    # TODO: track prediction errors + provide reconstructions for vis
    self.reset()
    self.models = dict(transition=models.TransitionModel(config.belief_size, config.state_size, action_size, config.hidden_size, config.embedding_size, config.activation_fn),
        observation=models.ObservationModel(config.state_type == "vector", observation_size, config.belief_size, config.state_size, config.embedding_size, config.activation_fn),
        encoder=models.Encoder(config.state_type == "vector", observation_size, config.embedding_size, config.activation_fn),
        reward=models.RewardModel(config.belief_size, config.state_size, config.hidden_size, config.activation_fn))
    self.experience = memory.ExperienceReplay(config.experience_size,
                                         config.state_type == "vector",
                                         observation_size,
                                         action_size,
                                         config.bit_depth,
                                         device)

    # move models to device and collect parameters
    self.params_list = list()
    for m in self.models.values():
      m.to(device=self.device)
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
    self.zero_mean = torch.zeros(config.batch_size, config.state_size, device=self.device)
    self.unit_var = torch.ones(config.batch_size, config.state_size, device=self.device)

  def load(self) -> str:
    models_path = self.models_path
    if os.path.isdir(models_path):
      paths = glob.glob(os.path.join(models_path, "*.pt"))
      if len(paths) == 0:
        raise Exception(f"Failed to load models in {models_path}")
      else:
        models_path = sorted(paths)[-1]

    print(f"Loading pre-trained model '{models_path}'")
    state_dicts = torch.load(models_path)
    for k, v in self.models.items():
      self.models[k].load_state_dict(state_dicts[k])

    return models_path


  def save(self, models_path: str):
    self.models_path = models_path
    torch.save({k: v.state_dict() for k, v in self.models.items()}, models_path)

  def set_models_to_eval(self):
    for m in self.models.values(): m.eval()

  def set_models_to_train(self):
    for m in self.models.values(): m.train()

  def record_experience(self, obs: torch.Tensor, action: torch.Tensor, rwd: float, done: bool):
    self.experience.append(_images_to_observation(obs.cpu().numpy(), bit_depth=5), action, rwd, done)

  def update(self) -> None : # TODO: update to compute_loss
    # sample data iid
    obs, actions, rewards, not_done = self.experience.sample(self.batch_size, self.chunk_size)

    # update models
    z_gt = expand(self.models["encoder"](flatten_sample(obs[1:])), obs[1:].shape) # embeddings of gt next state
    b0 = torch.zeros(self.batch_size, self.belief_size, device=self.device) # TODO: does this init matter?
    x0 = torch.zeros(self.batch_size, self.state_size, device=self.device)
    fwd_pred = self.models["transition"](x0, actions[:-1], b0, z_gt, not_done[:-1])
    beliefs, prior_x, prior_mean, prior_std = fwd_pred[:4]
    post_x, post_mean, post_std = fwd_pred[4:]
    # predict current reward and next obs
    r = expand(self.models["reward"](flatten_sample(beliefs), flatten_sample(post_x)), beliefs.shape)
    o = expand(self.models["observation"](flatten_sample(beliefs), flatten_sample(post_x)), beliefs.shape)
    # compute losses
    P = Normal(post_mean, post_std)
    Q = Normal(prior_mean, prior_std)
    nats = torch.full((1,), self.free_nats, dtype=torch.float32, device=self.device)
    mse_rewards = F.mse_loss(r, rewards[:-1], reduction='none')
    mse_pixels = F.mse_loss(o, obs[1:], reduction='none').sum(dim=(2, 3, 4))
    l_pixels = mse_pixels.mean(dim=(0, 1))
    l_rewards = mse_rewards.mean(dim=(0, 1))
    l_kl = torch.max(kl_divergence(P, Q).sum(dim=2), nats).mean(dim=(0, 1))
    L = l_pixels + l_rewards + l_kl
    # TODO: include overshooting
    #state_prior = torch.distributions.Normal(zero_mean, unit_var)

    # report losses (also clear mem)
    self.metrics.kl_loss.append(l_kl.item())
    self.metrics.observation_loss.append(l_pixels.item())
    self.metrics.reward_loss.append(l_rewards.item())
    return L

  def reset(self, *args, **kwargs):
    """ re-initialise generic policy """
    count = kwargs["count"] if "count" in list(kwargs.keys()) else self.batch_size
    self.b = torch.zeros(count, self.belief_size, device=self.device) # TODO: does this init matter?
    self.x = torch.zeros(count, self.state_size, device=self.device)
    self.a = torch.zeros(count, self.action_size, device=self.device)

  def clip_grads(self, grad_clip_norm, norm_type):
    nn.utils.clip_grad_norm_(self.params_list, grad_clip_norm, norm_type)

  def act(self, obs: torch.Tensor) -> torch.FloatTensor:
    # state estimation and fwd prediction
    obs = _images_to_observation(obs.cpu().numpy(), bit_depth=5)
    z = self.models["encoder"](obs.to(self.device)).unsqueeze(dim=0)
    action = self.a.unsqueeze(dim=0) # 1 x batch_size x a_size
    fwd_pred = self.models["transition"](self.x, action, self.b, z)
    self.b, self.x = fwd_pred[0].squeeze(dim=0), fwd_pred[4].squeeze(dim=0)
    # action selection
    self.a = self.planner(self.b, self.x)
    return self.a

  def sample_action(self, obs: torch.Tensor) -> torch.FloatTensor:
    # apply uniform exploration noise
    self.a = self.act(obs) + self.action_noise * torch.rand_like(self.a)
    self.a.clamp_(float(self.action_space.low[0]), float(self.action_space.high[0]))
    return self.a


class Dreamer:
  """
  Based on Dreamer-v2 (Hafner et al. 2021)
  https://github.com/jurgisp/pydreamer

  The point of this implementation is to run the bare bones of dreamer algo.
  Expect training to take ~1.5 weeks on _?_ tasks. Hopefully not longer...
  Also, pay attention to memory management in pydreamer...
  """
  def __init__(self, cfg, action_size, observation_size, device):
    self.name = "dreamer"

    self.horizon = cfg.imag_horizon
    self.grad_clip = cfg.grad_clip
    self.grad_clip_ac = cfg.grad_clip_ac
    self.batch_size = cfg.batch_size
    self.batch_len = cfg.batch_length
    self.hidden_size = cfg.hidden_dim

    # TODO: unused params
    self.action_noise = 3
    self.action_size = action_size
    self.action_space = None
    self.device = device
    self.models_path = cfg.models_path
    # TODO: track prediction errors + provide reconstructions for vis
    self.metrics = Metrics() # TODO: update
    self.worldmodel = WorldModel(cfg)
    self.agent = ActorCritic(in_dim=cfg.deter_dim + cfg.stoch_dim * (cfg.stoch_discrete or 1),
                            out_actions=cfg.action_dim,
                            layer_norm=cfg.layer_norm,
                            gamma=cfg.gamma,
                            lambda_gae=cfg.lambda_gae,
                            entropy_weight=cfg.entropy,
                            target_interval=cfg.target_interval,
                            actor_grad=cfg.actor_grad,
                            actor_dist=cfg.actor_dist)
    self.experience = memory.ExperienceReplay(cfg.experience_size,
                                         cfg.state_type == "vector",
                                         observation_size,
                                         action_size,
                                         cfg.bit_depth,
                                         device)

    self.reset()

  def load(self) -> str:
    models_path = self.models_path
    if os.path.isdir(models_path):
      paths = glob.glob(os.path.join(models_path, "*.pt"))
      if len(paths) == 0:
        raise Exception(f"Failed to load models in {models_path}")
      else:
        models_path = sorted(paths)[-1]

    print(f"Loading pre-trained model '{models_path}'")
    state_dicts = torch.load(models_path)
    self.worldmodel.load_state_dict(state_dicts)

    dir, name = os.path.dirname(models_path), os.path.basename(models_path).split('.')[0]
    self.agent.actor.load_state_dict(torch.load(os.path.join(dir, f"{name}-actor.pt")))
    self.agent.critic.load_state_dict(torch.load(os.path.join(dir, f"{name}-critic.pt")))
    self.agent.critic_target.load_state_dict(torch.load(os.path.join(dir, f"{name}-target.pt")))
    return models_path

  def save(self, models_path: str):
    self.models_path = models_path
    torch.save(self.worldmodel.state_dict(), models_path)
    dir, name = os.path.dirname(models_path), os.path.basename(models_path).split('.')[0]
    torch.save(self.agent.actor.state_dict(), os.path.join(dir, f"{name}-actor.pt"))
    torch.save(self.agent.critic.state_dict(), os.path.join(dir, f"{name}-critic.pt"))
    torch.save(self.agent.critic_target.state_dict(), os.path.join(dir, f"{name}-target.pt"))

  def set_models_to_eval(self):
    self.worldmodel.eval()

  def set_models_to_train(self):
    self.worldmodel.train()

  # TODO: custom memory
  def record_experience(self, obs: torch.Tensor, action: torch.Tensor, rwd: float, done: bool):
    self.experience.append(_images_to_observation(obs.cpu().numpy(), bit_depth=5), action, rwd, done)

  def update(self) -> None:
    obs, actions, rewards, not_done = self.experience.sample(self.batch_size, self.batch_len)


    x = self.worldmodel.decoder.image(torch.zeros(self.batch_size, 3072).to(self.device))
    L = F.mse_loss(obs, x)
    # TODO: include overshooting
    #state_prior = torch.distributions.Normal(zero_mean, unit_var)

    # report losses (also clear mem)
    #self.metrics.total_return.append(L.item())

    return L

  def reset(self, *args, **kwargs):
    """ re-initialise generic policy """
    self.worldmodel.to(device=self.device) # may be unnecessary here
    count = kwargs["count"] if "count" in list(kwargs.keys()) else self.batch_size

  def clip_grads(self, grad_clip_norm, norm_type):
    #nn.utils.clip_grad_norm_(self.params_list, self.grad_clip, norm_type=2) # TODO: spec norm_type?
    nn.utils.clip_grad_norm_(self.worldmodel.parameters(), self.grad_clip)
    nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.grad_clip or self.grad_clip_ac)
    nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.grad_clip or self.grad_clip_ac)

  def act(self, obs: torch.Tensor) -> torch.FloatTensor:
    # state estimation and fwd prediction
    obs = _images_to_observation(obs.cpu().numpy(), bit_depth=5)

    # action selection
    #self.a = self.agent(self.b, self.x)
    self.a = torch.zeros(self.action_size).unsqueeze(dim=0)
    return torch.FloatTensor(self.a)

  def sample_action(self, obs: torch.Tensor) -> torch.FloatTensor:
    # apply uniform exploration noise
    self.a = self.act(obs) + self.action_noise * torch.rand_like(self.a)
    #self.a.clamp_(float(self.action_space.low[0]), float(self.action_space.high[0]))
    return self.a