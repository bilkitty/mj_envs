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
from mj_envs_vision.utils.helpers import BasicTimer
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
    config.action_dim = action_size(env) # override config defaults for hand man tasks
    return Dreamer(config, action_size(env), observation_size(env), device)

def make_policy_optimisers(config: Config, policy_type: str, policy):
  optims = list()
  if policy_type == "ppo" or policy_type == "planet":
    optims.append(optim.Adam(policy.params_list, lr=config.learning_rate, eps=config.adam_epsilon))
  elif policy_type == "dreamer":
    # NOTE: currently ignoring probe models
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
    self.timer_ms = BasicTimer('ms')

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

  def update(self) -> None: # TODO: update to compute_loss
    pass

  def act(self, obs: torch.Tensor) -> torch.FloatTensor:
    return torch.FloatTensor(self.mlp.get_action(obs.numpy())[1]['evaluation'])

  def sample_action(self, obs: torch.Tensor) -> torch.FloatTensor:
    return torch.FloatTensor(self.mlp.get_action(obs.numpy())[0])


class PPOMetrics(Metrics):
  def __init__(self, group_size):
    Metrics.__init__(self, group_size)
    self.policy_gradient_loss = list()
    self.value_loss = list()
    self.entropy_loss = list()
    self.approx_kl = list()
    self.n_updates = list()
    self.clip_fraction = list()


class PPOBaseline:
  def __init__(self, config, custom_env: Env):
    self.name = "ppo"
    self.ppo = None
    self.env_name = config.env_name
    self.learning_rate = config.learning_rate
    self.n_steps = config.test_interval
    self.log_interval = config.test_interval
    self.epochs = config.sample_iters
    self.batch_size = config.batch_size
    self.target_kl = None
    self.entropy_coeff = config.entropy
    self.models_path = config.models_path
    self.metrics = PPOMetrics(group_size=1)
    self.experience = None
    self.timer_ms = BasicTimer('ms')

    if config.sample_iters > 10:
      print(f"WARN: consider decreasing sample_iters '{config.sample_iters}'")

    if config.model_type == "mlp":
      policy_type = ActorCriticPolicy#(observation_space, action_space, lr_schedule=lambda x: 0.0)
    elif config.model_type == "cnn":
      policy_type = ActorCriticCnnPolicy#(observation_space, action_space, lr_schedule=lambda x: 0.0)
    else:
      raise Exception(f"unsupported model type {config.model_type}")

    self.timer_ms.start("ppo-init")
    self.ppo = PPO(policy_type,
              custom_env,
              self.learning_rate,
              n_steps=self.n_steps,
              n_epochs=self.epochs,
              batch_size=self.batch_size,
              target_kl=self.target_kl,
              tensorboard_log=config.log_path)
    self.timer_ms.stop("ppo-init")

    self.params_list = [Parameter(torch.zeros(3))] # sb3 internally updates params

  def reset(self, *args, **kwargs):
    """ re-initialise generic policy """
    self.timer_ms.reset()

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
        models_path = sorted(paths)[0]

    print(f"Loading pre-trained model '{models_path}'")
    self.timer_ms.start("ppo-load")
    self.ppo.load(models_path)
    self.timer_ms.stop("ppo-load")

    return models_path

  def save(self, models_path: str):
    self.models_path = models_path
    self.timer_ms.start("ppo-save")
    self.ppo.save(models_path)
    self.timer_ms.stop("ppo-save")

  def update(self) -> None:
    self.timer_ms.start("ppo-learn")
    self.ppo.learn(self.n_steps, log_interval=self.log_interval, progress_bar=True)
    self.timer_ms.stop("ppo-learn")
    if self.ppo.logger.name_to_value and len(self.ppo.logger.name_to_value) > 0:
      self.metrics.value_loss.append(float(self.ppo.logger.name_to_value['train/value_loss']))
      self.metrics.entropy_loss.append(float(self.ppo.logger.name_to_value['train/entropy_loss']))
      self.metrics.policy_gradient_loss.append(float(self.ppo.logger.name_to_value['train/policy_gradient_loss']))
      self.metrics.approx_kl.append(float(self.ppo.logger.name_to_value['train/approx_kl']))
      self.metrics.n_updates.append(self.ppo.logger.name_to_value['train/n_updates'])
      self.metrics.clip_fraction.append(float(self.ppo.logger.name_to_value['train/clip_fraction']))

  def act(self, obs: torch.Tensor) -> torch.FloatTensor:
    self.timer_ms.start("ppo-act")
    action, state = self.ppo.predict(obs.numpy())
    self.timer_ms.stop("ppo-act")
    return torch.FloatTensor(action)

  def sample_action(self, obs: torch.Tensor) -> torch.FloatTensor:
    a = self.act(obs)
    return a + 0.3 * torch.rand_like(a)


class PlanetMetrics(Metrics):
  def __init__(self, group_size):
    Metrics.__init__(self, group_size)
    self.observation_loss = list()
    self.reward_loss = list()
    self.kl_loss = list()

  def update(self, metric: dict[str, torch.Tensor], should_clear: bool = False):
    for k,v in metric.items():
      if self.__dict__.get(k) is None:
        self.__dict__[k] = list()
      self.__dict__[k].extend([v.item()] if len(v.shape) == 0 else v.tolist())

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
    self.metrics = PlanetMetrics(config.sample_iters)
    self.timer_ms = BasicTimer('ms')

    # TODO: track prediction errors + provide reconstructions for vis
    self.reset()
    self.timer_ms.start("planet-init")
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
    self.timer_ms.stop("planet-init")

    # move models to device and collect parameters
    self.params_list = list()
    for m in self.models.values():
      m.to(self.device)
      self.params_list.extend(list(m.parameters()))

    # initialise mpc planner
    self.timer_ms.start("planet-init-mpc")
    self.planner = MPCPlanner(action_size,
                              config.planning_horizon,
                              config.optimisation_iters,
                              config.candidates,
                              config.top_candidates,
                              self.models["transition"],
                              self.models["reward"],
                              float(action_space.low[0]),
                              float(action_space.high[0]))
    self.timer_ms.stop("planet-init-mpc")

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
    self.timer_ms.start("planet-load")
    state_dicts = torch.load(models_path)
    for k, v in self.models.items():
      self.models[k].load_state_dict(state_dicts[k])
    self.timer_ms.stop("planet-load")

    return models_path


  def save(self, models_path: str):
    self.models_path = models_path
    self.timer_ms.start("planet-save")
    torch.save({k: v.state_dict() for k, v in self.models.items()}, models_path)
    self.timer_ms.stop("planet-save")

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
    self.timer_ms.start("planet-pred-fwd")
    z_gt = expand(self.models["encoder"](flatten_sample(obs[1:])), obs[1:].shape) # embeddings of gt next state
    b0 = torch.zeros(self.batch_size, self.belief_size, device=self.device) # TODO: does this init matter?
    x0 = torch.zeros(self.batch_size, self.state_size, device=self.device)
    fwd_pred = self.models["transition"](x0, actions[:-1], b0, z_gt, not_done[:-1])
    self.timer_ms.stop("planet-pred-fwd")
    beliefs, prior_x, prior_mean, prior_std = fwd_pred[:4]
    post_x, post_mean, post_std = fwd_pred[4:]
    # predict current reward and next obs
    self.timer_ms.start("planet-pred-rwd")
    r = expand(self.models["reward"](flatten_sample(beliefs), flatten_sample(post_x)), beliefs.shape)
    self.timer_ms.stop("planet-pred-rwd")
    self.timer_ms.start("planet-pred-obs")
    o = expand(self.models["observation"](flatten_sample(beliefs), flatten_sample(post_x)), beliefs.shape)
    self.timer_ms.stop("planet-pred-obs")
    # compute losses
    self.timer_ms.start("planet-loss")
    P = Normal(post_mean, post_std)
    Q = Normal(prior_mean, prior_std)
    nats = torch.full((1,), self.free_nats, dtype=torch.float32, device=self.device)
    mse_rewards = F.mse_loss(r, rewards[:-1], reduction='none')
    mse_pixels = F.mse_loss(o, obs[1:], reduction='none').sum(dim=(2, 3, 4))
    l_pixels = mse_pixels.mean(dim=(0, 1))
    l_rewards = mse_rewards.mean(dim=(0, 1))
    l_kl = torch.max(kl_divergence(P, Q).sum(dim=2), nats).mean(dim=(0, 1))
    L = l_pixels + l_rewards + l_kl
    self.timer_ms.stop("planet-loss")
    # TODO: include overshooting
    #state_prior = torch.distributions.Normal(zero_mean, unit_var)

    # report losses (also clear mem)
    self.metrics.update(dict(kl_loss=l_kl, observation_loss=l_pixels, reward_loss=l_rewards))
    return L

  def reset(self, *args, **kwargs):
    """ re-initialise generic policy """
    sample_count = kwargs["count"] if "count" in list(kwargs.keys()) else self.batch_size
    self.b = torch.zeros(sample_count, self.belief_size, device=self.device)
    self.x = torch.zeros(sample_count, self.state_size, device=self.device)
    self.a = torch.zeros(sample_count, self.action_size, device=self.device)
    self.timer_ms.reset()

  def clip_grads(self, grad_clip_norm, norm_type):
    nn.utils.clip_grad_norm_(self.params_list, grad_clip_norm, norm_type)

  def act(self, obs: torch.Tensor) -> torch.FloatTensor:
    # state estimation and fwd prediction
    obs = _images_to_observation(obs.cpu().numpy(), bit_depth=5)
    self.timer_ms.start("planet-act-est")
    z = self.models["encoder"](obs.to(self.device)).unsqueeze(dim=0)
    action = self.a.unsqueeze(dim=0) # 1 x batch_size x a_size
    fwd_pred = self.models["transition"](self.x, action, self.b, z)
    self.b, self.x = fwd_pred[0].squeeze(dim=0), fwd_pred[4].squeeze(dim=0)
    self.timer_ms.stop("planet-act-est")
    # action selection
    self.timer_ms.start("planet-act-sel")
    self.a = self.planner(self.b, self.x)
    self.timer_ms.stop("planet-act-sel")
    return self.a

  def sample_action(self, obs: torch.Tensor) -> torch.FloatTensor:
    # apply uniform exploration noise
    self.a = self.act(obs) + self.action_noise * torch.rand_like(self.a)
    self.a.clamp_(float(self.action_space.low[0]), float(self.action_space.high[0]))
    return self.a

class DreamerMetrics(Metrics):
  def __init__(self, group_size):
    Metrics.__init__(self, group_size)

  def update(self, metric: dict[str, torch.Tensor], should_clear: bool = False):
    for k,v in metric.items():
      if self.__dict__.get(k) is None:
        self.__dict__[k] = list()
      self.__dict__[k].extend([v.item()] if len(v.shape) == 0 else v.tolist())


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
    self.should_keep_state = cfg.keep_state
    self.ac_grad_mode = cfg.actor_grad

    self.action_noise = 3
    self.action_size = action_size
    self.observation_size = observation_size
    self.action_space = None
    self.state = None
    self.last_state_pred = None
    self.input_obs = None # named obs (see apply::preprocessing.py & wrappers.py)
    self.device = device
    self.models_path = cfg.models_path
    # TODO: track prediction errors + provide reconstructions for vis
    self.metrics = DreamerMetrics(cfg.sample_iters)
    self.timer_ms = BasicTimer('ms')

    assert cfg.action_dim == action_size, "please set actions for dex man"
    self.worldmodel = WorldModel(cfg)
    self.agent = ActorCritic(in_dim=cfg.deter_dim + cfg.stoch_dim * (cfg.stoch_discrete or 1),
                            out_actions=cfg.action_dim, #
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
    self.timer_ms.start("dreamer-load")
    state_dicts = torch.load(models_path)
    self.worldmodel.load_state_dict(state_dicts)

    dir, name = os.path.dirname(models_path), os.path.basename(models_path).split('.')[0]
    self.agent.actor.load_state_dict(torch.load(os.path.join(dir, f"{name}-actor.pt")))
    self.agent.critic.load_state_dict(torch.load(os.path.join(dir, f"{name}-critic.pt")))
    self.agent.critic_target.load_state_dict(torch.load(os.path.join(dir, f"{name}-target.pt")))
    self.timer_ms.stop("dreamer-load")
    return models_path

  def save(self, models_path: str):
    self.models_path = models_path
    torch.save(self.worldmodel.state_dict(), models_path)
    dir, name = os.path.dirname(models_path), os.path.basename(models_path).split('.')[0]
    #self.timer_ms.start("dreamer-save")
    torch.save(self.agent.actor.state_dict(), os.path.join(dir, f"{name}-actor.pt"))
    torch.save(self.agent.critic.state_dict(), os.path.join(dir, f"{name}-critic.pt"))
    torch.save(self.agent.critic_target.state_dict(), os.path.join(dir, f"{name}-target.pt"))
    #self.timer_ms.stop("dreamer-save")

  def set_models_to_eval(self):
    self.worldmodel.eval()

  def set_models_to_train(self):
    self.worldmodel.train()

  # TODO: custom memory
  #       see preprocessor.py for expected inputs
  def record_experience(self, image: torch.Tensor, action: torch.Tensor, rwd: float, done: bool):
    self.experience.append(_images_to_observation(image.cpu().numpy(), bit_depth=5), action, rwd, done)

  def update(self) -> None:
    #
    # Collect real samples
    #

    # Actions are passed to world model as part of observation
    # obs dims = (batch, traj, data)
    obs, actions, rewards, not_done = self.experience.sample(self.batch_size, self.batch_len)
    resets = torch.zeros_like(not_done).bool()
    resets[:, 0] = True
    resets = resets.squeeze(2)
    not_done = not_done.squeeze(2) # TODO: should we flip this for terminal?
    wm_obs = { "image" : obs, "action" : actions, "reward" : rewards, "terminal" : not_done, "reset" : resets }
    # TODO: could include 'vecobs'

    #
    # Predict state
    #

    self.timer_ms.start("dreamer-train-wm")
    predictions = self.worldmodel.training_step(wm_obs, self.state) # TODO: set iwae samples?
    wm_loss              = predictions[0] # or "loss_model" reconstruction & kl losses and entropies
    concat_state_preds    = predictions[1] # or "features" collection of concatenated hidden states and pr/po samples
    state_preds          = predictions[2] # or "states" collection of hidden states and pr/po samples
    self.last_state_pred = predictions[3] # or "out_state" the last hidden state and prior/posterior sample
    wm_metrics           = predictions[4] # or "metrics" a dict of ac (added later), reward and other losses
    self.metrics.update(wm_metrics)
    self.timer_ms.stop("dreamer-train-wm")

    #
    # Dream (i.e., generate latent trajectory rollouts for policy training)
    #

    # freeze wm in case policy updates depend on dynamics gradients
    self.worldmodel.requires_grad_(False)

    latents = []
    actions = []
    entropies = []
    aprobs = []
    state = tuple(s.reshape((-1,) + s.shape[-1:]) for s in state_preds)
    self.timer_ms.start("dreamer-pred-rollout")
    for i in range(self.horizon):
      hz = torch.cat(state, dim=-1)
      pi, val = self.agent.forward_actor(hz), self.agent.forward_value(hz)
      # use reparameterised sampling for dynamics-only backprop
      act = pi.sample() if self.ac_grad_mode == "reinforce" else pi.rsample()
      latents.append(hz)
      actions.append(act)
      entropies.append(pi.entropy())
      aprobs.append(pi.log_prob(act).exp())
      # make forward prediction (note: doesn't propagate grads through wm, see original code for details)
      _, state = self.worldmodel.core.cell.forward_prior(act, None, state)
    self.timer_ms.stop("dreamer-pred-rollout")

    # aggregate action entropy and probabilities over batch_size x chunk_size
    self.metrics.update(dict(entropy=torch.stack(entropies).mean().unsqueeze(0),
                             aprob=torch.stack(aprobs).mean().unsqueeze(0)))

    # record last point
    latents.append(torch.cat(state, dim=-1))
    latents = torch.stack(latents)
    actions = torch.stack(actions)
    self.timer_ms.start("dreamer-pred-rwd")
    rewards = self.worldmodel.decoder.reward.forward(latents).mean
    terminals = self.worldmodel.decoder.terminal.forward(latents).mean
    self.timer_ms.stop("dreamer-pred-rwd")

    # unfreeze wm
    self.worldmodel.requires_grad_(True)

    #
    # Policy iteration on latent rollouts
    #

    # NOTE: in dynamics backprop mode, the policy loss is computed from rewards and
    #       critic_target's output, which have requires_grad = False
    self.timer_ms.start("dreamer-train-ac")
    ac_loss, ac_metrics, _ = self.agent.training_step(latents.detach(),
                                                      actions.detach(),
                                                      rewards if self.ac_grad_mode == "dynamics" else rewards.detach(),
                                                      terminals.detach())
    self.timer_ms.stop("dreamer-train-ac")
    self.metrics.update(ac_metrics)

    # optionally update state
    self.state = self.last_state_pred if self.should_keep_state else self.worldmodel.init_state(self.batch_size)

    return wm_loss + ac_loss[0] + ac_loss[1]

  def reset(self, *args, **kwargs): # TODO: move up
    """ re-initialise generic policy """
    sample_count = kwargs["count"] if "count" in list(kwargs.keys()) else self.batch_size
    self.input_obs = dict(image=torch.zeros(sample_count, 1, self.observation_size).to(self.device),
                         action=torch.zeros(sample_count, 1, self.action_size).to(self.device),
                         reset=torch.zeros(sample_count, 1).bool().to(self.device), # (batch_size, batch_len, 0-dim)
                         reward=torch.ones(sample_count, 1) * -float('inf'),        # unused & 0-dim
                         terminal=torch.zeros(sample_count, 1) * -float('inf'))     # unused & 0-dim

    self.timer_ms.reset()
    #self.timer_ms.start("dreamer-reset")
    self.state = self.worldmodel.init_state(sample_count)
    self.worldmodel.to(self.device) # may be unnecessary here
    self.agent.to(self.device)
    #self.timer_ms.stop("dreamer-reset")

  def clip_grads(self, grad_clip_norm, norm_type):
    #nn.utils.clip_grad_norm_(self.params_list, self.grad_clip, norm_type=2) # TODO: spec norm_type?
    nn.utils.clip_grad_norm_(self.worldmodel.parameters(), self.grad_clip)
    nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.grad_clip_ac or self.grad_clip)
    nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.grad_clip_ac or self.grad_clip)

  def act(self, obs: torch.Tensor) -> torch.FloatTensor:

    # state estimation
    self.input_obs["image"] = _images_to_observation(obs.cpu().numpy(), bit_depth=5).unsqueeze(0).to(self.device)
    # TODO: should this be open-loop?
    self.timer_ms.start("dreamer-act-est")
    concat_state_preds, self.last_state_pred = self.worldmodel.forward(self.input_obs, self.state) # TODO: set iwae samples?
    self.timer_ms.stop("dreamer-act-est")

    # action selection
    self.timer_ms.start("dreamer-act-sel")
    pi = self.agent.forward_actor(concat_state_preds)
    val = self.agent.forward_value(concat_state_preds)
    action = pi.sample().squeeze(0)
    self.timer_ms.stop("dreamer-act-sel")
    val = val.squeeze(0)
    self.input_obs["action"] = action
    return action.squeeze(0)

  def sample_action(self, obs: torch.Tensor) -> torch.FloatTensor:
    action = self.act(obs)
    # apply uniform exploration noise
    action += self.action_noise * torch.rand_like(action).to(self.device)
    self.input_obs["action"] = action.unsqueeze(0)
    #a.clamp_(float(self.action_space.low[0]), float(self.action_space.high[0]))
    return action
