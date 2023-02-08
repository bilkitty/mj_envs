from gym import Env
from typing import List
import os
import glob
import torch
import pickle
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence
from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from dependencies.PlaNet import models
from dependencies.PlaNet.planner import MPCPlanner
from dependencies.PlaNet.env import _images_to_observation

# TODO: replace these with own generic implementation
from mjrl.policies.gaussian_mlp import MLP
from mj_envs_vision.utils.helpers import Metrics
from mj_envs_vision.utils.helpers import expand, flatten_sample
from mj_envs_vision.utils.helpers import action_size, observation_size
from mj_envs_vision.utils.config import Config
from mj_envs_vision.utils.wrappers import StateActionSpec


SUPPORTED_POLICIES = ["default", "dapg", "planet", "ppo"]


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

  def initialise(self, *args, **kwargs):
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

  def update(self, sample_batch: List, optimiser):
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

    named_model_parameters = self.ppo.get_parameters()["policy"]
    self.params_list = [Parameter(v) for k,v in named_model_parameters.items()]

  def initialise(self, *args, **kwargs):
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

  def update(self, sample_batch: List, optimiser):
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


# TODO: fix input incompatibility (ugh)
class Planet:
  def __init__(self, config, action_size, observation_size, action_space, device):
    self.name = "planet"
    self.free_nats = config.free_nats
    self.grad_clip_norm = config.grad_clip_norm
    self.batch_size = config.batch_size
    self.belief_size = config.belief_size
    self.state_size = config.state_size
    self.action_noise = config.action_noise
    self.action_size = action_size
    self.action_space = action_space
    self.device = device
    self.models_path = config.models_path
    self.metrics = PlanetMetrics()
    # TODO: track prediction errors + provide reconstructions for vis
    self.initialise()
    self.models = dict(transition=models.TransitionModel(config.belief_size, config.state_size, action_size, config.hidden_size, config.embedding_size, config.activation_fn),
        observation=models.ObservationModel(config.state_type == "vector", observation_size, config.belief_size, config.state_size, config.embedding_size, config.activation_fn),
        encoder=models.Encoder(config.state_type == "vector", observation_size, config.embedding_size, config.activation_fn),
        reward=models.RewardModel(config.belief_size, config.state_size, config.hidden_size, config.activation_fn))

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

  def update(self, sample_batch: List, optimiser):
    assert len(sample_batch) == 4 # TODO: pack/unpack data
    obs, actions, rewards, not_done = sample_batch # replay buffer outputs obs processed via _images_to_observation

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
    # update models
    optimiser.zero_grad()
    L.backward()
    nn.utils.clip_grad_norm_(self.params_list, self.grad_clip_norm, norm_type=2)
    optimiser.step()

    # report losses (also clear mem)
    self.metrics.kl_loss.append(l_kl.item())
    self.metrics.observation_loss.append(l_pixels.item())
    self.metrics.reward_loss.append(l_rewards.item())

  def initialise(self, *args, **kwargs):
    """ re-initialise generic policy """
    count = kwargs["count"] if "count" in list(kwargs.keys()) else self.batch_size
    self.b = torch.zeros(count, self.belief_size, device=self.device) # TODO: does this init matter?
    self.x = torch.zeros(count, self.state_size, device=self.device)
    self.a = torch.zeros(count, self.action_size, device=self.device)

  def act(self, obs: torch.Tensor) -> torch.FloatTensor:
    # state estimation and fwd prediction
    obs = _images_to_observation(obs.cpu().numpy(), bit_depth=5)
    z = self.models["encoder"](obs.to(self.device)).unsqueeze(dim=0)
    action = self.a.unsqueeze(dim=0)
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
