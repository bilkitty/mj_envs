import os
import torch
import time
import numpy as np
from PIL import Image
from gym.wrappers.time_limit import TimeLimit

from tqdm import tqdm
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence
from dependencies.PlaNet import memory
from dependencies.PlaNet import env
from mjrl.utils import gym_env
from mj_envs_vision.algos.baselines import Planet
from mj_envs_vision.algos.baselines import PlanetConfig
from mj_envs_vision.algos.baselines import PlanetMetrics


PROF = True

# Helpers for processing samples drawn from experience
def flatten_sample(x):
  sh = x.shape # TODO: same as size?
  return x.view(sh[0] * sh[1], *sh[2:])

def expand(x_flat, sh):
  return x_flat.view(sh[0], sh[1], *x_flat.shape[1:])


def train(config, experience, policy, optimiser, enable_overshooting=False):
  # TODO: move most of this to planet update models
  zero_mean = torch.zeros(config.batch_size, config.state_size, device=config.device)
  unit_var = torch.ones(config.batch_size, config.state_size, device=config.device)
  metrics = PlanetMetrics()
  for t in range(config.sample_iters):
    # sample data iid
    obs, actions, rewards, not_done = experience.sample(config.batch_size, config.chunk_size)
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
    metrics.observation_loss.append(mse_pixels.mean(dim=(0, 1)))
    metrics.reward_loss.append(mse_rewards.mean(dim=(0, 1)))
    # TODO: include overshooting
    #state_prior = torch.distributions.Normal(zero_mean, unit_var)
    metrics.kl_loss.append(torch.max(kl_divergence(P, Q).sum(dim=2), nats).mean(dim=(0, 1)))
    L = metrics.current_loss()
    # update models
    optimiser.zero_grad()
    L.backward()
    nn.utils.clip_grad_norm_(policy.params_list, config.grad_clip_norm, norm_type=2)

  return metrics


def evaluate(config, policy):
  # instantiate test env


  # close env
  pass


def collect_experience(config, E, experience, policy):
  with torch.no_grad():
    total_rwd = 0.0
    obs = E.reset()
    policy.initialise(**dict(count=1))
    # roll out policy and update experience
    for t in tqdm(range(config.max_episode_length // config.action_repeat)):
      action = policy.sample_action(obs).squeeze(dim=0).cpu()
      next_obs, rwd, done = E.step(action)
      experience.append(obs, action, rwd, done)
      total_rwd += rwd
      obs = next_obs

    return total_rwd


class EnvWrapper:
  def __init__(self, env, is_adroit):
    self.env = env
    self.is_adroit = is_adroit

  @property
  def action_space(self):
    return self.env.action_space if self.is_adroit else self.env._env.action_space

  @property
  def action_size(self):
    return self.env.action_dim if self.is_adroit else self.env.action_size

  @property
  def observation_size(self):
    return self.env.observation_dim if self.is_adroit else self.env.observation_size

  def reset(self):
    return self.env.reset()

  def step(self, action):
    obs, reward, done = self.env.step(action)[:3]
    return obs, rwd, done

  def sample_action(self):
    if self.is_adroit:
      return torch.FloatTensor(self.env.action_space.sample())
    else:
      return torch.FloatTensor(self.env._env.action_space.sample())


def make_env(config):
  if config.env_name in ['door-v0', 'pen-v0', 'haammer-v0', 'relocate-v0']:
    e = gym_env.GymEnv(config.env_name)
    e.set_seed(config.seed)
    if isinstance(e.env, TimeLimit):
      e = e.env.env
    return EnvWrapper(e, is_adroit=True)
  else:
    assert config.env_name in env.GYM_ENVS
    e = env.GymEnv(config.env_name,
                      config.state_type == "vector",
                      config.seed,
                      config.max_episode_length,
                      config.action_repeat,
                      config.bit_depth)
    if isinstance(e._env, TimeLimit):
      e._env = e._env.env
    return EnvWrapper(e, is_adroit=False)

def to_image_frame(obs: np.ndarray):
  """ converts image observation with pels in [-0.5, 0.5] to image with pels in [0, 255] """
  return (255 * (obs.transpose((1, 2, 0)) + 0.5)).astype('uint8')

def to_input_obs(frame: np.ndarray):
  """ converts image with pels in [0, 255] to image observation with pels in [-0.5, 0.5] """
  return (frame.transpose((2, 0, 1)) / 255 - 0.5).astype('float')

def visualise_batch_from_experience(id, config, experience, out_dir):
  # primarily for debugging
  pils = list()
  batch = experience.sample(min(config.batch_size, experience.idx - 1), min(config.chunk_size, experience.idx - 1))
  for obs in batch[0].reshape(-1, *E.observation_size).cpu():
    pils.append(Image.fromarray(to_image_frame(obs.numpy())))

  pils[0].save(os.path.join(out_dir, f'experience_{id}.gif'),
               append_images=pils, save_all=True, optimize=False, loop=True, duration=len(pils) / 10)


if __name__ == "__main__":
  # TODO: sanity check
  #       compare metrics of this training loop with those of PlaNet/main.py

  # load user defined parameters
  config = PlanetConfig()
  config.load("mj_envs_vision/utils/test_config.json")
  print(config.str())

  # validate params
  assert config.batch_size <= config.max_episode_length // config.action_repeat
  assert config.chunk_size <= config.max_episode_length // config.action_repeat

  # setup
  out_dir = os.path.join("results", f"train_{config.run_id}")
  os.makedirs(out_dir, exist_ok=True)
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  if torch.cuda.is_available() and not config.disable_cuda:
    config.device = torch.device('cuda')
    torch.cuda.manual_seed(config.seed)
  else:
    config.device = torch.device('cpu')

  # TODO: create worker setup and parallelise
  # instantiate env, policy, optimiser
  E = make_env(config)
  policy = Planet(config, E.action_size, E.observation_size, E.action_space)
  if config.models_path != "":
    policy.load_models(config.models_path)
  optimiser = optim.Adam(policy.params_list, lr=config.learning_rate, eps=config.adam_epsilon)

  # initialise experience replay buffer
  experience = memory.ExperienceReplay(config.experience_size,
                                       config.state_type == "vector",
                                       E.observation_size,
                                       E.action_size,
                                       config.bit_depth,
                                       config.device)

  for n in range(config.seed_episodes):
    t, rwd, done = 0, 0.0, False
    obs = E.reset()
    while not done and t < config.max_episode_length:
      action = E.sample_action()
      experience.append(obs, action, rwd, done)
      obs, rwd, done = E.step(action)
      t = t + 1

  train_time = list()
  eval_time = list()
  sim_time = list()
  for ep in tqdm(range(config.seed_episodes, config.max_episodes + 1)): # TODO: add total and initial?
    if PROF: tns = time.time_ns()
    train_metrics = train(config, experience, policy, optimiser)
    if PROF: train_time.append(time.time_ns() - tns)
    # TODO: dump metrics to tensorboard

    if PROF: tns = time.time_ns()
    collect_experience(config, E, experience, policy)
    if PROF: sim_time.append(time.time_ns() - tns)

    if ep % config.test_interval:
      if PROF: tns = time.time_ns()
      for m in policy.models.values(): m.eval()
      eval_metrics = evaluate(config, policy)
      for m in policy.models.values(): m.train()
      if PROF: eval_time.append(time.time_ns() - tns)
      # TODO: dump metrics to tensorboard

  # save performance metrics
  # visualise performance
  for i in range(5):
    visualise_batch_from_experience(i, config, experience, out_dir)

  if PROF:
    train_time, eval_time, sim_time = [t / 1e9 for t in train_time], [t / 1e9 for t in eval_time], [t / 1e9 for t in sim_time]
    print(f"iter time:\n\t{np.median(train_time): .2f}s\n\t{np.median(eval_time): .2f}s\n\t{np.median(sim_time): .2f}s")
    print(f"total time:\n\t1x tr\n\t{np.sum(eval_time)/np.sum(train_time): .2f}x tr\n\t{np.sum(sim_time)/np.sum(train_time): .2f}x tr")

  print("done :)")