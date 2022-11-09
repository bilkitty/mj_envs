import os
import torch
import time
import numpy as np

from tqdm import tqdm
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence
from dependencies.PlaNet import memory
from mj_envs_vision.algos.baselines import Planet
from mj_envs_vision.algos.baselines import PlanetConfig
from mj_envs_vision.algos.baselines import PlanetMetrics
from mj_envs_vision.utils.wrappers import make_env
from mj_envs_vision.utils.helpers import visualise_batch_from_experience
from mj_envs_vision.utils.helpers import visualise_trajectory
from mj_envs_vision.utils.helpers import plot_rewards


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
    optimiser.step()

  return metrics


def evaluate(config, policy, count=10):
  with torch.no_grad():
    total_rwds = []
    trajectories = []
    for i in range(count): # consider threading?
      # roll out policy
      rwd = 0.0
      traj = []
      T = make_env(config)
      obs = T.reset()
      policy.initialise(**dict(count=1))
      for t in tqdm(range(config.max_episode_length // config.action_repeat)):
        action = policy.sample_action(obs).squeeze(dim=0).cpu()
        next_obs, r, done = E.step(action)
        traj.append((obs.squeeze(dim=0), action, r))
        rwd += r
        obs = next_obs
        if done: break # TODO: is this bad?

      T.env.close()
      # record final obs
      traj.append((next_obs.squeeze(dim=0), torch.zeros_like(action), r))
      total_rwds.append(rwd)
      trajectories.append(traj)

    return total_rwds, trajectories


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
      if done: break # TODO: is this bad?

    return total_rwd


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

  # seed buffer with requested batch and chunk size
  rwd, done = 0.0, False
  obs = E.reset()
  while experience.idx <= max(config.batch_size, config.chunk_size):
    if done:
      rwd, done = 0.0, False
      obs = E.reset()

    action = E.sample_action()
    experience.append(obs, action, rwd, done)
    obs, rwd, done = E.step(action)

  train_time = list()
  eval_time = list()
  sim_time = list()
  exp_rewards = list()
  episode_rewards = list()
  episode_trajectories = list()
  for ep in tqdm(range(config.seed_episodes, config.max_episodes + 1)): # TODO: add total and initial?
    if PROF: tns = time.time_ns()
    train_metrics = train(config, experience, policy, optimiser)
    if PROF: train_time.append(time.time_ns() - tns)
    # TODO: dump metrics to tensorboard

    if PROF: tns = time.time_ns()
    train_reward = collect_experience(config, E, experience, policy)
    exp_rewards.append((ep, train_reward))
    if PROF: sim_time.append(time.time_ns() - tns)

    if ep % config.test_interval == 0:
      if PROF: tns = time.time_ns()
      for m in policy.models.values(): m.eval()
      rewards, trajs = evaluate(config, policy, count=10)
      for m in policy.models.values(): m.train()
      if PROF: eval_time.append(time.time_ns() - tns)

      episode_rewards.append((ep, np.mean(rewards)))
      episode_trajectories.append((ep, trajs))
      # TODO: dump metrics to tensorboard
      visualise_trajectory(ep, trajs[-1], out_dir)  # select worst
      # TODO: rm
      plot_rewards(exp_rewards).savefig(os.path.join(out_dir, "train_rewards.png"))
      plot_rewards(episode_rewards).savefig(os.path.join(out_dir, "eval_rewards.png"))

  E.close()

  # save performance metrics (TODO: pickle)

  # visualise performance
  plot_rewards(exp_rewards).savefig(os.path.join(out_dir, "train_rewards.png"))
  plot_rewards(episode_rewards).savefig(os.path.join(out_dir, "eval_rewards.png"))
  for i in range(5):
    visualise_batch_from_experience(i, config, experience, E.observation_size, out_dir)

  if PROF:
    train_time, eval_time, sim_time = [t / 1e9 for t in train_time], [t / 1e9 for t in eval_time], [t / 1e9 for t in sim_time]
    print(f"iter time:\n\t{np.median(train_time): .2f}s\n\t{np.median(eval_time): .2f}s\n\t{np.median(sim_time): .2f}s")
    print(f"total time:\n\t1.00x tr\n\t{np.sum(eval_time)/np.sum(train_time): .2f}x tr\n\t{np.sum(sim_time)/np.sum(train_time): .2f}x tr")

  print("done :)")