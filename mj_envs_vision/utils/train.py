import os
import torch
import time
import numpy as np

from tqdm import tqdm
from torch import optim
from dependencies.PlaNet import memory
from dependencies.PlaNet.env import _images_to_observation
from mj_envs_vision.algos.baselines import Planet
from mj_envs_vision.utils.wrappers import make_env
from mj_envs_vision.utils.helpers import visualise_batch_from_experience
from mj_envs_vision.utils.helpers import visualise_trajectory
from mj_envs_vision.utils.helpers import plot_rewards
from mj_envs_vision.utils.config import PlanetConfig


PROF = True


def train(config, experience, policy, optimiser):
  for t in range(config.sample_iters):
    # sample data iid
    obs, actions, rewards, not_done = experience.sample(config.batch_size, config.chunk_size)
    # # TODO: pack/unpack data
    policy.update(sample_batch=[obs, actions, rewards, not_done], optimiser=optimiser)

  return policy.metrics


def evaluate(config, policy, count=10):
  with torch.no_grad():
    total_rwds = []
    trajectories = []
    for i in range(count): # consider threading?
      # roll out policy
      rwd = 0.0
      traj = []
      T = make_env(config)
      obs, _ = T.reset()
      policy.initialise(**dict(count=1))
      for t in tqdm(range(config.max_episode_length // config.action_repeat)):
        action = policy.act(obs.squeeze(dim=0)).squeeze(dim=0).cpu()
        next_obs, r, done = T.step(action)
        traj.append((obs.squeeze(dim=0), action, r))
        rwd += r
        obs = next_obs

      T.env.close()
      # record final obs
      traj.append((next_obs.squeeze(dim=0), torch.zeros_like(action), r))
      total_rwds.append(rwd)
      trajectories.append(traj)

    return total_rwds, trajectories

def save_rewards_fig(rewards, path: str):
  fig = plot_rewards(rewards)
  fig.savefig(path)

def train_sb3_policy(config, E, policy, out_dir):
  train_time = list()
  eval_time = list()
  exp_rewards = list()
  episode_rewards = list()
  episode_trajectories = list()
  # initialise experience replay buffer
  experience = memory.ExperienceReplay(config.experience_size,
                                       config.state_type == "vector",
                                       E.observation_size,
                                       E.action_size,
                                       config.bit_depth,
                                       config.device)

  # populate buffer with requested batch and chunk size
  rwd, done = 0.0, False
  obs, _ = E.reset()

  for ep in tqdm(range(config.seed_episodes, config.max_episodes + 1)):
    if PROF: tns = time.time_ns()
    # # TODO: pack/unpack data
    policy.update(sample_batch=[], optimiser=None)
    exp_rewards.append((ep, policy.metrics.total_return))
    if PROF: train_time.append(time.time_ns() - tns)

    if ep % config.test_interval == 0:
      if PROF: tns = time.time_ns()
      policy.set_models_to_eval()
      rewards, trajs = evaluate(config, policy, count=10)
      policy.set_models_to_train()
      if PROF: eval_time.append(time.time_ns() - tns)

      episode_rewards.append((ep, rewards))
      episode_trajectories.append((ep, trajs))
      if config.state_type == "observation":
        visualise_trajectory(ep, trajs[-1], out_dir)  # select worst

      # TODO: dump metrics to tensorboard
      #save_rewards_fig(exp_rewards, os.path.join(out_dir, "train_rewards.png"))
      save_rewards_fig(episode_rewards, os.path.join(out_dir, "eval_rewards.png"))

  if PROF:
    train_time, eval_time = [t / 1e9 for t in train_time], [t / 1e9 for t in eval_time]
    print(f"iter time:\n\t{np.median(train_time): .2f}s\n\t{np.median(eval_time): .2f}s")
    print(f"total time:\n\t1.00x tr\n\t{np.sum(eval_time)/np.sum(train_time): .2f}x tr")

  return exp_rewards, episode_rewards, episode_trajectories


def train_policy(config, E, policy, optimiser, out_dir):
  train_time = list()
  eval_time = list()
  sim_time = list()
  exp_rewards = list()
  episode_rewards = list()
  episode_trajectories = list()
  # initialise experience replay buffer
  experience = memory.ExperienceReplay(config.experience_size,
                                       config.state_type == "vector",
                                       E.observation_size,
                                       E.action_size,
                                       config.bit_depth,
                                       config.device)
  # TODO: tmp hack to override hard coded image shape
  dtype = np.float32 if config.state_type == "vector" else np.uint8
  pil_format_shape = (E.observation_space.shape[2], *E.observation_space.shape[:2])
  experience.observations = np.empty((config.experience_size, *pil_format_shape), dtype)
  # assume models take inputs with channels at first dimension (i.e., CxHxW)
  transform_to_pil_like = lambda x: x.permute((2, 0, 1)) if E.is_adroit else _images_to_observation(x.numpy(), bit_depth=5)

  # populate buffer with requested batch and chunk size
  rwd, done = 0.0, False
  obs, _ = E.reset()
  obs = transform_to_pil_like(obs)
  print(f"Initialising experience replay with max(batch_size, chunk_size) samples")
  while experience.idx <= max(config.batch_size, config.chunk_size):
    if done:
      rwd, done = 0.0, False
      obs, _ = E.reset()
      obs = transform_to_pil_like(obs)

    action = E.sample_action()
    experience.append(obs, torch.FloatTensor(action), rwd, done)
    obs, rwd, done = E.step(action)
    obs = transform_to_pil_like(obs)

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
      policy.set_models_to_eval()
      rewards, trajs = evaluate(config, policy, count=10)
      policy.set_models_to_train()
      if PROF: eval_time.append(time.time_ns() - tns)

      episode_rewards.append((ep, rewards))
      episode_trajectories.append((ep, trajs))
      if config.state_type == "observation":
        visualise_trajectory(ep, trajs[-1], out_dir)  # select worst

      # TODO: dump metrics to tensorboard
      save_rewards_fig(exp_rewards, os.path.join(out_dir, "train_rewards.png"))
      save_rewards_fig(episode_rewards, os.path.join(out_dir, "eval_rewards.png"))

  # TODO: rm
  if config.state_type == "observation":
    for i in range(5):
      visualise_batch_from_experience(i, config, experience, out_dir)

  if PROF:
    train_time, eval_time, sim_time = [t / 1e9 for t in train_time], [t / 1e9 for t in eval_time], [t / 1e9 for t in sim_time]
    print(f"iter time:\n\t{np.median(train_time): .2f}s\n\t{np.median(eval_time): .2f}s\n\t{np.median(sim_time): .2f}s")
    print(f"total time:\n\t1.00x tr\n\t{np.sum(eval_time)/np.sum(train_time): .2f}x tr\n\t{np.sum(sim_time)/np.sum(train_time): .2f}x tr")

  return exp_rewards, episode_rewards, episode_trajectories


def collect_experience(config, E, experience, policy):
  # assume models take inputs with channels at first dimension (i.e., CxHxW)
  transform_to_pil_like = lambda x: x.permute((2, 0, 1)) if E.is_adroit else _images_to_observation(x.numpy(), bit_depth=5)
  with torch.no_grad():
    total_rwd = 0.0
    obs, _ = E.reset()
    policy.initialise(**dict(count=1))
    # roll out policy and update experience
    for t in tqdm(range(config.max_episode_length // config.action_repeat)):
      action = policy.sample_action(obs).squeeze(dim=0).cpu()
      next_obs, rwd, done = E.step(action)
      experience.append(transform_to_pil_like(obs), action, rwd, done)
      total_rwd += rwd
      obs = next_obs
      #if done: break # less time, but not good idea

    return total_rwd


if __name__ == "__main__":
  import sys

  # load user defined parameters
  config = PlanetConfig()
  if len(sys.argv) == 1:
    config.load("mj_envs_vision/utils/mini_config.json")
  else:
    config.load(sys.argv[1])
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
  # train policy on target environment
  exp_rewards, episode_rewards, episode_trajectories = train_policy(config, E, policy, optimiser, out_dir)
  E.close()

  # visualise performance
  plot_rewards(exp_rewards).savefig(os.path.join(out_dir, "train_rewards.png"))
  plot_rewards(episode_rewards).savefig(os.path.join(out_dir, "eval_rewards.png"))

  print("done :)")
