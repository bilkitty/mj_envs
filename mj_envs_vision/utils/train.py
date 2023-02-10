import os
import torch
import time
import numpy as np

from tqdm import tqdm
from torch import optim
from dependencies.PlaNet import memory
from dependencies.PlaNet.env import _images_to_observation
from mj_envs_vision.algos.baselines import Planet
from mj_envs_vision.utils.helpers import visualise_batch_from_experience
from mj_envs_vision.utils.helpers import visualise_trajectory
from mj_envs_vision.utils.helpers import plot_rewards
from mj_envs_vision.utils.helpers import make_env
from mj_envs_vision.utils.helpers import reset, step, observation_size, action_size
from mj_envs_vision.utils.config import PlanetConfig
from mj_envs_vision.utils.eval import evaluate


PROF = True


def train(config, experience, policy, optimiser):
  for t in range(config.sample_iters):
    # sample data iid
    obs, actions, rewards, not_done = experience.sample(config.batch_size, config.chunk_size)
    policy.update(sample_batch=[obs, actions, rewards, not_done], optimiser=optimiser)

  return policy.metrics.total_loss() # releases comp graph memory


def train_sb3_policy(config, E, policy, out_dir, device):
  train_time = list()
  eval_time = list()
  exp_rewards = list()
  episode_rewards = list()
  exp_successes = list()
  episode_successes = list()
  episode_trajectories = list()
  # initialise experience replay buffer
  experience = memory.ExperienceReplay(config.experience_size,
                                       config.state_type == "vector",
                                       observation_size(E),
                                       action_size(E),
                                       config.bit_depth,
                                       device)

  # populate buffer with requested batch and chunk size
  rwd, done = 0.0, False
  obs, _ = reset(E)

  for ep in tqdm(range(config.seed_episodes, config.max_episodes + 1)):
    if PROF: tns = time.time_ns()
    policy.update(sample_batch=[], optimiser=None)
    exp_rewards.append((ep, policy.metrics.total_loss()["value_loss"][-1]))
    #exp_successes.append((ep, policy.metrics.total_loss()["success"][-1]))
    if PROF: train_time.append(time.time_ns() - tns)

    if ep % config.test_interval == 0:
      if PROF: tns = time.time_ns()
      policy.set_models_to_eval()
      rewards, successes, trajs = evaluate(config, policy, count=10)
      policy.set_models_to_train()
      if PROF: eval_time.append(time.time_ns() - tns)

      episode_rewards.append((ep, rewards))
      episode_successes.append((ep, successes))
      episode_trajectories.append((ep, trajs))
      if config.state_type == "observation":
        visualise_trajectory(ep, trajs[-1], out_dir)  # select worst

      # TODO: dump metrics to tensorboard
      plot_rewards(exp_rewards, "total rewards").savefig(os.path.join(out_dir, "train_reward_loss.png"))
      plot_rewards(episode_rewards, "total rewards").savefig(os.path.join(out_dir, "eval_rewards.png"))
      plot_rewards(episode_successes, "success rate").savefig(os.path.join(out_dir, "eval_successes.png"))

      # save model
      if ep % config.checkpoint_interval == 0:
        policy.save(os.path.join(out_dir, f"{policy.name}-{config.state_type}-{config.env_name}-{ep}"))

  if PROF:
    train_time, eval_time = [t / 1e9 for t in train_time], [t / 1e9 for t in eval_time]
    print(f"iter time:\n\t{np.median(train_time): .2f}s\n\t{np.median(eval_time): .2f}s")
    print(f"total time:\n\t1.00x tr\n\t{np.sum(eval_time)/np.sum(train_time): .2f}x tr")

  plot_rewards(exp_rewards, "total rewards").savefig(os.path.join(out_dir, "train_reward_loss.png"))
  plot_rewards(episode_rewards, "total rewards").savefig(os.path.join(out_dir, "eval_rewards.png"))
  plot_rewards(episode_successes, "success rate").savefig(os.path.join(out_dir, "eval_successes.png"))

  return exp_rewards, episode_rewards, episode_trajectories


def train_policy(config, E, policy, optimiser, out_dir, device):
  # NOTE: all policies are responsible for preprocessing obs into input data
  train_time = list()
  eval_time = list()
  sim_time = list()
  exp_rewards = list()
  episode_rewards = list()
  exp_successes = list()
  episode_successes = list()
  episode_trajectories = list()
  # TODO: replace with simpler buffer (no discretisation)
  # initialise experience replay buffer
  experience = memory.ExperienceReplay(config.experience_size,
                                       config.state_type == "vector",
                                       observation_size(E),
                                       action_size(E),
                                       config.bit_depth,
                                       device)

  # populate buffer with requested batch and chunk size
  rwd, done = 0.0, False
  obs, _ = reset(E)
  print(f"Initialising experience replay with max(batch_size, chunk_size) samples")
  while experience.idx <= max(config.batch_size, config.chunk_size):
    if done:
      rwd, done = 0.0, False
      obs, _ = reset(E)

    action = E.action_space.sample()
    experience.append(_images_to_observation(obs.cpu().numpy(), bit_depth=5), torch.FloatTensor(action), rwd, done)
    obs, rwd, done, _ = step(E, action) # success rate is not given to algos

  for ep in tqdm(range(config.seed_episodes, config.max_episodes + 1)): # TODO: add total and initial?
    if PROF: tns = time.time_ns()
    train_metrics = train(config, experience, policy, optimiser)
    if PROF: train_time.append(time.time_ns() - tns)
    # TODO: dump metrics to tensorboard

    if PROF: tns = time.time_ns()
    train_reward, train_successes = collect_experience(config, E, experience, policy)
    exp_rewards.append((ep, train_reward))
    exp_successes.append((ep, train_successes))
    if PROF: sim_time.append(time.time_ns() - tns)

    if ep % config.test_interval == 0:
      if PROF: tns = time.time_ns()
      policy.set_models_to_eval()
      rewards, successes, trajs = evaluate(config, policy, count=10)
      policy.set_models_to_train()
      if PROF: eval_time.append(time.time_ns() - tns)

      episode_rewards.append((ep, rewards))
      episode_successes.append((ep, successes))
      episode_trajectories.append((ep, trajs))
      if config.state_type == "observation":
        visualise_trajectory(ep, trajs[-1], out_dir)  # select worst

      # TODO: dump metrics to tensorboard
      plot_rewards(exp_rewards, "total rewards").savefig(os.path.join(out_dir, "train_reward_loss.png"))
      plot_rewards(episode_rewards, "total rewards").savefig(os.path.join(out_dir, "eval_rewards.png"))
      plot_rewards(exp_successes, "success rate").savefig(os.path.join(out_dir, "train_success.png"))
      plot_rewards(episode_successes, "success rate").savefig(os.path.join(out_dir, "eval_successes.png"))


    # save model
    if ep % config.checkpoint_interval == 0:
      policy.save(os.path.join(out_dir, f"{policy.name}-{config.state_type}-{config.env_name}-{ep}.pt"))

  # TODO: rm
  if config.state_type == "observation":
    for i in range(5):
      visualise_batch_from_experience(i, config, experience, out_dir)

  if PROF:
    train_time, eval_time, sim_time = [t / 1e9 for t in train_time], [t / 1e9 for t in eval_time], [t / 1e9 for t in sim_time]
    print(f"iter time:\n\t{np.median(train_time): .2f}s\n\t{np.median(eval_time): .2f}s\n\t{np.median(sim_time): .2f}s")
    print(f"total time:\n\t1.00x tr\n\t{np.sum(eval_time)/np.sum(train_time): .2f}x tr\n\t{np.sum(sim_time)/np.sum(train_time): .2f}x tr")

  plot_rewards(exp_rewards, "total rewards").savefig(os.path.join(out_dir, "train_reward_loss.png"))
  plot_rewards(episode_rewards, "total rewards").savefig(os.path.join(out_dir, "eval_rewards.png"))
  plot_rewards(exp_successes, "success rate").savefig(os.path.join(out_dir, "train_success.png"))
  plot_rewards(episode_successes, "success rate").savefig(os.path.join(out_dir, "eval_successes.png"))

  return exp_rewards, episode_rewards, episode_trajectories


def collect_experience(config, E, experience, policy):
  with torch.no_grad():
    total_rwd = 0.0
    success = False
    obs, _ = reset(E)
    policy.initialise(**dict(count=1))
    # roll out policy and update experience
    for t in tqdm(range(config.max_episode_length // config.action_repeat)):
      action = policy.sample_action(obs).squeeze(dim=0).cpu()
      next_obs, rwd, done, suc = step(E, action)
      experience.append(_images_to_observation(obs.cpu().numpy(), bit_depth=5), action, rwd, done)
      total_rwd += rwd
      success |= suc
      obs = next_obs
      #if done: break # less time, but not good idea

    return total_rwd, success


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
    config.device_type = 'cuda'
    torch.cuda.manual_seed(config.seed)
  else:
    config.device_type = 'cpu'

  # TODO: create worker setup and parallelise
  # instantiate env, policy, optimiser
  E = make_env(config)
  device = torch.device(config.device_type)
  policy = Planet(config, action_size(E), observation_size(E), action_size(E), device)
  if config.models_path != "":
    policy.load_models(config.models_path)
  optimiser = optim.Adam(policy.params_list, lr=config.learning_rate, eps=config.adam_epsilon)
  # train policy on target environment
  exp_rewards, episode_rewards, episode_trajectories = train_policy(config,
                                                                    E,
                                                                    policy,
                                                                    optimiser,
                                                                    out_dir,
                                                                    device)
  E.close()

  # visualise performance
  plot_rewards(exp_rewards).savefig(os.path.join(out_dir, "train_rewards.png"))
  plot_rewards(episode_rewards).savefig(os.path.join(out_dir, "eval_rewards.png"))

  print("done :)")
