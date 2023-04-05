import os
import torch
import time
import json
import numpy as np
import pickle as pkl

from tqdm import tqdm
from mj_envs_vision.utils.helpers import visualise_batch_from_experience
from mj_envs_vision.utils.helpers import visualise_trajectory
from mj_envs_vision.utils.helpers import plot_rewards
from mj_envs_vision.utils.helpers import make_env
from mj_envs_vision.utils.helpers import reset, step, observation_size, action_size
from mj_envs_vision.utils.config import load_config
from mj_envs_vision.utils.eval import evaluate
from mj_envs_vision.algos.baselines import make_baseline_policy, make_policy_optimisers


PROF = True


def train(config, policy, optimiser):
  # NOTE: there will be config.sample_iters x (config.max_episodes - config.seed_episodes + 1)
  # items in policy metrics.
  for t in range(config.sample_iters):
    L = policy.update()
    for opt in optimiser: opt.zero_grad(set_to_none=True)
    L.backward()
    policy.clip_grads(config.grad_clip_norm, norm_type=2)
    for opt in optimiser: opt.step()


def train_sb3_policy(config, E, policy, out_dir, device):
  train_time = list()
  eval_time = list()
  exp_rewards = list()
  episode_rewards = list()
  exp_successes = list()
  episode_successes = list()
  episode_trajectories = list()

  # populate buffer with requested batch and chunk size
  rwd, done = 0.0, False
  obs, _ = reset(E)

  for ep in tqdm(range(config.seed_episodes, config.max_episodes + 1)):
    if PROF: tns = time.time_ns()
    policy.update()
    exp_rewards.append((ep, policy.metrics.items()["value_loss"][-1]))
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
      pkl.dump(policy.metrics.items(), open(os.path.join(out_dir, "train_metrics.pkl"), "wb"))
      pkl.dump(exp_rewards, open(os.path.join(out_dir, "train_rewards.pkl"), "wb"))
      pkl.dump(episode_rewards, open(os.path.join(out_dir, "eval_rewards.pkl"), "wb"))

      # save model
      if ep % config.checkpoint_interval == 0:
        policy.save(os.path.join(out_dir, f"{policy.name}-{config.state_type}-{config.env_name}-{ep}"))

  if PROF:
    train_time, eval_time = [t / 1e9 for t in train_time], [t / 1e9 for t in eval_time]
    print(f"iter time:\n\t{np.median(train_time): .2f}s\n\t{np.median(eval_time): .2f}s")
    print(f"total time:\n\t1.00x tr\n\t{np.sum(eval_time)/np.sum(train_time): .2f}x tr")

  plot_rewards(exp_rewards, "total rewards").savefig(os.path.join(out_dir, "train_reward.png"))
  plot_rewards(episode_rewards, "total rewards").savefig(os.path.join(out_dir, "eval_rewards.png"))
  plot_rewards(episode_successes, "success rate").savefig(os.path.join(out_dir, "eval_success.png"))

  return exp_rewards, episode_rewards, policy.metrics.items(), episode_trajectories


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

  # populate buffer with requested batch and chunk size
  print(f"Initialising experience replay with max(batch_size, chunk_size) samples")
  collect_experience(E, policy, sample_count=config.batch_size * config.chunk_size)
  n_samples = config.max_episode_length // config.action_repeat

  for ep in tqdm(range(config.seed_episodes, config.max_episodes + 1)): # TODO: add total and initial?
    if PROF: tns = time.time_ns()
    train(config, policy, optimiser)
    if PROF: train_time.append(time.time_ns() - tns)
    # TODO: dump metrics to tensorboard

    if PROF: tns = time.time_ns()
    train_reward, train_successes = collect_experience(E, policy, n_samples)
    exp_rewards.append((ep, [train_reward]))
    exp_successes.append((ep, [train_successes]))
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
      pkl.dump(policy.metrics.items(), open(os.path.join(out_dir, "train_metrics.pkl"), "wb"))
      pkl.dump(exp_rewards, open(os.path.join(out_dir, "train_rewards.pkl"), "wb"))
      pkl.dump(episode_rewards, open(os.path.join(out_dir, "eval_rewards.pkl"), "wb"))


    # save model
    if ep % config.checkpoint_interval == 0:
      policy.save(os.path.join(out_dir, f"{policy.name}-{config.state_type}-{config.env_name}-{ep}.pt"))

  # TODO: rm
  if config.state_type == "observation" and policy.experience is not None:
    for i in range(5):
      visualise_batch_from_experience(i, config, policy.experience, out_dir)

  if PROF:
    train_time, eval_time, sim_time = [t / 1e9 for t in train_time], [t / 1e9 for t in eval_time], [t / 1e9 for t in sim_time]
    print(f"iter time:\n\t{np.median(train_time): .2f}s\n\t{np.median(eval_time): .2f}s\n\t{np.median(sim_time): .2f}s")
    print(f"total time:\n\t1.00x tr\n\t{np.sum(eval_time)/np.sum(train_time): .2f}x tr\n\t{np.sum(sim_time)/np.sum(train_time): .2f}x tr")

  plot_rewards(exp_rewards, "total rewards").savefig(os.path.join(out_dir, "train_rewards.png"))
  plot_rewards(episode_rewards, "total rewards").savefig(os.path.join(out_dir, "eval_rewards.png"))
  plot_rewards(exp_successes, "success rate").savefig(os.path.join(out_dir, "train_success.png"))
  plot_rewards(episode_successes, "success rate").savefig(os.path.join(out_dir, "eval_success.png"))

  return exp_rewards, episode_rewards, policy.metrics.items(), episode_trajectories


def collect_experience(E, policy, sample_count):
  if not policy.experience:
    return 0.0, 0

  with torch.no_grad():
    total_rwd = 0.0
    success = False
    obs, _ = reset(E)
    # for sampling single actions
    policy.reset(**dict(count=1))
    # roll out policy and update experience
    for t in tqdm(range(sample_count)):
      action = policy.sample_action(obs).squeeze(dim=0).cpu()
      next_obs, rwd, done, suc = step(E, action)
      policy.record_experience(obs, action, rwd, done)
      total_rwd += rwd
      success |= suc
      obs = next_obs
      #if done: break # less time, but not good idea

    return total_rwd, success


if __name__ == "__main__":
  import sys

  # load user defined parameters
  if len(sys.argv) != 2:
    print("Usage: config_fp policy_type")
  policy_type = sys.argv[2]
  config = load_config(sys.argv[1], policy_type)
  print(config.str())

  # validate params
  assert config.batch_size <= config.max_episode_length // config.action_repeat
  assert config.chunk_size <= config.max_episode_length // config.action_repeat

  # setup
  out_dir = os.path.join("results", f"train_{policy_type}_{config.run_id}")
  os.makedirs(out_dir, exist_ok=True)
  print('\033[96m' + f"saving results to {out_dir}" + '\033[0m')

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
  policy = make_baseline_policy(config, policy_type, E, device)
  if config.models_path != "":
    assert policy_type in config.models_path
    policy.load()

  optimiser = make_policy_optimisers(config, policy_type, policy)
  # TODO: load optimisers

  # train policy on target environment
  exp_rewards, episode_rewards, train_metrics, episode_trajectories = train_policy(config,
                                                                    E,
                                                                    policy,
                                                                    optimiser,
                                                                    out_dir,
                                                                    device)
  E.close()

  summary_metrics = {k:v[::config.sample_iters*config.checkpoint_interval] for k,v in train_metrics.items()}
  json.dump(summary_metrics, open(os.path.join(out_dir, "train_metrics.json"), "w"))
  pkl.dump(train_metrics, open(os.path.join(out_dir, "train_metrics.pkl"), "wb"))
  pkl.dump(exp_rewards, open(os.path.join(out_dir, "train_rewards.pkl"), "wb"))
  pkl.dump(episode_rewards, open(os.path.join(out_dir, "eval_rewards.pkl"), "wb"))

  print("done :)")
