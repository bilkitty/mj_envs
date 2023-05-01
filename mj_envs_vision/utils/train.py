import os
import torch
import time
import json
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

from tqdm import tqdm
from mj_envs_vision.utils.helpers import visualise_batch_from_experience
from mj_envs_vision.utils.helpers import visualise_trajectory
from mj_envs_vision.utils.helpers import plot_rewards, plot_time
from mj_envs_vision.utils.helpers import make_env
from mj_envs_vision.utils.helpers import reset, step
from mj_envs_vision.utils.helpers import BasicTimer
from mj_envs_vision.utils.config import load_config
from mj_envs_vision.utils.eval import evaluate
from mj_envs_vision.algos.baselines import make_baseline_policy, make_policy_optimisers


def train(config, policy, optimiser):
  # NOTE: there will be config.train_epochs x (config.max_episodes - config.seed_episodes + 1)
  # items in policy metrics.
  for t in range(config.train_epochs):
    L = policy.update()
    for opt in optimiser: opt.zero_grad(set_to_none=True)
    L.backward()
    policy.clip_grads(config.grad_clip_norm, norm_type=2)
    for opt in optimiser: opt.step()

def train_sb3_policy_(config, E, policy, out_dir, PROF=False):
  exp_rewards = list()
  episode_rewards = list()
  exp_successes = list()
  episode_successes = list()
  episode_trajectories = list()
  timer_s = BasicTimer('s')
  train_timings_ms = dict()
  eval_timings_ms = dict()

  obs, _ = reset(E)
  test_env = make_env(config)

  for ep in tqdm(range(config.seed_episodes, config.max_episodes + 1)):
    if PROF: timer_s.start("train")
    policy.update()
    if PROF: timer_s.stop("train")
    exp_rewards.append((ep, policy.metrics.items()["value_loss"][-1]))
    #exp_successes.append((ep, policy.metrics.total_loss()["success"][-1]))
    train_timings_ms.update(policy.timer_ms.dump())

    if ep % config.test_interval == 0:
      if PROF: timer_s.start("eval")
      policy.set_models_to_eval()
      rewards, successes, trajs, eval_timings = evaluate(config, policy, test_env, count=10, should_time=PROF)
      policy.set_models_to_train()
      if PROF: timer_s.stop("eval")
      eval_timings_ms.update(eval_timings)

      episode_rewards.append((ep, rewards))
      episode_successes.append((ep, successes))
      episode_trajectories.append((ep, trajs))
      if config.state_type == "observation":
        visualise_trajectory(str(ep), trajs[-1], out_dir)  # select worst
        visualise_trajectory(f"{ep}-init", [x[0] for x in trajs], out_dir)

      # TODO: dump metrics to tensorboard
      train_metrics = policy.metrics.items()
      summary_metrics = {k:v[::config.checkpoint_interval] for k,v in train_metrics.items()}
      pkl.dump(summary_metrics, open(os.path.join(out_dir, f"train_metrics-{config.seed_episodes}.pkl"), "wb"))
      pkl.dump(exp_rewards, open(os.path.join(out_dir, f"train_rewards-{config.seed_episodes}.pkl"), "wb"))
      pkl.dump(episode_rewards, open(os.path.join(out_dir, f"eval_rewards-{config.seed_episodes}.pkl"), "wb"))
      pkl.dump(dict(total=timer_s.dump(), train=train_timings_ms, eval=eval_timings_ms),
               open(os.path.join(out_dir, f"timings-{config.seed_episodes}.pkl"), "wb"))

      # save model
      if ep % config.checkpoint_interval == 0:
        policy.save(os.path.join(out_dir, f"{policy.name}-{config.state_type}-{config.env_name}-{ep}"))

  if PROF:
    timings_s = timer_s.dump()
    print(f"\ttrain={np.median(timings_s['train']): .2f}s, test={np.median(timings_s['eval']): .2f}s")
    print(f"\ttrain={np.sum(timings_s['train']): .2f}s, test={np.sum(timings_s['eval']): .2f}s")
    plot_time(timings_s, config.max_episodes, "runtime (s)").savefig(os.path.join(out_dir, "time_total.png"), bbox_inches='tight')
    plot_time(train_timings_ms, config.max_episodes, "runtime (ms)").savefig(os.path.join(out_dir, "time_train.png"), bbox_inches='tight')
    plot_time(eval_timings_ms, config.max_episodes, "runtime (ms)").savefig(os.path.join(out_dir, "time_eval.png"), bbox_inches='tight')

  plot_rewards(exp_rewards, "total rewards").savefig(os.path.join(out_dir, "train_reward.png"), bbox_inches='tight')
  plot_rewards(episode_rewards, "total rewards").savefig(os.path.join(out_dir, "eval_rewards.png"), bbox_inches='tight')
  plot_rewards(episode_successes, "success rate").savefig(os.path.join(out_dir, "eval_success.png"), bbox_inches='tight')

  return exp_rewards, episode_rewards, policy.metrics.items(), episode_trajectories, timer_s.dump()

def train_sb3_policy(config, E, policy, out_dir, PROF=False):
  exp_rewards = list()
  episode_rewards = list()
  episode_successes = list()
  episode_trajectories = list()
  timer_s = BasicTimer('s')
  timer_ms = BasicTimer('ms')
  train_timings_ms = dict()
  eval_timings_ms = dict()

  timer_ms.start("eval-init")
  test_env = make_env(config)
  timer_ms.stop("eval-init")
  eval_timings_ms["eval-init"] = list(timer_ms.dump().items())

  for ep in tqdm(range(config.seed_episodes, config.max_episodes, config.test_interval)):
    if PROF: timer_s.start("train")
    policy.update()
    if PROF: timer_s.stop("train")
    exp_rewards.append((ep, policy.metrics.items()["value_loss"][-1]))
    for k,v in policy.timer_ms.dump().items():
      if train_timings_ms.get(k):
        train_timings_ms[k].append(np.mean(v))
      else:
        train_timings_ms[k] = [np.mean(v)]

    if PROF: timer_s.start("eval")
    policy.set_models_to_eval()
    rewards, successes, trajs, eval_timings = evaluate(config, policy, test_env, count=10, should_time=PROF)
    policy.set_models_to_train()
    if PROF: timer_s.stop("eval")
    for k,v in eval_timings.items():
      if eval_timings_ms.get(k):
        eval_timings_ms[k].append(np.mean(v))
      else:
        eval_timings_ms[k] = [np.mean(v)]

    episode_rewards.append((ep, rewards))
    episode_successes.append((ep, successes))
    episode_trajectories.append((ep, trajs))
    if config.state_type == "observation":
      visualise_trajectory(str(ep), trajs[-1], out_dir)  # select worst
      visualise_trajectory(f"{ep}-init", [x[0] for x in trajs], out_dir)

    # TODO: dump metrics to tensorboard
    train_metrics = policy.metrics.items()
    summary_metrics = {k:v[::config.checkpoint_interval] for k,v in train_metrics.items()}
    pkl.dump(summary_metrics, open(os.path.join(out_dir, f"train_metrics-{config.seed_episodes}.pkl"), "wb"))
    pkl.dump(exp_rewards, open(os.path.join(out_dir, f"train_rewards-{config.seed_episodes}.pkl"), "wb"))
    pkl.dump(episode_rewards, open(os.path.join(out_dir, f"eval_rewards-{config.seed_episodes}.pkl"), "wb"))
    pkl.dump(dict(total=timer_s.dump(), train=train_timings_ms, eval=eval_timings_ms),
             open(os.path.join(out_dir, f"timings-{config.seed_episodes}.pkl"), "wb"))

    # save model
    if ep % config.checkpoint_interval == 0:
      policy.save(os.path.join(out_dir, f"{policy.name}-{config.state_type}-{config.env_name}-{ep}"))

  if PROF:
    timings_s = timer_s.dump()
    print(f"\ttrain={np.median(timings_s['train']): .2f}s, test={np.median(timings_s['eval']): .2f}s")
    print(f"\ttrain={np.sum(timings_s['train']): .2f}s, test={np.sum(timings_s['eval']): .2f}s")
    plot_time(timings_s, config.max_episodes, "runtime (s)").savefig(os.path.join(out_dir, "time_total.png"), bbox_inches='tight')
    plot_time(train_timings_ms, config.max_episodes, "runtime (ms)").savefig(os.path.join(out_dir, "time_train.png"), bbox_inches='tight')
    plot_time(eval_timings_ms, config.max_episodes, "runtime (ms)").savefig(os.path.join(out_dir, "time_eval.png"), bbox_inches='tight')

  plot_rewards(exp_rewards, "total rewards").savefig(os.path.join(out_dir, "train_reward.png"), bbox_inches='tight')
  plot_rewards(episode_rewards, "total rewards").savefig(os.path.join(out_dir, "eval_rewards.png"), bbox_inches='tight')
  plot_rewards(episode_successes, "success rate").savefig(os.path.join(out_dir, "eval_success.png"), bbox_inches='tight')

  return exp_rewards, episode_rewards, policy.metrics.items(), episode_trajectories, timer_s.dump()


def train_policy(config, E, policy, optimiser, out_dir, PROF=False):
  # NOTE: all policies are responsible for preprocessing obs into input data
  exp_rewards = list()
  episode_rewards = list()
  exp_successes = list()
  episode_successes = list()
  episode_trajectories = list()
  timer_s = BasicTimer('s')
  timer_ms = BasicTimer('ms')
  train_timings_ms = dict()
  eval_timings_ms = dict()

  # populate buffer with requested batch and chunk size
  print(f"Initialising experience replay with max(batch_size, chunk_size) samples")
  if PROF: timer_s.start("sim")
  collect_experience(E, policy, sample_count=config.batch_size * config.chunk_size)
  if PROF: timer_s.stop("sim")
  n_samples = config.max_episode_length // config.action_repeat

  timer_ms.start("eval-init")
  test_env = make_env(config)
  timer_ms.stop("eval-init")
  eval_timings_ms["eval-init"] = list(timer_ms.dump().items())

  for ep in tqdm(range(config.seed_episodes, config.max_episodes + 1)): # TODO: add total and initial?
    if PROF: timer_s.start("train")
    train(config, policy, optimiser)
    if PROF: timer_s.stop("train")
    for k,v in policy.timer_ms.dump().items():
      if train_timings_ms.get(k):
        train_timings_ms[k].append(np.mean(v))
      else:
        train_timings_ms[k] = [np.mean(v)]
    # TODO: dump metrics to tensorboard

    if PROF: timer_s.start("sim")
    train_reward, train_successes = collect_experience(E, policy, n_samples)
    if PROF: timer_s.stop("sim")
    exp_rewards.append((ep, [train_reward]))
    exp_successes.append((ep, [train_successes]))

    if ep % config.test_interval == 0:
      if PROF: timer_s.start("eval")
      policy.set_models_to_eval()
      rewards, successes, trajs, eval_timings = evaluate(config, policy, test_env, count=10, should_time=PROF)
      policy.set_models_to_train()
      if PROF: timer_s.stop("eval")
      for k,v in eval_timings.items():
        if eval_timings_ms.get(k):
          eval_timings_ms[k].append(np.mean(v))
        else:
          eval_timings_ms[k] = [np.mean(v)]

      episode_rewards.append((ep, rewards))
      episode_successes.append((ep, successes))
      episode_trajectories.append((ep, trajs))
      if config.state_type == "observation":
        visualise_trajectory(str(ep), trajs[-1], out_dir)  # select worst
        visualise_trajectory(f"{ep}-init", [x[0] for x in trajs], out_dir)

      # TODO: dump metrics to tensorboard
      train_metrics = policy.metrics.items()
      summary_metrics = {k:v[::config.checkpoint_interval] for k,v in train_metrics.items()}
      pkl.dump(summary_metrics, open(os.path.join(out_dir, f"train_metrics-{config.seed_episodes}.pkl"), "wb"))
      pkl.dump(exp_rewards, open(os.path.join(out_dir, f"train_rewards-{config.seed_episodes}.pkl"), "wb"))
      pkl.dump(episode_rewards, open(os.path.join(out_dir, f"eval_rewards-{config.seed_episodes}.pkl"), "wb"))
      pkl.dump(dict(total=timer_s.dump(), train=train_timings_ms, eval=eval_timings_ms),
               open(os.path.join(out_dir, f"timings-{config.seed_episodes}.pkl"), "wb"))


    # save model
    if ep % config.checkpoint_interval == 0:
      policy.save(os.path.join(out_dir, f"{policy.name}-{config.state_type}-{config.env_name}-{ep}.pt"))

  # TODO: rm
  if config.state_type == "observation" and policy.experience is not None:
    for i in range(5):
      visualise_batch_from_experience(i, config, policy.experience, out_dir)

  if PROF:
    timings_s = timer_s.dump()
    print(f"\ttrain={np.median(timings_s['train']): .2f}s, sim={np.median(timings_s['sim']): .2f}s, test={np.median(timings_s['eval']): .2f}s")
    print(f"\t    ->{np.sum(timings_s['train']): .2f}s,   ->{np.sum(timings_s['sim']): .2f}s,    ->{np.sum(timings_s['eval']): .2f}s")
    plot_time(timings_s, config.max_episodes, "runtime (s)").savefig(os.path.join(out_dir, "time_total.png"), bbox_inches='tight')
    plot_time(train_timings_ms, config.max_episodes, "runtime (ms)").savefig(os.path.join(out_dir, "time_train.png"), bbox_inches='tight')
    plot_time(eval_timings_ms, config.max_episodes, "runtime (ms)").savefig(os.path.join(out_dir, "time_eval.png"), bbox_inches='tight')


  plot_rewards(exp_rewards, "total rewards").savefig(os.path.join(out_dir, "train_rewards.png"), bbox_inches='tight')
  plot_rewards(episode_rewards, "total rewards").savefig(os.path.join(out_dir, "eval_rewards.png"), bbox_inches='tight')
  plot_rewards(exp_successes, "success rate").savefig(os.path.join(out_dir, "train_success.png"), bbox_inches='tight')
  plot_rewards(episode_successes, "success rate").savefig(os.path.join(out_dir, "eval_success.png"), bbox_inches='tight')

  return exp_rewards, episode_rewards, policy.metrics.items(), episode_trajectories, timer_s.dump()


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
  results = train_sb3_policy(config, E, policy, out_dir, True) if policy_type == "ppo" \
    else train_policy(config, E, policy, optimiser, out_dir, True)
  E.close()

  exp_rewards, episode_rewards, train_metrics = results[:3]
  summary_metrics = {k:v[::config.checkpoint_interval] for k,v in train_metrics.items()}
  json.dump(summary_metrics, open(os.path.join(out_dir, f"train_metrics-{config.seed_episodes}.json"), "w"))
  pkl.dump(summary_metrics, open(os.path.join(out_dir, f"train_metrics-{config.seed_episodes}.pkl"), "wb"))
  pkl.dump(exp_rewards, open(os.path.join(out_dir, f"train_rewards-{config.seed_episodes}.pkl"), "wb"))
  pkl.dump(episode_rewards, open(os.path.join(out_dir, f"eval_rewards-{config.seed_episodes}.pkl"), "wb"))

  print("done :)")

