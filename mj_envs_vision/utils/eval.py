import torch
import click
import os
import numpy as np
import pickle as pkl
import multiprocessing
from multiprocessing import pool
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from mj_envs_vision.utils.helpers import make_env
from mj_envs_vision.utils.helpers import BasicTimer
from mj_envs_vision.utils.helpers import reset, step
from mj_envs_vision.utils.helpers import plot_rewards
from mj_envs_vision.utils.helpers import visualise_trajectory
from mj_envs_vision.utils.config import load_config
from mj_envs_vision.algos.baselines import make_baseline_policy


CORE_BATCHES = 2

@click.command(help="Policy evaluation\n\nUsage:\n\teval.py --config_path path_to_trained_policy]")
@click.option('--config_path', type=str, help='environment to load', required=True)
@click.option('--out_path', type=str, help='output directory', required=True)
@click.option('--policy_type', type=str, help='{default, dapg, planet, ppo}', default="default")
@click.option('--variation_type', type=str, help='{pos, size, mass}', default=None)
@click.option('--trials', type=int, help='number of trials to visualize', default=5)
@click.option('--checkpoint_count', type=int, help='number of checkpoints to test', default=1)
@click.option('--parallel', is_flag=True, help='enables multi-core processing')


def main(config_path, out_path, policy_type, trials, variation_type, checkpoint_count, parallel):
  # Setup config
  config = load_config(config_path, policy_type)
  config.variation_type = variation_type

  if torch.cuda.is_available() and not config.disable_cuda:
    config.device_type = 'cuda'
    torch.cuda.manual_seed(config.seed)
  else:
    config.device_type = 'cpu'
    print('\033[93m' + "\n[WARN] No cuda devices found. Please reboot if on gpu capable machine.\n" + '\033[0m')

  os.makedirs(out_path, exist_ok=True)

  # Load and run policy
  pi = make_baseline_policy(config, policy_type, make_env(config), torch.device(config.device_type))

  rewards = list()
  successes = list()
  T = make_env(config)
  for idx in range(checkpoint_count):
    pi.set_checkpoint_index(idx)
    models_path = pi.load()
    model_id = models_path.split(".")[-2].split('-v0')[-1] # TODO: don't hard-code env id
    model_id = int(model_id) if model_id != '' else 0
    print('\033[96m' + f"saving results to {out_path}" + '\033[0m')

    # reset seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # cuda manual seed?

    if parallel:
      print(f"~~~~~ sharing strategy {torch.multiprocessing.get_sharing_strategy()}")
      print(f"Enabled multi-core processing ({multiprocessing.cpu_count() // CORE_BATCHES} cores) of {trials} trials.")
      rwds, succs, trajs, _ = evaluate_parallel(config, pi, count=trials)
      traj = [[(xx, None, None) for xx in x] for x in trajs]
    else:
      rwds, succs, trajs, _ = evaluate(config, pi, T, count=trials)

    visualise_trajectory(str(model_id), trajs[-1], out_path, prefix=f"trajectory_")  # select worst
    visualise_trajectory(str(model_id), [x[0] for x in trajs], out_path, prefix=f"init-trajectory_")

    rewards.append((model_id, rwds))
    successes.append((model_id, succs))

  rewards.sort(key=lambda x: x[0])
  successes.sort(key=lambda x: x[0])

  exp_name = f"var-{variation_type or 'fixed'}"
  plot_rewards(rewards, "total rewards").savefig(os.path.join(out_path, f"eval_rewards_{exp_name}.png"), bbox_inches='tight')
  plot_rewards(successes, "success rate").savefig(os.path.join(out_path, f"eval_success_{exp_name}.png"), bbox_inches='tight')

  # save performance metrics
  pkl.dump(rewards, open(os.path.join(out_path, "eval_rewards.pkl"), "wb"))

  # save config
  config.save(os.path.join(out_path, "config.json"))

def single_eval(config, policy, worker_id):
  rwd = 0.0
  success = False
  config.seed += worker_id # TODO: verify
  test_env = make_env(config)
  frame = test_env.get_pixels().squeeze(dim=0)
  traj = torch.zeros((config.max_episode_length // config.action_repeat, *frame.shape))
  obs, _ = reset(test_env)
  policy.reset(**dict(count=1))

  r = -float('inf')
  action = policy.act(obs.squeeze(dim=0)).squeeze(dim=0).cpu()
  # NOTE: avoid using tqdm to spare memory
  for t in range(config.max_episode_length // config.action_repeat):
    action = policy.act(obs.squeeze(dim=0)).squeeze(dim=0).cpu()
    next_obs, r, done, s = step(test_env, action)
    # TODO: consider record image observation to disk, if memory issues persist
    traj[t, ...] = test_env.get_pixels().squeeze(dim=0)
    rwd += r
    success |= s
    obs = next_obs

  # record final obs, reward and success rates
  traj[-1, ...] = test_env.get_pixels().squeeze(dim=0)
  test_env.env.close()
  return dict(reward=rwd, success=success, trajectory=traj)

def evaluate_parallel(config, policy, count=10):
  # TODO: time batches
  # TODO: fix rendered output
  with torch.no_grad():
    total_rwds, successes, trajectories = [], [], []
    # use subset of available cores to allow enough memory for simulation loops
    # ex. resource allocation runtime error https://github.com/pytorch/pytorch/issues/973
    core_count = multiprocessing.cpu_count() // CORE_BATCHES
    # need spawn context in order to render in child processes
    context = multiprocessing.get_context('spawn')
    worker_pool = pool.Pool(processes=core_count, context=context, maxtasksperchild=None)
    print(f"running {count // core_count} batches of trials")
    for i in range(count // core_count):
      worker_args = list(zip([config] * core_count, [policy] * core_count, range(core_count)))
      res = worker_pool.starmap(single_eval, worker_args)[0]
      total_rwds.append(res['reward']), successes.append(int(res['success'])), trajectories.append(res['trajectory'])

    print(f"processing {count % core_count} remaining trials")
    if count % core_count != 0:
      remaining_count = count % core_count
      worker_args = list(zip([config] * remaining_count, [policy] * remaining_count, range(remaining_count)))
      res = worker_pool.starmap(single_eval, worker_args)[0]
      total_rwds.append(res['reward']), successes.append(int(res['success'])), trajectories.append(res['trajectory'])

    worker_pool.close()
    worker_pool.join()
    print("finished parallel eval")
  return total_rwds, successes, trajectories, dict()

def _eval(config, policy, T, count=10):
  with torch.no_grad():
    total_rwds = []
    successes = []
    trajectories = []
    for i in range(count): # consider threading?
      rwd = 0.0
      success = False
      traj = []
      obs, _ = reset(T)
      policy.reset(**dict(count=1))
      for t in tqdm(range(config.max_episode_length // config.action_repeat)):
        frame = T.get_pixels().squeeze(dim=0)
        action = policy.act(obs.squeeze(dim=0)).squeeze(dim=0).cpu()
        next_obs, r, done, s = step(T, action)
        traj.append((frame, action, r))
        rwd += r
        success |= s
        obs = next_obs

      # record final obs, reward and success rates
      traj.append((T.get_pixels().squeeze(dim=0), torch.zeros_like(action), r))
      total_rwds.append(rwd)
      successes.append(int(success))
      trajectories.append(traj)
      T.env.close()
  return total_rwds, successes, trajectories, dict()


def _eval_timed(config, policy, T, count=10):
  with torch.no_grad():
    total_rwds = []
    successes = []
    trajectories = []
    timer = BasicTimer('ms')
    for i in range(count): # consider threading?
      rwd = 0.0
      success = False
      traj = []

      obs, _ = reset(T)
      policy.reset(**dict(count=1))
      for t in tqdm(range(config.max_episode_length // config.action_repeat)):
        frame = T.get_pixels().squeeze(dim=0)
        timer.start("act")
        action = policy.act(obs.squeeze(dim=0)).squeeze(dim=0).cpu()
        timer.stop("act")
        timer.start("step")
        next_obs, r, done, s = step(T, action)
        timer.stop("step")
        traj.append((frame, action, r))
        rwd += r
        success |= s
        obs = next_obs

      # record final obs, reward and success rates
      traj.append((T.get_pixels().squeeze(dim=0), torch.zeros_like(action), r))
      total_rwds.append(rwd)
      successes.append(int(success))
      trajectories.append(traj)

    T.env.close()

  return total_rwds, successes, trajectories, timer.dump()

def evaluate(config, policy, T, count=10, should_time=False):
  if should_time:
    return _eval_timed(config, policy, T, count)
  else:
    return _eval(config, policy, T, count)


if __name__ == "__main__":
  main()
  print("done :)")
