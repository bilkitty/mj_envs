# todo:
# - training perf
# - eval perf
# - dump metrics to pkl
#
# - test on env
# - test on env variants (ood)
import torch
import click
import os
import sys
import numpy as np

from tqdm import tqdm
from mj_envs_vision.utils.helpers import make_env
from mj_envs_vision.utils.helpers import reset, step
from mj_envs_vision.utils.helpers import plot_rewards
from mj_envs_vision.utils.helpers import visualise_trajectory
from mj_envs_vision.utils.config import load_config
from mj_envs_vision.algos.baselines import make_baseline_policy

@click.command(help="Policy evaluation\n\nUsage:\n\teval.py --config_path path_to_trained_policy]")
@click.option('--config_path', type=str, help='environment to load', required=True)
@click.option('--policy_type', type=str, help='{default, dapg, planet, ppo}', default="default")
@click.option('--variation_type', type=str, help='{pos, size, mass}', default=None)
@click.option('--episodes', type=int, help='number of episodes to visualize', default=5)


def main(config_path, policy_type, episodes, variation_type):
  # Setup config
  config = load_config(config_path, policy_type)
  config.max_episodes = episodes
  config.variation_type = variation_type

  if torch.cuda.is_available() and not config.disable_cuda:
    config.device_type = 'cuda'
    torch.cuda.manual_seed(config.seed)
  else:
    config.device_type = 'cpu'
    print('\033[93m' + "\n[WARN] No cuda devices found. Please reboot if on gpu capable machine.\n" + '\033[0m')

  # Load and run policy
  pi = make_baseline_policy(config, policy_type, make_env(config), torch.device(config.device_type))
  models_path = pi.load()
  out_path = os.path.dirname(models_path)
  model_name = os.path.basename(models_path).replace('.', '_') + "_var-" + (variation_type or "fixed")
  print('\033[96m' + f"saving results to {out_path}" + '\033[0m')

  total_rewards = list()
  successes = list()
  for ep in tqdm(range(config.max_episodes)):
    # reset seeds
    np.random.seed(config.seed + ep)
    torch.manual_seed(config.seed + ep)
    # cuda manual seed?

    rwds, succs, trajs = evaluate(config, pi, count=10)
    total_rewards.append((ep, rwds))
    successes.append((ep, succs))

    visualise_trajectory(ep, trajs[-1], out_path, prefix=model_name)  # select worst

  plot_rewards(total_rewards, "total rewards").savefig(os.path.join(out_path, f"{model_name}_eval_rewards.png"))
  plot_rewards(successes, "success rate").savefig(os.path.join(out_path, f"{model_name}_eval_success.png"))

  # save performance metrics (TODO: pickle)


def evaluate(config, policy, count=10):
  with torch.no_grad():
    total_rwds = []
    successes = []
    trajectories = []
    for i in range(count): # consider threading?
      rwd = 0.0
      success = False
      traj = []
      T = make_env(config)
      obs, _ = reset(T)
      policy.initialise(**dict(count=1))
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

    return total_rwds, successes, trajectories


if __name__ == "__main__":
  main()
  print("done :)")
