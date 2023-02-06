# todo:
# - training perf
# - eval perf
# - dump metrics to pkl
#
# - test on env
# - test on env variants (ood)
import torch
import time
import os
import sys
import numpy as np

from tqdm import tqdm
from mj_envs_vision.utils.helpers import make_env
from mj_envs_vision.utils.helpers import reset, step
from mj_envs_vision.utils.helpers import plot_rewards
from mj_envs_vision.utils.config import load_config
from mj_envs_vision.algos.baselines import make_baseline_policy

def evaluate(config, policy, count=10):
  with torch.no_grad():
    total_rwds = []
    successes = []
    trajectories = []
    for i in range(count): # consider threading?
      rwd = 0.0
      success = 0.0
      traj = []
      T = make_env(config)
      obs, _ = reset(T)
      policy.initialise(**dict(count=1))
      for t in tqdm(range(config.max_episode_length // config.action_repeat)):
        action = policy.act(obs.squeeze(dim=0)).squeeze(dim=0).cpu()
        next_obs, r, done, s = step(T, action)
        traj.append((obs.squeeze(dim=0), action, r))
        rwd += r
        obs = next_obs
        if s:
          success += 1

      T.env.close()
      # record final obs, reward and success rates
      traj.append((next_obs.squeeze(dim=0), torch.zeros_like(action), r))
      total_rwds.append(rwd)
      successes.append(success / config.max_episode_length / config.action_repeat)
      trajectories.append(traj)

    return total_rwds, successes, trajectories


if __name__ == "__main__":
  MAX_EPISODES = 2
  if len(sys.argv) == 3:
    config_path = sys.argv[1]
    policy_type = sys.argv[2] # TODO: policy type should be loaded from config
  else:
    print("Usage:\n\teval.py [path_to_trained_policy] [policy_type]")
    sys.exit(-1)

  # Setup config
  config = load_config(config_path, policy_type)
  config.max_episodes = MAX_EPISODES

  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  if torch.cuda.is_available() and not config.disable_cuda:
    config.device_type = 'cuda'
    torch.cuda.manual_seed(config.seed)
  else:
    config.device_type = 'cpu'
    print('\033[93m' + "\n[WARN] No cuda devices found. Please reboot if on gpu capable machine.\n" + '\033[0m')

  # Load and run policy
  pi = make_baseline_policy(config, policy_type, make_env(config), torch.device(config.device_type))
  models_path = pi.load(config.models_path)

  total_rewards = list()
  successes = list()
  for ep in tqdm(range(config.max_episodes)):
    rwds, succs, trajs = evaluate(config, pi, count=10)
    total_rewards.append((ep, rwds))
    successes.append((ep, succs))

  out_path = os.path.dirname(models_path)
  model_name = os.path.basename(models_path)
  plot_rewards(total_rewards, "total rewards").savefig(os.path.join(out_path, f"{model_name}_eval_rewards.png"))
  plot_rewards(successes, "success rate").savefig(os.path.join(out_path, f"{model_name}_eval_success.png"))

  # save performance metrics (TODO: pickle)

  # visualise performance

  print("done :)")