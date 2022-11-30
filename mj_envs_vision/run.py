import os
import sys
import numpy as np
import torch
from torch import optim
from mj_envs_vision.utils.config import Config
from mj_envs_vision.utils.config import PPOConfig
from mj_envs_vision.utils.config import PlanetConfig
from mj_envs_vision.utils.train import train_policy
from mj_envs_vision.utils.train import train_sb3_policy
from mj_envs_vision.utils.helpers import plot_rewards
from mj_envs_vision.utils.helpers import make_env
from mj_envs_vision.algos.baselines import make_baseline_policy

def load_config(config_path, policy_type):
  if policy_type == "ppo":
    config = PPOConfig()
  elif policy_type == "planet":
    config = PlanetConfig()
  else:
    config = Config()
  config.load(config_path)
  print(config.str())

  return config


if __name__ == "__main__":
  # load user defined parameters
  if len(sys.argv) == 1:
    config_path = "mj_envs_vision/utils/test_config.json"
    policy_type = "ppo"
  elif len(sys.argv) <= 2:
    config_path = sys.argv[1]
    policy_type = "ppo"
  elif len(sys.argv) <= 3:
    config_path = sys.argv[1]
    policy_type = sys.argv[2]
  else:
    print("Usage:\n\trun.py [config_pathpath] [policy_type]")
    sys.exit(-1)

  config = load_config(config_path, policy_type)

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
  policy = make_baseline_policy(config, policy_type, E)
  if config.models_path != "":
    policy.load_models(config.models_path)
  optimiser = optim.Adam(policy.params_list, lr=config.learning_rate, eps=config.adam_epsilon)
  # train policy on target environment
  if policy_type == "planet":
    exp_rewards, episode_rewards, episode_trajectories = train_policy(config, E, policy, optimiser, out_dir)
  else:
    exp_rewards, episode_rewards, episode_trajectories = train_sb3_policy(config, E, policy, out_dir)
  E.close()

  # save performance metrics (TODO: pickle)

  # visualise performance
  plot_rewards(exp_rewards).savefig(os.path.join(out_dir, "train_rewards.png"))
  plot_rewards(episode_rewards).savefig(os.path.join(out_dir, "eval_rewards.png"))

  print("done :)")
