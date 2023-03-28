import os
import sys
import numpy as np
import torch
from mj_envs_vision.utils.config import load_config
from mj_envs_vision.utils.train import train_policy
from mj_envs_vision.utils.train import train_sb3_policy
from mj_envs_vision.utils.helpers import make_env
from mj_envs_vision.algos.baselines import make_baseline_policy, make_policy_optimisers


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
    print("Usage:\n\trun.py [config_path] [policy_type]")
    sys.exit(-1)

  config = load_config(config_path, policy_type)

  # validate params
  assert config.batch_size <= config.max_episode_length // config.action_repeat
  assert config.chunk_size <= config.max_episode_length // config.action_repeat

  # setup log dir
  out_dir = os.path.join("results", f"run_{policy_type}_{config.run_id}")
  os.makedirs(out_dir, exist_ok=True)
  print('\033[96m' + f"saving results to {out_dir}" + '\033[0m')

  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  if torch.cuda.is_available() and not config.disable_cuda:
    torch.cuda.manual_seed(config.seed)
  else:
    config.device_type = 'cpu'
    print('\033[93m' + "\n[WARN] No cuda devices found. Please reboot if on gpu capable machine.\n" + '\033[0m')

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
  if policy_type == "ppo":
    exp_rewards, episode_rewards, episode_trajectories = train_sb3_policy(config, E, policy, out_dir, device)
  else:
    exp_rewards, episode_rewards, episode_trajectories = train_policy(config, E, policy, optimiser, out_dir, device)
  E.close()

  # save (updated) run config (in case of updates)
  config.models_path = out_dir
  config.save(os.path.join(out_dir, "config.json"))

  # save performance metrics (TODO: pickle)

  # visualise performance

  # TODO: plot metrics
  # TODO: save perf to txt

  print("done :)")
