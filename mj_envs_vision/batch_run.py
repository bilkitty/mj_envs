import os
import sys
import json
import copy
import torch
from typing import List, Union
from multiprocessing import Process
from mj_envs_vision.utils.config import load_config
from mj_envs_vision.run import run


BASE_PATH = "/home/bilkit/Workspace/mj_envs_vision"


def new_experiment(c, parameter: str, vals: List[Union[int, str, float]]):
  param_sweep = list()
  for i, v in enumerate(vals):
    cnew = copy.deepcopy(c)
    cnew[parameter] = v
    if "run_id" in cnew.keys():
      cnew["run_id"] += f"-{parameter}-{i+1}"
    else:
      cnew["run_id"] = f"-{parameter}-{i+1}"
    param_sweep.append(cnew)
  return param_sweep


def test_proc(a):
  print(f"running {a}")


if __name__ == "__main__":
  # load user defined parameters
  if len(sys.argv) == 1:
    config_path = "exp.json"
  elif len(sys.argv) <= 2:
    config_path = sys.argv[1]
  else:
    print("Usage:\n\tbatch_run.py [config_path]")
    sys.exit(-1)

  exps = dict()
  exps_spec = json.load(open(config_path, 'r'))
  results_path = os.path.join(BASE_PATH, "results" , exps_spec["results_path"])
  for policy_type, config_file in zip(exps_spec["policy_types"], exps_spec["config_paths"]):
    exps[policy_type] = [dict(run_id=f"-{policy_type}")]

    # take cartesian product of hyperparameter settings
    if "hyper_parameters" in exps_spec.keys():
      for id, (k, v) in enumerate(exps_spec["hyper_parameters"].items()):
        if v[0] == "str":
          vals = v[1]
        elif v[0] == "float":
          vals = [float(f) for f in v[1]]
        elif v[0] == "int":
          vals = [int(f) for f in v[1]]
        else:
          raise Exception(f"Unknown parameter type{v[0]} ({k})")

        cartprod = list()
        for cfg in exps[policy_type]: cartprod.extend(new_experiment(cfg, k, vals))
        exps[policy_type] = cartprod

  # save batch-run config to results path
  os.makedirs(results_path, exist_ok=True)
  json.dump(exps_spec, open(os.path.join(results_path, "exps_spec.json"), 'w'), indent=2)
  json.dump(exps, open(os.path.join(results_path, "exps_launched.json"), 'w'), indent=2)

  workers = list()
  n_devices = torch.cuda.device_count()
  if n_devices == 0:
    print("No devices are available")
    sys.exit(1)
  else:
    print('\033[96m' + f"Preparing {len(list(exps.keys())) * len(list(exps.values())[0])} experiments" + '\033[0m')


  # TODO: more clever staggering of processes to avoid gaps left by fast algos
  for policy_type, config_file in zip(exps_spec["policy_types"], exps_spec["config_paths"]):
    default_config = load_config(os.path.join(BASE_PATH, config_file), policy_type)
    for i, exp in enumerate(exps[policy_type]):
      exp_config = copy.deepcopy(default_config)
      exp_config.__dict__.update(exp)
      exp_config.device_type = f"cuda:{i % n_devices}"

      args = (exp_config, policy_type, os.path.join(results_path, default_config.run_id + exp["run_id"]),)
      workers.append(Process(target=run, args=args))
      #workers.append(Process(target=test_proc, args=(exp_config,)))

  # run subsets of processes concurrently
  chunks = len(workers) // n_devices
  for i in range(chunks):
    for j in range(n_devices):
      workers[i*n_devices + j].start()
    for j in range(n_devices):
      workers[i*n_devices + j].join()

  # run remaining processes concurrently
  remainders = chunks * n_devices
  for i in range(remainders, len(workers)):
    workers[i].start()
  for i in range(remainders, len(workers)):
    workers[i].join()

  print("done :)")
