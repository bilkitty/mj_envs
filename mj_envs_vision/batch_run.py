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
  assert len(vals) > 1
  param_sweep = list()
  for i, v in enumerate(vals):
    cnew = copy.deepcopy(c)
    cnew[parameter] = v
    if "run_id" in cnew.keys():
      cnew["run_id"] += f"-{parameter}-{i+1}"
    else:
      cnew["run_id"] = f"{parameter}-{i+1}"
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
    print("Usage:\n\trun.py [config_path]")
    sys.exit(-1)

  exps_spec = json.load(open(config_path, 'r'))
  config_path = os.path.join(BASE_PATH, exps_spec["config_path"])
  results_path = os.path.join(BASE_PATH, "results" , exps_spec["results_path"])
  default_config = load_config(config_path, exps_spec["policy_type"])

  exps = [dict()]
  # take cartesian product of hyper parameter settings
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
      for cfg in exps: cartprod.extend(new_experiment(cfg, k, vals))
      exps = cartprod

  # save batch-run config to results path
  os.makedirs(results_path, exist_ok=True)
  json.dump(exps_spec, open(os.path.join(results_path, "exps_spec.json"), 'w'))
  json.dump(exps, open(os.path.join(results_path, "exps_launched.json"), 'w'))

  # setup runs with exp instance configs
  # cnew.__dict__[parameter] = v
  # cnew.run_id += f"-{parameter}-{i}"
  workers = list()
  n_devices = torch.cuda.device_count()
  if n_devices == 0:
    print("No devices are available")
    sys.exit(1)

  for i, exp in enumerate(exps):
    exp_config = copy.deepcopy(default_config)
    updated_run_id = exp_config.run_id + exp["run_id"]
    exp_config.__dict__.update(exp)
    exp_config.run_id = updated_run_id
    exp_config.device_type = f"cuda:{i % n_devices}"

    args = (exp_config, exps_spec["policy_type"], os.path.join(results_path, f"run_{i}"),)
    workers.append(Process(target=run, args=args))
    #workers.append(Process(target=test_proc, args=(exp_config,)))

  # run subsets of processes concurrently
  chunks = len(workers) // n_devices
  for i in range(chunks):
    for j in range(n_devices):
      print(f"running exp {i*n_devices + j}")
      workers[i*n_devices + j].start()
    for j in range(n_devices):
      print(f"ending {i*n_devices + j}")
      workers[i*n_devices + j].join()

  # run remaining processes concurrently
  remainders = chunks * n_devices
  for i in range(remainders, len(workers)):
    print(f"rem running exp {i}")
    workers[i].start()
  for i in range(remainders, len(workers)):
    print(f"rem ending {i}")
    workers[i].join()

  print("done :)")
