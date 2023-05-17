import torch
from tqdm import tqdm
from mj_envs_vision.utils.helpers import reset, step


def collect_online_experience(E, policy, sample_count):
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

def collect_offline_experience(E, policy, behaviour_policy, sample_count):
  if not policy.experience:
    return 0.0, 0

  with torch.no_grad():
    total_rwd = 0.0
    success = False
    obs, _ = reset(E)
    behaviour_obs = E.get_state() if behaviour_policy.name == "dapg" else obs
    # for sampling single actions
    behaviour_policy.reset(**dict(count=1))
    # roll out policy and update experience
    for t in tqdm(range(sample_count)):
      offline_action = behaviour_policy.sample_action(behaviour_obs).squeeze(dim=0).cpu()
      next_obs, rwd, done, suc = step(E, offline_action)
      policy.record_experience(obs, offline_action, rwd, done)
      total_rwd += rwd
      success |= suc
      obs = next_obs
      behaviour_obs = E.get_state() if behaviour_policy.name == "dapg" else obs
      #if done: break # less time, but not good idea

    return total_rwd, success
