import gym
import mj_envs
import click
import os
import gym
import numpy as np
import pickle
import torch
from mjrl.utils.gym_env import GymEnv
from torchvision.utils import save_image
from mjrl.policies.gaussian_mlp import MLP
from dependencies.PlaNet.env import _images_to_observation

DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python visualize_policy.py --env_name door-v0 \n
    $ python visualize_policy.py --env_name door-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', default=None)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('--episodes', type=int, help='number of episodes to visualize', default=10)
@click.option('--save_mode', type=int, default=0, help='flag to save renderings')

def main(env_name, policy, mode, seed, episodes, save_mode):
    e = GymEnv(env_name)
    e.set_seed(seed)
    if policy is not None:
        pi = pickle.load(open(policy, 'rb'))
    else:
        pi = MLP(e.spec, hidden_sizes=(32,32), seed=seed, init_log_std=-1.0)

    if save_mode == 1:
        e.env.reset()
        observation = _images_to_observation(e.env.render(mode='rgb_array'), bit_depth=5)
        save_image(torch.as_tensor(observation), os.path.join('/home/bilkit/Workspace/mj_envs_vision/results', f'test_render.png'))
        print(f"done")
    else:
        # render policy
        e.visualize_policy(pi, num_episodes=episodes, horizon=e.horizon, mode=mode)

if __name__ == '__main__':
    main()