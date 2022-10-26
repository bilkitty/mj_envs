import os
import cv2
import click
import pickle
import torch
from mjrl.utils.gym_env import GymEnv
from torchvision.utils import save_image
from mjrl.policies.gaussian_mlp import MLP

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
@click.option('--save_mode', type=int, default=1, help='flag to save renderings')

def main(env_name, policy, mode, seed, episodes, save_mode):
    try:
        e = GymEnv(env_name)
    except:
        print("retry create env")
        e = GymEnv(f'mj_envs_vision:{env_name}')

    e.set_seed(seed)
    if policy is not None:
        pi = pickle.load(open(policy, 'rb'))
    else:
        pi = MLP(e.spec, hidden_sizes=(32,32), seed=seed, init_log_std=-1.0)

    if save_mode == 1:
        e.env.reset()
        image = cv2.resize(e.env.render(mode='rgb_array'), (64, 64), interpolation=cv2.INTER_LINEAR) # crop/resize image
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # put channel first
        # TODO: fix these transforms
        image = 2 * image - 1 # zero center
        image = image.unsqueeze(dim=0)  # +batch dimension
        save_image(image, os.path.join('/home/bilkit/Workspace/mj_envs_vision/results', f'test_render.png'))
        print(f"done")
    else:
        # render policy
        e.visualize_policy(pi, num_episodes=episodes, horizon=e.horizon, mode=mode)

if __name__ == '__main__':
    main()