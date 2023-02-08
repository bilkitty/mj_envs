import os
import click
import pickle
import torch
from gym.wrappers.time_limit import TimeLimit
from PIL import Image
from mj_envs_vision.utils.config import load_config
from mj_envs_vision.algos.baselines import make_baseline_policy
from mj_envs_vision.utils.helpers import make_env, reset, step

max_door_offset = 25 # the number of iterations after which door policy freezes
horizons = {"door": 2000, "pen": 2000, "relocate": 1000, "hammer": 2000}
GIF_DURATION = 15
DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python visualize_policy.py --env_name door-v0 \n
    $ python visualize_policy.py --env_name door-v0 --variation_type pos\n
    $ python visualize_policy.py --env_name door-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
'''
# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', default=None)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('--episodes', type=int, help='number of episodes to visualize', default=1)
@click.option('--save_mode', type=int, default=1, help='flag to save renderings (0=no save, 1=save)')
@click.option('--variation_type', type=str, help='variation type for env parameters {mass, size, pos}', default=None)


def main(env_name, policy, mode, seed, episodes, save_mode, variation_type):
    # parse policy type
    config_path = "/home/bilkit/Workspace/mj_envs_vision/mj_envs_vision/utils/mini_config.json"
    if policy is None:
        policy_type = "default"
    elif "planet" in policy and ".pt" in policy:
        policy_type = "planet"
        config_path = os.path.join(os.path.dirname(policy), "config.json")
    elif "ppo" in policy and ".zip" in policy:
        policy_type = "ppo"
        config_path = os.path.join(os.path.dirname(policy), "config.json")
    else:
        policy_type = "dapg"
        config_path = os.path.join(os.path.dirname(policy), "config.json")

    if save_mode == 1 and "LD_PRELOAD" in os.environ and os.environ["LD_PRELOAD"] != "":
        raise Exception("Ensure that EGL is available. Use command 'unset $LD_PRELOAD'.")

    # setup env and policy with default (training) parameters
    config = load_config(config_path, policy_type)
    config.env_name = env_name
    config.nogui = save_mode == 1
    config.seed = seed
    config.max_episodes = episodes
    config.variation_type = variation_type
    config.state_type = 'vector'
    config.models_path = policy

    e = make_env(config)
    pi = make_baseline_policy(config, policy_type, e, device=torch.device('cpu'))
    pi.load()
    if save_mode == 1:
        record_policy(e, episodes, pi, env_name, policy_name='-'.join(policy.split('.')[0].split('/')[-2:]))
    else:
        visualise_policy(e, episodes, env_name, mode, pi)

    print(f"done")


def visualise_policy(e, episodes, env_name, mode, pi):
    env_base_name = env_name.split('-')[0]
    e.unwrapped.mujoco_render_frames = True
    for ep in range(episodes):
        t = 0
        o, _ = reset(e)

        # NOTE: strangely, task failure on
        # door open when delay is active
        # TODO: source of failure?
        if 'door' in env_name or 'hammer' in env_name:
            offset = max_door_offset - 3
        else:
            offset = horizons[env_base_name] / 10

        while t < offset:
            a_zeros = pi.sample_action(o) * 0
            step(e, a_zeros)
            t = t+1

        d = False
        score = 0.0
        while t < horizons[env_base_name] and d == False:
            a = pi.sample_action(o) if mode == 'exploration' else pi.act(o)
            o, r, d = step(e, a)[:3]
            t = t+1
            score = score + r
        print("Episode score = %f" % score)
        # TODO:  prompt for moving on
    e.unwrapped.mujoco_render_frames = False


ENABLE_RESIZE = False
RESULTS_DIR = '/home/bilkit/Workspace/mj_envs_vision/results'
def record_policy(gym_env, n_eps, policy, env_name="unk", policy_name="random", results_dir=RESULTS_DIR):
    # unwrap time limit envs for step
    if isinstance(gym_env.env, TimeLimit):
        gym_env.env = gym_env.env.env

    # setup env for off-screen rendering
    gym_env.env.mj_viewer_headless_setup()

    print('\033[96m' + f"saving renderings to {results_dir}" + '\033[0m')

    # roll out policy for k episodes
    trajectory = []
    for k in range(n_eps):
        t = 0
        term = False
        obs, _ = reset(gym_env)
        trajectory.append([gym_env.get_pixels().numpy()])
        while t < horizons[env_name.split('-')[0]] and term is False:
            action = policy.act(obs)
            obs, reward, term = step(gym_env, action)[:3]
            trajectory[-1].append(gym_env.get_pixels().numpy())
            t += 1

        # inspect the first frame of the first episode
        # to convert frames to WxHxC
        if trajectory[0][0].shape[0] == 3:
            pils = [Image.fromarray(frame.transpose(1, 2, 0).astype('uint8')) for frame in trajectory[-1]]
        else:
            pils = [Image.fromarray(frame.astype('uint8')) for frame in trajectory[-1]]
        pils[0].save(os.path.join(results_dir, f'visualise_{env_name}_{policy_name}_{k}.gif'),
                     append_images=pils, save_all=True, optimize=False, loop=True, duration=GIF_DURATION)


if __name__ == '__main__':
    main()
