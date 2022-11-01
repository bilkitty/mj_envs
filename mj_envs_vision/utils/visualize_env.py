import os
import click
import pickle
import mj_envs_vision
from gym.wrappers.time_limit import TimeLimit
from PIL import Image
from mjrl.utils.gym_env import GymEnv
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
    e = GymEnv(env_name)
    e.set_seed(seed)
    if policy is not None:
        pi = pickle.load(open(policy, 'rb'))
    else:
        pi = MLP(e.spec, hidden_sizes=(32,32), seed=seed, init_log_std=-1.0)

    if save_mode == 1:
        record_policy(e, num_episodes=episodes, mode='rgb_array', env_name=env_name, policy_name="random")
        print(f"done")
    else:
        # render policy
        e.visualize_policy(pi, num_episodes=episodes, horizon=e.horizon, mode=mode)


ENABLE_RESIZE = False
RESULTS_DIR = '/home/bilkit/Workspace/mj_envs_vision/results'
def record_policy(gym_env, num_episodes, mode, env_name="unk", policy_name="unk", policy=None):
    # unwrap time limit envs for step
    if isinstance(gym_env.env, TimeLimit):
        gym_env.env = gym_env.env.env

    gym_env.env.mj_viewer_headless_setup()

    # roll out policy for k episodes
    trajectory = []
    for k in range(num_episodes):
        t = 0
        term = False
        obs = gym_env.env.reset()
        trajectory.append([gym_env.env.render(mode, ENABLE_RESIZE).numpy() * 255])
        while t < gym_env.horizon and term is False:
            obs, reward, term, _ = gym_env.step(policy.get_action(obs)[0] if policy else gym_env.env.action_space.sample())
            trajectory[-1].append(gym_env.env.render(mode, ENABLE_RESIZE).numpy() * 255)
            t += 1

        pils = [Image.fromarray(frame.transpose(1, 2, 0).astype('uint8')) for frame in trajectory[-1]]
        pils[0].save(os.path.join(RESULTS_DIR, f'{env_name}_{policy_name}_{k}.gif'),
                     append_images=pils, save_all=True, optimize=False, loop=True, duration=len(pils) * 5)


if __name__ == '__main__':
    main()