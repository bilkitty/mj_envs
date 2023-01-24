import os
import click
import pickle
import torch
from gym.wrappers.time_limit import TimeLimit
from PIL import Image
from mjrl.policies.gaussian_mlp import MLP
from mj_envs_vision.utils.config import Config, PlanetConfig
from mj_envs_vision.algos.baselines import Planet, PPOBaseline, SUPPORTED_POLICIES
from mj_envs_vision.utils.helpers import make_env, action_size, observation_size

max_door_offset = 25 # the number of iterations after which door policy freezes
horizons = {"door": 2000, "pen": 2000, "relocate": 1000, "hammer": 2000}
GIF_DURATION = 15
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
@click.option('--episodes', type=int, help='number of episodes to visualize', default=1)
@click.option('--save_mode', type=int, default=1, help='flag to save renderings (0=no save, 1=save)')


def main(env_name, policy, mode, seed, episodes, save_mode):
    # parse policy type
    if policy is None:
        policy_type = "default"
    elif "planet" in policy and ".pt" in policy:
        policy_type = "planet"
        # TODO: currently can't render and use gui simultaneously
        # update save mode
        save_mode = 0 if policy.split('-')[1] == "observation" else 1
    else:
        policy_type = "pretrained"

    if save_mode == 0 and os.environ["LD_PRELOAD"] != "":
        raise Exception("Ensure that EGL is available. Use command 'unset $LD_PRELOAD'.")

    # setup env
    config = Config()
    config.env_name = env_name
    config.nogui = save_mode == 1
    config.seed = seed
    config.max_episodes = episodes
    e = make_env(config)

    # setup policy
    if policy_type == "default":
        config.state_type = 'vector'  # TODO: support for images
        pi = MLP(e.env_spec, hidden_sizes=(32, 32), seed=seed, init_log_std=-1.0)
    elif policy_type == "planet":
        pconfig = PlanetConfig() # TODO: copy ctr
        for k,v in config.__dict__.items():
            pconfig.__dict__[k] = v
        pi = Planet(pconfig, action_size(e), observation_size(e), e.action_space, torch.device(config.device_type))
        # TODO: move impl
        #pi.load_models(policy)
        pi.models = torch.load(policy)
    elif policy_type == "pretrained":
        config.state_type = 'vector'  # TODO: support for images
        pi = pickle.load(open(policy, 'rb'))
    else:
        raise Exception(f"Unsupported policy_type: {policy_type}")

    if save_mode == 1:
        policy_name = policy.split('.')[0].split('/')[-2:]
        record_policy(e, episodes, 'rgb_array', env_name, policy_name='-'.join(policy.split('.')[0].split('/')[-2:]))
    else:
        if os.path.basename(policy).split('-')[0] in SUPPORTED_POLICIES:
            visualise_policy(e, episodes, env_name, mode, pi)
        else:
            visualise_mlp_policy(e, episodes, env_name, mode, pi)

    print(f"done")


def visualise_policy(e, episodes, env_name, mode, pi):
    env_base_name = env_name.split('-')[0]
    e.unwrapped.mujoco_render_frames = True
    for ep in range(episodes):
        t = 0
        o = e.env.reset()
        if isinstance(o, tuple): o = o[0]

        # NOTE: strangely, task failure on
        # door open when delay is active
        # TODO: source of failure?
        if 'door' in env_name:
            offset = max_door_offset - 3
        else:
            offset = horizons[env_base_name] / 10

        while t < offset:
            x = torch.FloatTensor(o)
            a_zeros = pi.act(torch.FloatTensor(o))[0] * 0
            e.env.step(a_zeros)
            t = t+1

        d = False
        score = 0.0
        while t < horizons[env_base_name] and d == False:
            a = pi.act(torch.FloatTensor(o))[0] if mode == 'exploration' else pi.act(torch.FloatTensor(o))[1]['evaluation']
            o, r, d = e.env.step(a)[:3]
            t = t+1
            score = score + r
        print("Episode score = %f" % score)
        # TODO:  prompt for moving on
    e.unwrapped.mujoco_render_frames = False


def visualise_mlp_policy(e, episodes, env_name, mode, pi):
    env_base_name = env_name.split('-')[0]
    e.unwrapped.mujoco_render_frames = True
    for ep in range(episodes):
        t = 0
        o = e.env.reset()
        if isinstance(o, tuple): o = o[0]

        # NOTE: strangely, task failure on
        # door open when delay is active
        # TODO: source of failure?
        if 'door' in env_name:
            offset = max_door_offset - 3
        else:
            offset = horizons[env_base_name] / 10

        while t < offset:
            a_zeros = pi.get_action(o)[0] * 0
            e.env.step(a_zeros)
            t = t+1

        d = False
        score = 0.0
        while t < horizons[env_base_name] and d == False:
            a = pi.get_action(o)[0] if mode == 'exploration' else pi.get_action(o)[1]['evaluation']
            o, r, d = e.env.step(a)[:3]
            t = t+1
            score = score + r
        print("Episode score = %f" % score)
        # TODO:  prompt for moving on
    e.unwrapped.mujoco_render_frames = False


ENABLE_RESIZE = False
RESULTS_DIR = '/home/bilkit/Workspace/mj_envs_vision/results'
def record_policy(gym_env, n_eps, mode, env_name="unk", policy_name="random", policy=None, results_dir=RESULTS_DIR):
    # unwrap time limit envs for step
    if isinstance(gym_env.env, TimeLimit):
        gym_env.env = gym_env.env.env

    # setup env for off-screen rendering
    gym_env.env.mj_viewer_headless_setup()

    # roll out policy for k episodes
    trajectory = []
    for k in range(n_eps):
        t = 0
        term = False
        obs = gym_env.env.reset()
        trajectory.append([gym_env.env.render(mode, ENABLE_RESIZE)])
        while t < horizons[env_name.split('-')[0]] and term is False:
            action = policy.get_action(obs)[0] if policy else gym_env.env.action_space.sample()
            obs, reward, term = gym_env.env.step(action)[:3]
            trajectory[-1].append(gym_env.env.render(mode, ENABLE_RESIZE))
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
