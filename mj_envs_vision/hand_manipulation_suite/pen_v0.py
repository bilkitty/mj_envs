import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mj_envs_vision.utils.quatmath import quat2euler, euler2quat
from mj_envs_vision.hand_manipulation_suite.headless_observer import HeadlessObserver
from mujoco_py import MjViewer
import os

ADD_BONUS_REWARDS = True

class PenEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode, width=64, height=64, is_headless=False, variation_type=None):
        self.target_obj_bid = 0
        self.S_grasp_sid = 0
        self.eps_ball_sid = 0
        self.obj_bid = 0
        self.obj_t_sid = 0
        self.obj_b_sid = 0
        self.tar_t_sid = 0
        self.tar_b_sid = 0
        self.pen_length = 1.0
        self.tar_length = 1.0
        self.use_aerial_view = False
        self.observer = None # required for headless setup

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_pen.xml', 5)
        self.render_mode = render_mode
        self.is_headless = is_headless
        self.width = width
        self.height = height
        self.variation_type = variation_type

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        utils.EzPickle.__init__(self)

        # setup models and renderer (either window or headless aka 'nogui')
        if self.is_headless:
            self.observer = HeadlessObserver(self.sim, self.target_obj_bid)
            #self.observer.set_view('aerial')
        self.reset_model() # TODO: verify that this should be commented

        self.target_obj_bid = self.sim.model.body_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.eps_ball_sid = self.sim.model.site_name2id('eps_ball')
        self.obj_t_sid = self.sim.model.site_name2id('object_top')
        self.obj_b_sid = self.sim.model.site_name2id('object_bottom')
        self.tar_t_sid = self.sim.model.site_name2id('target_top')
        self.tar_b_sid = self.sim.model.site_name2id('target_bottom')

        self.pen_length = np.linalg.norm(self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])
        self.tar_length = np.linalg.norm(self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])

        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            starting_up = False
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            starting_up = True
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)

        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        desired_loc = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.pen_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length

        # pos cost
        dist = np.linalg.norm(obj_pos-desired_loc)
        reward = -dist
        # orien cost
        orien_similarity = np.dot(obj_orien, desired_orien)
        reward += orien_similarity

        if ADD_BONUS_REWARDS:
            # bonus for being close to desired orientation
            if dist < 0.075 and orien_similarity > 0.9:
                reward += 10
            if dist < 0.075 and orien_similarity > 0.95:
                reward += 50

        # penalty for dropping the pen
        done = False
        if obj_pos[2] < 0.075:
            reward -= 5
            done = True if not starting_up else False

        goal_achieved = True if (dist < 0.075 and orien_similarity > 0.95) else False

        return self.get_obs(), reward, done, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        qp = self.data.qpos.ravel()
        obj_vel = self.data.qvel[-6:].ravel()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        desired_pos = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.pen_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length
        # Ensure type is compatible with that of the default observation spec
        return np.concatenate([qp[:-6], obj_pos, obj_vel, obj_orien, desired_orien,
                               obj_pos-desired_pos, obj_orien-desired_orien]).astype('float32')

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.sim.forward()
        if self.is_headless:
            # NOTE: ensure that EGL rendering libs are referenced
            # (i.e., unset LD_PRELOAD)
            self.mj_viewer_headless_setup()
        else:
            # NOTE: ensure that rendering libs are referenced by
            # exporting libGLEW.so and libGL.so paths to LD_PRELOAD
            self.mj_viewer_setup()
        return self.get_obs(), {}

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        desired_orien = self.model.body_quat[self.target_obj_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, desired_orien=desired_orien)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        desired_orien = state_dict['desired_orien']
        self.set_state(qp, qv)
        self.model.body_quat[self.target_obj_bid] = desired_orien
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = -45
        self.sim.forward()
        self.viewer.cam.distance = 1.0

    def mj_viewer_headless_setup(self):
        self.observer.mj_viewer_headless_setup()

    def mj_viewer_headless_setup(self):
        # configure simulation cam
        self.sim.render(64, 64)
        self.sim.forward()

        # NOTE: rendered image  starts clipping at d < 4.5
        #       aerial view: elevation = -45 - deg
        #       fronto-parallel: azimuth = 90, elevation -45 + deg
        lookatv = self.sim.data.body_xpos[self.target_obj_bid] - self.sim.data.cam_xpos[-1]
        self.sim._render_context_offscreen.cam.azimuth = 90
        self.sim._render_context_offscreen.cam.distance = 4.5
        if self.use_aerial_view:
            self.sim._render_context_offscreen.cam.elevation = -45 - np.rad2deg(np.arccos(lookatv[0] / lookatv[2])) / 2
        else:
            self.sim._render_context_offscreen.cam.elevation = -45 + np.rad2deg(np.arccos(lookatv[0] / lookatv[2])) / 2


    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if pen within 15 degrees of target for 20 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 20:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage

    def render(self, *args, **kwargs):
        if self.is_headless:
            return self.observer.render(args, kwargs)
        else:
            self.viewer.render()