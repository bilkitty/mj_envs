import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
from mj_envs_vision.hand_manipulation_suite.headless_observer import HeadlessObserver
import os

ADD_BONUS_REWARDS = True
#DEFAULT_FRAME_SKIP = 5
DEFAULT_FRAME_SKIP = 1
DEFAULT_DT = 0.1


class DoorEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode, width=64, height=64, is_headless=False, variation_type=None):
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0
        self.observer = None # required for headless setup
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_door.xml', frame_skip=DEFAULT_FRAME_SKIP)
        # override rendering settings ---- but can't atm...
        self.metadata['video.frames_per_second'] = int(np.round(1.0 / DEFAULT_DT))
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
            self.observer = HeadlessObserver(self.sim, self.door_bid)
            #self.observer.set_view('aerial')
        self.reset_model()

        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])
        self.door_hinge_did = self.model.jnt_dofadr[self.model.joint_name2id('door_hinge')]
        self.grasp_sid = self.model.site_name2id('S_grasp')
        self.handle_sid = self.model.site_name2id('S_handle')
        self.door_bid = self.model.body_name2id('frame')


    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = self.data.qpos[self.door_hinge_did]

        # get to handle
        reward = -0.1*np.linalg.norm(palm_pos-handle_pos)
        # open door
        reward += -0.1*(door_pos - 1.57)*(door_pos - 1.57)
        # velocity cost
        reward += -1e-5*np.sum(self.data.qvel**2)

        if ADD_BONUS_REWARDS:
            # Bonus
            if door_pos > 0.2:
                reward += 2
            if door_pos > 1.0:
                reward += 8
            if door_pos > 1.35:
                reward += 10

        goal_achieved = True if door_pos >= 1.35 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = np.array([self.data.qpos[self.door_hinge_did]])
        if door_pos > 1.0:
            door_open = 1.0
        else:
            door_open = -1.0
        latch_pos = qp[-1]

        return np.concatenate([qp[1:-2], [latch_pos], door_pos, palm_pos, handle_pos, palm_pos - handle_pos, [door_open]]).astype('float32')

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.body_pos[self.door_bid,0] = self.np_random.uniform(low=-0.3, high=-0.2)
        self.model.body_pos[self.door_bid,1] = self.np_random.uniform(low=0.25, high=0.35)
        self.model.body_pos[self.door_bid,2] = self.np_random.uniform(low=0.252, high=0.35)
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
        door_body_pos = self.model.body_pos[self.door_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        self.model.body_pos[self.door_bid] = state_dict['door_body_pos']
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def mj_viewer_headless_setup(self):
        self.observer.mj_viewer_headless_setup()

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if door open for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage

    def render(self, *args, **kwargs):
        if self.is_headless:
            return self.observer.render(args, kwargs)
        else:
            self.viewer.render()
