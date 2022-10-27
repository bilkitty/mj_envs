import numpy as np
from gym import utils
import torch
import torchvision
#from contacts import _contact_data

class HeadlessObserver(utils.EzPickle):

    def __init__(self, sim, bid: int):
        self.sim = sim
        self.obj_bid = bid
        self.contact_type = 'none'
        self._resize = torchvision.transforms.Resize((64, 64))
        self._center_crop = torchvision.transforms.CenterCrop((128, 128))
        utils.EzPickle.__init__(self)

        # setup rendering
        self.mj_viewer_headless_setup()


    def mj_viewer_headless_setup(self):
        # configure simulation cam
        self.sim.render(64, 64)
        self.sim.forward()

        # NOTE: rendered image  starts clipping at d < 4.5
        #       aerial view: elevation = -45 - deg
        #       fronto-parallel: azimuth = 90, elevation -45 + deg
        self.sim._render_context_offscreen.cam.azimuth = 90
        self.sim._render_context_offscreen.cam.distance = 4.5
        self.set_view('default')


    def render(self, *args, **kwargs):
        # /opt/anaconda3/envs/planet-mjenv/lib/python3.9/site-packages/mujoco_py/mjviewer.py
        should_resize = 'enable_resize' in args[1].keys() and args[1]['enable_resize']
        image = self.sim.render(640, 480)
        if image is None:
            return np.random.randint((64, 64, 3)) if should_resize else np.random.randint((128, 128, 3))
        image = image[::-1, :, :] # rendered images are upside-down
        image = image.transpose(2, 0, 1).copy()  # put channel first
        image = self._center_crop(torch.tensor(image, dtype=torch.float32))     # crop
        if should_resize:
            image = self._resize(image)
        # TODO: fix these transforms
        return image / 255  # normalise + batch dimension


    def contact_type(self, contact_type: str):
        self.contact_type = contact_type


    def set_view(self, view_type: str):
        view_type = view_type.lower()
        lookatv = self.sim.data.body_xpos[self.obj_bid] - self.sim.data.cam_xpos[-1]
        if view_type == 'aerial' or view_type == 'top-down':
            self.sim._render_context_offscreen.cam.elevation = -45 - np.rad2deg(np.arccos(lookatv[0] / lookatv[2])) / 2
        elif view_type == 'default':
            self.sim._render_context_offscreen.cam.elevation = -45 + np.rad2deg(np.arccos(lookatv[0] / lookatv[2])) / 2
        else:
            raise Exception(f"Unsupported view type '{view_type}'")