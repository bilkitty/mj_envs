import numpy as np
from gym import utils
import cv2
import torch
import torchvision
from mujoco_py import MjRenderContextOffscreen
#from contacts import _contact_data

class HeadlessObserver(utils.EzPickle):
    NATIVE_H = 640 # wrt simulator, this is width and
    NATIVE_W = 480 # visa-versa (yields correct orientation)

    def __init__(self, sim, bid: int):
        self.sim = sim
        self.obj_bid = bid
        self.contact_type = 'none'
        utils.EzPickle.__init__(self)


    def mj_viewer_headless_setup(self):
        # configure simulation cam (instantiates camera)
        # instantiates mujoco_py.cymj.MjRenderContextOffscreen object
        self.sim.render(self.NATIVE_W, self.NATIVE_W)
        self.sim.forward()

        # NOTE: rendered image  starts clipping at d < 4.5
        #       aerial view: elevation = -45 - deg
        #       fronto-parallel: azimuth = 90, elevation -45 + deg
        self.sim._render_context_offscreen.cam.azimuth = 90
        self.sim._render_context_offscreen.cam.distance = 4.5
        self.set_view('default')


    def render(self, *args, **kwargs) -> np.ndarray:
        """ Returns an unnormalised and center cropped image """
        # /opt/anaconda3/envs/planet-mjenv/lib/python3.9/site-packages/mujoco_py/mjviewer.py
        if "height" in kwargs.keys() and "width" in kwargs.keys():
            # same convention as native resolution macros
            w, h = kwargs["height"], kwargs["width"]
            should_resize = 'enable_resize' in kwargs.keys() and kwargs['enable_resize']
        else:
            w, h = 64, 64
            should_resize = True

        resize = torchvision.transforms.Resize((w, h))
        center_crop = torchvision.transforms.CenterCrop((128, 128))

        image = self.sim.render(self.NATIVE_W, self.NATIVE_H)
        if image is None:
            return np.random.randint((w, h, 3)) if should_resize else np.random.randint((128, 128, 3))

        # mimic zoom by center cropping image
        image = torch.FloatTensor(image[::-1, :, :].copy()) # rendered images are upside-down
        pil_like_image = image.permute((2, 0, 1))
        pil_like_image = center_crop(pil_like_image)
        if should_resize:
            pil_like_image = resize(pil_like_image)
        image = pil_like_image.permute((1, 2, 0))

        # debug view
        #cv2.imshow("test", image.numpy().astype('uint8'))
        #cv2.waitKey(1)
        return image.numpy().astype('float') #/ 255


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