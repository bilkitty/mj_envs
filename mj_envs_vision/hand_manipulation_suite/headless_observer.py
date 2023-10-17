import numpy as np
import scipy as sci
from gym import utils
import cv2
import PIL
from PIL import ImageDraw, Image
import torch
import torchvision
from mujoco_py import MjRenderContextOffscreen
#from contacts import _contact_data

EPSILON = 1e-4
MAX_CONTACT = 5
S = 255//MAX_CONTACT

class HeadlessObserver(utils.EzPickle):
    NATIVE_H = 640 # wrt simulator, this is width and
    NATIVE_W = 480 # visa-versa (yields correct orientation)

    def __init__(self, sim, bid: int):
        self.sim = sim
        self.obj_bid = bid
        self.contact_type = 'none'
        self.state = None
        self.annotated_images = []
        utils.EzPickle.__init__(self)


    def mj_viewer_headless_setup(self):
        # configure simulation cam (instantiates camera)
        # instantiates mujoco_py.cymj.MjRenderContextOffscreen object
        self.sim.render(self.NATIVE_W, self.NATIVE_H)
        self.sim.forward()

        # NOTE: rendered image  starts clipping at d < 4.5
        #       aerial view: elevation = -45 - deg
        #       fronto-parallel: azimuth = 90, elevation -45 + deg
        self.sim._render_context_offscreen.cam.azimuth = 90
        self.sim._render_context_offscreen.cam.distance = 4.5
        self.set_view('default')


    def get_state(self):
        return self.state

    def get_annotated_images(self):
        x = self.annotated_images.copy()
        self.annotated_images = []
        return x


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
        image = pil_like_image.permute((1, 2, 0)).numpy()

        # debug view
        #cv2.imshow("test", image.astype('uint8'))
        #cv2.waitKey(1)

        # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html?highlight=MjContact#mjcontact
        i = 0
        text = ""
        self.state = [0] * 11 * MAX_CONTACT
        pil_annotated = Image.fromarray(image.astype('uint8'))
        for c in self.sim.data.contact:
            # c[11] = [dist, x, y, z, f1, f2, f3, f4, f5, g1, g2]
            if c.dist < -EPSILON or c.dist > EPSILON:
                if i+1 >= MAX_CONTACT:
                    break
                x = [c.dist, c.pos[0], c.pos[1], c.pos[2]]
                x.extend(c.friction)
                x.extend([c.geom1, c.geom2])
                self.state[i * 11: (i + 1) * 11] = x
                # TODO: move this into helpers
                bid1 = self.sim.model.geom_bodyid[x[-2]]
                bid2 = self.sim.model.geom_bodyid[x[-1]]
                bn1 = self.sim.model.body_names[self.sim.model.geom_bodyid[x[-2]]]
                bn2 = self.sim.model.body_names[self.sim.model.geom_bodyid[x[-1]]]
                if "Object" not in [bn1, bn2]:
                    continue
                #if "proximal" in bn1 and "proximal" in bn2:
                #    continue


                # fix transforms!!!


                # annotate in image with gradient of red to blue to distinguish contacts
                cid = 0
                body_names = self.sim.model.body_names
                xyz = np.array([[*x[1:4], 1]]);xyz[0,1] = 0
                # TODO: fix depth to y=0 ----- fix math below to avoid this
                #xyz = np.array([[*self.sim.data.body_xpos[self.sim.model.body_name2id('palm')], 1]]);xyz[0,1] = 0 # object (palm)
                # these are pos in world; object pos (0, -0.2, 0.035)
                xyz_ref1 = np.array([[*self.sim.data.body_xpos[self.sim.model.body_name2id('nail')], 1]]);xyz_ref1[0,1] = 0 # object (nail)
                xyz_ref2 = np.array([[*self.sim.data.geom_xpos[self.sim.model.geom_name2id('neck')], 1]]);xyz_ref2[0,1] = 0 # table (world)
                xyz_ref3 = np.array([[*self.sim.data.body_xpos[self.sim.model.body_name2id('Object')], 1]]) # table (world)
                # construct intrinsics using: FOV = 2 * arctan(h / 2 * f
                f = 0.5 * h / np.tanh(self.sim.model.cam_fovy[0] / 2 / 180 * np.pi)

                K = np.eye(3,3)
                K[:2, :3] = np.array([[f, 0, 0.5 * h], [0, f, 0.5 * h]])
                # construct extrinsics with view axis correction (unsure, x = x, +y = +z and +z = -y)
                # https://github.com/ARISE-Initiative/robosuite/blob/b9d8d3de5e3dfd1724f4a0e6555246c460407daa/robosuite/utils/camera_utils.py#L60C9-L60C9
                Rc = np.eye(4,4); Rc[1,1] = -1; Rc[2,2] = -1
                #Rc = np.eye(3,3); Rc[1,2] = -1; Rc[2,1] = 1; Rc[1,1] = 0; Rc[2,2] = 0
                #a = Rc @ xyz_ref1.T
                #b = Rc @ xyz_ref2.T

                R = sci.spatial.transform.Rotation.from_quat(self.sim.model.cam_quat[cid]).as_matrix() # in body frame
                t = self.sim.model.cam_pos[cid].reshape(-1,1)
                # construct projection mat P = K x I x Rt with axis-correction Q
                P = K @ np.concatenate([R, t], axis=1) @ Rc # 3x4
                # testing alternative axis correction, if we go with this, remove -1 scalar on v
                #P = K @ Rc @ np.concatenate([R, t], axis=1) # 3x4
                uvd = (P @ xyz.T).T.squeeze()
                uvd_ref1 = (P @ xyz_ref1.T).T.squeeze()
                uvd_ref2 = (P @ xyz_ref2.T).T.squeeze()
                uvd_ref3 = (P @ xyz_ref3.T).T.squeeze()
                # image origin is lower left corner
                #u_ref, v_ref = int(uvd_ref[0] / uvd_ref[2]), int(uvd_ref[1] / uvd_ref[2])
                u, v = int(uvd[0] / uvd[2]), -1 * int(uvd[1] / uvd[2])
                u_ref1, v_ref1 = int(uvd_ref1[0] / uvd_ref1[2]), -1 * int(uvd_ref1[1] / uvd_ref1[2])
                u_ref2, v_ref2 = int(uvd_ref2[0] / uvd_ref2[2]), -1 * int(uvd_ref2[1] / uvd_ref2[2])
                u_ref3, v_ref3 = int(uvd_ref3[0] / uvd_ref3[2]), -1 * int(uvd_ref3[1] / uvd_ref3[2])
                text += f"{bn1}->{bn2}: ({u}, {v})\n"
                ImageDraw.Draw(pil_annotated).rectangle((u - 2, v - 2, u + 2, v + 2), fill=(S * i, 0, 255 - S * i))
                ImageDraw.Draw(pil_annotated).rectangle((u_ref1 - 2, v_ref1 - 2, u_ref1 + 2, v_ref1 + 2), fill=(S * i * 2, 0, 255 - S * i * 2))
                ImageDraw.Draw(pil_annotated).rectangle((u_ref2 - 2, v_ref2 - 2, u_ref2 + 2, v_ref2 + 2), fill=(S * i * 3, 0, 255 - S * i * 3))
                ImageDraw.Draw(pil_annotated).rectangle((u_ref3 - 2, v_ref3 - 2, u_ref3 + 2, v_ref3 + 2), fill=(S * i * 3, 0, 255 - S * i * 3))
                i += 1

        # log annotated images for debugging
        self.annotated_images.append(np.array(pil_annotated))

        print('------------\n' + text)
        return image.astype('float') #/ 255


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