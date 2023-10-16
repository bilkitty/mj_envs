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
                bn = self.sim.model.body_names
                #xyz = np.array([[*x[1:4], 1]])
                # TODO: nail is spot on, but other objs are offset to bottom right...
                xyz = np.array([[*self.sim.data.body_xpos[-2], 1]]) # object (nail)
                #xyz = np.array([[*self.sim.data.body_xpos[5], 1]]) # object (palm)
                xyz_ref = np.array([[*self.sim.data.body_xpos[1], 1]]) # table (world)
                # construct intrinsics (FOV = 2 * arctan(h / 2 * f)
                f = 0.5 * h / np.tanh(self.sim.model.cam_fovy[0] / 2 / 180 * np.pi)
                t = self.sim.model.cam_pos[cid].reshape(-1,1)

                K = np.eye(3,3)
                K[:2, :3] = np.array([[f, 0, 0.5 * h], [0, f, 0.5 * h]])
                # construct extrinsics with view axis correction
                # https://github.com/ARISE-Initiative/robosuite/blob/b9d8d3de5e3dfd1724f4a0e6555246c460407daa/robosuite/utils/camera_utils.py#L60C9-L60C9
                Rc = np.eye(4,4); Rc[1,1] = -1; Rc[2,2] = -1
                R = sci.spatial.transform.Rotation.from_quat(self.sim.model.cam_quat[cid]).as_matrix() # in body frame
                # construct projection mat P = K x I x Rt with axis-correction Q
                P = np.concatenate([R, t], axis=1)
                P = K @ P @ Rc
                uvd = np.matmul(P, xyz.T).T.squeeze()
                uvd_ref = np.matmul(P, xyz_ref.T).T.squeeze()
                #uvd = np.matmul(xyz, K).squeeze() + self.sim.model.cam_pos[0]
                # image origin is lower left corner
                u, v = int(uvd[0] / uvd[2]), -1 * int(uvd[1] / uvd[2])
                u_ref, v_ref = int(uvd_ref[0] / uvd_ref[2]), -1 * int(uvd_ref[1] / uvd_ref[2])
                #u_ref, v_ref = int(uvd_ref[0] / uvd_ref[2]), int(uvd_ref[1] / uvd_ref[2])
                text += f"{bn1}->{bn2}: ({u}, {v})\n"
                #u, v = max(0, min(w, u)), max(0, min(h, v))
                ImageDraw.Draw(pil_annotated).rectangle((u - 2, v - 2, u + 2, v + 2), fill=(S * i, 0, 255 - S * i))
                ImageDraw.Draw(pil_annotated).rectangle((u_ref - 2, v_ref - 2, u_ref + 2, v_ref + 2), fill=(S * i, 0, 255 - S * i))
                #text += f"{x[-2]}->{x[-1]}: ({x[1]: .2f}, {x[2]: .2f}, {x[3]: .2f})\n"
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