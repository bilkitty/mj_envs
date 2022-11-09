import os
import numpy as np
from PIL import Image
from typing import List, Tuple
from matplotlib import pyplot as plt



class Metrics:
  def __init__(self):
    self.total_return = list()


def to_image_frame(obs: np.ndarray):
  """ converts image observation with pels in [-0.5, 0.5] to image with pels in [0, 255] """
  return (255 * (obs.transpose((1, 2, 0)) + 0.5)).astype('uint8')

def to_input_obs(frame: np.ndarray):
  """ converts image with pels in [0, 255] to image observation with pels in [-0.5, 0.5] """
  return (frame.transpose((2, 0, 1)) / 255 - 0.5).astype('float')

def save_as_gif(frames: List[np.ndarray], gif_path: str):
  pils = list()
  for frame in frames:
    pils.append(Image.fromarray(to_image_frame(frame)))
  pils[0].save(gif_path, append_images=pils, save_all=True, optimize=False, loop=True, duration=len(pils) / 10)

def plot_rewards(rewards: List[Tuple]):
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  ep = [x[0] for x in rewards]
  rwd = [x[1] for x in rewards]
  ax.plot(ep, rwd)
  ax.set_xlabel('epochs')
  ax.set_ylabel('total reward')
  return fig

#
# primarily for debugging
#

def visualise_batch_from_experience(id, config, experience, observation_size, out_dir):
  batch = experience.sample(min(config.batch_size, experience.idx - 1), min(config.chunk_size, experience.idx - 1))
  save_as_gif(batch[0].reshape(-1, *observation_size).cpu().numpy(), os.path.join(out_dir, f'experience_{id}.gif'))

def visualise_trajectory(id, trajectory: List, out_dir):
  save_as_gif([x[0].cpu().numpy() for x in trajectory], os.path.join(out_dir, f'trajectory_{id}.gif'))
