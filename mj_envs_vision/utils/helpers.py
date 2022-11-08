import os
import numpy as np
from PIL import Image


def to_image_frame(obs: np.ndarray):
  """ converts image observation with pels in [-0.5, 0.5] to image with pels in [0, 255] """
  return (255 * (obs.transpose((1, 2, 0)) + 0.5)).astype('uint8')

def to_input_obs(frame: np.ndarray):
  """ converts image with pels in [0, 255] to image observation with pels in [-0.5, 0.5] """
  return (frame.transpose((2, 0, 1)) / 255 - 0.5).astype('float')

def visualise_batch_from_experience(id, config, experience, observation_size, out_dir):
  # primarily for debugging
  pils = list()
  batch = experience.sample(min(config.batch_size, experience.idx - 1), min(config.chunk_size, experience.idx - 1))
  for obs in batch[0].reshape(-1, *observation_size).cpu():
    pils.append(Image.fromarray(to_image_frame(obs.numpy())))

  pils[0].save(os.path.join(out_dir, f'experience_{id}.gif'),
               append_images=pils, save_all=True, optimize=False, loop=True, duration=len(pils) / 10)