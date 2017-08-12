#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

import matplotlib.pyplot as plt


class Sampler(object):
  def __init__(self, filename, patch_size=5):
    self.patch_size = patch_size
    self.img = imread(filename)/255.-0.5
    self.sy, self.sx, channels = self.img.shape
    if channels==4:
      self.img = self.img[:,:,:3]

  def sample(self, num):
    """ Samples from image
        returns position, data
    """

    if isinstance(num, int):
      batch_size = num
      res_pos = np.random.rand(batch_size,2)
      res_pos = np.asarray(res_pos *
                 [self.sy-self.patch_size,
                  self.sy-self.patch_size], dtype="int")
    else:
      res_pos = num
      batch_size = res_pos.shape[0]

    res_data = np.zeros((batch_size, self.patch_size*self.patch_size, 3))
    for i, (y,x) in enumerate(res_pos):
      res_data[i] = self.img[y:y+self.patch_size, x:x+self.patch_size]\
          .reshape(self.patch_size*self.patch_size,3)

    return res_pos, res_data


if __name__ == "__main__":
  s = Sampler("images/cat.png",patch_size=5)

  pos, data = s.sample(100)

  #pos, data = s.sample(np.asarray([[2,2],[10,10],[100,6],[99,45]]))

  #lin = np.arange(0,s.sx,s.patch_size)
  #pos = np.vstack((
  #  np.tile(lin, s.sx/s.patch_size),
  #  np.repeat(lin, s.sx/s.patch_size))).T
  #pos, data = s.sample(pos)

  result = np.zeros_like(s.img)
  for (y,x), d in zip(pos,data):
    result[y:y+s.patch_size,x:x+s.patch_size] = \
        d.reshape(s.patch_size, s.patch_size, 3)
    plt.imshow(result+0.5)

  plt.show()

