#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

import matplotlib.pyplot as plt


class Sampler(object):
    def __init__(self, filename):
        img = imread(filename)/255.-0.5
        self.sy, self.sx, channels = img.shape
        if channels==4:
            img = img[:,:,:3]
        self.flat_image = img.reshape((self.sx*self.sy,3))
        self.pos = 0


    def sample(self, batch_size, steps):
        res_xr = np.zeros((batch_size, steps, 2))
        res_xs = np.zeros((batch_size, steps, 2))
        res_y = np.zeros((batch_size, steps, 3))

        for i in xrange(batch_size):
            r = np.arange(self.pos, self.pos+steps)
            
            res_xr[i,:,0] = (r % self.sx)
            res_xr[i,:,1] = (r / self.sx)

            res_xs[i,:,0] = res_xr[i,:,0] / float(self.sx)-0.5
            res_xs[i,:,1] = res_xr[i,:,1] / float(self.sy)-0.5

            if self.pos+steps < self.sy*self.sx:
                res_y[i:i+steps,:,:] = self.flat_image[self.pos:self.pos+steps,:]
                self.pos += steps
            else:
                self.pos = 0

        return res_xr.astype("int"), res_xs, res_y


    def test(self):
        batches = 10
        steps = 12
        
        i=0
        orig = np.zeros((self.sy, self.sx, 3))
        result = np.zeros((self.sy*self.sx, 3))
        resultx = np.zeros((self.sy*self.sx))
        resulty = np.zeros((self.sy*self.sx))

        while i+batches*steps < self.sy*self.sx:
            res_xr, _, res_y = self.sample(batches,steps)

            for batch_res_xr, batch_res_y in zip(res_xr, res_y):
                for p,v in zip(batch_res_xr, batch_res_y):
                    orig[p[1],p[0]] = v
                result[i:i+steps] = batch_res_y
                resultx[i:i+steps] = batch_res_xr[:,0]
                resulty[i:i+steps] = batch_res_xr[:,1]
                i += steps

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.set_title("orig"), ax1.imshow(orig+0.5)
        ax2.set_title("reshaped"), ax2.imshow(result.reshape(self.sy,self.sx,3)+0.5)
        ax3.set_title("x"), ax3.imshow(resultx.reshape(self.sy,self.sx))
        ax4.set_title("y"), ax4.imshow(resulty.reshape(self.sy,self.sx))
        plt.show()

        return orig

