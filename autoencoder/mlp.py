#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from sampler import Sampler

import matplotlib.pyplot as plt

class MLP(object):
    def __init__(self):
        self.batch_size = 128
        self.state_size = 16
        self.num_in, self.num_out = 100*3, 100*3
        self.learning_rate = 0.001

        self.x = tf.placeholder(tf.float32,
                [None, self.num_in],
                name='in_placeholder')
        self.y = tf.placeholder(tf.float32,
                [None, self.num_out],
                name='out_placeholder')

        weights = {
            'w1': tf.Variable(tf.random_normal([self.num_in, self.state_size],  0, np.sqrt(1./20.))),
            'w2': tf.Variable(tf.random_normal([self.state_size, self.num_out], 0, np.sqrt(1./20.))),
        }
        biases = {
            'b1': tf.Variable(tf.ones([self.state_size])*0.1),
            'b2': tf.Variable(tf.ones([self.num_out])*0.1),
        }

        self.layer1 = tf.nn.relu(tf.add(tf.matmul(self.x,      weights["w1"]), biases["b1"]))
        #self.layer2 = tf.nn.relu(tf.add(tf.matmul(self.layer1, weights["w2"]), biases["b2"]))
        self.output = tf.add(tf.matmul(self.layer1, weights["w2"]), biases["b2"])

        self.total_loss = tf.reduce_mean(tf.square(self.y-self.output))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)\
                .minimize(self.total_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def train(self, s):
        for i in xrange(25000*10):
          pos, data = s.sample(self.batch_size)
          data = data.reshape((-1,self.num_in))
          loss, _ = self.sess.run([self.total_loss,self.optimizer],
                    feed_dict={self.x:data, self.y:data})

          if 0<i and i%100==0:
            print "%d: %f" % (i, loss)

          if i%10000==0:
              self.sample(s)


    def sample(self, s):
        lin = np.arange(0,s.sx,s.patch_size)
        pos = np.vstack((
          np.tile(lin, s.sx/s.patch_size),
          np.repeat(lin, s.sx/s.patch_size))).T
        pos, data = s.sample(pos)
        data = data.reshape((-1,self.num_in))

        res_y, loss = self.sess.run([self.output, self.total_loss],
                feed_dict={self.x:data, self.y:data})
        result = np.zeros_like(s.img)

        for (y,x), d in zip(pos,res_y):
          result[y:y+s.patch_size,x:x+s.patch_size] = \
              d.reshape(s.patch_size, s.patch_size, 3)
          plt.imshow(np.clip(result+0.5,0,1))

        plt.show()


def main():
    sampler = Sampler("images/cat.png", patch_size=10)
    mlp = MLP()
    #mlp.sample(sampler)
    mlp.train(sampler)


if __name__ == "__main__":
    main()

