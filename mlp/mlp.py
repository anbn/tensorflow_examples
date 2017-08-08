#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from sampler import Sampler

import matplotlib.pyplot as plt

class MLP(object):
    def __init__(self):
        self.batch_size = 10
        self.state_size = 20
        self.num_in, self.num_out = 2,3
        self.learning_rate = 0.001

        self.x = tf.placeholder(tf.float32,
                [None, self.num_in],
                name='in_placeholder')
        self.y = tf.placeholder(tf.float32,
                [None, self.num_out],
                name='out_placeholder')

        weights = {
            'w1': tf.Variable(tf.random_normal([self.num_in, self.state_size])),
            'w2': tf.Variable(tf.random_normal([self.state_size, self.state_size], 0, np.sqrt(1./20.))),
            'w3': tf.Variable(tf.random_normal([self.state_size, self.state_size], 0, np.sqrt(1./20.))),
            'w4': tf.Variable(tf.random_normal([self.state_size, self.state_size], 0, np.sqrt(1./20.))),
            'w5': tf.Variable(tf.random_normal([self.state_size, self.state_size], 0, np.sqrt(1./20.))),
            'w6': tf.Variable(tf.random_normal([self.state_size, self.state_size], 0, np.sqrt(1./20.))),
            'w7': tf.Variable(tf.random_normal([self.state_size, self.state_size], 0, np.sqrt(1./20.))),
            'w8': tf.Variable(tf.random_normal([self.state_size, self.num_out],    0, np.sqrt(1./20.))),
        }
        biases = {
            'b1': tf.Variable(tf.ones([self.state_size])*0.1),
            'b2': tf.Variable(tf.ones([self.state_size])*0.1),
            'b3': tf.Variable(tf.ones([self.state_size])*0.1),
            'b4': tf.Variable(tf.ones([self.state_size])*0.1),
            'b5': tf.Variable(tf.ones([self.state_size])*0.1),
            'b6': tf.Variable(tf.ones([self.state_size])*0.1),
            'b7': tf.Variable(tf.ones([self.state_size])*0.1),
            'b8': tf.Variable(tf.ones([self.num_out])*0.1),
        }

        self.layer1 = tf.nn.relu(tf.add(tf.matmul(self.x,      weights["w1"]), biases["b1"]))
        self.layer2 = tf.nn.relu(tf.add(tf.matmul(self.layer1, weights["w2"]), biases["b2"]))
        self.layer3 = tf.nn.relu(tf.add(tf.matmul(self.layer2, weights["w3"]), biases["b3"]))
        self.layer4 = tf.nn.relu(tf.add(tf.matmul(self.layer3, weights["w4"]), biases["b4"]))
        self.layer5 = tf.nn.relu(tf.add(tf.matmul(self.layer4, weights["w5"]), biases["b5"]))
        self.layer6 = tf.nn.relu(tf.add(tf.matmul(self.layer5, weights["w6"]), biases["b6"]))
        self.layer7 = tf.nn.relu(tf.add(tf.matmul(self.layer6, weights["w7"]), biases["b7"]))
        self.output = tf.add(tf.matmul(self.layer5, weights["w8"]), biases["b8"])

        self.total_loss = tf.reduce_mean(tf.square(self.y-self.output))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)\
                .minimize(self.total_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def train(self, sampler):
        self.num_steps = self.batch_size
        sampler.pos, i = 0, 0
        result = np.zeros((sampler.sy, sampler.sx, 3))
        for i in xrange(25000*10):
            sampler.pos = np.random.randint(sampler.sx*sampler.sy-1)
            res_xr, res_x, res_y = sampler.sample(self.batch_size, self.num_steps)
            res_xr = res_xr.reshape(-1,2)
            res_x = res_x.reshape(-1,2)
            res_y = res_y.reshape(-1,3)
            loss, _ = self.sess.run([self.total_loss,self.optimizer],
                    feed_dict={self.x:res_x, self.y:res_y})

            if i%100==0:
                print "%d: %f" % (i, loss)

            if i%10000==0:
                self.sample(sampler)

            if i%100000==0:
                self.learning_rate /= 2



    def sample(self, sampler):
        result = np.zeros((sampler.sy, sampler.sx, 3))
        res_x = np.zeros((self.batch_size, self.num_steps, 2))
        sampler.pos, j = 0, 0
        while j<sampler.sx*sampler.sy:
            res_xr, res_x, res_t = sampler.sample(self.batch_size, self.num_steps)
            res_xr = res_xr.reshape(-1,2)
            res_x = res_x.reshape(-1,2)
            res_t = res_t.reshape(-1,3)
            res_y, loss = self.sess.run([self.output, self.total_loss],
                    feed_dict={self.x:res_x, self.y:res_t})

            for n,o in zip(res_xr, res_y):
                p1, p2 = n
                result[p2,p1] = o
            j+=self.batch_size*self.num_steps

        img = result.reshape(sampler.sy,sampler.sx,3)+0.5
        img = np.clip(img, 0., 1.)
        plt.figure(0), plt.imshow(img)
        plt.show()


def main():
    mlp = MLP()
    sampler = Sampler("cat.png")
    sampler.test()
    mlp.train(sampler)


if __name__ == "__main__":
    main()

