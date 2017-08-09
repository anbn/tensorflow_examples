#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from sampler import Sampler

import matplotlib.pyplot as plt

class LSTM(object):
    def __init__(self):
        self.batch_size = 10
        self.state_size = 25
        self.num_steps = 50
        self.num_in, self.num_out = 3,3
        self.learning_rate = 0.001

        self.cut = 30


        self.x = tf.placeholder(tf.float32,
                [self.batch_size, self.num_steps, self.num_in],
                name='in_placeholder')
        self.y = tf.placeholder(tf.float32,
                [self.batch_size, self.num_steps, self.num_out],
                name='out_placeholder')

        with tf.variable_scope('input'):
            Win = tf.get_variable('W', [self.num_in, self.state_size])
            bin = tf.get_variable('b', [self.state_size],
                    initializer=tf.constant_initializer(0.0))

        self.init_state = tf.contrib.rnn.LSTMStateTuple(
                            tf.zeros([self.batch_size, self.state_size]),
                            tf.zeros([self.batch_size, self.state_size]))

        rnn_inputs = tf.reshape(
            tf.matmul(tf.reshape(self.x, [-1, self.num_in]), Win) + bin,
            [self.batch_size, self.num_steps, self.state_size])

        cell1 = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple=True)
        cell2 = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple=True)
        cell3 = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple=True)

        cell = rnn.MultiRNNCell([cell1,cell2,cell3])

        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs,
                initial_state=tuple([self.init_state, self.init_state, self.init_state]))

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self.state_size, self.num_out])
            b = tf.get_variable('b', [self.num_out],
                    initializer=tf.constant_initializer(0.0))

        self.output = tf.reshape(
            tf.matmul(tf.reshape(rnn_outputs, [-1, self.state_size]), W) + b,
            [self.batch_size, self.num_steps, self.num_out])

        self.total_loss = tf.reduce_mean(tf.square(self.y-self.output))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)\
                .minimize(self.total_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def train(self, sampler, orig):
        sampler.pos, i = 0, 0
        result = np.zeros((sampler.sy, sampler.sx, 3))
        for i in xrange(25000*10):
            sampler.pos = np.random.randint(sampler.sx*sampler.sy-1)
            res_xr, res_xs, res_y = sampler.sample(self.batch_size, self.num_steps)
            #res_xr = res_xr.reshape(-1,2)
            #res_xs = res_xs.reshape(-1,2)
            #res_y = res_y.reshape(-1,3)

            res_x = np.copy(res_y)
            res_x[:,self.cut:,:] = 0

            loss, _ = self.sess.run([self.total_loss,self.optimizer],
                    feed_dict={self.x:res_x, self.y:res_y})

            if i%100==0:
                print "%d: %f" % (i, loss)

            if i%10000==0:
                self.sample(sampler, orig)


    def sample(self, sampler, orig):
        result = np.zeros((sampler.sy*sampler.sx, 3))
        sampler.pos, j = 0, 0
        while j+self.batch_size*self.num_steps < sampler.sy*sampler.sx:
            res_xr, res_x, res_t = sampler.sample(self.batch_size, self.num_steps)

            res_x = np.copy(res_t)
            res_x[:,self.cut:,:] = 0

            res_y = self.sess.run(self.output,
                    feed_dict={self.x:res_x, self.y:res_t})

            for batch_res_xr, batch_res_y in zip(res_xr, res_y):
                result[j:j+self.num_steps] = batch_res_y
                j+=self.num_steps

        img = result.reshape(sampler.sy,sampler.sx,3)+0.5
        img = np.clip(img, 0., 1.)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.set_title("image"), ax1.imshow(img)
        ax2.set_title("orig"), ax2.imshow(orig+0.5)
        ax3.set_title("diff"), ax3.imshow(np.sum(orig-img+0.5, axis=2))
        for i in range(0, sampler.sx, self.num_steps):
            for a in [ax1, ax2, ax3]:
                a.axvline(i+self.cut)
        plt.show()


def main():
    lstm = LSTM()
    sampler = Sampler("images/bloemaert2.tif")
    orig = sampler.test()
    lstm.train(sampler, orig)


if __name__ == "__main__":
    main()

