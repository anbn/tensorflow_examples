#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from sampler import Sampler

import matplotlib.pyplot as plt

class MLP(object):
    def __init__(self, sampler):
        self.sampler = sampler
        self.batch_size = 128
        self.state_size = 16
        self.num_in, self.num_out = 100*3, 100*3
        self.learning_rate = 0.001
        topology = { 0:100*3, 1:128, 2:64, 3:32, 4:16}
        max_steps = 10
        for i in range(len(topology), max_steps):
          topology[i] = 16

        self.i = 0
        self.variables = {}
        for d in range(max_steps-1,max_steps):
          print "-- %d ----------" % d
          self.variables, new_variables = \
                self.build_model(topology, depth=d, variables=self.variables)
          new_vars = [v for n in new_variables for v in tf.trainable_variables() if n in v.name]
          print new_variables

          self.total_loss = tf.reduce_mean(tf.square(self.y-self.output))
          #self.total_loss = self.total_loss + regularizer
          self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)\
                  .minimize(self.total_loss, var_list=tf.trainable_variables())

          self.sess = tf.Session()
          self.sess.run(tf.global_variables_initializer())
          self.train()
        self.sample()


    def build_model(self, topology, depth, variables):
      self.x = tf.placeholder(tf.float32, [None, topology[0]],
              name='in_placeholder')
      self.y = tf.placeholder(tf.float32, [None, topology[0]],
              name='out_placeholder')

      previous = self.x
      rng = range(min(len(topology), depth))
      new_variables = []
      #regularizer = 0

      for n in rng:
        wname, bname = "w_in_%d" % n, "b_in_%d" % n
        print "%d %s:  %d->%d" % (n, wname, topology[n], topology[n+1])
        if not(wname in variables and bname in variables):
          variables[wname] = tf.Variable(tf.random_normal([topology[n], topology[n+1]],
                    0, np.sqrt(1./topology[n])), name=wname)
          variables[bname] = tf.Variable(tf.ones([topology[n+1]])*0.1, name=bname),
          new_variables.append(wname)
          new_variables.append(bname)
        previous = tf.add(tf.matmul(previous, variables[wname]), variables[bname])
        previous = tf.nn.relu(previous)
        #regularizer = regularizer + tf.nn.l2_loss(previous)

      for n in reversed(rng):
        wname, bname = "w_out_%d" % n, "b_out_%d" % n
        print "%d %s: %d->%d" % (n, wname, topology[n+1], topology[n])
        if not(wname in variables and bname in variables):
          variables[wname] = tf.Variable(tf.random_normal([topology[n+1], topology[n]],
                    0, np.sqrt(1./topology[n+1])), name=wname)
          variables[bname] = tf.Variable(tf.ones([topology[n]])*0.1, name=bname),
          new_variables.append(wname)
          new_variables.append(bname)
        previous = tf.add(tf.matmul(previous, variables[wname]), variables[bname])
        if n > 0: previous = tf.nn.relu(previous)
        #regularizer = regularizer + tf.nn.l2_loss(previous)

      self.output = previous
      return variables, new_variables


    def train(self):
      loss = 0, 1.
      losses = []
      while loss > 0.001:
        pos, data = self.sampler.sample(self.batch_size)
        data = data.reshape((-1,self.num_in))
        loss, _ = self.sess.run([self.total_loss,self.optimizer],
                  feed_dict={self.x:data, self.y:data})

        losses.append(loss)
        if self.i%1000==0:
          print "%d: %f" % (self.i, loss)

        self.i+=1

      print "%d: %f" % (self.i, loss)


    def sample(self):
        plt.figure("sample")
        lin = np.arange(0,self.sampler.sx,self.sampler.patch_size)
        pos = np.vstack((
          np.tile(lin, self.sampler.sx/self.sampler.patch_size),
          np.repeat(lin, self.sampler.sx/self.sampler.patch_size))).T
        pos, data = self.sampler.sample(pos)
        data = data.reshape((-1,self.num_in))

        res_y, loss = self.sess.run([self.output, self.total_loss],
                feed_dict={self.x:data, self.y:data})
        result = np.zeros_like(self.sampler.img)

        for (y,x), d in zip(pos,res_y):
          result[y:y+self.sampler.patch_size,x:x+self.sampler.patch_size] = \
              d.reshape(self.sampler.patch_size, self.sampler.patch_size, 3)
          plt.imshow(np.clip(result+0.5,0,1))

        plt.show()


def main():
    sampler = Sampler("images/cat.png", patch_size=10)
    mlp = MLP(sampler)
    #mlp.sample(sampler)


if __name__ == "__main__":
    main()

