#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":

    features = 2
    classes = 3

    g1 = np.random.normal(0.0, 0.4, [1000,2]) + (1, 2)
    g2 = np.random.normal(0.0, 0.3, [1000,2]) + (4, 2)
    g3 = np.random.normal(0.0, 0.8, [1000,2]) + (5, 4)

    plt.figure("real")
    plt.scatter(g1[:,0], g1[:,1], color="r")
    plt.scatter(g2[:,0], g2[:,1], color="g")
    plt.scatter(g3[:,0], g3[:,1], color="b")

    W = tf.Variable(tf.zeros([features, classes]))
    b = tf.Variable(tf.zeros([classes]))
    x = tf.placeholder(tf.float32, [None, features])
    y_ = tf.placeholder(tf.float32, [None, classes])

    y = tf.matmul(x,W) + b
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = tf.reduce_sum(softmax)
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    x_train = np.vstack((g1,g2,g3))
    y_train = np.zeros((3000,classes))
    y_train[   0:1000,0]=1
    y_train[1000:2000,1]=1
    y_train[2000:3000,2]=1

    with tf.Session() as sess:
        sess.run(init)

        for i in range(1000):
            sess.run(train, feed_dict={x: x_train, y_: y_train})
            test_loss, test_y = sess.run([loss, y], feed_dict={x: x_train, y_: y_train})

        c_dict = {0:"r", 1:"g", 2:"b"}
        colors = [c_dict[np.argmax(i)] for i in test_y]

        plt.figure("test"),plt.scatter(x_train[:,0], x_train[:,1], color=colors)
        plt.show()
