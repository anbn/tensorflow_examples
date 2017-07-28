#!/usr/bin/env python
#-*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import tensorflow as tf

def linear_regression():
    xx = np.random.rand(1000,3)
    yy = xx[:,0]*8.7 + xx[:,1]*2.4 + xx[:,2]*2 + 6

    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_ = tf.placeholder(tf.float32, shape=[None])

    W = tf.Variable(tf.random_uniform([3,1]),name="W")
    b = tf.Variable(tf.zeros([1,1]),name="b")
    y = tf.matmul(x, W) + b

    cost_function = tf.reduce_mean(tf.square(tf.squeeze(y) - y_))
    train_function = tf.train.AdamOptimizer(0.01).minimize(cost_function)

    tf.summary.scalar("loss", cost_function)
    merged = tf.summary.merge_all()

    config = tf.ConfigProto(device_count = {'GPU': 0})
    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(
                'logs/'+datetime.now().strftime("%Y%m%d-%H%M%S"), sess.graph)

        sess.run(tf.initialize_all_variables())
        for i in range(10000):
            summary_str, train_loss = sess.run([merged,train_function], feed_dict={x:xx, y_:yy})
            summary_writer.add_summary(summary_str, i)

            if i % 100 == 0:
                print(sess.run([cost_function,W,b], feed_dict={x:xx, y_:yy}))


if __name__ == "__main__":
    linear_regression()
