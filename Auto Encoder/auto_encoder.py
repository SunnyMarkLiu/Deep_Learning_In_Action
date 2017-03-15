#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
Autoencoders implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

@author: MarkLiu
@time  : 17-3-15 下午5:38
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Autoencoders(object):
    """
    Autoencoders model
    """

    def __init__(self, input_layer_size, activation_fun):
        self.input_layer_size = input_layer_size
        self.activation_fun = activation_fun

    def build_model(self):
        # input features
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_layer_size], name='input_layer')
        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        hidden_layer_1_size = 100
        hidden_layer_2_size = 100
        with tf.variable_scope('hidden_layer_1') as scope:
            weights = tf.get_variable('weights', shape=[self.input_layer_size, hidden_layer_1_size], trainable=True)
            biases = tf.get_variable('biases', [hidden_layer_1_size], trainable=True)
            fc = tf.nn.xw_plus_b(self.x, weights, biases, name=scope.name)
            self.hidden_layer_1 = self.activation_fun(fc)

        with tf.variable_scope('hidden_layer_2') as scope:
            weights = tf.get_variable('weights', shape=[hidden_layer_1_size, hidden_layer_2_size], trainable=True)
            biases = tf.get_variable('biases', [hidden_layer_2_size], trainable=True)
            fc = tf.nn.xw_plus_b(self.hidden_layer_1, weights, biases, name=scope.name)
            self.hidden_layer_2 = self.activation_fun(fc)

        # reconstruction
        with tf.variable_scope('reconstruction') as scope:
            weights = tf.get_variable('weights', shape=[hidden_layer_2_size, self.input_layer_size],
                                      trainable=True)
            biases = tf.get_variable('biases', [self.input_layer_size], trainable=True)
            fc = tf.nn.xw_plus_b(self.hidden_layer_2, weights, biases, name=scope.name)
            self.reconstruction = self.activation_fun(fc)

    def init_train_test_op(self):
        # loss function
        self.loss_function = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        # training op
        self.training_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_function)

    def init(self):
        self.build_model()
        self.init_train_test_op()
        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def train(self, x, learning_rate, keep_prob=0.5):
        cost, _ = self.sess.run((self.loss_function, self.training_op), feed_dict={self.x: x,
                                                                                   self.learning_rate: learning_rate,
                                                                                   self.keep_prob: keep_prob})
        return cost

    def encode(self, x):
        """
        encode input
        """
        return self.sess.run(self.hidden_layer_2, feed_dict={self.x: x,
                                                             self.keep_prob: 1})

    def decode(self, x):
        """
        decode hidden layer output
        """
        return self.sess.run(self.reconstruction, feed_dict={self.x: x,
                                                             self.keep_prob: 1})

    def calc_total_cost(self, x):
        return self.sess.run(self.loss_function, feed_dict={self.x: x,
                                                            self.keep_prob: 1})

    def get_weights(self, scope_name):
        with tf.variable_scope(scope_name, reuse=True):
            weights = tf.get_variable('weights')
        return self.sess.run(weights)

    def get_biases(self, scope_name):
        with tf.variable_scope(scope_name, reuse=True):
            weights = tf.get_variable('biases')
        return self.sess.run(weights)
