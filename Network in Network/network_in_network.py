#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
Network in Network(NIN) model implementation example using TensorFlow library.

NIN Paper: Network In Network(https://arxiv.org/abs/1312.4400)

@author: MarkLiu
@time  : 17-3-12 上午9:36
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class NetworkInNetwork(object):
    """
    Network in Network(NIN) model
    """

    def __init__(self, input_height, input_width, input_channels, num_classes, activation=tf.nn.relu):
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.activation = activation

    def create_weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)

    def create_bias_variable(self, shape, name):
        initial = tf.constant(np.random.rand(), shape=shape)
        return tf.Variable(initial, name=name)

    def mlp_conv(self, x, filter_height, filter_width, num_filters, output_height, output_width, micro_layer_size,
                 name):
        """
        multi layer perceptron convolution.

        :param num_filters: number of micro_net filter
        :param micro_layer_size: [hidden_layer]
        :param name:
        :return:
        """

        # get local patch
        inputs = x
        inputs = tf.reshape(inputs, [-1, filter_height * filter_width])
        filter_i_outs = []
        for filter_i in range(0, num_filters):
            for hidden_i, hidden_layer_size in enumerate(micro_layer_size):
                with tf.variable_scope(name + '_hidden_' + str(hidden_i)):
                    # create mlp weights and bias
                    mlp_weights = self.create_weight_variable([filter_height * filter_width, hidden_layer_size],
                                                              name='mlp_weights')
                    mlp_bias = self.create_bias_variable([hidden_layer_size], name='mlp_bias')
                    # adding activation function to make nonlinear
                    hidden_i_out = self.activation(tf.add(tf.matmul(inputs, mlp_weights), mlp_bias))
                    inputs = hidden_i_out
            # every filtering output
            filter_i_outs.append(inputs)

        # concate output
        return tf.concat(filter_i_outs, axis=3, name=name + '_mlp_out')

    def golbal_average_pooling(self, x):
        """
        golbal average pooling
        :param x: [batch, height, width, channels]
        """
        ksize_height = x.get_shape()[0]
        ksize_width = x.get_shape()[0]
        return tf.nn.avg_pool(x, ksize=[1, ksize_height, ksize_width, 1], strides=[1, 1, 1, 1], padding='SAME')

    def build_model(self):
        # input features
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width, self.input_channels],
                                name='input_layer')
        self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='output_layer')

        # learning_rate placeholder
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # dropout layer: keep probability, vgg default value:0.5
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.nin_lay_1 = self.mlp_conv(self.x, filter_height=11, filter_width=11, num_filters=96, output_height=55,
                                       output_width=55, micro_layer_size=[96, 96], name='nin_lay_1')

        self.nin_lay_2 = self.mlp_conv(self.nin_lay_1, 5, 5, 256, 27, 27, [256, 256], name='nin_lay_2')
        self.nin_lay_3 = self.mlp_conv(self.nin_lay_2, 3, 3, 384, 13, 13, [384, 384], name='nin_lay_3')
        self.nin_lay_4 = self.mlp_conv(self.nin_lay_3, 3, 3, self.num_classes, 6, 6, [1024, 1024], name='nin_lay_4')

        # golbal average pooling
        self.digits = self.golbal_average_pooling(self.nin_lay_4)
        # softmax
        self.read_out_logits = tf.nn.softmax(self.digits)

