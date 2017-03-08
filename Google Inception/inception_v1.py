#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
Google Inception V1 model implementation example using TensorFlow library.

Inception V1 Paper: Going Deeper with Convolutions(https://arxiv.org/abs/1409.4842)

Mnist Dataset: http://yann.lecun.com/exdb/mnist/

Pre-trained model layers infor：
['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']

@author: MarkLiu
@time  : 17-3-8 下午1:35
"""

import tensorflow as tf
import numpy as np


class GoogleInceptionV1(object):
    """
    Google Inception V1 model
    """

    def __init__(self, num_classes, skip_layer, pre_trained_model='DEFAULT'):
        self.num_classes = num_classes
        # 指定跳过加载 pre-trained 层
        self.skip_layer = skip_layer
        if pre_trained_model == 'DEFAULT':
            self.pre_trained_model = 'vgg16.npy'
        else:
            self.pre_trained_model = pre_trained_model

    def conv2d(self, x, filter_height, filter_width, num_filters, stride_y, stride_x,
               name, padding='SAME'):
        """
        卷积层
        :param x: [batch, in_height, in_width, in_channels]
        :param num_filters: filters 的数目,[filter_height, filter_width, in_channels, out_channels]
        :param stride_y, stride_x: 每一维度滑动的步长,strides[0]=strides[3]=1
        """
        # Get number of input channels
        input_channels = int(x.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('filter',
                                      shape=[filter_height, filter_width, input_channels, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])
            conv = tf.nn.conv2d(x, weights,
                                strides=[1, stride_y, stride_x, 1],
                                padding=padding)
            conv_bias = tf.nn.bias_add(conv, biases)

            # Apply activation function
            relu = tf.nn.relu(conv_bias, name=scope.name)

            return relu

    def max_pool(self, x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        """
        pooling 层, 当 stride = ksize， padding='SAME' 时输出 tensor 大小减半
        :param x: [batch, height, width, channels]
        :param filter_height, filter_width: [1, height, width, 1]
        :param stride_y, stride_x: [1, stride, stride, 1]
        """
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)

    def fully_connected(self, x, num_out, name, activation=True):
        """
        全连接层， n_units 指定输出神经元的数目
        """
        with tf.variable_scope(name) as scope:
            shape = x.get_shape().as_list()
            num_in = 1
            for d in shape[1:]:
                num_in *= d
            x = tf.reshape(x, [-1, num_in])

            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)

            fc = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
            if activation:
                fc = tf.nn.relu(fc)

            return fc

    def dropout(self, x, keep_prob):
        """
        dropout layer
        """
        return tf.nn.dropout(x, keep_prob)

    def build_model(self):
        """
        build basic inception v1 model
        """
        # input features
        self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input_layer')
        self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='output_layer')

        # learning_rate placeholder
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # dropout layer: keep probability, vgg default value:0.5
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Conv2d_1a_7x7
