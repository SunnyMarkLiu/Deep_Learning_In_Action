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
import tensorflow.contrib.slim as slim


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

    def mlp_conv(self, x, kernel_size, stride, num_filters, micro_layer_size, name,
                 final_pooling_kernel_size=None, final_pooling_stride=None, pooling=True):
        """
        multi layer perceptron convolution.

        :param num_filters: number of micro_net filter
        :param micro_layer_size: [hidden_layer]
        :return:
        """
        with tf.variable_scope(name, values=[x]):
            # first convolution
            net = slim.conv2d(inputs=x, num_outputs=num_filters, kernel_size=[kernel_size, kernel_size],
                              stride=stride, scope='first_conv')
            # cccp layer
            with slim.arg_scope([slim.conv2d], kernel_size=[1, 1], stride=1,
                                padding='SAME', activation_fn=tf.nn.relu):
                for hidden_i, hidden_size in enumerate(micro_layer_size):
                    net = slim.conv2d(net, hidden_size, scope='hidden_' + str(hidden_i))
            # final max-pooling
            if pooling:
                net = slim.max_pool2d(net, kernel_size=final_pooling_kernel_size,
                                      stride=final_pooling_stride, name='max_pooling')
        return net

    def golbal_average_pooling(self, x):
        """
        golbal average pooling
        :param x: [batch, height, width, channels]
        """
        ksize_height = x.get_shape()[0]
        ksize_width = x.get_shape()[0]
        return tf.nn.avg_pool(x, ksize=[1, ksize_height, ksize_width, 1], strides=[1, 1, 1, 1], padding='SAME')

    def build_nin_model(self):
        # input features
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width, self.input_channels],
                                name='input_layer')
        self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='output_layer')

        # learning_rate placeholder
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # dropout layer: keep probability, vgg default value:0.5
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.nin_lay_1 = self.mlp_conv(self.x, kernel_size=11, stride=4, num_filters=96,
                                       micro_layer_size=[96, 96], final_pooling_kernel_size=3,
                                       final_pooling_stride=2, name='nin_lay_1')

        self.nin_lay_2 = self.mlp_conv(self.nin_lay_1, kernel_size=5, stride=1, num_filters=256,
                                       micro_layer_size=[256, 256], final_pooling_kernel_size=3,
                                       final_pooling_stride=2, name='nin_lay_2')

        self.nin_lay_3 = self.mlp_conv(self.nin_lay_2, kernel_size=3, stride=1, num_filters=384,
                                       micro_layer_size=[384, 384], final_pooling_kernel_size=3,
                                       final_pooling_stride=2, name='nin_lay_3')

        # dropout
        self.dropout = slim.dropout(self.nin_lay_3, keep_prob=self.dropout)

        self.nin_lay_4 = self.mlp_conv(self.dropout, kernel_size=3, stride=1, num_filters=1024,
                                       micro_layer_size=[1024, self.num_classes], pooling=False,
                                       name='nin_lay_4')
        # golbal average pooling
        self.digits = self.golbal_average_pooling(self.nin_lay_4)
        # softmax
        self.read_out_logits = tf.nn.softmax(self.digits)

    def init_train_test_op(self):
        # some loss functions and all -> total loss
        self.loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                                                    logits=self.read_out_logits))
        # training op
        self.training_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_function)
        self.predict_op = tf.arg_max(self.read_out_logits, 1)
        # predict
        predict_matches = tf.equal(tf.arg_max(self.y, dimension=1),
                                   tf.arg_max(self.read_out_logits, 1))
        # accuracy metric
        self.accuracy = tf.reduce_mean(tf.cast(predict_matches, tf.float32))

    def train(self, x, y, learning_rate, keep_prob=0.5):
        feed_dict = {
            self.x: x,
            self.y: y,
            self.keep_prob: keep_prob,
            self.learning_rate: learning_rate
        }
        _, train_loss = self.sess.run([self.training_op, self.loss_function], feed_dict=feed_dict)
        train_accuracy = self.get_accuracy(x, y)
        return train_loss, train_accuracy

    def classify(self, features_x):
        feed_dict = {self.x: features_x, self.keep_prob: 1.0}
        predict_y, prob = self.sess.run([self.predict_op, self.read_out_logits], feed_dict=feed_dict)
        return predict_y, prob

    def get_accuracy(self, x, y):
        feed_dict = {
            self.x: x,
            self.y: y,
            self.keep_prob: 1.0
        }
        accuracy = self.sess.run(self.accuracy, feed_dict=feed_dict)
        return accuracy

    def init(self):
        self.build_nin_model()
        self.init_train_test_op()
        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
