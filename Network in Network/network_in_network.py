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

    def mlp_conv(self, x, kernel_size, stride, num_filters, micro_layer_size, name):
        """
        multi layer perceptron convolution.

        :param num_filters: number of micro_net filter
        :param micro_layer_size: [hidden_layer]
        :return:
        """
        with tf.variable_scope(name, values=[x]):
            # first convolution
            net = slim.conv2d(inputs=x, num_outputs=num_filters, kernel_size=[kernel_size, kernel_size],
                              stride=stride, scope='first_conv', padding='SAME')
            # cccp layer
            with slim.arg_scope([slim.conv2d], kernel_size=[1, 1], stride=1,
                                padding='VALID', activation_fn=tf.nn.relu):
                for hidden_i, hidden_size in enumerate(micro_layer_size):
                    net = slim.conv2d(net, hidden_size, scope='hidden_' + str(hidden_i))
        return net

    def golbal_average_pooling(self, x):
        """
        golbal average pooling
        :param x: [batch, height, width, channels]
        """
        shapes = x.get_shape().as_list()
        kernel_height = shapes[1]
        kernel_width = shapes[2]
        return slim.avg_pool2d(x, kernel_size=[kernel_height, kernel_width], stride=1, padding='VALID',
                               scope='golbal_average_pooling')

    def build_nin_model(self):
        # input features
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width, self.input_channels],
                                name='input_layer')
        self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='output_layer')

        # learning_rate placeholder
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # dropout layer: keep probability, vgg default value:0.5
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        print('x:' + str(self.x.get_shape().as_list()))

        self.nin_lay_1 = self.mlp_conv(self.x, kernel_size=11, stride=2, num_filters=96,
                                       micro_layer_size=[96, 96], name='nin_lay_1')
        # add dropout
        dropout = slim.dropout(self.nin_lay_1, keep_prob=self.keep_prob)
        self.maxpooling_1 = slim.max_pool2d(dropout, kernel_size=3, stride=2, padding='SAME')
        print('maxpooling_1:' + str(self.maxpooling_1.get_shape().as_list()))

        self.nin_lay_2 = self.mlp_conv(self.maxpooling_1, kernel_size=5, stride=1, num_filters=256,
                                       micro_layer_size=[256, 256], name='nin_lay_2')
        # add dropout
        dropout = slim.dropout(self.nin_lay_2, keep_prob=self.keep_prob)
        self.maxpooling_2 = slim.max_pool2d(dropout, kernel_size=3, stride=2, padding='SAME')
        print('maxpooling_2:' + str(self.maxpooling_2.get_shape().as_list()))

        self.nin_lay_3 = self.mlp_conv(self.maxpooling_2, kernel_size=3, stride=1, num_filters=384,
                                       micro_layer_size=[384, 384], name='nin_lay_3')
        # NO dropout
        self.maxpooling_3 = slim.max_pool2d(self.nin_lay_3, kernel_size=3, stride=2, padding='SAME')
        print('maxpooling_3:' + str(self.maxpooling_3.get_shape().as_list()))

        self.nin_lay_4 = self.mlp_conv(self.maxpooling_3, kernel_size=3, stride=1, num_filters=1024,
                                       micro_layer_size=[1024, self.num_classes], name='nin_lay_4')
        self.maxpooling_4 = slim.max_pool2d(self.nin_lay_4, kernel_size=3, stride=2, padding='SAME')
        print('maxpooling_4:' + str(self.maxpooling_4.get_shape().as_list()))
        # golbal average pooling
        self.digits = self.golbal_average_pooling(self.nin_lay_4)
        self.digits = self.digits[:, 0, 0, :]
        print('golbal_average_pooling:' + str(self.digits.get_shape().as_list()))
        # softmax
        self.read_out_logits = tf.nn.softmax(self.digits)
        print('read_out_logits:' + str(self.read_out_logits.get_shape().as_list()))

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
