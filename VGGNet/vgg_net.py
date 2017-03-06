#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
VGG net implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits

VGG net Paper: https://arxiv.org/pdf/1409.1556.pdf
Mnist Dataset: http://yann.lecun.com/exdb/mnist/

@author: MarkLiu
@time  : 17-3-4 下午3:22
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf


class Vgg16(object):
    """
    VggNet-16
    """

    def __init__(self, num_classes, activation, skip_layer, weights_path='DEFAULT'):
        self.NUM_CLASSES = num_classes
        self.ACTIVATION = activation
        # 指定跳过加载 pre-trained weight 层
        self.SKIP_LAYER = skip_layer
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'vgg16.npy'
        else:
            self.WEIGHTS_PATH = weights_path

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
            relu = self.ACTIVATION(conv_bias, name=scope.name)

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
                fc = self.ACTIVATION(fc)

            return fc

    def dropout(self, x, keep_prob):
        """
        dropout layer
        """
        return tf.nn.dropout(x, keep_prob)

    def build_model(self):
        """
        构建模型
        """
        # input features
        self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input_layer')
        self.y = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='output_layer')

        # learning_rate placeholder
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # dropout layer: keep probability, vgg default value:0.5
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # build model
        # conv1: conv1_1 + conv1_2 + pool1
        conv1_1 = self.conv2d(self.x, 3, 3, 64, 1, 1, padding='SAME', name='conv1_1')
        conv1_2 = self.conv2d(conv1_1, 3, 3, 64, 1, 1, padding='SAME', name='conv1_2')
        pool1 = self.max_pool(conv1_2, 3, 3, 2, 2, padding='SAME', name='pool1')

        # conv2: conv2_1 + conv2_2 + pool2
        conv2_1 = self.conv2d(pool1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_1')
        conv2_2 = self.conv2d(conv2_1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_2')
        pool2 = self.max_pool(conv2_2, 3, 3, 2, 2, padding='SAME', name='pool2')

        # conv3: conv3_1 + conv3_2 + conv3_3 + pool3
        conv3_1 = self.conv2d(pool2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_1')
        conv3_2 = self.conv2d(conv3_1, 3, 3, 256, 1, 1, padding='SAME', name='conv3_2')
        conv3_3 = self.conv2d(conv3_2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_3')
        pool3 = self.max_pool(conv3_3, 3, 3, 2, 2, padding='SAME', name='pool3')

        # conv4: conv4_1 + conv4_2 + conv4_3 + pool4
        conv4_1 = self.conv2d(pool3, 3, 3, 512, 1, 1, padding='SAME', name='conv4_1')
        conv4_2 = self.conv2d(conv4_1, 3, 3, 512, 1, 1, padding='SAME', name='conv4_2')
        conv4_3 = self.conv2d(conv4_2, 3, 3, 512, 1, 1, padding='SAME', name='conv4_3')
        pool4 = self.max_pool(conv4_3, 3, 3, 2, 2, padding='SAME', name='pool4')

        # conv5: conv5_1 + conv5_2 + conv5_3 + pool5
        conv5_1 = self.conv2d(pool4, 3, 3, 512, 1, 1, padding='SAME', name='conv5_1')
        conv5_2 = self.conv2d(conv5_1, 3, 3, 512, 1, 1, padding='SAME', name='conv5_2')
        conv5_3 = self.conv2d(conv5_2, 3, 3, 512, 1, 1, padding='SAME', name='conv5_3')
        pool5 = self.max_pool(conv5_3, 3, 3, 2, 2, padding='SAME', name='pool5')

        # fc6
        fc6 = self.fully_connected(pool5, 4096, name='fc6')
        dropout6 = self.dropout(fc6, self.keep_prob)
        # fc7
        fc7 = self.fully_connected(dropout6, 4096, name='fc7')
        dropout7 = self.dropout(fc7, self.keep_prob)

        # fc8
        read_out_digits = self.fully_connected(dropout7, self.NUM_CLASSES, activation=False, name='fc8')
        self.read_out_logits = tf.nn.softmax(read_out_digits, name="prob")

    def init_train_test_op(self):
        # loss function
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

    def classify(self, features_x):
        """
        分类预测
        """
        feed_dict = {self.x: features_x, self.keep_prob: 1.0}
        predict_y, prob = self.sess.run([self.predict_op, self.read_out_logits], feed_dict=feed_dict)
        return predict_y, prob

    def train(self, x, y, learning_rate, keep_prob=0.5):
        """
        训练
        """
        feed_dict = {
            self.x: x,
            self.y: y,
            self.keep_prob: keep_prob,
            self.learning_rate: learning_rate
        }
        _, train_loss = self.sess.run([self.training_op, self.loss_function], feed_dict=feed_dict)
        train_accuracy = self.get_accuracy(x, y)
        return train_loss, train_accuracy

    def get_accuracy(self, x, y):
        """
        获取测试数据的精度
        """
        feed_dict = {
            self.x: x,
            self.y: y,
            self.keep_prob: 1.0
        }
        accuracy = self.sess.run(self.accuracy, feed_dict=feed_dict)
        return accuracy

    def init(self):
        self.build_model()
        self.init_train_test_op()
        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def load_initial_weights(self):
        """
        As the weights from https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM come
        as a dict of lists (e.g. weights['conv1_1'] is a list) and not as dict of
        dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
        need a special load function

        """
        print('Load the pretrained weights into the non-trainable layer...')
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse=True):

                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            print('load bias' + op_name)
                            var = tf.get_variable('biases', trainable=False)
                            self.sess.run(var.assign(data))
                        # full connected layer weights
                        elif len(data.shape) == 2:
                            print('load Weights' + op_name)
                            var = tf.get_variable('weights', trainable=False)
                            self.sess.run(var.assign(data))
                        # cnn layer filters
                        else:
                            print('load filter' + op_name)
                            var = tf.get_variable('c', trainable=False)
                            self.sess.run(var.assign(data))
