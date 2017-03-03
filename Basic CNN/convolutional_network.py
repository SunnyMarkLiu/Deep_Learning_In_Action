#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

@author: MarkLiu
@time  : 17-3-3 下午8:43
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from utils import mnist_dir


class ConvolutionalNetwork(object):
    """
    convolutional network model
    """
    def __init__(self, image_width, image_height, labels_size, activation):
        self.image_width = image_width
        self.image_height = image_height
        self.labels_size = labels_size
        self.activation = activation

    def create_weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)

    def create_bias_variable(self, shape, name):
        initial = tf.constant(np.random.rand(), shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, x, filter_w, stride):
        """
        卷积层
        :param x: [batch, in_height, in_width, in_channels]
        :param filter_w: [filter_height, filter_width, in_channels, out_channels]
        :param stride: 每一维度滑动的步长,strides[0]=strides[3]=1
        :return:
        """
        return tf.nn.conv2d(x, filter_w, strides=[1, stride, stride, 1], padding='SAME')

    def max_pool(self, x, ksize, stride):
        """
        pooling 层, 当 stride = ksize， padding='SAME' 时输出 tensor 大小减半
        :param x: [batch, height, width, channels]
        :param ksize: [1, height, width, 1]
        :param stride: [1, stride, stride, 1]
        """
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')

    def build_model(self):
        """
        构建模型
        """
        # input features
        self.x = tf.placeholder(tf.float32, shape=[None, self.image_width * self.image_height], name='input_layer')
        self.y = tf.placeholder(tf.float32, [None, self.labels_size], name='output_layer')

        # reshape features to 2d shape
        self.x_image = tf.reshape(self.x, [-1, self.image_width, self.image_height, 1])

        # learning_rate placeholder
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # dropout layer: keep probability
        self.keep_prob = tf.placeholder(tf.float32)

        # conv + conv + pool
        # 32 个 filters
        self.W_conv1 = self.create_weight_variable([3, 3, 1, 32], 'W_conv1')
        self.b_conv1 = self.create_bias_variable([32], 'b_conv1')
        conv1 = self.activation(self.conv2d(self.x_image, self.W_conv1, stride=1) + self.b_conv1)

        self.W_conv2 = self.create_weight_variable([3, 3, 32, 64], 'W_conv2')
        self.b_conv2 = self.create_bias_variable([64], 'b_conv2')
        conv2 = self.activation(self.conv2d(conv1, self.W_conv2, stride=1) + self.b_conv2)

        self.pool = self.max_pool(conv2, ksize=2, stride=2)

        # fully-connected layer + dropout
        self.W_fc1 = self.create_weight_variable([14 * 14 * 64, 256], 'W_fc1')
        self.b_fc1 = self.create_bias_variable([256], 'b_fc1')
        self.pool = tf.reshape(self.pool, [-1, 14 * 14 * 64])
        full_con_1 = tf.nn.relu(tf.matmul(self.pool, self.W_fc1) + self.b_fc1)

        # dropout
        dropout = tf.nn.dropout(full_con_1, self.keep_prob)

        self.W_readout = self.create_weight_variable([256, 10], 'W_readout')
        self.b_read_out = self.create_bias_variable([10], 'b_read_out')
        self.digits = tf.matmul(dropout, self.W_readout) + self.b_read_out

        # softmax
        self.read_out_logits = tf.nn.softmax(self.digits)

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
        predict_y = self.sess.run(self.predict_op, feed_dict=feed_dict)
        return predict_y

    def train(self, features_x, y, learning_rate, keep_prob=0.8):
        """
        训练
        """
        feed_dict = {
            self.x: features_x,
            self.y: y,
            self.keep_prob: keep_prob,
            self.learning_rate: learning_rate
        }
        _, loss = self.sess.run([self.training_op, self.loss_function], feed_dict=feed_dict)
        return loss

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


def main():
    print('load datas...')
    mnist = input_data.read_data_sets(mnist_dir, one_hot=True)

    image_width = 28
    image_height = 28
    labels_size = 10

    # Parameters
    learning_rate = 0.001
    training_epochs = 10
    batch_size = 200
    display_step = 1

    total_batch = int(mnist.train.num_examples / batch_size)

    cnn = ConvolutionalNetwork(image_width, image_height, labels_size, activation=tf.nn.relu)

    cnn.init()
    for epoch in range(0, training_epochs):
        avg_cost = 0.
        for i in range(0, total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            cost = cnn.train(batch_x, batch_y, learning_rate, 0.8)
            avg_cost += cost / total_batch

        if epoch % display_step == 0:
            print("Epoch: %04d, cost=:%.9f" % (epoch + 1, avg_cost))
        if epoch % 4 == 0:
            learning_rate /= 2

    print("Training Finished!")
    print('Predict...')
    accuracy = cnn.get_accuracy(x=mnist.test.images, y=mnist.test.labels)
    print("accuracy = %.3f" % accuracy)


if __name__ == '__main__':
    main()
