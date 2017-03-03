#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
http://ml.informatik.uni-freiburg.de/_media/teaching/ss10/05_mlps.printer.pdf

@author: MarkLiu
@time  : 17-3-3 下午4:19
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from utils import mnist_dir
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


class MultilayerPerceptron(object):
    """
    multilayer perceptron model
    """

    def __init__(self, layer_size, activation_fun):
        """
        :param layer_size: [input_layer,[hidden_layer], output_layer]
        """
        self.layer_size = layer_size
        self.activation_fun = activation_fun

    def create_weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)

    def create_bias_variable(self, shape, name):
        initial = tf.constant(np.random.rand(), shape=shape)
        return tf.Variable(initial, name=name)

    def build_model(self):
        """
        构建模型
        """
        input_layer_size = self.layer_size[0]
        hidden_layer_sizes = self.layer_size[1]
        output_layer_size = self.layer_size[2]

        # input features
        self.x = tf.placeholder(tf.float32, shape=[None, input_layer_size], name='input_layer')
        self.y = tf.placeholder(tf.float32, [None, output_layer_size], name='output_layer')

        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_layer = self.x
        if hidden_layer_sizes:
            for i in range(0, len(hidden_layer_sizes)):
                # create hidden layer
                hidden_w = self.create_weight_variable(shape=[input_layer_size, hidden_layer_sizes[i]],
                                                       name='hidden_layer_w_' + str(i))
                hidden_b = self.create_bias_variable(shape=[hidden_layer_sizes[i]],
                                                     name='hidden_layer_b_' + str(i))
                hidden_layer = tf.add(tf.matmul(input_layer, hidden_w), hidden_b)
                hidden_layer = self.activation_fun(hidden_layer)

                input_layer = hidden_layer
                input_layer_size = hidden_layer_sizes[i]
        # dropout
        dropout = tf.nn.dropout(input_layer, self.keep_prob)
        # output layer
        output_w = self.create_weight_variable(shape=[input_layer_size, output_layer_size],
                                               name='output_layer_w')
        output_b = self.create_bias_variable(shape=[output_layer_size],
                                             name='output_layer_b')
        digits = tf.add(tf.matmul(dropout, output_w), output_b)
        read_out = self.activation_fun(digits)
        # softmax layer
        self.read_out_logits = tf.nn.softmax(read_out)

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


if __name__ == '__main__':
    print('load datas...')
    mnist = input_data.read_data_sets(mnist_dir, one_hot=True)

    # Parameters
    learning_rate = 0.001
    training_epochs = 10
    batch_size = 200
    display_step = 1

    total_batch = int(mnist.train.num_examples / batch_size)

    multilayer_net = MultilayerPerceptron(layer_size=[784, [1024, 1024], 10],
                                          activation_fun=tf.nn.relu)

    multilayer_net.init()
    for epoch in range(0, training_epochs):
        avg_cost = 0.
        for i in range(0, total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            cost = multilayer_net.train(batch_x, batch_y, learning_rate, 0.8)
            avg_cost += cost / total_batch

        if epoch % display_step == 0:
            print("Epoch: %04d, cost=:%.9f" % (epoch + 1, avg_cost))
        if epoch % 4 == 0:
            learning_rate /= 2

    print("Training Finished!")
    print('Predict...')
    accuracy = multilayer_net.get_accuracy(x=mnist.test.images, y=mnist.test.labels)
    print("accuracy = %.3f" % accuracy)
