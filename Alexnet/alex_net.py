#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
Alexnet implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits

AlexNet Paper: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
Mnist Dataset: http://yann.lecun.com/exdb/mnist/

@author: MarkLiu
@time  : 17-3-3 下午10:12
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import utils
from functools import reduce
from tensorflow.examples.tutorials.mnist import input_data


class Alexnet(object):
    """
    convolutional network model
    """

    def __init__(self, labels_size, activation, image_width=227, image_height=227):
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

    def conv2d(self, x, filter_size, filters_out_channels, stride, activation, name):
        """
        卷积层
        :param x: [batch, in_height, in_width, in_channels]
        :param filter_size: filter 的大小
        :param filters_out_channels: filters 的数目,[filter_height, filter_width, in_channels, out_channels]
        :param stride: 每一维度滑动的步长,strides[0]=strides[3]=1
        """
        # x 获取最后一维的 shape
        input_final_shape = utils.get_incoming_shape(x)[-1]
        w_conv = self.create_weight_variable([filter_size, filter_size, input_final_shape, filters_out_channels],
                                             name + '_w')
        b_conv = self.create_bias_variable([filters_out_channels], name + '_b')
        conv = tf.add(tf.nn.conv2d(x, w_conv, strides=[1, stride, stride, 1], padding='SAME'), b_conv)
        return activation(conv)

    def max_pool(self, x, ksize, stride, name):
        """
        pooling 层, 当 stride = ksize， padding='SAME' 时输出 tensor 大小减半
        :param x: [batch, height, width, channels]
        :param ksize: [1, height, width, 1]
        :param stride: [1, stride, stride, 1]
        """
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

    def local_response_normalization(self, feature_map, depth_radius, bias, alpha, beta, name):
        """
        对卷积输出的 feature map 进行归一化： 将 feature map 中和其他卷积层的同位置的进行 normalize，即平滑处理。
        公式：http://img.blog.csdn.net/20161206153458678
        :param feature_map: 卷积层输出的 feature map
        :param depth_radius: 所要 normalize 的深度半径，对应公式中的 n/2
        """
        return tf.nn.local_response_normalization(input=feature_map, depth_radius=depth_radius, bias=bias,
                                                  alpha=alpha, beta=beta, name=name)

    def fully_connected(self, incoming, n_units, activation, name):
        """
        全连接层， n_units 指定输出神经元的数目
        """
        input_shapes = utils.get_incoming_shape(incoming)
        input_size = reduce(lambda x, y: x * y, input_shapes[1:])
        incoming = tf.reshape(incoming, [-1, input_size])
        fc_w = self.create_weight_variable(shape=[input_size, n_units], name=name + '_w')
        fc_b = self.create_bias_variable(shape=[n_units], name=name + '_b')
        full_con = activation(tf.add(tf.matmul(incoming, fc_w), fc_b))
        return full_con

    def dropout(self, incoming, keep_prob):
        """
        dropout layer
        """
        return tf.nn.dropout(incoming, keep_prob)

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

        conv1 = self.conv2d(self.x_image, filter_size=11, filters_out_channels=96, stride=4,
                            activation=self.activation, name='conv1')
        # over-lapping pooling
        pool1 = self.max_pool(conv1, ksize=3, stride=2, name='pool1')
        # local_response_normalization
        lrn_1 = self.local_response_normalization(pool1, depth_radius=2.5, bias=2,
                                                  alpha=0.0001, beta=0.75, name='norm1')

        conv2 = self.conv2d(lrn_1, filter_size=5, filters_out_channels=256, stride=1,
                            activation=self.activation, name='conv2')
        # over-lapping pooling
        pool2 = self.max_pool(conv2, ksize=3, stride=2, name='pool2')
        # local_response_normalization
        lrn_2 = self.local_response_normalization(pool2, depth_radius=2.5, bias=2,
                                                  alpha=0.0001, beta=0.75, name='norm2')

        # 3 conv
        conv3 = self.conv2d(lrn_2, filter_size=3, filters_out_channels=384, stride=1,
                            activation=self.activation, name='conv3')
        conv4 = self.conv2d(conv3, filter_size=3, filters_out_channels=384, stride=1,
                            activation=self.activation, name='conv4')
        conv5 = self.conv2d(conv4, filter_size=3, filters_out_channels=256, stride=1,
                            activation=self.activation, name='conv5')
        pool5 = self.max_pool(conv5, ksize=3, stride=2, name='pool5')
        lrn_5 = self.local_response_normalization(pool5, depth_radius=2.5, bias=2,
                                                  alpha=0.0001, beta=0.75, name='norm5')

        # fully-connected layer + dropout
        full_con_6 = self.fully_connected(lrn_5, 4096, activation=self.activation, name='fc6')
        dropout6 = self.dropout(full_con_6, self.keep_prob)
        full_con_7 = self.fully_connected(dropout6, 4096, activation=self.activation, name='fc7')

        # dropout
        dropout7 = self.dropout(full_con_7, self.keep_prob)

        read_out_digits = self.fully_connected(dropout7, self.labels_size, activation=self.activation,
                                               name='read_out_digits')
        # softmax
        self.read_out_logits = tf.nn.softmax(read_out_digits)

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

    def load_initial_weights(self, pre_trained_weight_path, skip_layers=None):
        """
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come
        as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of
        dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
        need a special load function

        skip_layers: 指定不进行加载初始化的层的
        """

        # Load the weights into memory
        weights_dict = np.load(pre_trained_weight_path, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in skip_layers:
                print(op_name)
                with tf.variable_scope(op_name, reuse=True):

                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:

                            var = tf.get_variable(op_name + '_w', trainable=False)
                            self.sess.run(var.assign(data))

                        # Weights
                        else:

                            var = tf.get_variable(op_name + '_b', trainable=False)
                            self.sess.run(var.assign(data))


def main():
    print('load datas...')

    image_width = 227
    image_height = 227
    labels_size = 17
    mnist = input_data.read_data_sets(utils.mnist_dir, one_hot=True)

    # Parameters
    learning_rate = 0.001
    training_epochs = 10
    batch_size = 200
    display_step = 1

    total_batch = int(mnist.train.num_examples / batch_size)

    alexnet = Alexnet(labels_size=labels_size, activation=tf.nn.relu,
                      image_width=image_width, image_height=image_height)

    alexnet.init()
    # Load the pretrained weights into the non-trainable layer
    alexnet.load_initial_weights('bvlc_alexnet.npy')
    for epoch in range(0, training_epochs):
        avg_cost = 0.
        for i in range(0, total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            cost = alexnet.train(batch_x, batch_y, learning_rate, 0.8)
            avg_cost += cost / total_batch

        if epoch % display_step == 0:
            print("Epoch: %04d, cost=:%.9f" % (epoch + 1, avg_cost))
        if epoch % 4 == 0:
            learning_rate /= 2

    print("Training Finished!")
    print('Predict...')
    accuracy = alexnet.get_accuracy(x=mnist.test.images, y=mnist.test.labels)
    print("accuracy = %.3f" % accuracy)


if __name__ == '__main__':
    main()
