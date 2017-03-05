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

from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import utils


class Alexnet(object):
    """
    convolutional network model
    """

    def __init__(self, num_classes, activation, skip_layer,
                 weights_path='DEFAULT'):
        self.NUM_CLASSES = num_classes
        self.ACTIVATION = activation
        self.SKIP_LAYER = skip_layer
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

    def create_weight_variable(self, shape, name):
        with tf.variable_scope(name) as scope:
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial, name=name)

    def create_bias_variable(self, shape, name):
        initial = tf.constant(np.random.rand(), shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, x, filter_height, filter_width, num_filters, stride_y, stride_x,
               name, padding='SAME', groups=1):
        """
        卷积层
        Ref: https://github.com/ethereon/caffe-tensorflow
        :param x: [batch, in_height, in_width, in_channels]
        :param num_filters: filters 的数目,[filter_height, filter_width, in_channels, out_channels]
        :param stride_y, stride_x: 每一维度滑动的步长,strides[0]=strides[3]=1
        """
        # Get number of input channels
        input_channels = int(x.get_shape()[-1])

        # Create lambda function for the convolution
        def convolve(input_x, filters):
            return tf.nn.conv2d(input_x, filters,
                                strides=[1, stride_y, stride_x, 1],
                                padding=padding)

        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights',
                                      shape=[filter_height, filter_width, input_channels / groups, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])

            if groups == 1:
                conv = convolve(x, weights)

            # In the cases of multiple groups, split inputs & weights and
            else:
                # Split input and weights and convolve them separately
                input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
                weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

                # Concat the convolved output together again
                conv = tf.concat(axis=3, values=output_groups)

            # Add biases
            bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

            # Apply activation function
            relu = self.ACTIVATION(bias, name=scope.name)

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

    def local_response_normalization(self, feature_map, depth_radius, bias, alpha, beta, name):
        """
        对卷积输出的 feature map 进行归一化： 将 feature map 中和其他卷积层的同位置的进行 normalize，即平滑处理。
        公式：http://img.blog.csdn.net/20161206153458678
        :param feature_map: 卷积层输出的 feature map
        :param depth_radius: 所要 normalize 的深度半径，对应公式中的 n/2
        """
        return tf.nn.local_response_normalization(input=feature_map, depth_radius=depth_radius, bias=bias,
                                                  alpha=alpha, beta=beta, name=name)

    def fully_connected(self, x, num_in, num_out, name, activation=True):
        """
        全连接层， n_units 指定输出神经元的数目
        """
        with tf.variable_scope(name) as scope:
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
        self.x = tf.placeholder(tf.float32, shape=[None, 227 * 227], name='input_layer')
        self.y = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='output_layer')

        # reshape features to 2d shape
        self.x_image = tf.reshape(self.x, [-1, 227, 227, 1])

        # learning_rate placeholder
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # dropout layer: keep probability
        self.keep_prob = tf.placeholder(tf.float32)

        # build model
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = self.conv2d(self.x_image, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        # over-lapping pooling
        pool1 = self.max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        # local_response_normalization
        lrn_1 = self.local_response_normalization(pool1, depth_radius=2.5, bias=2,
                                                  alpha=0.0001, beta=0.75, name='norm1')
        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = self.conv2d(lrn_1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        # over-lapping pooling
        pool2 = self.max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        # local_response_normalization
        lrn_2 = self.local_response_normalization(pool2, depth_radius=2.5, bias=2,
                                                  alpha=0.0001, beta=0.75, name='norm2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = self.conv2d(lrn_2, 3, 3, 384, 1, 1, name='conv3')
        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = self.conv2d(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = self.conv2d(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = self.max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        lrn_5 = self.local_response_normalization(pool5, depth_radius=2.5, bias=2,
                                                  alpha=0.0001, beta=0.75, name='norm5')
        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(lrn_5, [-1, 6 * 6 * 256])
        fc6 = self.fully_connected(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = self.dropout(fc6, self.keep_prob)
        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = self.fully_connected(dropout6, 4096, 4096, name='fc7')
        dropout7 = self.dropout(fc7, self.keep_prob)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        read_out_digits = self.fully_connected(dropout7, 4096, self.NUM_CLASSES, activation=False, name='fc8')
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

    def load_initial_weights(self):
        """
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come
        as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of
        dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
        need a special load function

        skip_layers: 指定不进行加载初始化的层的
        """
        print('Load the pretrained weights into the non-trainable layer...')
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        print(weights_dict['conv1'])
        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if the layer is one of the layers that should be reinitialized
            if skip_layers is None or op_name not in skip_layers:
                print(op_name)
                with tf.variable_scope(op_name, reuse=True):

                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            print('load bias' + op_name)
                            var = tf.get_variable('biases', trainable=False)
                            self.sess.run(var.assign(data))

                        # Weights
                        else:
                            print('load Weights' + op_name)
                            var = tf.get_variable('weights', trainable=False)
                            self.sess.run(var.assign(data))

    def get_weights(self, layer_name):
        var = tf.get_variable(layer_name + '_w')
        return self.sess.run(var)


def main():
    print('load datas...')

    image_width = 28
    image_height = 28
    num_classes = 10
    mnist = input_data.read_data_sets(utils.mnist_dir, one_hot=True)

    # Parameters
    learning_rate = 0.01
    training_epochs = 10
    batch_size = 200
    display_step = 1
    train_layers = ['fc8', 'fc7']

    total_batch = int(mnist.train.num_examples / batch_size)

    alexnet = Alexnet(num_classes=num_classes, activation=tf.nn.relu,
                      skip_layer=train_layers, weights_path=utils.pre_trained_alex_model)

    alexnet.init()
    # Load the pretrained weights into the non-trainable layer
    alexnet.load_initial_weights()
    print('=============================')
    print(alexnet.get_weights('conv1'))
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
    print("accuracy = %.5f" % accuracy)


if __name__ == '__main__':
    main()
