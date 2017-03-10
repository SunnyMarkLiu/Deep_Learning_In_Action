#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
Google Inception V1 model implementation example using TensorFlow library.

Inception V1 Paper: Going Deeper with Convolutions(https://arxiv.org/abs/1409.4842)

Pre-trained model layers infor：
['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
  'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
  'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
  'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']

@author: MarkLiu
@time  : 17-3-8 下午1:35
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow


class GoogleInceptionV1(object):
    """
    Google Inception V1 model
    """

    def __init__(self, num_classes, skip_layer, pre_trained_model='DEFAULT'):
        self.num_classes = num_classes
        # 指定跳过加载 pre-trained 层
        self.skip_layer = skip_layer
        if pre_trained_model == 'DEFAULT':
            self.pre_trained_model = './inception_v1.ckpt'
        else:
            self.pre_trained_model = pre_trained_model

    def inception_v1_base(self, inputs, final_endpoint='Mixed_5c', scope='InceptionV1'):
        """
        flexible inception v1 base model. Extract softmax layer outside the inception_base
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v1.py
        :param inputs: a tensor of size [batch_size, height, width, channels].
        :param final_endpoint: final out put layer
                            ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
                              'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
                              'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
                              'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
        :return: A dictionary from components of the network to the corresponding activation.
        """
        ent_point_nets = {}
        with tf.variable_scope(scope, default_name='InceptionV1', values=[inputs]):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu,
                                weights_initializer=slim.initializers.variance_scaling_initializer()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='SAME'):
                    end_point = 'Conv2d_1a_7x7'
                    net = slim.conv2d(inputs, num_outputs=64, kernel_size=[7, 7], stride=2, scope=end_point)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    end_point = 'MaxPool_2a_3x3'
                    net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope=end_point)
                    # LocalRespNorm
                    net = slim.nn.local_response_normalization(net, depth_radius=2.5, bias=2,
                                                               alpha=0.0001, beta=0.75, name='norm1')
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    end_point = 'Conv2d_2b_1x1'
                    net = slim.conv2d(net, 64, [1, 1], scope=end_point)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    end_point = 'Conv2d_2c_3x3'
                    net = slim.conv2d(net, 192, [3, 3], scope=end_point)
                    net = slim.nn.local_response_normalization(net, depth_radius=2.5, bias=2,
                                                               alpha=0.0001, beta=0.75, name='norm1')
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    end_point = 'MaxPool_3a_3x3'
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    # build inception bolck
                    end_point = 'Mixed_3b'
                    with tf.variable_scope(end_point):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                            # in paper, the kernel_size = [5, 5]
                            branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')

                        # DepthConcat
                        net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    end_point = 'Mixed_3c'
                    with tf.variable_scope(end_point):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                            # in paper, the kernel_size = [5, 5]
                            branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')

                        # DepthConcat
                        net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    end_point = 'MaxPool_4a_3x3'
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    end_point = 'Mixed_4b'
                    with tf.variable_scope(end_point):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')

                        # DepthConcat
                        net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    end_point = 'Mixed_4c'
                    with tf.variable_scope(end_point):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                        # DepthConcat
                        net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    # for flexible, extract softmax layer outside the inception_base

                    end_point = 'Mixed_4d'
                    with tf.variable_scope(end_point):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                        # DepthConcat
                        net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    end_point = 'Mixed_4e'
                    with tf.variable_scope(end_point):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                        # DepthConcat
                        net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    # softmax 1

                    end_point = 'Mixed_4f'
                    with tf.variable_scope(end_point):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                        # DepthConcat
                        net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    end_point = 'MaxPool_5a_2x2'
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    end_point = 'Mixed_5b'
                    with tf.variable_scope(end_point):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                        # DepthConcat
                        net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                    end_point = 'Mixed_5c'
                    with tf.variable_scope(end_point):
                        with tf.variable_scope('Branch_0'):
                            branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                        with tf.variable_scope('Branch_1'):
                            branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                            branch_1 = slim.conv2d(
                                branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_2'):
                            branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
                        with tf.variable_scope('Branch_3'):
                            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                        # DepthConcat
                        net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
                    ent_point_nets[end_point] = net
                    if final_endpoint == end_point:
                        return net, ent_point_nets

                        # softmax 2

                raise ValueError('Unknown final endpoint %s' % final_endpoint)

    def build_inception_v1(self, prediction_fn=tf.nn.relu, scope='InceptionV1'):
        """
        build basic inception v1 model
        """
        # input features [batch_size, height, width, channels]
        self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input_layer')
        self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='output_layer')

        # learning_rate placeholder
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # dropout layer: keep probability, vgg default value:0.5
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope(name_or_scope=scope, reuse=False) as scope:
            net, ent_point_nets = self.inception_v1_base(self.x, scope=scope)
            with tf.variable_scope('Logits'):
                net = slim.avg_pool2d(net, kernel_size=[7, 7], stride=1, scope='MaxPool_0a_7x7')
                net = slim.dropout(net, self.keep_prob, scope='Dropout_0b')
                self.logits = slim.fully_connected(net, num_outputs=1024)
                ent_point_nets['Logits'] = self.logits
                ent_point_nets['Predictions'] = prediction_fn(self.logits, name='Predictions')
                self.read_out_logits = ent_point_nets['Predictions']

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
        self.build_inception_v1()
        self.init_train_test_op()
        self.sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def load_pretrained_model(self):
        """
        Load the pretrained weights into the non-trainable layer
        :return:
        """
        print('Load the pretrained weights into the non-trainable layer...')
        trainable_variables = slim.get_trainable_variables()

        reader = pywrap_tensorflow.NewCheckpointReader('./inception_v1.ckpt')
        pretrained_model_variables = reader.get_variable_to_shape_map()
        for variable in trainable_variables:
            variable_name = variable.name.split(':')[0]
            if variable_name in self.skip_layer:
                continue
            if variable_name not in pretrained_model_variables:
                continue
            with tf.variable_scope('', reuse=True):
                var = tf.get_variable(variable_name, trainable=False)
                data = reader.get_tensor(variable_name)
                self.sess.run(var.assign(data))
