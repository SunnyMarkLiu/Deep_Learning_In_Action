#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-3-3 下午4:41
"""
import numpy as np
import tensorflow as tf

base_dir = '/home/sunnymarkliu/projects/deeplearning/'
tensorboard_dir = base_dir + 'tensorboard/'
checkpoint_path = base_dir + 'checkpoint/'
best_checkpoint_path = checkpoint_path + 'best_checkpoint/'

# datasets
mnist_dir = base_dir + 'datasets/mnist/'
reshape_mnist_alexnet_dir = base_dir + 'datasets/mnist/reshape_mnist_alexnet.h5'

# model
pre_trained_alex_model = '/home/sunnymarkliu/projects/deeplearning/pre_trained_model/bvlc_alexnet.npy'
