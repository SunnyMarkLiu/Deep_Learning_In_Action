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
import utils
from functools import reduce
from tensorflow.examples.tutorials.mnist import input_data


