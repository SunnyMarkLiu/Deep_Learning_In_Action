#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
Alexnet predict ImageNet demo.

@author: MarkLiu
@time  : 17-3-6 上午11:23
"""
from alex_net import Alexnet
from caffe_classes import class_names
import tensorflow as tf
import utils
import numpy as np
import os
import cv2


def main():
    alexnet = Alexnet(num_classes=1000, activation=tf.nn.relu,
                      skip_layer=[], weights_path=utils.pre_trained_alex_model)

    alexnet.init()
    # Load the pretrained weights into the non-trainable layer
    alexnet.load_initial_weights()

    # mean of imagenet dataset in BGR
    imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

    current_dir = os.getcwd()
    image_dir = os.path.join(current_dir, 'images')
    # get list of all images
    img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpeg')]

    # load all images
    imgs = []
    for f in img_files:
        imgs.append(cv2.imread(f))

    for i, image in enumerate(imgs):
        # Convert image to float32 and resize to (227x227)
        image = cv2.resize(image.astype(np.float32), (227, 227))
        # Subtract the ImageNet mean
        image -= imagenet_mean
        image = image.reshape((1, 227, 227, 3))
        predict_y, prob = alexnet.classify(image)
        class_name = class_names[int(predict_y)]
        print('image ' + str(i) + ':' + class_name)
