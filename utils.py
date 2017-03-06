#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-3-3 下午4:41
"""
base_dir = '/home/sunnymarkliu/projects/deeplearning/'
tensorboard_dir = base_dir + 'tensorboard/'
checkpoint_path = base_dir + 'checkpoint/'
best_checkpoint_path = checkpoint_path + 'best_checkpoint/'

# datasets
mnist_dir = base_dir + 'datasets/mnist/'
train_mnist_2_imagenet_size_file = base_dir + 'datasets/mnist/train_mnist_2_imagenet_size.h5'
test_mnist_2_imagenet_size_file = base_dir + 'datasets/mnist/test_mnist_2_imagenet_size.h5'
train_mnist_2_vggnet_size_file = base_dir + 'datasets/mnist/train_mnist_2_vggnet_size.h5'
test_mnist_2_vggnet_size_file = base_dir + 'datasets/mnist/test_mnist_2_vggnet_size.h5'

# model
pre_trained_alex_model = '/home/sunnymarkliu/projects/deeplearning/pre_trained_model/bvlc_alexnet.npy'
pre_trained_vgg16_model = '/home/sunnymarkliu/projects/deeplearning/pre_trained_model/vgg16.npy'
