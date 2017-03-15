#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-3-15 下午7:18
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from auto_encoder import Autoencoders
import numpy as np
import utils
import matplotlib as mpl
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt

print('load datas...')
mnist = input_data.read_data_sets(utils.mnist_dir, one_hot=True)
image_width = 28
image_height = 28
labels_size = 10

learning_rate = 0.001
training_epochs = 100
batch_size = 200
display_step = 1

total_batch = int(mnist.train.num_examples / batch_size)

auto_encoder = Autoencoders(image_height * image_width, tf.nn.relu)
auto_encoder.init()

for epoch in range(0, training_epochs):
    avg_cost = 0.
    for i in range(0, total_batch):
        batch_x, _ = mnist.train.next_batch(batch_size)

        cost = auto_encoder.train(batch_x, learning_rate, 0.8)
        avg_cost += cost / total_batch

    if epoch % display_step == 0:
        print("Epoch: %04d, cost=%.9f" % (epoch + 1, avg_cost))

    if epoch % 40 == 0:
        learning_rate /= 2

print("Training Finished!")
total_cost = auto_encoder.calc_total_cost(x=mnist.test.images)
print("total cost = %.3f" % total_cost)

# visualize weights
hidden_layer1_weights = auto_encoder.get_weights('hidden_layer_1')
preprocessor = prep.MinMaxScaler().fit(hidden_layer1_weights)
hidden_layer1_weights = preprocessor.transform(hidden_layer1_weights)

hidden_layer1_weights = hidden_layer1_weights.reshape((28, 28, 100))
hidden_layer1_weights /= np.sum(np.power(hidden_layer1_weights, 2))
hidden_layer1_weights *= 255
fig = plt.figure()
for i in range(100):
    weight = hidden_layer1_weights[:, :, i]
    cmap = mpl.cm.gray_r
    ax = fig.add_subplot(10, 10, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=-0.7)
    ax.imshow(weight, cmap=cmap)
fig.savefig('cnn_weight.png', dpi=75)
