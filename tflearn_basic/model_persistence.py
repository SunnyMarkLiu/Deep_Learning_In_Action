#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-3-1 下午8:54
"""
from __future__ import absolute_import, division, print_function

import os

import tflearn
import tflearn.datasets.mnist as mnist

base_dir = '/home/sunnymarkliu/Projects/PycharmProjects/deeplearning_in_action/'
tensorboard_dir = base_dir + 'tensorboard/model_persistence/'
checkpoint_path = base_dir + 'checkpoint/model_persistence/'
best_checkpoint_path = checkpoint_path + 'best_checkpoint/'
data_dir = base_dir + 'input_datas/mnist/'
saved_model_path = checkpoint_path + 'mnist_model.ckpt'

for path in [data_dir, tensorboard_dir, checkpoint_path, best_checkpoint_path]:
    if not os.path.exists(path):
        os.makedirs(path)

X, Y, testX, testY = mnist.load_data(data_dir=data_dir, one_hot=True)


def build_model():
    input_layer = tflearn.input_data(shape=[None, 784], name='input')
    fc1 = tflearn.fully_connected(input_layer, n_units=128, activation='relu', name='fc1')
    fc2 = tflearn.fully_connected(fc1, n_units=128, activation='relu', name='fc2')
    softmax = tflearn.fully_connected(fc2, 10, activation='softmax')

    regression = tflearn.regression(softmax, optimizer='adam',
                                    learning_rate=0.01,
                                    loss='categorical_crossentropy')

    # Define classifier, with model checkpoint (autosave)
    mnist_model = tflearn.DNN(network=regression,
                              tensorboard_dir=tensorboard_dir,
                              checkpoint_path=checkpoint_path,
                              best_checkpoint_path=best_checkpoint_path)

    return mnist_model


def save_model():
    mnist_model = build_model()

    mnist_model.fit(X, Y,
                    n_epoch=10,
                    validation_set=(testX, testY),
                    validation_batch_size=100,
                    show_metric=True,
                    snapshot_epoch=True,  # Snapshot (save & evaluate) model every epoch.
                    snapshot_step=200,  # Snapshot (save & evalaute) model every 500 steps.
                    run_id='mnist_model_train',
                    )

    # save model
    mnist_model.save(saved_model_path)


def load_model():
    mnist_model = build_model()
    mnist_model.load(saved_model_path)

    # 在加载的模型的基础之上再次训练模型
    mnist_model.fit(X, Y,
                    n_epoch=10,
                    validation_set=(testX, testY),
                    validation_batch_size=100,
                    show_metric=True,
                    run_id='mnist_model_train')

if __name__ == '__main__':
    # save_model()
    load_model()
