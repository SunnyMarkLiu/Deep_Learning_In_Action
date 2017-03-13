#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-3-12 下午10:03
"""
import h5py
import utils
from network_in_network import NetworkInNetwork
from datautil import DataWapper

print('load train datas...')

num_classes = 10
train_split = 0.85  # training/validation split

data = h5py.File(utils.train_mnist_2_vggnet_size_file, 'r')
images = data['images'][:]
labels = data['labels'][:]
del data
# split data into training and validation sets
train_samples = int(len(images) * train_split)
train_features = images[:train_samples]
validation_features = images[train_samples:]
del images
train_labels = labels[:train_samples]
validation_labels = labels[train_samples:]
del labels
# Parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 100
display_step = 1
total_batch = int(train_samples / batch_size)

nin = NetworkInNetwork(224, 224, 3, 10)
nin.init()

train_datas = DataWapper(train_features, train_labels)
validate_datas = DataWapper(validation_features, validation_labels)

# training
print('Train model ...')
for epoch in range(0, training_epochs):
    avg_loss = 0.
    for i in range(0, total_batch):
        batch_x, batch_y = train_datas.next_batch(batch_size)
        train_loss, train_accuracy = nin.train(batch_x, batch_y, learning_rate, keep_prob=0.8)
        avg_loss += train_loss / total_batch
        print("train loss = %.9f, train accuracy = %.5f" % (train_loss, train_accuracy))

    if epoch % display_step == 0:
        batch_x, batch_y = validate_datas.next_batch(batch_size * 2)
        accuracy = nin.get_accuracy(x=batch_x, y=batch_y)
        print("Epoch: %04d, train loss = %.9f, validation accuracy = %.5f" % (epoch + 1, avg_loss, accuracy))
    if epoch % 4 == 0:
        learning_rate /= 2

print('Train end.')
print('Predict ...')
print('load test datas...')
data = h5py.File(utils.test_mnist_2_vggnet_size_file, 'r')
test_images = data['images'][:]
test_labels = data['labels'][:]
print('load datas done!')
predict_accuracy = nin.get_accuracy(x=test_images, y=test_labels)
print('predict_accuracy = %.5f' % predict_accuracy)
