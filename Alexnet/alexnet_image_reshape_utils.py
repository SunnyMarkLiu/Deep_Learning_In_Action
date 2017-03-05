# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import progressbar as pbar
import Image
import h5py

img_rows, img_cols = 28, 28
input_rows, input_cols = 224, 224
data = pd.read_csv('train.csv')
images = data.iloc[:, 1:].values
images = images.astype(np.float)
image_reshape = np.ndarray(shape=(images.shape[0], input_rows, input_cols, 3), dtype=np.float16)
images = images.reshape(images.shape[0], img_rows, img_cols)
labels_flat = data[[0]].values.ravel()

widgets = ['Test: ', pbar.Percentage(), ' ', pbar.Bar('>'), ' ', pbar.ETA()]
image_bar = pbar.ProgressBar(widgets=widgets, maxval=images.shape[0]).start()

for j in range(images.shape[0]):
    pil_im = Image.fromarray(images[j])
    im_resize = pil_im.resize((input_rows, input_cols), Image.ANTIALIAS)
    im = np.array(im_resize.convert('RGB'), dtype=np.float16)
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    # 'RGB'->'BGR'
    im = im[:, :, ::-1]
    image_reshape[j, :, :, :] = im
    image_bar.update(j + 1)
image_bar.finish()
print('image_vgg shape:', image_reshape.shape)

try:
    with h5py.File('reshape_mnist_alexnet.h5', 'w') as f:
        f.create_dataset('images', data=image_reshape)
        f.create_dataset('labels', data=labels_flat)
except Exception as e:
    print('Unable to save images:', e)
