# coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# see top_30 labels
for i in range(30):
    one_hot_label = mnist.train.labels[i, :]
    label = np.argmax(one_hot_label)
    print('mnist_train_%d.jpg label: %d' % (i, label))