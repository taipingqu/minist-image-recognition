# coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# show train data size
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

# show validation data size
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)

# show test data size
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

# print vector representation of picture 0
print(mnist.train.images[0, :])

# print label of picture 0
print(mnist.train.labels[0, :])
