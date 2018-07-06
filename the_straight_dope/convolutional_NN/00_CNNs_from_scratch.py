# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 19:02:50 2018

@author: Taylor
"""


import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon

ctx = mx.cpu()

mx.random.seed(1)



batch_size = 64  # mini batch per SGD update
num_inputs = 784  # 28 x 28 pixel image size
num_outputs = 10  # 10 classes to output
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float64)


# this is subtle, but now the transform function is different
# than before... this is what it was before
# there was no transpose to (2,0,1) axis before...
def transform_old(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)


# I want to explore what makes this different..
train_data = gluon.data.DataLoader(
        dataset = gluon.data.vision.MNIST(train=True, transform=transform),
        batch_size = batch_size,
        shuffle=True)

test_data = gluon.data.DataLoader(
        dataset = gluon.data.vision.MNIST(train=False, transform=transform),
        batch_size = batch_size,
        shuffle=True)


train_data_old = gluon.data.DataLoader(
        dataset = gluon.data.vision.MNIST(train=True, transform=transform_old),
        batch_size = batch_size,
        shuffle = True)


for i, (data, label) in enumerate(train_data):
    data = data
    break




# ?nd.transpose
# ?gluon.data.DataLoader



