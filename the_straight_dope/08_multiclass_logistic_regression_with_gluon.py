# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 17:33:58 2018

@author: Taylor
"""


"""
Introducing a few subtle tricks that weren't
covered in the logistic regression with gluon
script
"""

import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np


data_ctx = mx.cpu()
model_ctx = mx.cpu()
# model_ctx = mx.gpu()

model_ctx



batch_size = 64
num_inputs = 784
num_outputs = 10
num_examples = 60000

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

train_data = mx.gluon.data.DataLoader(
        dataset = mx.gluon.data.vision.MNIST(train=True, transform=transform), 
        batch_size = batch_size,
        shuffle = True)

test_data = mx.gluon.data.DataLoader(
        dataset = mx.gluon.data.vision.MNIST(train=False, transform=transform),
        batch_size = batch_size,
        shuffle = True)


# define the net architecture - it will infer input shape
net = gluon.nn.Dense(num_outputs)

# these will fail
net.collect_params()
# net.weight.data()
# net.bias.data()

# call an initializer for the parameters of the network
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)


# set up loss function
# this is why this didn't work last time...
# I was choosing the wrong function
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


# set up an optimizer
trainer = gluon.Trainer(
        params = net.collect_params(),
        optimizer = 'sgd',
        optimizer_params= {'learning_rate': 0.1})



# this is a quick snippet to understand the reshape operator here...
for i, (data, label) in enumerate(train_data):
    data2 = data.as_in_context(model_ctx).reshape((-1, 784))
    break


# (64, 28, 28, 1)  to  (64, 784)
print(data.shape, data2.shape)



# set up evaluation metric - because logloss
# isn't as meaningful as accuracy for humans to read
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data2 = data.as_in_context(model_ctx).reshape((-1, 784))
        break
    break
















