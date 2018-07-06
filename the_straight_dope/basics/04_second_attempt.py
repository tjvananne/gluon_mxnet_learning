# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 18:49:16 2018

@author: Taylor
"""


""" 
Ok, this one is working, but the one I tried to code
myself is not working... I'd like to figure out what
I did wrong in the other script because I think that
will be a good learning experience.
"""


from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)


data_ctx = mx.cpu()
model_ctx = mx.cpu()

num_inputs = 2
num_outputs = 1
num_examples = 10000

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

X = nd.random_normal(shape=(num_examples, num_inputs), ctx=data_ctx)
noise = .1 * nd.random_normal(shape=(num_examples,), ctx=data_ctx)
y = real_fn(X) + noise


import matplotlib.pyplot as plt
plt.scatter(X[:, 1].asnumpy(),y.asnumpy())
plt.show()



# construct the DataLoader with gluon
batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                      batch_size=batch_size, shuffle=True)

# build the function to represent the network (raw NDArray pieces...)
def net(X):
    return mx.nd.dot(X, w) + b

# build the function to represent the loss function that will be differentiated
def square_loss(yhat, y):
    return nd.mean((yhat - y) ** 2)

# build the stochastic gradient descent function that will update parameters in place
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
        

# initialize the weights and bias using random normal distribution
w = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
b = nd.random_normal(shape=num_outputs, ctx=model_ctx)
params = [w, b]


# attach the gradient to the parameters to be used for autograd and backward()
for param in params:
    param.attach_grad()
    
    
# set up hyper parameters (# of epochs, learning rate, and number of batches)
epochs = 10
learning_rate = .0001
num_batches = num_examples/batch_size


# write the training loop
for e in range(epochs):
    cumulative_loss = 0
    # inner loop
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += loss.asscalar()
    print(cumulative_loss / num_batches)




