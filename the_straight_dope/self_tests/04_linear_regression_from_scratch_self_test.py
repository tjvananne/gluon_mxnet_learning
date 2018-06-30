# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 20:50:16 2018

@author: Taylor
"""

"""
1. set up all of the functions to fake the data (peek for this)
2. construct the DataLoader with gluon
3. build the function to represent the network (raw NDArray pieces...)
4. build the function to represent the loss function that will be differentiated
5. build the stochastic gradient descent function that will update parameters in place
6. initialize the weights and bias using random normal distribution (model context)
7. attach the gradient to the parameters to be used for autograd and backward()
8. set up hyper parameters (# of epochs, learning rate, and number of batches)
9. write the training loop

"""

# this is a challenge to see if I learned the material

import mxnet as mx
from mxnet import nd, autograd, gluon
# import gluon
mx.random.seed(1)



data_ctx = mx.cpu()
model_ctx = mx.cpu()

num_inputs = 2
num_outputs = 1
num_examples = 10000

def real_fn(X):
    return  ((2 * X[:, 0]) - (3.4 * X[:, 1]) + 4.2)

fake_X = nd.arange(6).reshape((3,2))
print(fake_X)
print(fake_X[:,0])
real_fn( X = fake_X)

X = nd.random_normal(shape=(num_examples, num_inputs), ctx=data_ctx)
noise = .1 * nd.random_normal(shape=(num_examples,), ctx=data_ctx)
y = real_fn(X) + noise

print(y)

# build the data loader
batch_size = 10
data_array = gluon.data.ArrayDataset(X, y)
train_data = gluon.data.DataLoader(data_array, batch_size=4, shuffle=True)


# function to represent the network
def net(X):
    return nd.dot(X, w) + b


# function to represent the loss function (mean squared error)
def squared_error(yhat, y):
    return nd.mean((yhat - y) ** 2)

squared_error(nd.array([1, 2, 3]), nd.array([1.5, 1.5, 5]))


# function to represent gradient descent that will update parameters in place
def SGD(params, lr):
    for param in params:
        param[:] = param - (lr * param.grad)


# hyper params for the network
number_inputs = 2
number_outputs = 1
w = nd.random_normal(loc=0, scale=1, shape=(number_inputs, number_outputs), ctx=model_ctx)
b = nd.random_normal(loc=0, scale=1, shape=(number_outputs,), ctx=model_ctx)

params = [w, b]


for param in params:
    param.attach_grad()

epochs = 10
learning_rate = 0.0001
num_batches = num_examples / batch_size


for i, (data, label) in enumerate(train_data):
    print(data)
    print(label.reshape((-1, 1)))
    break


print(data)
print(type(data))


"""
Hyper params to tune here would be:
    1. epochs
    2. learning rate
    3. mini-batch size for SGD
    4. number of inputs (if we had more features)
"""



for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        with autograd.record():
            output = net(data)
            loss = squared_error(output, label)
        loss.backward()             # w.grad and b.grad are updated here...
        SGD(params, learning_rate)  # apply that update to w and b simultaneously here...
        cumulative_loss += loss.asscalar()
    print(cumulative_loss / num_batches)












