# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 16:45:30 2018

@author: Taylor
"""



"""
Notes: 
    Multiclass logistic regression ==
    softmax regression ==
    multinomial regression
    
    if we train with batch size == 1:
        z = xW + b
        
        z will have dimension 1 x k
        x will have dimension 1 x d
        W will have dimension d x k
        b will have dimension 1 x k
        
        (k is number of outputs)
        (d is number of inputs)
        
    if we train with batch_size == m:
        
        Z = XW + B
        
        Z will have dimension m x k
        X will have dimension m x d
        W will have dimension d x k
        B will have dimension m x k
        
        (m is the number of training examples per batch)
        
    Note we're still talking about a "dense" two layer network
    The first layer (inputs) fully connects to the second
    layer (the output).
    Every single node of the input connects to every single
    node of the output - hence "dense."
    
    we'll need the softmax function to map all of the
    outputs to sum to a probability of 1:
        
        softmax(z) = e^z  /  for i:k sum all e^zi
        
    Let's do it.
"""



import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)

# set data and model contexts up
data_ctx = mx.cpu()
model_ctx = mx.cpu()

# we'll be using MNIST for this
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test  = gluon.data.vision.MNIST(train=False, transform=transform)

type(mnist_train)

# two parts of the data set for training and testing
# each part has N items and each item is a tuple of an 
# image and a label
image, label = mnist_train[0]
print(image.shape, label)

# for color images, shape would be (28, 28, 3) - but we 
# only have one dimension for color since these are black and white


# record data and label shapes
num_inputs = 784
num_outputs = 10
num_examples = 60000


# two fully connected layers
# input: 784 row by 1 column
# output: 10 row by 1 column


# ML libraries generally expect to find images in
# (batch, channel, height, width) format but libraries
# for visualization prefer (height, width, channel)

# need to transpose our image into the expected shape. In
# this case, matplotlib expects either (height, width) or
# (height, width, channel) with RGB channels, so let's 
# broadcast our single channel to 3.

# ?mx.nd.tile

# mx.nd.tile repeats the array a certain number of times
# to map to the new dimensions...
# so it was a (28, 28, 1) and when we tile it with (1, 1, 3)
# it becomes a (28, 28, 3) because it repeats that (28, 28, 1)
# shape 3 times
im = mx.nd.tile(image, (1,1,3))
print(im.shape)


import matplotlib.pyplot as plt
# ?plt.imshow
plt.imshow(im.asnumpy())
plt.show()



# create the data loader
batch_size = 64
train_data = mx.gluon.data.DataLoader(
        mx.gluon.data.ArrayDataset(mnist_train),  # label already built in?
        batch_size=batch_size,
        shuffle=True)

test_data = mx.gluon.data.DataLoader(
        mx.gluon.data.ArrayDataset(mnist_test),
        batch_size=batch_size,
        shuffle=True)


# allocate model parameters - this is the "from scratch" piece
W = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
b = nd.random_normal(shape=num_outputs, ctx=model_ctx)
# b = nd.random_normal(shape=(num_outputs,1), ctx=model_ctx)  # what I would have done


print(W.shape, b.shape) 
# i would have thought b.shape would need to be 1, 10...

# combine into list
params = [W, b]


# attach our gradient
for param in params:
    param.attach_grad()


# another refresher on what reshape((-1,1)) does
tjv_example = mx.nd.array([1, 2, 3, 4])
print(tjv_example, tjv_example.shape)
tjv_example = tjv_example.reshape((-1, 1))
print(tjv_example, tjv_example.shape)


# try a few examples of different axis in nd.max
tjv_example = nd.arange(9).reshape((3, 3))
print(tjv_example)
print(nd.max(tjv_example, axis=1))  # max per row
print(nd.max(tjv_example, axis=0))  # max per col 

# ?nd.max

# define the softmax implementation to squash the sum of
# all class probabilities into 1 for each predicted example
# I don't fully understand this code yet...
def softmax(y_linear):
    # assuming axis=1 means by row
    exp = nd.exp(y_linear - nd.max(y_linear, axis=1).reshape((-1,1)))
    norms = nd.sum(exp, axis=1).reshape((-1, 1))
    return exp / norms


# deconstructing the softmax function for my understanding
sample_y_linear = nd.arange(10).reshape(shape=(2, 5))
print(sample_y_linear)
sample_yhat = softmax(sample_y_linear)
print(sample_yhat)


max_class_value = nd.max(sample_y_linear, axis=1).reshape((-1,1))
max_now_zero = sample_y_linear - max_class_value
exp_max_zero = nd.exp(max_now_zero)
print(exp_max_zero)
norms_example = nd.sum(exp_max_zero, axis=1).reshape((-1,1))
norms_example
print(exp_max_zero / norms_example)
sample_y_linear


print(nd.sum(sample_yhat, axis=1))


# define the model
# basically the definition of a forward pass (prediction)
def net(X):
    y_linear = nd.dot(X, W) + b
    yhat = softmax(y_linear)
    return yhat


# much more time needs to be spent on 
# understanding loss functions and their
# gradients / derivatives
def cross_entropy(yhat, y):
    # can't log a zero - so add a tiny value to it
    return - nd.sum(y * nd.log(yhat+1e-6))

yhat_example = nd.array([0, 0, 0, 1, 0])
y_example    = nd.array([0, 0, 0, 0, 1])

cross_entropy(yhat_example, y_example)


def SGD(params, lr):
    for param in params:
        param[:] = param - (lr * param.grad)



# log loss isn't human friendly for understanding
# how our model is doing. let's define accuracy as well
def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1,784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()


# should be ~0.10 due to randomness
# there are 10 classes
evaluate_accuracy(test_data, net)


# training loop

epochs = 5
learning_rate = 0.005

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        # print(label)
        label_one_hot = nd.one_hot(label, 10)
        # print(label_one_hot)
        
        break
    break
        
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))



"""
Dimensionality is a little confusing still.

I get a little tripped up on whether I want something to
have a dimension/shape of (64,) or (64,1) or (1,64)

I think with more practice and a better understanding
of linear algebra, that will get better with time

I'm going to become a practitioner first and then
work my way through the theory one step at a time...
starting with intuition of loss functions and what
their gradients look like... then deeper knowledge
of linear algebra and things like matrix decomposition.

"""






