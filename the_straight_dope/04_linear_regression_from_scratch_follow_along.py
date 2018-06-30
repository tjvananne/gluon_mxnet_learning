# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 12:17:19 2018

@author: Taylor
"""


import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)


# specify contexts - this is a good habit apparently
# ctx is short for context
data_ctx = mx.cpu()
model_ctx = mx.cpu()

print(type(data_ctx))
print(data_ctx)



# now let's cook up some fake data
num_inputs = 2
num_outputs = 1
num_examples = 10000

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

X = nd.random_normal(shape=(num_examples, num_inputs), ctx=data_ctx)
print(X)
print(len(X))  # len is always first-axis
print(X.shape)


noise = .1 * nd.random_normal(shape=(num_examples,), ctx=data_ctx)
print(noise)
print(noise.shape)

y = real_fn(X) + noise


import matplotlib.pyplot as plt
plt.scatter(X[:, 1].asnumpy(), y.asnumpy())
plt.show()


# setting up data iterators
batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
        batch_size=batch_size, shuffle=True)

# DataLoader objects don't have many attributes or methods...
print(train_data)
print(type(train_data))


# we can treat DataLoader objects much like any other
# iterable object in python like a list
# that means we can use for loops and enumerate
for i, (data, label) in enumerate(train_data):
    print(i)
    # print(data)
    # print(type(data))
    print(label)
    print(label.shape)
    print(type(label))
    print(label.reshape((-1, 1)))
    print(label.reshape((-1, 1)).shape)
    print(label.T)
    break


tjv = nd.arange(10)
tjv
tjv.reshape((2, 5))
tjv.reshape((-1, 3))
tjv.reshape((1,3))
tjv.reshape((3, -1))
tjv.reshape((3, 1))
tjv.reshape((1, 4))
tjv.reshape((-1, 4))


# we can count the number of batches by setting up a counter
counter = 0  # counter does nothing?
for i, (data, label) in enumerate(train_data):
    pass
print(i + 1)



# now let's allocate memory for parameters and set
# their initial values...
# this should be done on the model_ctx context
w = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
b = nd.random_normal(shape=num_outputs, ctx=model_ctx)

print(w)
print(b)

params = [w, b]

# these reference the same object...
print(id(w))
print(id(params[0]))


# params is just a list to wrap them together


# we're going to update these parameters to better fit
# the data. This involves taking the gradient (a multi-
# dimensional derivative) of some loss function with 
# respect to the parameters. Then we'll update each
# parameter in the direction that reduces the loss

# for now, lets allocate some memory for each gradient
for param in params:
    param.attach_grad()


# our net is basically just the definition of a linear model
def net(X):
    return mx.nd.dot(X, w) + b


# it would be xTw if x were a one-dimensional vector
# review linear regression section again if that doesn't
# make sense
    

# now we should define the loss function
def square_loss(yhat, y):
    return nd.mean((yhat - y) ** 2)


# Stochastic gradient descent
# this is our learner that will update parameters
# based on the gradient of the loss function
def SGD(params, lr):
    # "simultaneous update"
    for param in params:
        param[:] = param - lr * param.grad


print(param[0].grad)
print(type(param[0].grad))


epochs = 10
learning_rate = .0001
num_batches = num_examples/batch_size



# e = 0-9 loop
for e in range(epochs):
    print(e)
    cumulative_loss = 0
    # inner loop
    # i = 0-2499 loop
    for i, (data, label) in enumerate(train_data):
        if i % 1000 == 0: 
            print(i)
            # print("length of label: ", len(label))
        data = data.as_in_context(model_ctx)
        # reshape((-1, 1)) basically melts/casts this into a column vector
        # makes the column equal to 1 instead of blank... 
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        if i % 1000 == 0: 
            print("b.grad before: ", b.grad)
        with autograd.record():
            # calculate the output of the net with current weight/bias params
            output = net(data)
            # calculate the loss based on output and label comparison
            loss = square_loss(output, label)
            # print("type of output: ", type(output))
            # print("type of loss: ", type(loss))
        # backward function 
        if i % 1000 == 0: 
            print("b.grad after: ", b.grad)
        loss.backward()  # this is what calculates the gradients
        SGD(params, learning_rate)
        cumulative_loss += loss.asscalar()
    print("current cumulative loss / num_batch: ", cumulative_loss / num_batches)
    
    
    
    
print(w)
print(w.grad)
print(type(loss))
?loss.backward
?autograd.record



# training loop with no comments (clean)
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




# training loop with no debug steps but all comments
for e in range(epochs):
    cumulative_loss = 0
    # inner loop
    for i, (data, label) in enumerate(train_data):
        # processing four rows of X and four labels at a time
        # move those data we're enumerating into the model context
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        # forward pass through the network with autograd recording
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        # back propogation to recalculate all of the gradients of
        # the parameters that have autograd attached...
        # confusion on whether this belongs in the "with" block above.
        loss.backward()
        # so now w.grad (two parameters)
        # and the b.grad (single parameter) 
        # are updated...
        
        # this is where we use the param.grad to slowly change
        # the parameters based on the learning rate
        # SGD updates the parameters IN-PLACE!
        SGD(params, learning_rate)
        cumulative_loss += loss.asscalar()
    print(cumulative_loss / num_batches)








