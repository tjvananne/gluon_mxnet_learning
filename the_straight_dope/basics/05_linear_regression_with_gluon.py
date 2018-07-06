# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 20:42:59 2018

@author: Taylor
"""

# Building linear regression with gluon api


import mxnet as mx
from mxnet import nd, gluon, autograd
mx.random.seed(1)

data_ctx = mx.cpu()
model_ctx = mx.cpu()


num_inputs = 2
num_outputs = 1
num_examples = 10000

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2


# what are the defaults for loc and scale? 0 and 1?
# is the comma in shape of noise necessary?
# do we need to define the context for these?
X = nd.random_normal(shape=(num_examples, num_inputs))
noise = nd.random_normal(shape=(num_examples,))
y = real_fn(X) + noise

print(X.shape)
print(noise.shape)

# did we need to specify the data context?
print(X.context)
print(noise.context)


# set up data array and data loader
batch_size = 4
array_dataset = gluon.data.ArrayDataset(X, y)
train_data = gluon.data.DataLoader(array_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)

# Now this is where we'll use gluon instead of raw MXNEt

# ?gluon.nn.Dense
# units is dimensionality of output space
# in_units is shape/size of input space
# in_units will be inferred on first forward pass
# if left blank
net = gluon.nn.Dense(1, in_units=2)

#  weights and bias are set up for us magically!
print(net.weight)
print(net.bias)


# we can collect all of the parameters of the network like this
net.collect_params()

# this returns a dictionary of the parameters (key-value pairs)
type(net.collect_params())

# it is important to note that parameters haven't been
# initialized yet. They just have a place in memory and 
# that is all.

# if we try invoking the model, we'll get an error
net(nd.array([[0, 1]]))


# we haven't told gluon what the initial values for the 
# parameters should be

net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

# paremeters have now been given an initializer, BUT the
# network won't actually initialize values for them until
# the network is called to make a forward pass.
net.weight.data()
net.bias.data()

# I don't think these above commands were supposed to work
# but they did...


# notice the difference between these manually generated NDArrays
nd.array([4, 7]).shape    # (2,)
nd.array([[4, 7]]).shape  # (1,2)

# the second one from above is what we need to pass data through the model
# this is just an example of a simple forward pass...
example_data = nd.array([[4, 7]])
net(example_data)  
#  you just call the net like it's a function and it will
# assume you want to execute a forward pass with that data

# Now we can accss these values (even though we technically could before...)
print(net.weight.data())
print(net.bias.data())


# we didn't have to specify the input shape of this dense network...
# let's see what that would look like to let gluon infer that info
net = gluon.nn.Dense(1)
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)


# define the loss we want to use..
square_loss = gluon.loss.L2Loss()
# L2 is most common (squared error basically)
# L1 is next most common (more robust against outliers - absolute error)


# optimizer - we don't have to write our own stochastic gradient
# descent function - just assign it to a gluon.Trainer
# ?gluon.Trainer
# http://mxnet.io/api/python/optimization/optimization.html#the-mxnet-optimizer-package

opti_params = {'learning_rate': 0.0001}
trainer = gluon.Trainer(params=net.collect_params(), 
                        optimizer='sgd',
                        optimizer_params=opti_params)

# ?trainer.step - just takes batch_size


# set up other hyper parameters for training loop
epochs = 10
loss_sequence = []
num_batches = num_examples / batch_size

for e in range(epochs):
    cumulative_loss = 0
    #inner loop
    # this for loop below will run the whole data set
    # every time (epoch)
    # it will run 2500 batches of 4 observations (10,000 total)
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        # record the forward pass and loss calc so we can back prop
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward() # this updates parameter gradients
        trainer.step(batch_size)
        # if you don't do asscalar, you'll get an NDArray instead...
        cumulative_loss += nd.mean(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss / num_batches))
    loss_sequence.append(cumulative_loss)


import matplotlib.pyplot as plt

plt.figure(num=None, figsize=(8,6))
plt.plot(loss_sequence)

# adding some bells and whistles
plt.grid(True, which="both")
plt.xlabel('epoch', fontsize=14)
plt.ylabel('average loss', fontsize=14)


loss_sequence



# accessing the learned parameters (weights / biases)
params = net.collect_params()

print('The type of "params" is a ', type(params))
type(params.values())
print(params.values())

for param in params.values():
    print(param.name, param.data())





"""
Notes:
    
    1. start with your data you want to train on
    2. build your DataLoader with a ArrayData - this will
    define the batch size of your data so you can use
    the enumerate() function to capture the correct
    number of X/y observations during training
    3. build the network with # inputs / # outputs and
    the number of layers desired
    4. set up the initializer for parameters
    <net>.collect_params().initialize(mx.init.Normal(1.), context=<context>)
    5. set up the loss function: gluon.loss.L2loss
    6. set up the trainer/learner/optimizer
    gluon.Trainer(params, optimizer, optimizer_params):
        1. the params are the output of <net>.collect_params()
        2. the optimizer is a character string representing the algo ('sgd')
        3. the optimizer params are parameters for the optimize algo (learning rate)
    7. set up the training loop:
        1. loop through each epoch
        2. enumerate the DataLoader and move data/label into model context
        3. begin recording for autograd
        4. forward pass + loss calculation
        5. OUTSIDE OF RECORDING - <loss function>.backward()
        6. Trainer.step - pass in the batch size for proper step
"""



