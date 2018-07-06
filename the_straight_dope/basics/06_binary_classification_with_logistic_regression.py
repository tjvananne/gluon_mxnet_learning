# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 13:43:31 2018

@author: Taylor
"""


import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt
from urllib.request import urlopen

def logistic(z):
    # the lower z is, the bigger the denominator
    return 1. / (1. + nd.exp(-z))


# investigating properties of logistic sigmoid
# log loss == cross entropy
x = nd.arange(-5, 5, .1)
y = logistic(x)

plt.plot(x.asnumpy(), y.asnumpy())
plt.show()


data_ctx = mx.cpu()
model_ctx = mx.cpu()

# https://stackoverflow.com/questions/1393324/in-python-given-a-url-to-a-text-file-what-is-the-simplest-way-to-read-the-cont
# https://stackoverflow.com/questions/4981977/how-to-handle-response-encoding-from-urllib-request-urlopen/41231735
train_raw = urlopen("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a").read().decode('utf-8')
print(train_raw)


with open("C:/Users/Taylor/Downloads/a1a.t") as f:
    test_raw = f.read()


type(train_raw)
type(test_raw)


#with open("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a") as f:
#    train_raw = f.read()
#    
#with open("C:/Users/Taylor/Desktop/adult_test.csv") as f:
#    test_raw = f.read()



# circle back around and decompose this to see
# what it's doing to improve on python wrangling
def process_data(raw_data):
    train_lines = raw_data.splitlines()
    num_examples = len(train_lines)
    num_features = 123  # we just had to know this?
    X = nd.zeros((num_examples, num_features), ctx=data_ctx)
    Y = nd.zeros((num_examples, 1), ctx=data_ctx)
    for i, line in enumerate(train_lines):
        tokens = line.split()
        # change label from {-1,1} to {0,1}
        label = (int(tokens[0]) + 1) / 2  
        Y[i] = label
        # start at tokens[1:] because [0] is the label
        for token in tokens[1:]:
        
            # examples "64:1" - the index is 64, so index is equal to
            # the characters of the token UP TO where the last two characters
            # begin (hence the [:-2] == all but the last two characters)
            index = int(token[:-2]) - 1
            X[i, index] = 1
    return X, Y



## procedural version of the function above...
#print(train_raw)
#train_lines = train_raw.splitlines()
#?train_raw.splitlines()
#print(train_lines)
#
## ok so now they are lists.. each line is a list
#train_raw[0:30]    # index here are raw characters within the string
#train_lines[0]     # index here is a whole row
#type(train_lines)  # now we're working with each line as a list of a string
#type(train_lines[0])  # this is a str
#train_lines[0][0:15]
#print(len(train_lines))
#X = nd.zeros((len(train_lines), 123), ctx=data_ctx)
#Y = nd.zeros((len(train_lines), 1), ctx=data_ctx)
#
#for i, line in enumerate(train_lines):
#    # i should give us index; line should give us the line as a str
#    print(i)
#    print(line)
#    tokens = line.split()
#    print(tokens[0])
#    print(tokens[1])
#    break


Xtrain, Ytrain = process_data(train_raw)
Xtest, Ytest = process_data(test_raw)

# inspect the shapes..
print(Xtrain.shape)
print(Ytrain.shape)
print(Xtest.shape)
print(Ytest.shape)

# inspect the breakdown of the target variable
# ~ 24% are 1's and ~76% are 0's
print(nd.sum(Ytrain)/len(Ytrain))
print(nd.sum(Ytest)/len(Ytest))


# Now let's build our data Loader
batch_size = 64
train_data = gluon.data.DataLoader(
        gluon.data.ArrayDataset(Xtrain, Ytrain),
        batch_size = batch_size,
        shuffle=True)
test_data = gluon.data.DataLoader(
        gluon.data.ArrayDataset(Xtest, Ytest),
        batch_size = batch_size,
        shuffle=True)

# set up the network and give the network parameters an initializer
net = gluon.nn.Dense(1)
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

# instantiate the optimizer
# ?gluon.Trainer

trainer = gluon.Trainer(
        params = net.collect_params(),
        optimizer = 'sgd',
        optimizer_params = {'learning_rate': 0.01})


# build log loss function
# not sure why we aren't using this...
# gluon.loss.LogisticLoss
def log_loss(output, y):
    yhat = logistic(output)  # a squashing sigmoid function
    return  - nd.nansum(  y * nd.log(yhat) + (1-y) * nd.log(1-yhat))


# tjv - what happens if we use logistic loss instead?
# log_loss = gluon.loss.LogisticLoss()


# why no num_batches?
epochs = 30
loss_sequence = []
num_examples = len(Xtrain)

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = log_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss))
    loss_sequence.append(cumulative_loss)



# plot the convergence of the estimated loss function
# %matplotlib inline
# import matplotlib
import matplotlib.pyplot as plt

plt.figure(num=None,figsize=(8, 6))
plt.plot(loss_sequence)

# Adding some bells and whistles to the plot
plt.grid(True, which="both")
plt.xlabel('epoch',fontsize=14)
plt.ylabel('average loss',fontsize=14)


num_correct = 0.0
num_total = len(Xtest)
for i, (data, label) in enumerate(test_data):
    data = data.as_in_context(model_ctx)
    label = label.as_in_context(model_ctx)
    output = net(data)
    prediction = (nd.sign(output) + 1) / 2
    num_correct += nd.sum(prediction == label)
print("Accuracy: %0.3f (%s/%s)" % (num_correct.asscalar()/num_total, num_correct.asscalar(), num_total))

