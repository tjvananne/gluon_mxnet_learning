# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 12:05:45 2018

@author: Taylor
"""

# looking to differentiate the function:
# 2 * (x ** 2)
# 2 * (x^2)




import mxnet as mx
from mxnet import nd, autograd

mx.random.seed(1)

x = nd.array([[1, 2], [3, 4]])



print(x)

# 2x2 matrix looks like this:
# 1  2
# 3  4


print(x[0,:])
print(x[:,1])

print(x.shape)


x.attach_grad()


print(x)


# this is where my issue is... what is autograd.record doing and how
# are we supposed to know how to decompose the function into these pieces?
# where is the link between the calculus theory and how to represent it
# in python code in this framework? 
# I'm going to move on and hopefully some of the "from scratch" deep learning
# tutorials will make this more clear to me.
with autograd.record():
    y = x * 2
    z = y * x
    # y = x * x
    # z = y * x


print(z)

# this part makes more sense... this is the actual function output..
# 2x2 matrix looks like this:
# 2   8
# 18  32    
    
z.backward()


print(x.grad)

# 2x2 matrix
# 4   8
# 12  16


