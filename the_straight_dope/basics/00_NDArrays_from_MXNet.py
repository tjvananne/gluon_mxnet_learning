# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 09:57:51 2018

@author: Taylor
"""

# import mxnet
from mxnet import nd



# this is just going to grab random rubbish from memory...
x = nd.empty(shape=(2, 3))
print(x)
print(type(x))


# this gives zeros
x = nd.zeros(shape = (3, 2))
print(x)
print(type(x))


# ones
x = nd.ones(shape = (2, 3))
print(x)


# ?nd.random_normal
# loc = mean
# scale = standard deviation
x = nd.random_normal(loc=5, scale=1, shape=(3, 4))
print(x)



print(x.shape)
print(x.size)



x = nd.random_normal(loc=0, scale=1, shape=(3,4))
print(x)

y = nd.ones((3,4))
print(y)

print(x + y)
print(x * y)



# in place operations
# here's the gist. every time we do an operation like this:

# id() is telling us where in memory the object is saved
print(id(x))
x = x + y
print(id(x))

# the location in memory of x changes. We're allocating memory every
# time we do an operation... in deep learning we'll be doing way too many
# operations to work like this.


# there are workarounds to make sure we're using the same "bucket" in memory

# 1st, there is the slice operator [:] 
print(id(x))
x[:] = x + y
print(id(x))

# notice how the location in memory did not change.. cool!
# however.. we're still creating a temporary memory buffer area for the
# result of (x + y) before it's stored back in the same location as x


# to get around that, we can use some of the built in functions from
# mxnet.nd and specify the "out" parameter to be the variable we're looking
# to update in place
print(id(x))
print(x)
nd.elemwise_add(lhs=x, rhs=y, out=x)
print(id(x))
print(x)

# using the function straight from mxnet.nd is the most efficient (from
# what i've read so far...)


# BROADCASTING ----------------------------------

# how does element wise work when our elements are different dimensions?
x = nd.ones((3, 3))
y = nd.arange(3)

print(x)
print(y)

print(x + y)

# it will default to trying to match the vector to columns of the matrix
# in other words... it's "casting" the y vector to a [1, 3] matrix

y = y.reshape((1, 3))

print(y)
print(y.shape)
print(x.shape)

print(x + y)

# if we specifically force y to have a shape of (3, 1) then it will behave differently
y = y.reshape((3, 1))

print(x + y)

# I call these matching by columns or matching by rows...
# There is a gap in my understanding when it comes to
# "left-most" axis and "right-most" axis...
# I don't understand the linear algebra terminology in programming terms...


# Converting between numpy and MXNet ndarrays ----------------------------

# they are called "ndarrays" in both settings...
print(type(x))
z = x.asnumpy()
print(z)
print(type(z))

# easy enough, this is how to convert numpy into mxnet ndarray
w = nd.array(z)
print(w)
print(type(w))




# Accessing context ------------------------------------------------------

print(x.context)

# Knowledge gap: Is there a way to list out all of the available contexts?
# Also, how do I build/install mxnet to use/access my GPU?

?x.copyto  # i think this creates the copy no matter what
?x.as_in_context # this checks if the thing already exists in the desired context










