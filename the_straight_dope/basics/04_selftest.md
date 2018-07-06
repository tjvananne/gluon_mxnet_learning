

# Linear regression from scratch


This is where the self-learning gets really interesting/frustrating...


I have already built up some knowledge-debt since I didn't fully understand
how the calculus was represented in the autograd.record() and z.backward()
steps of the last chapter on autograd. I'm moving forward regardless to
hopefully gain a better understanding through the "from-scratch" tutorials.


1. set up all of the functions to fake the data (peek for this)
2. construct the DataLoader with gluon
3. build the function to represent the network (raw NDArray pieces...)
4. build the function to represent the loss function that will be differentiated
5. build the stochastic gradient descent function that will update parameters in place
6. initialize the weights and bias using random normal distribution (model context)
7. attach the gradient to the parameters to be used for autograd and backward()
8. set up hyper parameters (# of epochs, learning rate, and number of batches)
9. write the training loop


