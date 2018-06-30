

# Progress check-in 00:

I've made it through the first 7 sections 
(00-06) and have a much better understanding
of how modern deep learning frameworks work as
well as some of the theoretical knowledge
for "under the hood."


Current knowledge gaps:

* linear algebra + multivariate calculus
* advanced algebra + basics of exponential algebraic rules


Main pieces of deep learning:

1. the data
2. data structures for accessing the data
    1. DataLoader + ArrayDataset (allows you to enumerate over 
    the data in mini batches)
3. parameters (weights/biases) - these can be defined on 
their own as MDArrays or they automatically be assigned 
if using models from gluon.nn.{model}
4. the model - can be a function that implements mathematical expressions
with NDArrays OR it can be from the gluon.nn.{model} pre-built layers
5. trainer/learner- the algorithm that optimizes the 
parameters (weights/biases)
6. loss function - the function to score the model's current progress
at learning the function to map data to labels/values.
    1. this is also what we take the gradient of to figure out how to
    update the parameters (weights/biases) so we can improve the
    scores in the future on subsequent forward passes
    2. the "backward()" method of the loss function is what handles
    the calculation of the gradient (partial derivative of each param).
    3. the trainer.step(batch_size) is what then updates the parameters
    by subtracting (learning_rate * gradient of param) to get us closer
    to where the parameter needs to be for higher accuracy
    

Weird epiphany:

* the log in logloss simply makes it easier to use an additive
algorithm like stochastic gradient descent. normally, with 
independent probabilities, we'd need to multiply them all 
together, but if we log them all, then we can add them and
have basically the same result. I need to deep-dive the
log-loss (cross-entropy) function a little more to really
understand why/how it's used. 


Confused about:

1. why did we implement our own log loss function instead of using
the one from gluon? When I tried to use the one from gluon in the
tutorial about logistic regression, I got significantly worse scores.
    

