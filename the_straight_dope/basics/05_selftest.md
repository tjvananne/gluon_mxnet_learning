

# Building linear regression with Gluon API



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

