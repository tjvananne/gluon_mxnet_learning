# gluon_mxnet_learning

Exploring and learning the mxnet deep learning framework - specifically using the gluon API

## Environment setup


1. Install miniconda
2. `conda create -n gluon35 python=3.5
3. activate gluon35
3. conda install spyder
4. conda install jupyter
5. conda install mxnet
6. conda install cython
7. conda install gluon  (might need to pip this one...)
8. conda install numpy
9. conda install pandas
10. conda install matplotlib
11. conda install seaborn


will probably add to this list later, but this should be plenty for now. 


Because I anticipate having many scripts, I think it will be nicer to use
spyder instead of jupyter. But if I feel like the I'm missing the markdown
capabilities of jupyter, I may switch at some point.



## Tutorials to work through 


https://gluon.mxnet.io/chapter01_crashcourse/preface.html 


Gluon - the straight dope



## Reason for MXNet + Gluon

MXNet claims to be friendly enough for quick-iteration research (much
like Keras) while also being friendly for production deployment. This
framework also plays well with the ONNX (open neural network exchange) that
enables you to "export" model architecture and weights so you can switch
between frameworks. I'm skeptical, but that is something I'd like to
see and experience first-hand. 


There are also plans for the friendly Gluon API wrapper to support
CNTK (Microsoft's cognitive toolkit framework for deep learning).


Plus. Everyone and their dog go the Keras + Tensorflow route. And for whatever
reason, I really don't feel like using a framework from Facebook. So by 
process of elimination, it sounds like MXNet + Gluon (and eventually CNTK) 
are a good fit for my needs. 



## Self-test method


I want to actually learn this material. When I say "learn," I mean a combination
of rote memorization (method names, their arguments, what they accomlish, etc) as
well as a (fairly deep) fundamental understanding of what is happening under
the hood (both from a mathematical perspective as well as a computational
perspective).


* rote memorization equates to learning the alphabet
* fundamental math/computational understanding equates to writing poetry


To accomplish the memorization, I'd like each python script to be accompanied
by a markdown file stating the conceptual task, which I'll later have to 
recall which python classes/methods/arguments fit together to solve that concept.


1. read through the tutorial and take notes of the high-level concepts going on in the "chapter"
2. open a blank script and try to rebuild the examples from scratch
3. get really frustrated
4. go back to the example and strengthen the concept



