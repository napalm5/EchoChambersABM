# EchoChambersABM
Implementation of a model aiming to model social interactions in a group, and in particular investigate the phenomenon of Echo Chambers; it is being written as part of my Master's thesis in Physics.

The model was first introduced in [Sîrbu, A., Loreto, V., Servedio, V.D.P., Tria, F., 2013. Cohesion, consensus and extreme information in opinion dynamics](https://arxiv.org/abs/1302.4872). It features attractive and repulsive interaction between agents and the possibility of introducing and modulating an external influence, both from one or multiple sources. In addition to the paper, here we introduced the possibility of adding agents with peculiar behavior; so far have been implemented agents that have a fixed probability of agreeing or disagreeing with others.

`main.py` launches the simulation, `echocABM/model.py` contains the definitions of the classes that create the model, and `echocABM/functions.py` contains a few simple functions used in the rest of the code, plus some functions that are now deprecated, and will be removed in the future. These deprecated functions are pretty disordered, since I have not polished their code, so please ignore them.

`condor_parser.py`, is the script I used to quickly launch simulations with the resource manager Condor. It is an adaptation of an already existing script beautifully written by Gabriele Bozzola: [https://github.com/Sbozzolo/RNSA/blob/master/condor_parser.py](https://github.com/Sbozzolo/RNSA/blob/master/condor_parser.py)

