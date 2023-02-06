# Pulse Optimization with Noisy Quantum Gates
This repository contains a collection of experiments to understand and leverage the effect of the pulse shape on quantum computations. The main tools in this investigation are the [quantum-gates](https://pypi.org/project/quantum-gates/) library for generating the noisy quantum gates for specific pulse shapes, and Qiskit for creating for the quantum circuits, and IBMQ for experiments on real hardware. The aim is to leverage the findings for improving the Noisy Gates approach, enabling device- and circuit-specific optimizations, and find new Quantum Error Mitigation schemes.


## Overview
At the moment, we plan to conduct three main experiments to answer the following research question: 

### A) Does the pulse shape in the Noisy Gates approach change the outcome of the simulation? 

On a high level, we simply have to run circuits and see if the result changes according to the pulse shape. More precicely, we will simulate a sequence of gates which are involutions (X, SX, CNOT, H) with varying depth with various pulse shapes (constant, Gaussian, sin**2, linear), and compare the results to the noisy free solution. For parametrizated pulses, we can then plot the matrix elements as a function of the parameters. 


## How to use
### Requirements
The Python version should be 3.9 or later. Find your Python version by typing python or python3 in the CLI. We recommend using the repo together with an [IBM Quantum Lab](https://quantum-computing.ibm.com/lab) account, as it necessary for circuit compilation with Qiskit in many cases. Install the dependencies with `pip install requirements.txt`. 

### Usage
To get the most out of this repo, we recommend reading introduction of the initial publication, study the experiment descriptions, and look into the [documentation](https://quantum-gates.readthedocs.io/en/latest/index.html) of the [quantum-gates](https://pypi.org/project/quantum-gates/).


## Contributors
This project has been developed thanks to the effort of the following people:

* Roman Wixinger (roman.wixinger@gmail.com)
* Michele Grossi (michele.grossi@cern.ch)
* Giovanni Di Bartolomeo (dibartolomeo.giov@gmail.com)
* Michele Vischi (vischimichele@gmail.com)
