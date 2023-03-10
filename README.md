# Pulse Optimization with Noisy Quantum Gates
This repository contains a collection of experiments to understand and leverage the effect of the pulse shape on quantum computations. The main tools in this investigation are the [quantum-gates](https://pypi.org/project/quantum-gates/) library for generating the noisy quantum gates for specific pulse shapes, and Qiskit for creating for the quantum circuits, and IBMQ for experiments on real hardware. The aim is to leverage the findings for improving the Noisy Gates approach, enabling device- and circuit-specific optimizations, and find new Quantum Error Mitigation schemes.


## Overview
At the moment, we plan to conduct three main experiments to answer the following research question: 

### A) Does the pulse shape in the Noisy Gates approach change the outcome of the simulation? 

On a high level, we simply have to run circuits and see if the result changes according to the pulse shape. More precicely, we will simulate a sequence of gates which are involutions (X, SX, CNOT, H) with varying depth with various pulse shapes (constant, Gaussian, sin**2, linear), and compare the results to the noisy free solution. For parametrizated pulses, we can then plot the matrix elements as a function of the parameters. 

### B) Does the accuracy of simulating real hardware improve when the same pulses are used? 

To answer this question, we repeat the simulation done to answer question A) on real hardware. Then we can set up a matrix $M_ij = H(p_i^{SIM}, p_j^{REAL})$ for pulse shapes i, j which represents the Hellinger distances between simulation results and real hardware for various choices for the pulses on each. In case that the diagonal has lower values than the off-diagonal, this is an indication that the pulses are accurately described in the Noisy Gates approach, and that the resulting differences are at least partly physical.


## How to use
Here we explain how to create and activate a virtual environment for the project, install the dependencies, and validate the installation. 

### Requirements
The Python version should be 3.9 or later. Find your Python version by typing python or python3 in the CLI. We recommend using the repo together with an [IBM Quantum Lab](https://quantum-computing.ibm.com/lab) account, as it necessary for circuit compilation with Qiskit in many cases. 

### Virtual environment with `virtualenv`
We recommend virtualenv if you use pip as package manager. Open your CLI and check your Python version by starting `python` and closing `quit()` Python. Navigate to this repository. Install virtualenv with `pip install virtualenv` and check the installation using `virtualenv --version`. Create a new virtual environment with `virtualenv venv` as described in this [guide](https://docs.python-guide.org/dev/virtualenvs/) for virtual environments. Activate the environment with `.\venv\Scripts\activate`, and you should see the name `(venv)` displayed in your CLI to the left of your input. Now that we are inside the virtual environment, we can install the dependencies without polluting other projects. We simply type `python -m pip install -r requirements.txt`. Check your installation by starting Python with `python` and import the quantum-gates package with `import quantum_gates`, which is part of the requirements. To deactivate the virtual environment, simply type `deactivate`. 

### Virtual environment with Conda
We recommend Conda if you use it also as package manager. Open the Anaconda prompt and navigate to the repository. Verify that you have the correct Python version, and that you are in the base environment(base). Following the [documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), we create a new virtual environment with `conda env create -f environment.yml`.

### Usage
To get the most out of this repo, we recommend reading introduction of the initial publication, study the experiment descriptions, and look into the [documentation](https://quantum-gates.readthedocs.io/en/latest/index.html) of the [quantum-gates](https://pypi.org/project/quantum-gates/).


## Contributors
This project has been developed thanks to the effort of the following people:

* Roman Wixinger (roman.wixinger@gmail.com)
* Michele Grossi (michele.grossi@cern.ch)
* Giovanni Di Bartolomeo (dibartolomeo.giov@gmail.com)
* Michele Vischi (vischimichele@gmail.com)
