# Pulse Optimization with Noisy Quantum Gates
This repository contains a collection of experiments to understand and leverage the effect of the pulse shape on quantum computations. The main tools in this investigation are the [quantum-gates](https://pypi.org/project/quantum-gates/) library for generating the noisy quantum gates for specific pulse shapes, and Qiskit for creating for the quantum circuits, and IBMQ for experiments on real hardware. The aim is to leverage the findings for improving the Noisy Gates approach, enabling device- and circuit-specific optimizations, and find new Quantum Error Mitigation schemes.

## Documentation
Check the latest version of the [documentation](https://pulse-optimization.readthedocs.io/en/latest/) made with Sphinx. 

## How to use
Here we explain how to create and activate a virtual environment for the project, install the dependencies, and validate the installation. 

### Requirements
The Python version should be 3.9 or later. Find your Python version by typing python or python3 in the CLI. We recommend using the repo together with an [IBM Quantum Lab](https://quantum-computing.ibm.com/lab) account, as it necessary for circuit compilation with Qiskit in many cases. Take the `token-template.py`, add your token and save it as `token.py`. Consider using a secrets manager.

### Virtual environment with `virtualenv`
We recommend virtualenv if you use pip as package manager. Open your CLI and check your Python version by starting `python` and closing `quit()` Python. Navigate to this repository. Install virtualenv with `pip install virtualenv` and check the installation using `virtualenv --version`. Create a new virtual environment with `virtualenv venv` as described in this [guide](https://docs.python-guide.org/dev/virtualenvs/) for virtual environments. Activate the environment with `.\venv\Scripts\activate`, and you should see the name `(venv)` displayed in your CLI to the left of your input. Now that we are inside the virtual environment, we can install the dependencies without polluting other projects. We simply type `python -m pip install -r requirements.txt`. Check your installation by starting Python with `python` and import the quantum-gates package with `import quantum_gates`, which is part of the requirements. To deactivate the virtual environment, simply type `deactivate`. 

### Virtual environment with Conda
We recommend Conda if you use it also as package manager. Open the Anaconda prompt and navigate to the repository. Verify that you have the correct Python version, and that you are in the base environment(base). Following the [documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), we create a new virtual environment with `conda env create -f environment.yml`.

### Usage
To get the most out of this repo, we recommend reading introduction of the [initial publication](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.043210), study the experiment descriptions, and look into the [documentation](https://quantum-gates.readthedocs.io/en/latest/index.html) of the [quantum-gates](https://pypi.org/project/quantum-gates/).


## Contributors
This project has been developed thanks to the effort of the following people:

* Roman Wixinger (roman.wixinger@gmail.com)
* Michele Grossi (michele.grossi@cern.ch)
* Giovanni Di Bartolomeo (dibartolomeo.giov@gmail.com)
* Michele Vischi (vischimichele@gmail.com)
