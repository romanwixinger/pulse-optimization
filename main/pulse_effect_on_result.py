"""
We run a simulation to answer the following question:
- Does the pulse shape in the Noisy Gates approach change the outcome of the simulation?

Notes:
- On a high level, we run circuits with different pulses and see if the result changes according to the pulse shape.
- We will simulate a sequence of gates which are involutions (X, SX, H, CNOT) with various pulse shapes such as
constant, Gaussian, sin**2, linear, and compare the results to the noisy free solution.
- For parametrizated pulses, we can then plot the probability vector elements as a function of the parameter.
"""

import numpy as np

from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.pulses import Pulse, GaussianPulse, standard_pulse
from quantum_gates.circuits import EfficientCircuit
from quantum_gates.utilities import multiprocessing_parallel_simulation

from src.pulses import triangle_pulse, sin_squared_pulse, linear_pulse, reversed_linear_pulse, gaussian_pulse_lookup
from src.visualizations import plot_pulses, plot_parametrizations


pulse_lookup = {
    "standard_pulse": standard_pulse,
    "triangle_pulse": triangle_pulse,
    "sin_squared_pulse": sin_squared_pulse,
    "linear_pulse": linear_pulse,
    "reversed_linear_pulse": reversed_linear_pulse
}


def main():
    """ Executes the experiment.
    """

    # Setup arguments

    # Maps arguments on simulation

    # Postprocessing

    return None


def simulation():
    """ Executes a run of the simulation. """

    # Transform arguments

    # Run simulator

    # Postprocessing

    return None


if __name__ == "__main__":

    # Plot pulses
    print("Plot normal pulses")
    plot_pulses(pulse_lookup)
    plot_parametrizations(pulse_lookup)

    print("Plot gaussian pulses")
    plot_pulses(gaussian_pulse_lookup)
    plot_parametrizations(gaussian_pulse_lookup)

    print("Finished")

    # Load configuration

    # Prepare arguments
    args = {}

    # Run experiment
    main(**args)
