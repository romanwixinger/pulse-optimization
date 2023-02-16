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
import matplotlib.pyplot as plt
from collections import defaultdict

from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.pulses import standard_pulse
from quantum_gates.gates import Gates
from quantum_gates.circuits import EfficientCircuit
from quantum_gates.utilities import DeviceParameters

from src.pulses.pulses import triangle_pulse, sin_squared_pulse, linear_pulse, reversed_linear_pulse, gaussian_pulse_lookup
from src.pulses.visualizations import plot_pulses, plot_parametrizations
from src.algorithms.algorithms import n_x_gates

pulse_lookup = {
    "standard_pulse": standard_pulse,
    "triangle_pulse": triangle_pulse,
    "sin_squared_pulse": sin_squared_pulse,
    "linear_pulse": linear_pulse,
    "reversed_linear_pulse": reversed_linear_pulse
}


def main(pulse_lookup: dict,
         shots: int,
         experiments: int,
         circuit_generator: callable,
         circuit_generator_args: dict,
         device_param: dict):
    """ Executes the experiment.
    """

    result_lookup = defaultdict(list)

    for name, pulse in pulse_lookup.items():
        for i in range(experiments):
            print(f"Simulate {name}")
            run_args = {
                "t_qiskit_circ": circuit_generator(**circuit_generator_args),
                "qubits_layout": [0],
                "psi0": np.array([1.0, 0.0]),
                "shots": shots,
                "device_param": device_param,
                "nqubit": 1
            }

            res = simulation(pulse, run_args)
            result_lookup[name].append(res)
            print(f"Finished {name}: {res}")

    return result_lookup


def simulation(pulse, run_args: dict):
    """ Executes a run of the simulation. """

    # Transform arguments
    gates = Gates(pulse=pulse)

    # Run simulator
    sim = MrAndersonSimulator(gates=gates, CircuitClass=EfficientCircuit)
    p = sim.run(**run_args)

    return p


if __name__ == "__main__":

    # Plot pulses
    print("Plot normal pulses")
    plot_pulses(pulse_lookup, "plots/pulses/normal_pulses.png")
    plot_parametrizations(pulse_lookup, "plots/pulses/normal_parametrizations.png")

    print("Plot gaussian pulses")
    plot_pulses(gaussian_pulse_lookup, "plots/pulses/gaussian_pulse.png")
    plot_parametrizations(gaussian_pulse_lookup, "plots/pulses/gaussian_parametrizations.png")

    # Load configuration
    qubits_layout = [0, 1, 4, 7, 10, 12, 15, 18, 21, 23, 24, 25, 22, 19, 16, 14, 11, 8, 5, 3, 2]
    device_param = DeviceParameters(qubits_layout=qubits_layout)
    device_param.load_from_json(location="configuration/")
    device_param = device_param.__dict__()

    # Prepare arguments
    args = {
        "pulse_lookup": gaussian_pulse_lookup,
        "shots": 1000,
        "experiments": 16,
        "circuit_generator": n_x_gates,
        "circuit_generator_args": {"nqubits": 1, "N": 128},
        "device_param": device_param
    }

    # Run experiment
    result_lookup = main(**args)

    # Analyze result
    y_list, y_std_list = analyze_result_lookup(result_lookup)
