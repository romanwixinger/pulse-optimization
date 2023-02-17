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
from collections import defaultdict

from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.gates import Gates
from quantum_gates.circuits import EfficientCircuit

from configuration.device_parameters.lookup import device_param_lookup_20221208 as device_param_lookup
from src.pulses.pulses import gaussian_pulse_lookup
from src.algorithms.algorithms import n_x_gates
from src.algorithms.experiments import analyze_result_lookup
from src.algorithms.visualizations import plot_result_lookup


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

    # Prepare arguments
    args = {
        "pulse_lookup": gaussian_pulse_lookup,
        "shots": 1000,
        "experiments": 16,
        "circuit_generator": n_x_gates,
        "circuit_generator_args": {"nqubits": 1, "N": 128},
        "device_param": device_param_lookup
    }

    # Run experiment
    result_lookup = main(**args)

    # Analyze result
    y_list, y_std_list = analyze_result_lookup(result_lookup)
    plot_result_lookup(y_list, y_std_list)
