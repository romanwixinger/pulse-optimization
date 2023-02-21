"""
Experiment on level gate. Generates the results which are to be visualized.
"""

import os
import time
import numpy as np

from quantum_gates.gates import Gates, NoiseFreeGates


def x_gate_experiment(args):
    """ Wrapper to the _x_gate_experiment() function to be able to call the function with a single argument.
        This is necessary for using the multiprocesing.map_unordered method. 
    """
    return _x_gate_experiment(**args)


def cnot_gate_experiment(args):
    """ Wrapper to the _cnot_gate_experiment() function to be able to call the function with a single argument.
        This is necessary for using the multiprocesing.map_unordered method.
    """
    return _cnot_gate_experiment(**args)


def _x_gate_experiment(pulse_lookup: dict, n: 10000, gate_args: dict):
    """ Samples n X gates each for the pulses defined in the lookup. Uses the noise parameters given in the gate_args
        lookup.

        Returns a lookup table with the same keys as the pulse_lookup and the results (mean, std, unc. of the mean)
        as value, stored as lookup.
    """
    # Set random seed, otherwise each experiment gets the same result
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    # Generate the gate factories.
    gate_factory_lookup = dict()
    for name, pulse in pulse_lookup.items():
        # We have to use a default parameter in the lambda, otherwise the expression is evaluated to late and each
        # key gets the same argument.
        gateset_instance = Gates(pulse=pulse)
        gate_factory = lambda gateset=gateset_instance: gateset.X(**gate_args)
        gate_factory_lookup[name] = gate_factory

    # Compute the noise free reference
    noise_free_gate = NoiseFreeGates().X(**gate_args)
    result_lookup = _gate_experiment(gate_factory_lookup, n, noise_free_gate)

    return result_lookup


def _cnot_gate_experiment(pulse_lookup: dict, n: 10000, gate_args: dict):
    """ Samples n SX gates each for the pulses defined in the lookup. Uses the noise parameters given in the arguments.

        Returns a lookup table with the same keys as the pulse_lookup and the results (mean, std, unc. of the mean)
        as value, stored as lookup.
    """
    # Set random seed, otherwise each experiment gets the same result
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    # Generate the gate factories.
    gate_factory_lookup = dict()
    for name, pulse in pulse_lookup.items():
        # We have to use a default parameter in the lambda, otherwise the expression is evaluated to late and each
        # key gets the same argument.
        gateset_instance = Gates(pulse=pulse)
        gate_factory = lambda gateset=gateset_instance: gateset.CNOT(**gate_args)
        gate_factory_lookup[name] = gate_factory

    # Compute the noise free reference
    noise_free_gate = NoiseFreeGates().CNOT(**gate_args)
    result_lookup = _gate_experiment(gate_factory_lookup, n, noise_free_gate)

    return result_lookup


def _gate_experiment(gate_factory_lookup: dict,
                    n: int,
                    reference_gate: np.array) -> dict:
    """ Sample n gates each with gate factories provided in a lookup. Subtracts a reference matrix from the result if
        provided.

        Returns a lookup of the results, with keys being lookups of the form
            mean: np.array with mean of the sampled gates                               Mean of the population
            std: np.array with standard deviation of the sampled gates                  Empirical standard deviation of the population
            std_sqrt(n): np.array with uncertainty of the mean of the sampled gates     Empirical uncertainty of the mean of the population.
    """
    result_lookup = dict()

    # Sample gates and calculate mean, std and uncertainty of the mean
    for name, gate_factory in gate_factory_lookup.items():

        # Sample gates
        sampled_gates = [gate_factory() for i in range(n)]

        # Subtract noise free result
        subtracted_gates = [arr - reference_gate for arr in sampled_gates]

        # Transform complex 2x2 (4x4) arrays to real 8x1 (32x1) vector
        dimension = reference_gate.shape[0]**2
        stacked_gates = [
            np.concatenate((arr.real.reshape(dimension), arr.imag.reshape(dimension))) for arr in subtracted_gates
        ]
        flattened_gates = [arr.reshape(2 * dimension) for arr in stacked_gates]

        # Compute metrics and save in lookup table
        result_lookup[name] = {
            "mean": np.mean(flattened_gates, axis=0),
            "std": np.std(flattened_gates, axis=0),
            "std over sqrt(n)": np.std(flattened_gates, axis=0) / np.sqrt(n)
        }

    return result_lookup

