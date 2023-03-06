"""Experiments on level gate. Generates the results which are to be visualized.
"""

import os
import time
import numpy as np

from src.gates.factories import GateFactory
from src.gates.utilities import (
    perform_parallel_simulation,
    aggregate_results,
    save_results,
    save_aggregated_results,
)


def simulate_gate(GateFactoryClass: type[GateFactory],
                  gate_args: dict,
                  pulse_lookup: dict,
                  run: str,
                  samples: int=10000,
                  runs: int=50,
                  prefix: str="X"):
    """Compute a gate for a specific level of noise and various pulses to a high precision.

    Args:
        GateFactoryClass (type[GateFactory]):
        gate_args (dict): Arguments like theta, T1, p_cnot that are used to construct the gate stored as lookup.
        pulse_lookup (str): Lookup with the pulse name or parametrization value as key and the pulse as value.
        run (str): Name of the current run of the experiment.
        samples (int): Number of gates to be sampled for each run.
        runs (int): Number of repetitions of the experiments.
        prefix (str): Gate name or prefix to be added to the filenames of the results.
    """

    # Prepare arguments
    args = [
        {
            "pulse_lookup": pulse_lookup,
            "GateFactoryClass": GateFactoryClass,
            "gate_args": gate_args,
            "samples": samples,
        } for i in range(runs)
    ]

    print("args", args)

    # Compute in parallel
    results = perform_parallel_simulation(args=args, simulation=_gate_experiment_with_single_argument, max_workers=50)

    # Aggregate results
    aggregated = aggregate_results(results)

    # Create folder to save results
    result_folder = f"results/gates/{run}"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Save results
    save_results(results=results, folder=result_folder, prefix=prefix)
    save_aggregated_results(result=aggregated, folder=result_folder, prefix=prefix)

    return


def _gate_experiment_with_single_argument(args):
    """Wrapper to the _gate_experiment() function to be able to call the function with a single argument.

        The fact that this method has just a single argument is necessary for using the multiprocesing.map_unordered
        method.

    Args:
        args (dict): All the arguments of the gate experiment saved as a lookup, where the key is the argument name
        as str and the value is the argument value in the corresponding type.
    """
    return _gate_experiment(**args)


def _gate_experiment(pulse_lookup: dict,
                     GateFactoryClass: type[GateFactory],
                     gate_args: dict,
                     samples: int=10000):
    """Samples n gates with the GateFactoryClass for each pulses defined in the lookup.

    Uses the noise parameters given in the gate_args lookup.

    Args:
        pulse_lookup (str): Lookup with the pulse name or parametrization value as key and the pulse as value.
        GateFactoryClass (type[GateFactory]):
        gate_args (dict): Arguments like theta, T1, p_cnot that are used to construct the gate stored as lookup.
        run (str): Name of the current run of the experiment.
        samples (int): Number of gates to be sampled for each run.
        runs (int): Number of repetitions of the experiments.
        prefix (str): Gate name or prefix to be added to the filenames of the results.

    Returns:
        A lookup of the results with the name of the pulses as key (str) and with values being lookups itself.
        Each of these lookup has mean, std, and std_sqrt(n) as keys, and arrays as values. More precicely:
            mean: np.array with mean of the sampled gates                               Mean of the population
            std: np.array with standard deviation of the sampled gates                  Empirical standard deviation of the population
            std_sqrt(n): np.array with uncertainty of the mean of the sampled gates     Empirical uncertainty of the mean of the population.
    """
    # Set random seed, otherwise each experiment gets the same result
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    # Generate the results
    result_lookup = dict()
    for name, pulse in pulse_lookup.items():
        # We have to use a default parameter in the lambda, otherwise the expression is evaluated to late and each
        # key gets the same argument.
        gate_factory = GateFactoryClass(pulse=pulse, gate_args=gate_args)

        # Sample gates
        sampled_gates = [gate_factory.construct() for i in range(samples)]

        # Transform complex 2x2 (4x4) arrays to real 8x1 (32x1) vector
        flattened_gates = [_reshape_gate(gate) for gate in sampled_gates]

        # Compute metrics and save in lookup table
        result_lookup[name] = {
            "mean": np.mean(flattened_gates, axis=0),
            "std": np.std(flattened_gates, axis=0),
            "std over sqrt(n)": np.std(flattened_gates, axis=0) / np.sqrt(n)
        }

    return result_lookup


def _reshape_gate(gate: np.array) -> np.array:
    """Takes a complex 2x2 (4x4) array and turns it to a vector with 8 (32) real entries, which represent the real and
        complex part of the array.

        Example input:
            np.array([1, J, 0, 0])

        Example output:
            np.array([1, 0, 0, 0, 0, 1, 0, 0])

        Args:
            gate (np.array): Complex 2x2 (4x4) array

        Returns:
            vector (np.array): Vector with 8 (32) real entries, which represent the real and complex part of the array.
    """
    dim = gate.shape[0]**2
    return np.concatenate((gate.real.reshape(dim), gate.imag.reshape(dim))).reshape(2 * dim)
