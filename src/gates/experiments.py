"""
Experiment on level gate. Generates the results which are to be visualized.
"""

import numpy as np

from quantum_gates.gates import Gates, NoiseFreeGates


def gate_experiment(T1: float, T2: float, p: float, pulse_lookup: dict, n: 10000) -> dict:
    """ Sample many X gates for specific noise values and pulses. Returns a lookup of the results.
    """
    result_lookup = dict()

    # Sample gates and calculate mean, std and uncertainty of the mean
    for name, pulse in pulse_lookup.items():
        # Create gateset from pulse
        gateset = Gates(pulse=pulse)

        # Sample gates
        sampled_gates = [gateset.X(phi=0, p=p, T1=T1, T2=T2) for i in range(n)]

        # Subtract noise free result
        noise_free_gate = NoiseFreeGates().X(phi=0, p=p, T1=T1, T2=T2)
        subtracted_gates = [arr - noise_free_gate for arr in sampled_gates]

        # Transform complex 2x2 arrays to real 8x1 vector
        stacked_gates = [np.concatenate((arr.real.reshape(4), arr.imag.reshape(4))) for arr in subtracted_gates]
        flattened_gates = [arr.reshape(8) for arr in stacked_gates]

        # Compute metrics
        x_mean = np.mean(flattened_gates, axis=0)
        x_std = np.std(flattened_gates, axis=0)
        x_unc = x_std / np.sqrt(n)

        # Save metrics to lookup table
        result_lookup[name] = {
            "x_mean": x_mean,
            "x_std": x_std,
            "x_unc": x_unc
        }

    return result_lookup

