""" Runs quantum circuit with scaled versions of the optimized Fourier pulses.

The idea is to use a weighted average of the optimized pulse and constant pulse. This prevents issues with the large peak of the optimized pulse
and we can still observe whether it works better than the default.
"""

import numpy as np
import pandas as pd

from qiskit import pulse
from qiskit.pulse import Waveform

from quantum_gates.utilities import fix_counts, setup_backend, load_config

from pulse_opt.algorithms.algorithms import n_x_gates
from pulse_opt.integrals.utilities import load_pulses
from configuration.token import IBM_TOKEN


def load_optimal_pulse(run: str, theta=3.14):
    """ Loads the optimal pulse (hardcoded) found in a run.

    Todo: Make dynamic.
    Todo: Fix filtering (float precision).
    """
    def filter_function(df: pd.DataFrame) -> pd.Series:
        """ Filters a dataframe of pulses, retrieving the resulting rows as Series.
        """
        df_filtered = df.loc[round(df['args.theta'], 2) == theta]
        filtered_pulses = df_filtered.iloc[[19]]
        return filtered_pulses

    pulses = load_pulses(run=run, filter_function=filter_function, folder_path=None)
    assert len(pulses) == 1, "Expected to just get the most optimal pulse."
    best_pulse = pulses[0]
    optimal_waveform = best_pulse.get_pulse()
    return optimal_waveform


def setup_weighted_pulse(pulse1: callable, pulse2: callable, scaling: float):
    """ Creates the waveform of a weighted combination of two pulses.

    The idea is to use this construction to check how the resulting circuit changes when we smoothly replace one pulse
    waveform with the other.
    """
    weighted_waveform = lambda t: scaling * pulse2(t) + (1 - scaling) * pulse1(t)
    scaled_waveform = lambda s: weighted_waveform(s/160)
    x = np.linspace(0.0, 160.0, 160)
    y = np.array([np.pi/(4*160) * scaled_waveform(s) for s in x])
    drive_pulse = Waveform(y, name='Weighted_Pulse', limit_amplitude=False)
    return drive_pulse

def main_with_args(backend, scaling_options: list[float], lengths: list[int], shots: int, folder: str, run: str):
    """
    Performs runs on real hardware of weighted average of optimized pulse and constant pulse.

    Each scaling between 0 and 1 determines the weight of the optimized pulse. The function saves the results as txt
    files.

    Args:
        backend: IBM real hardware backend.
        scaling_options (list): Each scaling between 0 and 1 determines the weight of the optimized pulse.
        lengths (list): Each length will be tried as a number of X gates in the circuit.
        shots (int): The number of times each circuit is ran.
    """
    optimal_waveform = load_optimal_pulse(run=run, theta=3.14)
    results = []
    for scaling in scaling_options:

        # Setup weighted pulse
        constant = lambda t: 1
        drive_pulse = setup_weighted_pulse(pulse1=constant, pulse2=optimal_waveform, scaling=scaling)

        # Prototype and results
        qc_list = []
        r00_device = []
        r11_device = []

        # Run quantum circuits
        with pulse.build(backend, name='x-gate') as x_q0:
            pulse.play(drive_pulse, pulse.drive_channel(0))

            # Create circuits of different length
            for i in lengths:
                qc = n_x_gates(nqubits=1, N=i, add_barrier=True, add_measurement=True)
                qc.add_calibration('x', [0], x_q0)
                qc_list.append(qc)

            # Run and wait for result
            job = backend.run(qc_list, shots=shots)
            result = job.result()

            # Perform postprocessing
            for i, length in enumerate(lengths):
                counts_0 = result.get_counts(qc_list[i])
                counts = fix_counts(counts_0, 1)
                p_real = [counts[j][1]/shots for j in range(0,2)]
                r00_device.append(p_real[0])
                r11_device.append(p_real[1])

        # Add results
        results.append({"r00": r00_device, "r11": r11_device})

    # Save results in suitable format
    # Todo: Implement

    return


def main(run: str):
    """ Performs runs on real hardware of weighted average of optimized pulse and constant pulse.

    Args:
        run (str): Specifies the configuration that should be used.
    """
    # Extract config
    config = load_config(filename=run)

    # Setup folders
    # Todo: Add.

    # Save configs
    # Todo: Add.

    # Save current commit
    # Todo: Add.

    # Setup backend.
    backend = setup_backend(
        Token=IBM_TOKEN,
        hub=config["hub"],
        group=config["group"],
        project=config["project"],
        device_name=config["device_name"]
    )

    # Perform run.
    main_with_args(
        backend=backend,
        scaling_options=config["scaling_options"],
        lengths=config["lengths"],
        shots=config["shots"],
        folder=...,
        run=run
    )

