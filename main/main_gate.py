"""
We compute the expectation value of the gates for various pulse shapes to understand
the effect of the pulses.

Note:
- Just sampling a single gate has the advantage that we do not have an accumulation of
  effects. Moreover, it is much cheaper.
"""

import os
import json
import numpy as np

from quantum_gates.utilities import load_config

import configuration.device_parameters.lookup as dpl
from src.pulses.pulses import gaussian_pulse_lookup
from src.gates.experiments import x_gate_experiment
from src.gates.experiments import cnot_gate_experiment
from src.gates.visualizations import plot_gates_mean, plot_gates_standard_deviation, plot_gates_mean_reverse
from src.gates.utilities import construct_x_gate_args, construct_cnot_gate_args, perform_parallel_simulation


if __name__ == "__main__":

    # Load configuration
    run = "single_gate_high_statistics"
    config = load_config(f"gates/{run}.json")

    # Load device parameters
    device_param_lookup = dpl.device_param_lookup_20221208
    assert device_param_lookup["version"] == config["content"]["device_parameters"]

    # Prepare args
    x_gate_args = construct_x_gate_args(device_param_lookup, noise_scaling=1.0, phi=0.0)
    cnot_gate_args = construct_cnot_gate_args(device_param_lookup, noise_scaling=1.0, phi_ctr=0.0, phi_trg=0.0)
    x_args = [
        {"pulse_lookup": gaussian_pulse_lookup, "n": config["content"]["samples"], "gate_args": x_gate_args}
        for i in range(config["content"]["runs"])
    ]
    cnot_args = [
        {"pulse_lookup": gaussian_pulse_lookup, "n": config["content"]["samples"], "gate_args": cnot_gate_args}
        for i in range(config["content"]["runs"])
    ]

    x_results = perform_parallel_simulation(args=x_args, simulation=x_gate_experiment, max_workers=2)
    cnot_results = perform_parallel_simulation(args=cnot_args, simulation=cnot_gate_experiment, max_workers=2)

    # Create folder to save results
    result_folder = f"results/gates/{run}"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)


    # Create folder to save plots
    plots_folder = f"plots/gates/{run}"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    # Just plot the first result: TODO: Fix
    x_result_lookup = x_results[0]
    cnot_result_lookup = cnot_results[0]
    plot_gates_mean(x_result_lookup, plots_folder, filename="x_gate_mean.png")
    plot_gates_standard_deviation(x_result_lookup, plots_folder, filename="x_gate_std.png")
    plot_gates_mean_reverse(x_result_lookup, plots_folder, filename="cnot_gate_mean_reversed.png")

    plot_gates_mean(cnot_result_lookup, plots_folder, filename="cnot_gate_mean.png")
    plot_gates_standard_deviation(cnot_result_lookup, plots_folder, filename="cnot_gate_std.png")
    plot_gates_mean_reverse(cnot_result_lookup, plots_folder, filename="cnot_gate_mean_reversed.png")

    # Save configurations
