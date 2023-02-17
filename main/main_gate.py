"""
We compute the expectation value of the gates for various pulse shapes to understand
the effect of the pulses.

Note:
- Just sampling a single gate has the advantage that we do not have an accumulation of
  effects. Moreover, it is much cheaper.
"""

import os
import json

from quantum_gates.utilities import load_config

import configuration.device_parameters.lookup as dpl
from src.pulses.pulses import gaussian_pulse_lookup
from src.gates.experiments import x_gate_experiment, cnot_gate_experiment
from src.gates.visualizations import plot_gates_mean, plot_gates_standard_deviation, plot_gates_mean_reverse
from src.gates.utilities import construct_x_gate_args, construct_cnot_gate_args


if __name__ == "__main__":

    # Load configuration
    run = "single_gate_high_statistics"
    config = load_config(f"configuration/gates/{run}.json")
    version = config["version"]

    # Load device parameters
    device_param_lookup = dpl.device_param_lookup_20221208
    assert device_param_lookup["version"] == config["device_parameters"]

    # Prepare args
    x_gate_args = construct_x_gate_args(device_param_lookup, noise_scaling=1.0, phi=0.0)
    cnot_gate_args = construct_cnot_gate_args(device_param_lookup, noise_scaling=1.0, phi_ctr=0.0, phi_trg=0.0)

    # Perform experiments
    x_result_lookup = x_gate_experiment(
        pulse_lookup=gaussian_pulse_lookup,
        n=config["n"],
        gate_args=x_gate_args
    )
    cnot_result_lookup = cnot_gate_experiment(
        pulse_lookup=gaussian_pulse_lookup,
        n=config["n"],
        gate_args=cnot_gate_args
    )

    # Create folder to save results
    result_folder = f"results/gates/{run}"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Save results: TODO: Implement

    # Create folder to save plots
    plots_folder = f"plots/gates/{run}"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    # Plot and save X gate result
    plot_gates_mean(x_result_lookup, plots_folder, filename="x_gate_mean.png")
    plot_gates_standard_deviation(x_result_lookup, plots_folder, filename="x_gate_std.png")
    plot_gates_mean_reverse(x_result_lookup, plots_folder, filename="cnot_gate_mean_reversed.png")

    # Plot and save CNOT gate result
    plot_gates_mean(cnot_result_lookup, plots_folder, filename="cnot_gate_mean.png")
    plot_gates_standard_deviation(cnot_result_lookup, plots_folder, filename="cnot_gate_std.png")
    plot_gates_mean_reverse(cnot_result_lookup, plots_folder, filename="cnot_gate_mean_reversed.png")

    # Save configurations
    json.dump(config, f"{result_folder}/{run}.json", indent=6)
    json.dump(config, f"{plots_folder}/{run}.json", indent=6)
