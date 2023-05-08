"""Load and visualize the results of the gate experiments.
"""

import numpy as np
import os
import json

from quantum_gates.utilities import load_config
from quantum_gates.gates import NoiseFreeGates

from pulse_opt.gates.visualizations import (
    plot_gates_mean,
    plot_gates_std,
    plot_gates_mean_reverse,
    plot_gates_std_reverse,
    plot_gates_mean_confidence_interval,
    plot_gates_std_confidence_interval,
    plot_hellinger,
)
from pulse_opt.gates.utilities import load_aggregated_results, construct_x_gate_args, construct_cnot_gate_args
from configuration.device_parameters.lookup import device_param_lookup_20221208


def plot(run: str, config: dict):
    """Create all plots of the results.

    Args:
        run (str): Name of the run of the experiment.
        config (dict): Configuration file.
    """

    # Load aggregated results
    result_folder = f"results/gates/{run}"
    agg_results = load_aggregated_results(result_folder)
    x_aggregated = agg_results["x"]
    cnot_aggregated = agg_results["cnot"]

    # Create folder to save plots
    plots_folder = f"plots/gates/{run}"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    # Plot and save first X gate result
    plot_gates_mean(x_aggregated, plots_folder, filename="x_gate_mean.pdf")
    plot_gates_std(x_aggregated, plots_folder, filename="x_gate_std.pdf")
    plot_gates_mean_reverse(x_aggregated, plots_folder, filename="x_gate_mean_reversed.pdf")
    plot_gates_std_reverse(x_aggregated, plots_folder, filename="x_gate_std_reversed.pdf")

    # Plot and save first CNOT gate result
    plot_gates_mean(cnot_aggregated, plots_folder, filename="cnot_gate_mean.pdf")
    plot_gates_std(cnot_aggregated, plots_folder, filename="cnot_gate_std.pdf")
    plot_gates_mean_reverse(cnot_aggregated, plots_folder, filename="cnot_gate_mean_reversed.pdf")
    plot_gates_std_reverse(cnot_aggregated, plots_folder, filename="cnot_gate_std_reversed.pdf")

    # Plot Hellinger((X_noisy psi)**2, (X_noise_free psi)**2)
    x_gate_args = construct_x_gate_args(device_param_lookup_20221208, noise_scaling=1e-12, phi=0.0)
    x_noise_free = NoiseFreeGates().X(**x_gate_args)
    x_noise_free = np.concatenate((x_noise_free.real.reshape(4), x_noise_free.imag.reshape(4)))
    plot_hellinger(
        x_aggregated,
        folder=plots_folder,
        filename="hellinger_distance_cnot_0.pdf",
        noise_free_gate=x_noise_free,
        psi=np.array([1, 0]),
        psi_name="|0>",
        gate_name="X"
    )
    plot_hellinger(
        x_aggregated,
        folder=plots_folder,
        filename="hellinger_distance_cnot_1.pdf",
        noise_free_gate=x_noise_free,
        psi=np.array([0, 1]),
        psi_name="|1>",
        gate_name="X"
    )
    plot_hellinger(
        x_aggregated,
        folder=plots_folder,
        filename="hellinger_distance_cnot_0_plus_1.pdf",
        noise_free_gate=x_noise_free,
        psi=np.array([1, 1]) / np.sqrt(2),
        psi_name="(|1> + |0>)/sqrt(2)",
        gate_name="X"
    )
    # Plot Hellinger((CNOT_noisy psi)**2, (CNOT_noise_free psi)**2)
    cnot_gate_args = construct_cnot_gate_args(device_param_lookup_20221208, noise_scaling=1e-12, phi_ctr=0.0, phi_trg=0.0)
    cnot_noise_free = NoiseFreeGates().CNOT(**cnot_gate_args)
    cnot_noise_free = np.concatenate((cnot_noise_free.real.reshape(16), cnot_noise_free.imag.reshape(16)))
    plot_hellinger(
        cnot_aggregated,
        folder=plots_folder,
        filename="hellinger_distance_cnot_00.pdf",
        noise_free_gate=cnot_noise_free,
        psi=np.array([1, 0, 0, 0]),
        psi_name="|00>",
        gate_name="CNOT"
    )
    plot_hellinger(
        cnot_aggregated,
        folder=plots_folder,
        filename="hellinger_distance_cnot_01.pdf",
        noise_free_gate=cnot_noise_free,
        psi=np.array([0, 1, 0, 0]),
        psi_name="|01>",
        gate_name="CNOT"
    )
    plot_hellinger(
        cnot_aggregated,
        folder=plots_folder,
        filename="hellinger_distance_cnot_10.pdf",
        noise_free_gate=cnot_noise_free,
        psi=np.array([0, 0, 1, 0]),
        psi_name="|10>",
        gate_name="CNOT"
    )
    plot_hellinger(
        cnot_aggregated,
        folder=plots_folder,
        filename="hellinger_distance_cnot_11.pdf",
        noise_free_gate=cnot_noise_free,
        psi=np.array([0, 0, 0, 1]),
        psi_name="|11>",
        gate_name="CNOT"
    )

    # Save configuration
    with open(f"{plots_folder}/{run}.json", 'w', encoding='utf8') as file:
        json.dump(config, file, indent=6)

    return


def plot_conf_intervals(run: str, config: dict):
    """Create just some new plots.

    Args:
        run (str): Name of the run of the experiment.
        config (dict): Configuration file.
    """

    # Load aggregated results
    result_folder = f"results/gates/{run}"
    agg_results = load_aggregated_results(result_folder)
    x_aggregated = agg_results["x"]
    cnot_aggregated = agg_results["cnot"]

    # Create folder to save plots
    plots_folder = f"plots/gates/{run}"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    # Plot and save first X gate result
    plot_gates_mean_confidence_interval(x_aggregated, plots_folder, filename="x_gate_mean_confidence_interval.pdf")
    plot_gates_std_confidence_interval(x_aggregated, plots_folder, filename="x_gate_std_confidence_interval.pdf")

    # Plot and save first CNOT gate result
    plot_gates_mean_confidence_interval(cnot_aggregated, plots_folder, filename="cnot_gate_mean_confidence_interval.pdf")
    plot_gates_std_confidence_interval(cnot_aggregated, plots_folder, filename="cnot_gate_std_confidence_interval.pdf")


    # Save configuration
    with open(f"{plots_folder}/{run}.json", 'w', encoding='utf8') as file:
        json.dump(config, file, indent=6)

    return


def main():
    runs = [
        "single_gate_boosted_1",
        "single_gate_boosted_0.1",
        "single_gate_boosted_10",
        "single_gate_boosted_100",
        "single_gate_boosted_1000"
    ]

    for run in runs:
        config = load_config(f"gates/{run}.json")
        plot_conf_intervals(run=run, config=config)


if __name__ == "__main__":

    main()
