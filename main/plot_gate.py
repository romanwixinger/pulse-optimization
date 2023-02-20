"""
Load and visualize the results of the gate experiments.
"""



import os
import json

from quantum_gates.utilities import load_config

import configuration.device_parameters.lookup as dpl
from src.gates.visualizations import (
    plot_gates_mean,
    plot_gates_standard_deviation,
    plot_gates_mean_reverse,
    plot_gates_std_reverse
)
from src.gates.utilities import load_results, load_aggregated_results


def plot(run: str, config: dict):
    """ Create all plots of the results. """

    # Load results
    result_folder = f"results/gates/{run}"
    results = load_results(result_folder)
    x_results = results["x"]
    cnot_results = results["cnot"]

    # Load aggregated results
    agg_results = load_aggregated_results(result_folder)
    x_aggregated = agg_results["x"]
    cnot_aggregated = agg_results["cnot"]

    # Create folder to save plots
    plots_folder = f"plots/gates/{run}"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    # Plot and save first X gate result
    plot_gates_mean(x_aggregated, plots_folder, filename="x_gate_mean.png")
    plot_gates_standard_deviation(x_aggregated, plots_folder, filename="x_gate_std.png")
    plot_gates_mean_reverse(x_aggregated, plots_folder, filename="x_gate_mean_reversed.png")
    plot_gates_std_reverse(x_aggregated, plots_folder, filename="x_gate_std_reversed.png")

    # Plot and save first CNOT gate result
    plot_gates_mean(cnot_aggregated, plots_folder, filename="cnot_gate_mean.png")
    plot_gates_standard_deviation(cnot_aggregated, plots_folder, filename="cnot_gate_std.png")
    plot_gates_mean_reverse(cnot_aggregated, plots_folder, filename="cnot_gate_mean_reversed.png")
    plot_gates_std_reverse(cnot_aggregated, plots_folder, filename="cnot_gate_std_reversed.png")

    # Save configuration
    with open(f"{plots_folder}/{run}.json", 'w', encoding='utf8') as file:
        json.dump(config, file, indent=6)


if __name__ == "__main__":

    # Load configuration
    run_ = "single_gate_high_statistics"
    config_ = load_config(f"gates/{run_}.json")

    # Plot
    plot(run=run_, config=config_)
