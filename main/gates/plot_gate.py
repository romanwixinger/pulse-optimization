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
from pulse_opt.gates.utilities import (
    load_aggregated_results,
    construct_x_gate_args,
    construct_cnot_gate_args,
    construct_cr_gate_args,
)
from configuration.device_parameters.lookup import device_param_lookup


def plot(run: str, config: dict):
    """Create all plots of the results.

    Args:
        run (str): Name of the run of the experiment.
        config (dict): Configuration file.
    """

    # Load aggregated results
    result_folder = f"results/gates/{run}"
    agg_results = load_aggregated_results(result_folder)
    device_param_name = config["content"]["device_parameters"]
    device_param = device_param_lookup[device_param_name]

    # Create folder to save plots
    plots_folder = f"plots/gates/{run}"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    # Plot for each gate
    for gate_name in config["content"]["gates"]:

        # Load results and construct noise free gate
        aggregated = agg_results[gate_name]

        # Construct noise free gate
        noise_free = construct_noise_free_reference(gate_name=gate_name.lower(), device_param=device_param)

        # Matrix elements
        plot_gates_mean(
            result_lookup=aggregated,
            folder=plots_folder,
            filename=f"{gate_name.lower()}_gate_mean.pdf",
            reference_gate=noise_free,
            gate_name=gate_name
        )
        plot_gates_std(
            result_lookup=aggregated,
            folder=plots_folder,
            filename=f"{gate_name.lower()}_gate_std.pdf",
            gate_name=gate_name
        )

        # Matrix elements reversed
        plot_gates_mean_reverse(
            result_lookup=aggregated,
            folder=plots_folder,
            filename=f"{gate_name.lower()}_gate_mean_reverse.pdf",
            reference_gate=noise_free,
            gate_name=gate_name
        )
        plot_gates_std_reverse(
            result_lookup=aggregated,
            folder=plots_folder,
            filename=f"{gate_name.lower()}_gate_std_reverse.pdf",
            gate_name=gate_name
        )

        # Hellinger distance to the ideal case
        noise_free = construct_noise_free_reference(gate_name=gate_name.lower(), device_param=device_param)
        psi_lookup = construct_psi(gate_name.lower())
        for psi_name, psi in psi_lookup.items():
            plot_hellinger(
                result_lookup=aggregated,
                folder=plots_folder,
                noise_free_gate=noise_free,
                psi=psi,
                psi_name=psi_name,
                gate_name=gate_name,
                filename=f"hellinger_{gate_name.lower()}_{psi_name.replace('|', '').replace('>', '')}.pdf")

    # Plot stuff with confidence interval (Todo: Check)
    # plot_conf_intervals(run=run, config=config)

    # Save configuration
    with open(f"{plots_folder}/{run}.json", 'w', encoding='utf8') as file:
        json.dump(config, file, indent=6)


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


def construct_noise_free_reference(gate_name: str, device_param: dict):
    if gate_name == "x":
        x_gate_args = construct_x_gate_args(device_param, noise_scaling=1e-12)
        x_noise_free = NoiseFreeGates().X(**x_gate_args)
        return np.concatenate((x_noise_free.real.reshape(4), x_noise_free.imag.reshape(4)))
    if gate_name == "sx":
        sx_gate_args = construct_x_gate_args(device_param, noise_scaling=1e-12)
        sx_noise_free = NoiseFreeGates().SX(**sx_gate_args)
        return np.concatenate((sx_noise_free.real.reshape(4), sx_noise_free.imag.reshape(4)))
    if gate_name == "cnot":
        cnot_gate_args = construct_cnot_gate_args(device_param_lookup=device_param, noise_scaling=1e-12)
        cnot_noise_free = NoiseFreeGates().CNOT(**cnot_gate_args)
        return np.concatenate((cnot_noise_free.real.reshape(16), cnot_noise_free.imag.reshape(16)))
    if gate_name == "cnot_inv":
        cnot_inv_gate_args = construct_cnot_gate_args(device_param_lookup=device_param, noise_scaling=1e-12)
        cnot_inv_noise_free = NoiseFreeGates().CNOT_inv(**cnot_inv_gate_args)
        return np.concatenate((cnot_inv_noise_free.real.reshape(16), cnot_inv_noise_free.imag.reshape(16)))
    if gate_name == "cr":
        cr_gate_args = construct_cr_gate_args(device_param_lookup=device_param, noise_scaling=1e-12)
        cr_noise_free = NoiseFreeGates().CR(**cr_gate_args)
        return np.concatenate((cr_noise_free.real.reshape(16), cr_noise_free.imag.reshape(16)))
    raise Exception(f"Invalid gate name found: {gate_name}.")


def construct_psi(gate_name: str) -> dict:
    """ Returns a lookup with the name of psi as key (str) and the corresponding array as value (np.array).

    Args:
        gate_name (str): Name of the gate as all lowercase string.
    """
    if gate_name in ["x", "sx"]:
        return {
            "|0>": np.array([1.0, 0.0]),
            "|1>": np.array([0.0, 1.0]),
            "|+>": np.array([1.0, 1.0])/np.sqrt(2),
            "|->": np.array([1.0, -1.0])/np.sqrt(2)
        }
    if gate_name in ["cnot", "cnot_inv", "cr"]:
        return {
            "|00>": np.array([1.0, 0.0, 0.0, 0.0]),
            "|11>": np.array([0.0, 0.0, 0.0, 1.0]),
        }
    raise Exception(f"Invalid gate name found: {gate_name}.")


def main():
    runs = [
        "test_configuration"
    ]

    for run in runs:
        config = load_config(f"gates/{run}.json")
        plot(run, config=config)


if __name__ == "__main__":

    main()
