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
from src.gates.experiments import x_gate_experiment
from src.gates.experiments import cnot_gate_experiment
from src.gates.utilities import (
    construct_x_gate_args,
    construct_cnot_gate_args,
    perform_parallel_simulation,
    save_results,
    save_aggregated_results,
    load_results,
    load_aggregated_results,
    aggregate_results
)


def main(run: str,
         config: dict,
         noise_scaling: float,
         pulse_lookup: dict,
         phi: float,
         device_param_lookup: dict):
    """ Compute the X and CNOT gate for a specific level of noise and
    """

    assert device_param_lookup["version"] == config["content"]["device_parameters"]

    # Compute
    x_gate_args = construct_x_gate_args(device_param_lookup, noise_scaling=noise_scaling, phi=phi)
    cnot_gate_args = construct_cnot_gate_args(device_param_lookup, noise_scaling=noise_scaling, phi_ctr=phi, phi_trg=phi)
    x_args = [
        {"pulse_lookup": pulse_lookup, "n": config["content"]["samples"], "gate_args": x_gate_args}
        for i in range(config["content"]["runs"])
    ]
    cnot_args = [
        {"pulse_lookup": pulse_lookup, "n": config["content"]["samples"], "gate_args": cnot_gate_args}
        for i in range(config["content"]["runs"])
    ]

    # Compute in parallel
    x_results = perform_parallel_simulation(args=x_args, simulation=x_gate_experiment, max_workers=50)
    cnot_results = perform_parallel_simulation(args=cnot_args, simulation=cnot_gate_experiment, max_workers=50)

    # Aggregate results
    x_aggregated = aggregate_results(x_results)
    cnot_aggregated = aggregate_results(cnot_results)

    # Create folder to save results
    result_folder = f"results/gates/{run}"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Save results
    save_results(results=x_results, folder=result_folder, prefix="x")
    save_results(results=cnot_results, folder=result_folder, prefix="cnot")
    save_aggregated_results(result=x_aggregated, folder=result_folder, prefix="x")
    save_aggregated_results(result=cnot_aggregated, folder=result_folder, prefix="cnot")

    lookup = load_results(result_folder)
    print("lookup", lookup)

    lookup_agg = load_aggregated_results(result_folder)
    print("lookup_agg", lookup_agg)

    # Save configurations
    with open(f"{result_folder}/{run}.json", 'w', encoding='utf8') as file:
        json.dump(config, file, indent=6)

    return


if __name__ == "__main__":

    # Configuration
    run_ = "single_gate_high_statistics"
    config_ = load_config(f"gates/{run_}.json")
    noise_scaling_ = 1.0
    phi_ = 0.0
    pulse_lookup_ = gaussian_pulse_lookup

    # Load device parameters
    device_param_lookup_ = dpl.device_param_lookup_20221208

    # Run
    main(run=run_,
         config=config_,
         pulse_lookup=pulse_lookup_,
         noise_scaling=noise_scaling_,
         phi=phi_,
         device_param_lookup=device_param_lookup_)
