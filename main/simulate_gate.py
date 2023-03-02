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
from src.gates.experiments import x_gate_experiment, cnot_gate_experiment, simulate_gate
from src.gates.utilities import (
    construct_x_gate_args,
    construct_cnot_gate_args,
)


def main():

    # Configuration
    run = "single_gate_high_statistics"
    config = load_config(f"gates/{run}.json")
    content = config["content"]
    phi = 0.0
    pulse_lookup = gaussian_pulse_lookup

    # Load device parameters
    device_param = dpl.device_param_lookup_20221208
    assert device_param["version"] == content["device_parameters"], \
        "Wrong device parameter version."

    # Construct gate parameters
    gate_param_x = construct_x_gate_args(device_param, noise_scaling=content["noise_scaling"], phi=phi)
    gate_param_cnot = construct_cnot_gate_args(device_param, noise_scaling=content["noise_scaling"],
                                               phi_ctr=phi, phi_trg=phi)

    # Run X Gate
    simulate_gate(
        run=run,
        gate_parameters=gate_param_x,
        samples=content["samples"],
        runs=content["runs"],
        pulse_lookup=pulse_lookup,
        simulation=x_gate_experiment,
        prefix="x"
    )
    # Run CNOT gate
    simulate_gate(
        run=run,
        gate_parameters=gate_param_cnot,
        samples=content["samples"],
        runs=content["runs"],
        pulse_lookup=pulse_lookup,
        simulation=cnot_gate_experiment,
        prefix="cnot"
    )

    # Save configurations
    with open(f"results/gates/{run}/{run}.json", 'w', encoding='utf8') as file:
        json.dump(config, file, indent=6)


if __name__ == "__main__":
    main()
