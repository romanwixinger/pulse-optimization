""" Compute the expectation value of the gates for various pulse shapes to understand the effect of the pulses.
"""

import json

from quantum_gates.utilities import load_config

from configuration.device_parameters.lookup import device_param_lookup
from pulse_opt.pulses.legacy_pulses import gaussian_pulse_lookup
from pulse_opt.gates.experiments import simulate_gate
from pulse_opt.gates.utilities import gate_args_constructor_lookup
from pulse_opt.gates.factories import factory_class_lookup


def main(run: str):
    """Executes all computations for a run of the experiment.

    Args:
        run (str): Name of the run.
    """

    # Configuration
    config = load_config(f"gates/{run}.json")
    content = config["content"]

    # Load instances
    pulse_lookup = gaussian_pulse_lookup[content["pulse_lookup"]]
    device_param = device_param_lookup[content["device_parameters"]]

    # Construct gate parameters
    gate_args_lookup = {
        name: gate_args_constructor_lookup[name](
            device_param_lookup=device_param,
            noise_scaling=content["noise_scaling"],
            **content["extra_gate_args"][name]
        ) for name in content["gates"]
    }

    # Execute simulations
    for name in content["gates"]:
        print(f"Simulate {name} gate.")
        simulate_gate(
            GateFactoryClass=factory_class_lookup[name],
            gate_args=gate_args_lookup[name],
            pulse_lookup=pulse_lookup,
            run=run,
            samples=content["samples"],
            runs=content["runs"],
            prefix=name,
        )

    # Save configurations
    with open(f"results/gates/{run}/{run}.json", 'w', encoding='utf8') as file:
        json.dump(config, file, indent=6)

    return


if __name__ == "__main__":

    runs = ["test_configuration"]

    for run in runs:
        print(f"Start run with configuration {run}")
        main(run=run)
