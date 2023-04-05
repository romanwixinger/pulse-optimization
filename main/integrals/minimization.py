""" Minimizes the Ito integrals by finding the optimal coefficient of the parametrized pulses.
"""

import importlib

from quantum_gates.utilities import load_config

from pulse_opt.integrals.minimization import optimization
from pulse_opt.configuration.argument_constructor import construct_args


def main(run: str):
    """Executes all computations for a run of the experiment.

    Args:
        run (str): Name of the run.
    """

    # Configuration
    config = load_config(f"integrals/{run}.json")
    content = config["content"]

    # Load loss class
    module_name = content["loss_path"]
    class_name = content["loss"]
    LossClass = getattr(importlib.import_module(module_name), class_name)

    # Setup arguments
    static_args = content["static_args"]
    variable_args = content["variable_args"]
    loss_arg_list = construct_args(static_args=static_args, variable_args=variable_args)

    # Execute simulations
    for loss_arg in loss_arg_list:
        loss = LossClass(**loss_arg)
        start_coeff = loss.default_coefficients
        constraints = loss.constraints
        bounds = loss.bounds
        res = optimization(loss=loss, start_coeff=start_coeff, constraints=constraints, bounds=bounds)

    # Save configurations
    pass

    return


if __name__ == "__main__":

    runs = ['power', 'fourier', 'gaussian']

    for run in runs:
        print(f"Start run with {run} configuration.")
        main(run=run)

