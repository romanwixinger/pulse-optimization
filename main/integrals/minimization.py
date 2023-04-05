""" Minimizes the Ito integrals by finding the optimal coefficient of the parametrized pulses.
"""

import os
import importlib
import multiprocessing
import json

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
    items = [(loss_arg, LossClass) for loss_arg in loss_arg_list]

    print("items: ", items)

    # Setup multiprocessing and execute simulation
    processes = max(2, os.cpu_count() // 4)
    with multiprocessing.Pool(processes=processes) as pool:
        print("Started multiprocessing pool. ")
        result = pool.starmap_async(simulation, items, chunksize=10)
        print("Here")

        # Save configurations
        results = []
        for res_lookup in result.get():
            res = res_lookup["res"]
            loss_arg = res_lookup["loss_arg"]
            print(f"For arguments {loss_arg} we receive the result {res}.", flush=True)
            save_result(res_lookup=res_lookup, config=config, variable_args=list(config["content"]["variable_args"].keys()))
            print("Saved results", flush=True)
            results.append(res_lookup)

    return results


def simulation(loss_arg, LossClass) -> dict:
    """ Function to perform the minimization, will be executed in parallel.

    Creates the loss function, extracts the constraints, bounds and default coefficient of the pulse ansatz. Applies
    the optimization procedure to minimize the loss. Finally, returns the result as lookup.

    Note:
        This function is defined here so that we can use it in the multiprocessing module, see below.

    Example:
        .. code:: python

            import multiprocessing

            from main.integrals.minimization import simulation

            if __name__ == '__main__':

                with multiprocessing.Pool(processes=4) as pool:

                    loss_arg1 = los_arg2 = ...
                    args = [(loss_arg1, PowerLoss), (los_arg2, PowerLoss)]
                    result = pool.starmap(simulation, args)
                    print(result)

    Returns:
        A lookup with keys res and loss_arg, which have the optimization result (OptimizeResult) and the arguments
        (dict) as values, respectively.
    """

    loss = LossClass(**loss_arg)
    start_coeff = loss.default_coefficients
    constraints = loss.constraints
    bounds = loss.bounds
    res = optimization(loss=loss, start_coeff=start_coeff, constraints=constraints, bounds=bounds)
    return {"res": res, "loss_arg": loss_arg}


def save_result(res_lookup: dict, config: dict, variable_args: list):
    """ Saves the result of the simulation in a single json files.

    Uses the information in loss_arg and content to create the filename of the json as follows:
    filename = {loss}_{variable-arg-name}_{variable-arg-value}_..._{variable-arg-name}_{variable-arg-value}.json

    Saves the result in the folder results/integrals/name where the name is the one specified in the config.

    Args:
        result_lookup (dict): Lookup with keys res and loss_arg as produced by the simulation function.
        config (dict): Configuration file.
        variable_args (list[str]): All names of the arguments that are variable in the config.
        folder (str): Folder in which the json should be saved.
    """
    # Input validation

    loss_arg = res_lookup["loss_arg"]
    assert all((arg in loss_arg for arg in variable_args)), \
        f"Expected all variable args {variable_args} to come up in the results but found otherwise: {loss_arg}."

    # Construct filename
    loss = config["content"]['loss']
    variable_args_as_key_value_pairs = '_'.join([f'{key}_{loss_arg[key]}' for key in variable_args])
    filename = f"{loss}_{variable_args_as_key_value_pairs}.json"
    print("Filename: ", filename)

    # Construct folder
    folder = f"results/integrals/{config['name']}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save json
    with open(f"{folder}/{filename}", "w") as f:
        json.dump(res_lookup, f)

    return


if __name__ == "__main__":

    runs = ['power_small', 'power', 'fourier', 'gaussian']
    all_results = [None for run in runs]

    for i, run in enumerate(runs):
        print(f"Start run with {run} configuration.")
        results = main(run=run)
        all_results[i] = results
