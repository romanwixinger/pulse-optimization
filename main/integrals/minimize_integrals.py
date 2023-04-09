""" Minimizes the Ito integrals by finding the optimal coefficient of the parametrized pulses.
"""

import os
import importlib
import multiprocessing

from quantum_gates.utilities import load_config

from pulse_opt.integrals.optimizers import optimize_with_hard_constraint, optimize_with_penalty_constraint
from pulse_opt.integrals.utilities import create_table, save_result
from pulse_opt.configuration.argument_constructor import construct_args


def main(run: str):
    """Executes all computations for a run of the experiment.

    Saves the results as many json files, as well as a pandas dataframe in the form of a pickle object and a csv file.

    Note:
        This function assumes the existence of a configuration file "integrals/{run}.json".

    Args:
        run (str): Name of the run.

    Returns:
        List of the results of the optimizations.
    """

    # Configuration
    config = load_config(f"integrals/{run}.json")
    content = config["content"]

    # Load loss class
    module_name = content["loss_path"]
    class_name = content["loss"]
    LossClass = getattr(importlib.import_module(module_name), class_name)
    optimizer = optimize_with_hard_constraint

    # Setup arguments
    static_args = content["static_args"]
    variable_args = content["variable_args"]
    loss_arg_list = construct_args(static_args=static_args, variable_args=variable_args)
    items = [(loss_arg, LossClass, optimizer) for loss_arg in loss_arg_list]

    # Setup multiprocessing and execute simulation
    results = []
    processes = max(2, os.cpu_count() // 4)
    with multiprocessing.Pool(processes=processes) as pool:
        result = pool.starmap_async(simulation, items, chunksize=10)
        for res_lookup in result.get():
            save_result(
                res_lookup=res_lookup,
                config=config,
                variable_args=list(config["content"]["variable_args"].keys())
            )
            results.append(res_lookup)

    # Convert to dataframe
    df = create_table(results=results, config=config)

    # Save dataframe
    df.to_csv(f"results/integrals/{run}/results.csv")
    df.to_pickle(f"results/integrals/{run}/results.pkl")

    return results


def simulation(loss_arg, LossClass, optimizer: callable) -> dict:
    """ Function to perform the minimization, will be executed in parallel.

    Creates the loss function, extracts the constraints, bounds and default coefficient of the pulse ansatz. Applies
    the optimization procedure to minimize the loss. Finally, returns the result as lookup.

    Note:
        This function is defined here so that we can use it in the multiprocessing module, see below.

    Args:
        loss_arg (dict): Arguments for setting up an instance of the lossClass.
        LossClass (Loss): Class that represents the loss, containing all information about the pulse, the bounds,
            constraints and starting coefficients for the minimization.
        optimizer (callable): Method to minimize the loss.

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
    res = optimizer(loss=loss, start_coeff=start_coeff, constraints=constraints, bounds=bounds)
    return {"res": res, "loss_arg": loss_arg}


if __name__ == "__main__":

    runs = [
        'power_test',
        'fourier_test',
        'gaussian_test',
        'power_small',
        'fourier_small',
        'gaussian_small',
        'power',
        'fourier',
        'gaussian'
    ]
    all_results = [None for run in runs]

    for i, run in enumerate(runs):
        print(f"Start run with {run} configuration.")
        results = main(run=run)
        all_results[i] = results
