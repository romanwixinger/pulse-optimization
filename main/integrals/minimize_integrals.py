""" Minimizes the Ito integrals by finding the optimal coefficient of the parametrized pulses.
"""

import logging

from quantum_gates.utilities import load_config

from pulse_opt.integrals.utilities import (
    load_function_or_class,
    create_table,
    setup_logging,
    run_with_multiprocessing,
    run_without_multiprocessing,
)
from pulse_opt.configuration.argument_constructor import construct_args

logger = logging.getLogger()


def main(run: str, use_multiprocessing: bool=True):
    """Executes all computations for a run of the experiment.

    Saves the results as many json files, as well as a pandas dataframe in the form of a pickle object and a csv file.

    Note:
        This function assumes the existence of a configuration file "integrals/{run}.json".

    Args:
        run (str): Name of the run.
        use_multiprocessing (bool): Should the simulations be executed in parallel.

    Returns:
        List of the results of the optimizations.
    """

    # Configuration
    config = load_config(f"integrals/{run}.json")
    content = config["content"]

    # Set up logging
    setup_logging(run)

    # Load functions and classes dynamically
    loss_class = load_function_or_class(module_name=content["loss_path"], name=content["loss"])
    optimizer = load_function_or_class(
        module_name="pulse_opt.integrals.optimizers",
        name="optimize_with_hard_constraint"
    )

    # Setup arguments

    # Execute simulation
    runner = run_with_multiprocessing if use_multiprocessing else run_without_multiprocessing
    results = runner(simulation=simulation, items=items, config=config)

    # Convert to dataframe
    df = create_table(results=results, config=config)

    # Save dataframe
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
    try:
        res = optimizer(loss=loss, start_coeff=start_coeff, constraints=constraints, bounds=bounds)
    except RuntimeError as e:
        logger.error(f"Encountered RunTimeError: {e}")
        return {"res": None, "loss_arg": loss_arg, "successful": False}
    except Exception as e:
        logger.error(f"Encountered Exception: {e}")
        return {"res": None, "loss_arg": loss_arg, "successful": False}
    result_lookup = {"res": res, "loss_arg": loss_arg, "successful": True}
    return result_lookup


if __name__ == "__main__":

    runs = [
        'power_test',
        'fourier_test',
        'gaussian_test',
    ]

    for i, run in enumerate(runs):
        logger.info(f"Start run with {run} configuration.")
