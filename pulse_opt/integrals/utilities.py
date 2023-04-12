"""Utilities used for characterizing the integrals.

This script defines pulses and makes them available through lookup. The key is either the name of the pulse or the
name plus a parameteter in the case of parametrized pulses.

Attributes:
    integrands (list): List of integrands used in the Ito integrals of the quantum-gates library.

    markers (list): List of matplotlib.pyplot markers used for visualizing the integration values.
"""

import os
import pandas as pd
import json
import logging
import numpy as np
import multiprocessing
import datetime


integrand_lookup = {
    "sin(theta/a)**2": lambda theta, a=1.0: np.sin(theta/a)**2,
    "sin(theta/(2*a))**4": lambda theta, a=1.0: np.sin(theta/(2*a))**4,
    "sin(theta/a)*sin(theta/(2*a))**2": lambda theta, a=1.0: np.sin(theta/a)*np.sin(theta/(2*a))**2,
    "sin(theta/(2*a))**2": lambda theta, a=1.0: np.sin(theta/(2*a))**2,
    "cos(theta/a)**2": lambda theta, a=1.0: np.cos(theta/a)**2,
    "sin(theta/a)*cos(theta/a)": lambda theta, a=1.0: np.sin(theta/a)*np.cos(theta/a),
    "sin(theta/a)":  lambda theta, a=1.0: np.sin(theta/a),
    "cos(theta/(2*a))**2": lambda theta, a=1.0: np.cos(theta/(2*a))**2,
}

integrands = list(integrand_lookup.keys())

markers = [".", "^", "o", "2", "*", "D", "x", "X", "+"]

logger = logging.getLogger()


def flatten_dict(d, parent_key='', sep='.'):
    """Flattens a dictionary that may contain nested dictionaries with recursion.
    """
    if d is None:
        logging.WARN("Encountered d=None in flatten_dict function.")
        return {}
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_table(results: list[dict], config) -> pd.DataFrame:
    """ Create a list of result dicts, flattens them, and return a pandas data frame with the data.

    Args:
        results (list[dict]): List of results of the simulation.
        config (dict): Configuration file loaded in main.
    """

    config_info = {
        'name': config['name'],
        'description': config['description'],
        'loss': config['content']['loss'],
        'loss_path': config['content']['loss_path'],
    }
    flattened_dicts = [
        config_info | flatten_dict(res_lookup['loss_arg']) | flatten_dict(res_lookup['res']) for res_lookup in results
    ]
    return pd.DataFrame(flattened_dicts)


class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder that can deal with np.arrays by converting them to list.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_result(res_lookup: dict, config: dict, variable_args: list):
    """ Saves the result of the simulation in a single json files.

    Uses the information in loss_arg and content to create the filename of the json as follows:
    filename = {loss}_{variable-arg-name}_{variable-arg-value}_..._{variable-arg-name}_{variable-arg-value}.json

    Saves the result in the folder results/integrals/name where the name is the one specified in the config.

    Args:
        res_lookup (dict): Lookup with keys 'res' and 'loss_arg' as produced by the simulation function.
        config (dict): Configuration file.
        variable_args (list[str]): All names of the arguments that are variable in the config.
    """
    # Input validation
    loss_arg = res_lookup["loss_arg"]
    assert all((arg in loss_arg for arg in variable_args)), \
        f"Expected all variable args {variable_args} to come up in the results but found otherwise: {loss_arg}."

    # Construct filename
    loss = config["content"]['loss']
    variable_args_as_key_value_pairs = '_'.join([f'{key}_{loss_arg[key]}' for key in variable_args])
    filename = f"{loss}_{variable_args_as_key_value_pairs}.json"

    # Construct folder
    folder = f"results/integrals/{config['name']}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    subfolder = f"results/integrals/{config['name']}/raw"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # Save json
    with open(f"{subfolder}/{filename}", "w") as f:
        json.dump(res_lookup, f, cls=CustomEncoder)
    return


def setup_logging(run: str):
    """ Sets up log file and prepares the logging module.

    Note:
        The log files are saved at 'logs/integrals/{run}.log'
    """
    # Setup folder
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists("logs/integrals"):
        os.makedirs("logs/integrals")
    if not os.path.exists(f"logs/integrals/{run}"):
        os.makedirs(f"logs/integrals/{run}")

    # Set up logging configuration
    now = datetime.datetime.now()
    logging.basicConfig(
        filename=f"logs/integrals/{run}/{run}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log",
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

    # Start logging
    logger.info("Start logging.")
    return


def run_with_multiprocessing(simulation: callable, items: list, config: dict) -> list:
    """Runs the simulation on the items with parallel processing, and saves each result as json.

    Args:
        simulation (callable): Function with a single argument of the form of an item.
        items (list): List of items to be feed to the simulation.
        config (dict): Additional configurations that are saved.

    Returns:
        List of the results of the simulation.
    """
    logging.info("Start multiprocessing.")
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
    logger.info("Finished optimization.")
    return results


def run_without_multiprocessing(simulation: callable, items: list, config: dict) -> list:
    """Runs the simulation on the items sequentially, and saves each result as json.

    Args:
        simulation (callable): Function with a single argument of the form of an item.
        items (list): List of items to be feed to the simulation.
        config (dict): Additional configurations that are saved.

    Returns:
        List of the results of the simulation.
    """
    logging.info("Start sequential processing.")
    results = []
    for item in items:
        res_lookup = simulation(*item)
        save_result(
            res_lookup=res_lookup,
            config=config,
            variable_args=list(config["content"]["variable_args"].keys())
        )
        results.append(res_lookup)
    logger.info("Finished optimization.")
    return results
