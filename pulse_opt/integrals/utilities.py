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
import importlib


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


def load_function_or_class(module_name: str, name: str):
    """ Imports and returns a lossClass.

    Args:
        module_name (str): Name of the module.
        name (str): Name of the function or class.

    Returns:
        The function or class at {module_name}.{name}.
    """
    obj = getattr(importlib.import_module(module_name), name)
    return obj


def flatten_dict(d, parent_key='', sep='.'):
    """Flattens a dictionary that may contain nested dictionaries with recursion.
    """
    if d is None:
        logging.warning("Encountered d=None in flatten_dict function.")
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

    Note:
        The resulting dataframe reflect the three types of columns 'config', 'args' and 'results' with prefixes.
        Moreover, the nested structure of the dicts is reflected in the column names with '.' as separator.

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
        add_prefix(config_info, "config")
        | add_prefix(flatten_dict(res_lookup['loss_arg']), "args")
        | add_prefix(flatten_dict(res_lookup['res']), "results")
        for res_lookup in results
    ]
    return pd.DataFrame(flattened_dicts)


def add_prefix(flat_lookup: dict, prefix: str):
    """ Adds a prefix to each of the keys of the dictionary.
    """
    assert isinstance(prefix, str), f"Assumed prefix to be of type str but found {type(prefix)}."
    assert all((isinstance(key, str) for key in flat_lookup.keys())), \
        "Assumed keys of lookup to be of type str but found otherwise."
    return {(prefix + key): value for key, value in flat_lookup.items()}


def save_table_as_csv(df: pd.DataFrame, run: str, folder_path: str=None):
    """ Saves data frame representing the result of an integral minimization run as csv file.

    Note:
        The argument folder_path can be used to overwrite the normally expected path deduced from run. It is used for
        testing.

    Args:
        df (pd.DataFrame): Dataframe that should be saved.
        run (str): Name of the run in which the df was generated.
        folder_path (str): Can be used to overwrite the default place to save the csv file.

    Raises:
        Exception if the folder 'results/integrals/{run}' does not exist.
    """
    folder_path = f"results/integrals/{run}" if folder_path is None else folder_path
    if not os.path.exists(folder_path):
        raise Exception(f"Tried to save table to a folder {folder_path} that does not exist.")
    df.to_csv(f"{folder_path}/results.csv")
    return


def load_table_from_csv(run: str, folder_path: str=None) -> pd.DataFrame:
    """ Retrieves the data frame that represents the result of an integral minimization run from a csv file.

    Note:
        The argument folder_path can be used to overwrite the normally expected path deduced from run. It is used for
        testing.

    Args:
        run (str): Name of the run to which we want to get the result.
        folder_path (str): Overwrite the usual path at which we expect the file results.csv.

    Returns:
        The data frame with the results of the run.

    Raises:
        An exception if the file 'results/integrals/{run}/results.csv' does not exist.
    """
    folder_path = f"results/integrals/{run}" if folder_path is None else folder_path
    filepath = f"{folder_path}/results.csv"
    if not os.path.isfile(filepath):
        raise Exception(f"Expected to find the file {filepath} but it was not there.")
    df = pd.read_csv(filepath_or_buffer=filepath)
    return df


def save_table_as_pickle(df: pd.DataFrame, run: str, folder_path: str=None):
    """ Pickles the data frame that represents the result of an integral minimization run.

    Note:
        The argument folder_path can be used to overwrite the normaly expected path deduced from run. It is used for
        testing.

    Args:
        df (pd.DataFrame): Dataframe that should be pickled.
        run (str): Name of the run in which the df was generated.
        folder_path (str): Can be used to overwrite the default place to save the pickled object.

    Raises:
        Exception if the folder 'results/integrals/{run}' does not exist.
    """
    folder_path = f"results/integrals/{run}" if folder_path is None else folder_path
    if not os.path.exists(folder_path):
        raise Exception(f"Tried to save table to a folder {folder_path} that does not exist.")
    df.to_pickle(f"{folder_path}/results.pkl")
    return


def load_table_from_pickle(run: str, folder_path: str=None) -> pd.DataFrame:
    """ Retrieves the data frame that represents the result of an integral minimization run from a pickle object.

    Note:
        The argument folder_path can be used to overwrite the normally expected path deduced from run. It is used for
        testing.

    Args:
        run (str): Name of the run to which we want to get the result.
        folder_path (str): Overwrite the usual path at which we expect the file results.pkl.

    Returns:
        The data frame with the results of the run.

    Raises:
        An exception if the file 'results/integrals/{run}/results.pkl' does not exist.
    """
    folder_path = f"results/integrals/{run}" if folder_path is None else folder_path
    filepath = f"{folder_path}/results.pkl"
    if not os.path.isfile(filepath):
        raise Exception(f"Expected to find the file {filepath} but it was not there.")
    df = pd.read_pickle(filepath_or_buffer=filepath)
    return df


class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder that can deal with np.arrays by converting them to list.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_result_as_json(res_lookup: dict, config: dict, variable_args: list):
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
    filename = construct_filename(
        loss=config["content"]['loss'],
        variable_args=variable_args,
        loss_arg=loss_arg,
        filetype="json"
    )

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


def construct_filename(loss: str, variable_args: list, loss_arg: dict, filetype: str="json") -> str:
    """ Parses the variable arguments and their values and construct a filename to store the result.

    Args:
        loss (str): Name of the loss which was used in the minimization.
        variable_args (list[str]): All the name of the variables which were varied to create many different ansätze.
        loss_arg (dict): Lookup which takes the variable_arg as key and returns its value.
        filetype (str): Ending of the filename.
    """
    variable_args_as_key_value_pairs = '_'.join([f'{key}_{loss_arg[key]}' for key in variable_args])
    filename = f"{loss}_{variable_args_as_key_value_pairs}.{filetype}"
    return filename


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
            save_result_as_json(
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
        save_result_as_json(
            res_lookup=res_lookup,
            config=config,
            variable_args=list(config["content"]["variable_args"].keys())
        )
        results.append(res_lookup)
    logger.info("Finished optimization.")
    return results
